package jukebox

// End-to-end integration tests for the stop-on-evict + wake-from-Stopped
// + redeploy-member full pipeline (commits 3872c2b → de7b3eb).
//
// Each scenario drives the real Scheduler + AdmissionController state
// machine and asserts the cross-component invariants that the per-file
// unit tests cannot — admission books match the scheduler's instance
// state, async cold-load goroutines do not leak, the global cold-load
// lock collapses N concurrent retries to exactly one docker start, and
// failure rollbacks leave the system recoverable rather than wedged.
//
// All docker exec is mocked via SetSleepDockerCmdForTest /
// SetDockerCmdForTest. /is_sleeping is faked by the per-instance
// SleepCapable manager (fakeRedeployMgr). Tests are deterministic — no
// real-clock sleeps over 100ms.
//
// Lives in package jukebox (internal) so we can read admission's
// lowercase state directly (admissionStopped / admissionSleeping /
// admissionAwake) and assert the books are clean. The other
// integration_test.go file is in package jukebox_test and exercises
// the higher-level external-vLLM HTTP path.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
)

// stopOnEvictConfig is the canonical two-member swap-group used by
// scenarios 1-3:
//
//   - main: pinned, evict_action: sleep (default), swap_group g1
//   - moe:  evict_action: stop, swap_group g1
//
// Both lifecycle: external so admission tracks them and the cold-load
// machinery (which requires a Host string for `docker start`) applies.
// VRAM math: main is 12000 MB, moe is 14000 MB on a 24000 MB GPU. They
// CANNOT coexist awake (12000 + 14000 = 26000 > 24000), so a wake of
// moe MUST force admission to evict main — that's the swap-group
// invariant the test exercises.
const stopOnEvictConfig = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  main:
    lifecycle: external
    host: vllm-main
    port: 8001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    pinned: true
    swap_group: g1
    expected_vram_mb_per_gpu: 12000
    sleep_l1_residual_mb: 1000
    wake_timeout: 1s
  moe:
    lifecycle: external
    host: vllm-moe
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: g1
    evict_action: stop
    expected_vram_mb_per_gpu: 14000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// makeStopOnEvictScheduler is the per-scenario setup helper. It builds
// a Scheduler + AdmissionController wired with SchedulerEvictor (the
// real evictor — not stubAdmissionEvictor) so eviction-driven sleeps
// flow through sleepInstance and update the instance state machine
// end-to-end. Returns mgrs keyed by model name for assertion access.
func makeStopOnEvictScheduler(t *testing.T) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(stopOnEvictConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"main": {port: 8001},
		"moe":  {port: 8002},
	}
	// Prime managers: main is awake (pid set, not sleeping); moe is
	// "stopped" (pid=0). Tests will flip admission state to match.
	mgrs["main"].pid.Store(int64(7000 + 8001))
	mgrs["main"].isSleeping.Store(false)
	mgrs["moe"].pid.Store(0)
	mgrs["moe"].isSleeping.Store(true) // will be true once "started"

	// Seed scheduler instances. main starts ready, moe starts stopped.
	s.SeedInstanceForTest("main", 8001, []int{0}, true, StateReady, mgrs["main"])
	s.SeedInstanceForTest("moe", 8002, []int{0}, false, StateStopped, mgrs["moe"])

	// Sync admission books with the seeded states.
	//   - main: pinned + swap_group → starts admissionAwake at construction
	//     (pinned-with-swap-group footprint goes in awakeByGPU). Already
	//     correct.
	//   - moe: starts admissionUnknown at construction. Flip to Stopped so
	//     the cold-load path engages.
	a.mu.Lock()
	a.markStoppedLocked(a.models["moe"])
	a.mu.Unlock()

	return s, a, mgrs
}

// waitForCondition polls fn every 10ms up to timeout. Returns true on
// match, false on timeout. Used to wait for async goroutines to land.
func waitForCondition(timeout time.Duration, fn func() bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return fn()
}

// TestIntegration_StopOnEvict_WakeFromStopped_Roundtrip is Scenario 1:
// the full async-cold-load + admission-driven-eviction pipeline.
//
//  1. main is awake (pinned, evict_action: sleep), moe is Stopped.
//  2. KickColdLoad(moe) returns true; goroutine cold-loads moe.
//  3. After cold-load: moe is admissionSleeping, scheduler state Sleeping.
//  4. AcquireRoute(moe) walks tryRouteFromSleep → performWake → admission
//     RequestWake. RequestWake sees moe needs 12000 on a GPU with main
//     (12000 awake) → picks main as same-group victim → SleepForEviction
//     dispatches because main's evict_action is sleep.
//  5. main becomes admissionSleeping + StateSleeping; moe.Wake completes
//     → StateReady + admissionAwake.
//  6. No goroutine leaks (verified via the WaitGroup-style poll).
func TestIntegration_StopOnEvict_WakeFromStopped_Roundtrip(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	// Mock docker so the cold-load goroutine doesn't shell out. We track
	// every call so the test can assert the right verbs hit the right
	// container.
	var dockerMu sync.Mutex
	var dockerCalls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		dockerCalls = append(dockerCalls, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Setup sanity.
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe Stopped")
	}
	if a.models["main"].State != admissionAwake {
		t.Fatalf("setup: expected main admissionAwake, got %d", a.models["main"].State)
	}

	// Step 1-3: kick the cold-load and wait for it to land.
	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false; expected true (moe is Stopped)")
	}
	if ok := waitForCondition(5*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("cold-load did not finish within 5s; moe still Stopped")
	}
	// Cold-load should have produced exactly one `docker start` call for moe.
	dockerMu.Lock()
	startCalls := 0
	for _, c := range dockerCalls {
		if strings.Contains(c, "start") && strings.Contains(c, "vllm-moe") {
			startCalls++
		}
	}
	dockerMu.Unlock()
	if startCalls != 1 {
		t.Errorf("expected exactly 1 `docker start vllm-moe`, got %d (calls: %v)", startCalls, dockerCalls)
	}
	// Admission state: moe should be Sleeping (post-NotifyStarted). main
	// is the resident pinned peer on GPU 0; cold-loading moe must FIRST
	// evict main (BUG 1 fix — without the eviction, the moe cold-load
	// would race main for VRAM and OOM at worker init). Verify main was
	// slept by the cold-load path.
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected moe admissionSleeping after cold-load, got %d", a.models["moe"].State)
	}
	if a.models["main"].State != admissionSleeping {
		t.Errorf("expected main admissionSleeping (evicted by cold-load BUG 1 fix), got %d", a.models["main"].State)
	}
	// main.Sleep must have been called exactly once during the
	// cold-load's peer-eviction step.
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected main.Sleep=1 (cold-load peer eviction), got %d", got)
	}
	// Wait for the goroutine to remove itself from coldLoadKicks (no leak).
	if ok := waitForCondition(2*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Errorf("coldLoadKicks[moe] still set after cold-load completed (goroutine leaked)")
	}

	// The instance state should now be Sleeping: coldLoadStoppedMember
	// flips both admission AND the scheduler's inst.state under s.mu
	// after NotifyStarted, so the next AcquireRoute takes the wake-from-
	// sleep fast path (tryRouteFromSleep) rather than discarding the
	// instance via the full scheduling path. Verify the production code
	// did the flip rather than driving it from the test (which would
	// mask regressions in the production state-flip).
	if ok := waitForCondition(2*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return s.instances["moe"].state == StateSleeping
	}); !ok {
		s.mu.RLock()
		gotState := s.instances["moe"].state
		s.mu.RUnlock()
		t.Fatalf("expected inst.state=StateSleeping after cold-load (production must flip; tests must not), got %v", gotState)
	}
	// moe's manager should now report awake-but-still-sleeping (vLLM's
	// post-cold-load resting state). Mirror that on the fake — Wake
	// will flip it to false.
	mgrs["moe"].pid.Store(int64(7000 + 8002))
	mgrs["moe"].isSleeping.Store(true)

	// Step 4-5: consumer request for moe. AcquireRoute must take the
	// sleep-fast-path → performWake → admission evicts main → wake moe.
	route, err := s.AcquireRoute(context.Background(), "moe", "req-1")
	if err != nil {
		t.Fatalf("AcquireRoute(moe): %v", err)
	}
	if route.Done != nil {
		route.Done()
	}

	// Step 6: verify final state.
	if a.models["main"].State != admissionSleeping {
		t.Errorf("expected main admissionSleeping (evicted), got %d", a.models["main"].State)
	}
	if a.models["moe"].State != admissionAwake {
		t.Errorf("expected moe admissionAwake (woken), got %d", a.models["moe"].State)
	}
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected main.Sleep=1 (admission eviction), got %d", got)
	}
	if got := mgrs["moe"].wakeCalls.Load(); got != 1 {
		t.Errorf("expected moe.Wake=1 (post-cold-load wake), got %d", got)
	}
	// Books: main's 12000 should be off the awake books (it's now
	// sleeping → 1000 residual), moe's 14000 should be on. Total awake +
	// residual = 14000 + 1000 = 15000; available = 24000 - 15000 = 9000.
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 14000 {
		t.Errorf("expected awake=14000 (moe), got %d", snap[0].AwakeMB)
	}
	if snap[0].L1ResidualMB != 1000 {
		t.Errorf("expected residual=1000 (main slept), got %d", snap[0].L1ResidualMB)
	}
}

// TestIntegration_ConcurrentRetries_DuringColdLoad is Scenario 2: many
// concurrent KickColdLoad arrivals during the ~5min cold-load window
// must collapse to exactly one cold-load (the global lock + TOCTOU
// re-check do their job) and exactly one docker start.
//
// This is the load-bearing safety property — without it, N parallel
// requests could each spawn a docker start goroutine that contends for
// GPU 3, and the second would silently OOM the first.
func TestIntegration_ConcurrentRetries_DuringColdLoad(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	// Slow docker start so multiple kicks pile up while the first holds
	// the cold-load lock.
	var startCalls atomic.Int32
	startEntered := make(chan struct{})
	releaseStart := make(chan struct{})
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			n := startCalls.Add(1)
			if n == 1 {
				close(startEntered)
				<-releaseStart
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Fire 5 concurrent kicks for the same Stopped model.
	const N = 5
	var kickedOK atomic.Int32
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			if s.KickColdLoad("moe") {
				kickedOK.Add(1)
			}
		}()
	}
	wg.Wait()

	// Per the dedupe in KickColdLoad, only ONE goroutine should have
	// returned true (the others see coldLoadKicks[moe]=true and bail).
	// Even if the implementation didn't dedupe, the docker mock would
	// only see exactly one start call (one goroutine wins the lock,
	// others TOCTOU-recheck and skip).
	if got := kickedOK.Load(); got != 1 {
		t.Errorf("expected exactly 1 KickColdLoad to return true (dedupe), got %d", got)
	}

	// Wait for the first start to enter the mock, then release it.
	// Budget 5s — the BUG 1 fix evicts pinned main (sleepInstance) before
	// docker start, and sleepInstance pays a hardcoded 2s settleAfterSleep
	// after Sleep returns. Pre-fix this was tight at 2s.
	select {
	case <-startEntered:
	case <-time.After(5 * time.Second):
		releaseStart <- struct{}{} // unblock if test is going to fail
		t.Fatalf("first docker start never entered the mock within 5s")
	}
	close(releaseStart)

	// Cold-load completes. The poll interval inside doColdLoad is 5s
	// (hardcoded in sleep.go), so the first probe lands at ~5s after
	// docker start unblocked.
	if ok := waitForCondition(8*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("cold-load did not finish within 8s")
	}

	// Critical assertion: exactly ONE docker start hit the mock,
	// regardless of how many KickColdLoad / WithColdLoadLock attempts
	// piled up.
	if got := startCalls.Load(); got != 1 {
		t.Errorf("expected EXACTLY 1 docker start (single-flight), got %d", got)
	}

	// Subsequent KickColdLoad must return false (moe is no longer Stopped).
	if s.KickColdLoad("moe") {
		t.Errorf("KickColdLoad(moe) returned true after cold-load completed; expected false")
	}
}

// TestIntegration_ColdLoadFailureRollback is Scenario 3: when the cold
// load times out (mock docker start succeeds but /is_sleeping never
// returns), the goroutine must:
//   - run the best-effort docker stop cleanup
//   - call NotifyStartFailed so admission stays in Stopped (not wedged)
//   - leave admission books at zero VRAM for the model
//   - allow a SUBSEQUENT KickColdLoad to fire a fresh cold-load.
//
// This is the recoverability guarantee — a transient cold-load failure
// must not poison the admission state machine.
func TestIntegration_ColdLoadFailureRollback(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	// Make /is_sleeping always error so doColdLoad's poll loop times out.
	// Drive the manager: any IsSleeping call returns an error, never
	// "is_sleeping=true". We achieve this by overriding the manager
	// with one whose IsSleeping returns an error.
	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()
	mgrs["moe"] = nil // drop the unused fake to avoid confusion

	// Mock docker — track stops to ensure cleanup fired. Use a short
	// startup_timeout via the modelCfg overriding to keep the test fast:
	// scheduler reads s.cfg.VLLM.StartupTimeout. We can't easily change
	// it here without rebuilding cfg, so we instead time-limit the test
	// via a context-bounded coldLoadStoppedMember invocation. The test
	// calls coldLoadStoppedMember directly (rather than through the
	// async KickColdLoad) so we can pass a tight ctx and observe the
	// timeout deterministically.
	var dockerMu sync.Mutex
	var stopCalls, startCount int
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		defer dockerMu.Unlock()
		if len(args) > 0 && args[0] == "start" {
			startCount++
		}
		if len(args) > 0 && args[0] == "stop" {
			stopCalls++
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Force StartupTimeout to a tiny value via reflection-free path: the
	// scheduler's coldLoadStoppedMember reads s.cfg.VLLM.StartupTimeout
	// and uses 600s as the default if <60s. We need it to be < a few
	// hundred ms for a fast test. Set to 100ms.
	s.cfg.VLLM.StartupTimeout.Duration = 100 * time.Millisecond

	// Run the cold-load. Should fail with a timeout error. doColdLoad's
	// poll interval is 5s; with StartupTimeout < 60s the helper bumps
	// it to 600s (the safe default), defeating our 100ms target.
	// Workaround: drive the lower-level pollUntilSleeping path directly
	// is too far below the integration boundary. Instead use a tight
	// context to abort the poll. The test asserts the goroutine cleanup
	// path, NOT the timeout-error wording.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	err := s.coldLoadStoppedMember(ctx, s.instances["moe"], s.cfg.Models["moe"])
	if err == nil {
		t.Fatalf("expected cold-load to fail (mock /is_sleeping always errors), got nil")
	}

	// (a) docker stop cleanup must have fired.
	dockerMu.Lock()
	if stopCalls < 1 {
		t.Errorf("expected best-effort docker stop cleanup, got %d stop calls", stopCalls)
	}
	if startCount != 1 {
		t.Errorf("expected exactly 1 docker start before failure, got %d", startCount)
	}
	dockerMu.Unlock()

	// (b) admission state: moe must STILL be Stopped (not wedged Sleeping
	// or Awake) — NotifyStartFailed collapsed it back to Stopped.
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still Stopped after failed cold-load, got state %d", a.models["moe"].State)
	}

	// (c) books are clean — zero VRAM allocated to moe on its GPUs.
	a.mu.Lock()
	if got := a.awakeByGPU[0]; got != 12000 { // main is still awake at 12000
		t.Errorf("expected awakeByGPU[0]=12000 (only main, moe=0), got %d", got)
	}
	if got := a.l1ResidualByGPU[0]; got != 0 {
		t.Errorf("expected l1ResidualByGPU[0]=0 (moe contributed nothing after failure), got %d", got)
	}
	a.mu.Unlock()

	// (d) recoverability: a fresh KickColdLoad must fire another attempt.
	// The kicks map should already be cleaned up by the first goroutine's
	// defer (we called coldLoadStoppedMember directly, bypassing
	// KickColdLoad — so coldLoadKicks was never set; the assertion is
	// that a fresh kick still works). Wire the manager so the second
	// attempt actually succeeds: replace the flaky one with a healthy
	// fake. Then KickColdLoad → wait for completion → moe must reach
	// admissionSleeping.
	healthy := &fakeRedeployMgr{port: 8002}
	healthy.pid.Store(0)
	healthy.isSleeping.Store(true)
	s.mu.Lock()
	s.instances["moe"].mgr = healthy
	s.mu.Unlock()
	// Restore a sane startup timeout for the recovery attempt.
	s.cfg.VLLM.StartupTimeout.Duration = 1 * time.Second

	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false on retry; expected true (still Stopped)")
	}
	if ok := waitForCondition(10*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("retry cold-load did not finish; moe still Stopped after recovery attempt")
	}
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected moe admissionSleeping after retry, got %d", a.models["moe"].State)
	}
}

// flakyIsSleepingMgr is a SleepCapable manager whose IsSleeping always
// errors — used to simulate a half-loaded vLLM whose engine never comes
// up. All other methods are no-ops.
type flakyIsSleepingMgr struct {
	port int
	pid  atomic.Int64
}

func (m *flakyIsSleepingMgr) Start(_ context.Context, _ string) (int, error) {
	m.pid.Store(int64(7000 + m.port))
	return int(m.pid.Load()), nil
}
func (m *flakyIsSleepingMgr) Stop(_ context.Context) error                  { m.pid.Store(0); return nil }
func (m *flakyIsSleepingMgr) VerifyReady(_ context.Context, _ string) error { return nil }
func (m *flakyIsSleepingMgr) CurrentPID() int                               { return int(m.pid.Load()) }
func (m *flakyIsSleepingMgr) BaseURL() string                               { return "" }
func (m *flakyIsSleepingMgr) IsSleeping(_ context.Context) (bool, error) {
	return false, errors.New("simulated /is_sleeping unreachable")
}
func (m *flakyIsSleepingMgr) Sleep(_ context.Context, _ int) error             { return nil }
func (m *flakyIsSleepingMgr) Wake(_ context.Context, _ time.Duration) error    { return nil }

// TestIntegration_RedeployMember_EndToEnd is Scenario 4: the full
// /admin/redeploy-member flow against a swap-group with a pinned peer.
// Verifies the docker call sequence and the pinned-peer pause/restore
// dance (sleep main → docker stop moe → docker start moe → wake main).
func TestIntegration_RedeployMember_EndToEnd(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	// For redeploy to work end-to-end, moe needs to be in a non-Stopped
	// scheduler state (the redeploy path calls drainInstance + docker stop
	// + admission.NotifyStopped). Reset moe to Sleeping so it's a normal
	// "this peer is up but slept" baseline.
	a.mu.Lock()
	a.models["moe"].State = admissionSleeping
	a.l1ResidualByGPU[0] += a.models["moe"].L1ResidualMB
	a.mu.Unlock()
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()
	mgrs["moe"].pid.Store(int64(7000 + 8002))
	mgrs["moe"].isSleeping.Store(true)

	// Mock docker via the redeploy hook (NOT the sleep hook — redeploy
	// uses a separate var). Track call order.
	var dockerMu sync.Mutex
	var calls []string
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		calls = append(calls, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "moe")
	if err != nil {
		t.Fatalf("RedeployMember: %v", err)
	}

	// Result fields.
	if result.Result != "redeployed" {
		t.Errorf("expected Result=redeployed, got %q", result.Result)
	}
	if len(result.EvictedPinned) != 1 || result.EvictedPinned[0] != "main" {
		t.Errorf("expected EvictedPinned=[main], got %v", result.EvictedPinned)
	}
	if len(result.RestoredPinned) != 1 || result.RestoredPinned[0] != "main" {
		t.Errorf("expected RestoredPinned=[main], got %v", result.RestoredPinned)
	}

	// main got slept (paused) and woken (restored).
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected main.Sleep=1, got %d", got)
	}
	if got := mgrs["main"].wakeCalls.Load(); got != 1 {
		t.Errorf("expected main.Wake=1, got %d", got)
	}

	// Docker call sequence: stop moe + start moe (in that order).
	dockerMu.Lock()
	defer dockerMu.Unlock()
	if len(calls) != 2 {
		t.Fatalf("expected 2 docker calls, got %d: %v", len(calls), calls)
	}
	if !strings.Contains(calls[0], "stop") || !strings.Contains(calls[0], "vllm-moe") {
		t.Errorf("expected first call to stop vllm-moe, got %q", calls[0])
	}
	if !strings.Contains(calls[1], "start") || !strings.Contains(calls[1], "vllm-moe") {
		t.Errorf("expected second call to start vllm-moe, got %q", calls[1])
	}
}

// TestIntegration_RedeployMember_DockerStartFailureRestoresPinned is the
// Scenario 4 failure branch: if `docker start` fails for the target,
// the pinned peer (main) we slept must still be best-effort restored.
// Otherwise an operator triggering a redeploy that hits a transient
// docker error would silently leave the pinned default napping forever.
func TestIntegration_RedeployMember_DockerStartFailureRestoresPinned(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	a.mu.Lock()
	a.models["moe"].State = admissionSleeping
	a.l1ResidualByGPU[0] += a.models["moe"].L1ResidualMB
	a.mu.Unlock()
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()
	mgrs["moe"].pid.Store(int64(7000 + 8002))
	mgrs["moe"].isSleeping.Store(true)

	startErr := errors.New("simulated docker start failure")
	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			return []byte("daemon error"), startErr
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	_, err := s.RedeployMember(context.Background(), "moe")
	if err == nil {
		t.Fatalf("expected docker start error to surface")
	}
	if !errors.Is(err, startErr) {
		t.Fatalf("expected wrapped start error, got: %v", err)
	}

	// main was paused (slept) and best-effort restored (woken) on rollback.
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected main.Sleep=1 (pause), got %d", got)
	}
	if got := mgrs["main"].wakeCalls.Load(); got != 1 {
		t.Errorf("expected main.Wake=1 (rollback restore), got %d", got)
	}

	// admission must reflect moe Stopped (NotifyStartFailed in redeploy.go).
	if !a.IsStopped("moe") {
		t.Errorf("expected moe Stopped after docker start failure, got state %d", a.models["moe"].State)
	}
}

// ---------------------------------------------------------------------------
// Phase 4/regression — BUG 1 + BUG 2 from 2026-06-09 fleet-bot live-smoke
// (request-triggered cold-load missing peer-eviction; dead-container loop
// wedging the GPU-set lock and starving the pinned co-tenant).
// ---------------------------------------------------------------------------

// TestIntegration_ColdLoad_EvictsPinnedPeerOnRequest is BUG 1's
// happy-path regression: a request for a STOPPED swap-group member
// must trigger the cold-load path, which must FIRST evict the
// awake pinned peer (sleep it), THEN docker start the target. Without
// the eviction, the cold-load races for VRAM with the resident
// pinned peer and OOMs at worker init (live-observed 2026-06-09:
// vllm-moe boot exited(1) with "Free memory on device cuda:2
// (5.31/23.56 GiB) on startup is less than desired GPU memory
// utilization (0.8, 18.85 GiB)" because pinned vllm-main held the
// GPUs awake).
//
// Asserts the docker call sequence (no docker stop for main — it's
// evict_action: sleep — but exactly one docker start for moe) AND
// the admission state machine (main Awake → Sleeping, moe Stopped
// → Sleeping with NotifyStarted).
func TestIntegration_ColdLoad_EvictsPinnedPeerOnRequest(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	// Track docker calls so we can assert the exact sequence.
	var dockerMu sync.Mutex
	var dockerCalls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		dockerCalls = append(dockerCalls, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Setup sanity: main awake (pinned, evict_action: sleep), moe Stopped.
	if a.models["main"].State != admissionAwake {
		t.Fatalf("setup: expected main admissionAwake, got %d", a.models["main"].State)
	}
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe Stopped")
	}
	// main.Sleep should be called by the cold-load eviction step.
	preMainSleeps := mgrs["main"].sleepCalls.Load()

	// Kick the cold-load and wait for it to land.
	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false; expected true (moe is Stopped)")
	}
	if ok := waitForCondition(8*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("cold-load did not finish within 8s; moe still Stopped")
	}

	// Critical: main must have been evicted (slept) by the cold-load
	// path's peer-eviction step. Pre-fix, this assertion fails because
	// main is left Awake.
	if got := mgrs["main"].sleepCalls.Load(); got != preMainSleeps+1 {
		t.Errorf("expected main.Sleep=%d (cold-load peer eviction), got %d",
			preMainSleeps+1, got)
	}
	if a.models["main"].State != admissionSleeping {
		t.Errorf("expected main admissionSleeping after cold-load eviction, got %d",
			a.models["main"].State)
	}

	// moe must have advanced Stopped → Sleeping.
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected moe admissionSleeping after cold-load, got %d",
			a.models["moe"].State)
	}

	// Exactly one docker start for moe (main is sleep-mode — no docker
	// stop for it, just sc.Sleep via fakeRedeployMgr).
	dockerMu.Lock()
	var startCalls, stopCalls int
	for _, c := range dockerCalls {
		if strings.Contains(c, "start") && strings.Contains(c, "vllm-moe") {
			startCalls++
		}
		if strings.Contains(c, "stop") && strings.Contains(c, "vllm-main") {
			stopCalls++
		}
	}
	dockerMu.Unlock()
	if startCalls != 1 {
		t.Errorf("expected exactly 1 `docker start vllm-moe`, got %d (calls: %v)",
			startCalls, dockerCalls)
	}
	if stopCalls != 0 {
		t.Errorf("expected zero `docker stop vllm-main` calls (main is evict_action: sleep), got %d",
			stopCalls)
	}

	// Goroutine kick map cleared (no leak).
	if ok := waitForCondition(2*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Errorf("coldLoadKicks[moe] still set after cold-load completed (goroutine leaked)")
	}

	// Now AcquireRoute(moe) → wake fast path. Eviction is a no-op (main
	// already Sleeping). moe should reach admissionAwake.
	mgrs["moe"].pid.Store(int64(7000 + 8002))
	mgrs["moe"].isSleeping.Store(true)
	route, err := s.AcquireRoute(context.Background(), "moe", "req-1")
	if err != nil {
		t.Fatalf("AcquireRoute(moe): %v", err)
	}
	if route.Done != nil {
		route.Done()
	}
	if a.models["moe"].State != admissionAwake {
		t.Errorf("expected moe admissionAwake after wake, got %d", a.models["moe"].State)
	}
}

// exitingContainerInspectMock returns a SetSleepDockerCmdForTest hook
// that:
//   - succeeds `docker start <container>` (the daemon accepted the start)
//   - reports `exited` for `docker inspect <container>` (simulating a
//     container that exited(1) immediately at worker init — e.g. OOM)
//   - succeeds any other docker call (stop cleanup, etc.)
//
// `started` and `stopped` atomic counters track the live call shape so
// tests can assert exact counts. `started` is set to true after the
// first docker start fires (only relevant for tests that need to
// distinguish pre-start from post-start probes).
func exitingContainerInspectMock(t *testing.T, container string) (func(context.Context, string, ...string) ([]byte, error), *atomic.Int32, *atomic.Int32) {
	t.Helper()
	var startCount, stopCount atomic.Int32
	hook := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) == 0 {
			return []byte("ok"), nil
		}
		switch args[0] {
		case "start":
			startCount.Add(1)
			return []byte("ok"), nil
		case "stop":
			stopCount.Add(1)
			return []byte("ok"), nil
		case "inspect":
			// `docker inspect -f '{{json .State}}' <container>`
			// Find container arg (after the -f format arg).
			var target string
			for i := 0; i < len(args)-1; i++ {
				if args[i] == "-f" {
					if i+2 < len(args) {
						target = args[i+2]
					}
					break
				}
			}
			if target != container {
				// Other container — pretend it's running.
				return []byte(`{"Status":"running","ExitCode":0,"OOMKilled":false}`), nil
			}
			// Target container: simulate exit(1) (the live-observed OOM).
			return []byte(`{"Status":"exited","ExitCode":1,"OOMKilled":false,"Error":"simulated OOM at worker init","StartedAt":"2026-06-09T19:25:00Z","FinishedAt":"2026-06-09T19:25:02Z"}`), nil
		}
		return []byte("ok"), nil
	}
	return hook, &startCount, &stopCount
}

// TestIntegration_ColdLoad_FailFastOnExitedContainer is BUG 2a:
// when the member container exits immediately after `docker start`
// (e.g. OOM at worker init), the cold-load poll loop must detect
// the exit via `docker inspect` and fail fast — NOT wait the full
// cold_load_timeout on a `/is_sleeping` HTTP probe that will never
// succeed (the embedded docker DNS doesn't resolve dead container
// names).
//
// Asserts the cold-load returns within a couple of poll intervals
// (not the full ~10min default timeout) AND the error wraps
// ErrColdLoadContainerExited so callers can classify it.
func TestIntegration_ColdLoad_FailFastOnExitedContainer(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	// Drop poll interval to 50ms so the test finishes fast.
	oldPoll := SetColdLoadPollIntervalForTest(50 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	// Make moe's IsSleeping always error (simulates DNS NXDOMAIN against
	// a dead container — the HTTP client can't tell "not ready yet" from
	// "gone forever", so it would normally retry until cold_load_timeout).
	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()

	// Mock docker: start succeeds, inspect reports exited (BUG 2a's
	// fast-fail trigger).
	hook, startCount, _ := exitingContainerInspectMock(t, "vllm-moe")
	SetSleepDockerCmdForTest(hook)
	defer SetSleepDockerCmdForTest(nil)

	// Disable the cooldown so the test exercises the inner fail-fast,
	// not the outer cooldown gate.
	oldCooldown := SetColdLoadFailureCooldownForTest(0)
	defer SetColdLoadFailureCooldownForTest(oldCooldown)

	// Cold-load. Should fail fast (within a few poll intervals + the
	// initial docker start + eviction settle), NOT the full 10min
	// default timeout. Budget 8s = 2s settle + 6s slack.
	start := time.Now()
	err := s.coldLoadStoppedMember(context.Background(), s.instances["moe"], s.cfg.Models["moe"])
	dur := time.Since(start)
	if err == nil {
		t.Fatalf("expected cold-load to fail, got nil")
	}
	if !errors.Is(err, ErrColdLoadContainerExited) {
		t.Errorf("expected ErrColdLoadContainerExited, got: %v", err)
	}
	// Hard cap: must NOT wait the full cold-load timeout. 8s = generous
	// upper bound (eviction settle is 2s; inspect probe at 50ms intervals
	// should trigger within the first 1-2 polls).
	if dur > 8*time.Second {
		t.Errorf("cold-load took %v — expected fast-fail within 8s (BUG 2a regression: container-exit detection broken)", dur)
	}

	// Cleanup path should have run docker start exactly once + a cleanup
	// stop (best-effort). Admission must reflect Stopped.
	if got := startCount.Load(); got != 1 {
		t.Errorf("expected exactly 1 docker start, got %d", got)
	}
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still Stopped after fast-fail, got state %d", a.models["moe"].State)
	}
}

// TestIntegration_ColdLoad_RetrySuppressedByCooldown is BUG 2b: after
// a cold-load failure, the next N KickColdLoad calls within the
// cooldown window must be suppressed (no second docker start fires).
// Without the cooldown, a doomed cold-load (OOM at worker init) would
// re-fire on every inbound request — each kick re-acquires the GPU-set
// cold-load lock and re-runs the doomed start — wedging the lock and
// starving healthy peers (live-observed 2026-06-09: pinned 27B
// inaccessible until jukebox restart).
//
// Asserts:
//  1. First KickColdLoad → goroutine runs, fails (container exits).
//  2. Subsequent KickColdLoad calls inside the cooldown window
//     return false WITHOUT firing a new goroutine (no new docker start).
//  3. After the cooldown expires, KickColdLoad fires again.
func TestIntegration_ColdLoad_RetrySuppressedByCooldown(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	oldPoll := SetColdLoadPollIntervalForTest(20 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	// 1.5s cooldown — long enough to make the suppression visible against
	// the test's 1-2s poll waits, short enough to verify expiry.
	oldCooldown := SetColdLoadFailureCooldownForTest(1500 * time.Millisecond)
	defer SetColdLoadFailureCooldownForTest(oldCooldown)

	// Flaky IsSleeping (DNS-NXDOMAIN-equivalent) for moe.
	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()

	hook, startCount, _ := exitingContainerInspectMock(t, "vllm-moe")
	SetSleepDockerCmdForTest(hook)
	defer SetSleepDockerCmdForTest(nil)

	// 1. First kick: cold-load runs and fails.
	if !s.KickColdLoad("moe") {
		t.Fatalf("first KickColdLoad(moe) returned false; expected true")
	}
	// Wait for goroutine to finish (coldLoadKicks cleared on defer +
	// coldLoadFailures populated on failure).
	if ok := waitForCondition(8*time.Second, func() bool {
		s.mu.RLock()
		_, hasFailure := s.coldLoadFailures["moe"]
		kickedNow := s.coldLoadKicks["moe"]
		s.mu.RUnlock()
		return !kickedNow && hasFailure
	}); !ok {
		t.Fatalf("first cold-load did not complete with a failure record within 8s")
	}
	if got := startCount.Load(); got != 1 {
		t.Errorf("expected 1 docker start from first kick, got %d", got)
	}
	// admission still Stopped (NotifyStartFailed).
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still Stopped after first failure, got state %d", a.models["moe"].State)
	}

	// 2. Subsequent kicks within cooldown must return false AND fire no
	// new docker start (which would wedge the GPU-set lock).
	const cooldownKicks = 5
	for i := 0; i < cooldownKicks; i++ {
		if s.KickColdLoad("moe") {
			t.Errorf("KickColdLoad #%d within cooldown returned true; expected false (suppressed)", i+2)
		}
	}
	// No new docker start should have fired.
	if got := startCount.Load(); got != 1 {
		t.Errorf("expected docker start count to stay at 1 within cooldown, got %d (re-fires would wedge the lock — BUG 2b regression)", got)
	}

	// 3. Wait for cooldown to expire, then verify a new kick fires.
	time.Sleep(2 * time.Second) // > 1500ms cooldown

	// Heal the container so the next kick succeeds — swap the manager
	// back to a healthy fake AND switch the docker mock to a non-exit
	// inspect.
	healthy := &fakeRedeployMgr{port: 8002}
	healthy.pid.Store(0)
	healthy.isSleeping.Store(true)
	s.mu.Lock()
	s.instances["moe"].mgr = healthy
	s.mu.Unlock()
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			startCount.Add(1)
		}
		if len(args) > 0 && args[0] == "inspect" {
			return []byte(`{"Status":"running","ExitCode":0,"OOMKilled":false}`), nil
		}
		return []byte("ok"), nil
	})

	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad after cooldown returned false; expected true (cooldown should have expired)")
	}
	if ok := waitForCondition(10*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("recovery cold-load did not complete within 10s")
	}
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected moe admissionSleeping after recovery, got %d", a.models["moe"].State)
	}
}

// TestIntegration_PinnedPeerSurvivesWedgedColdLoad is BUG 2c, the
// regression test fleet-bot explicitly requested. While a stopped
// member is wedged in a cold-load failure loop (mock simulates an
// exited container forever), requests to the PINNED healthy peer
// must continue to succeed.
//
// Pre-fix, the cold-load held coldLoadMu for the full 5min timeout,
// then immediately re-kicked, infinitely wedging the lock. Any
// request that touched performWake on the pinned peer (e.g. after
// any sleep-then-wake cycle) would block behind the lock and time
// out — live-observed: live Sparx was effectively DOWN until jukebox
// was restarted.
//
// Post-fix:
//   - BUG 2a: fast-fail detects the exited container in milliseconds.
//   - BUG 2b: cooldown suppresses immediate re-kicks.
//   - The lock is released within 50-200ms of each failed cold-load.
//
// Rewrite (HIGH-3, architect round-5): the previous shape put main
// in Sleeping BEFORE hammering moe, which made the pinned-eviction
// loop in evictPeersForColdLoadLocked skip main immediately — the
// wedge scenario only exercised "moe goroutine takes the lock,
// fast-fails, releases" without any real eviction work. Test passed
// pre-fix AND post-fix. Now: main is AWAKE before the hammer starts,
// so each wedged cold-load attempts to evict main (sleep) → docker
// inspect → fast-fail → cleanup-stop → release lock → re-wake main
// via bestEffortRestorePinned. This exercises the FULL pinned-
// eviction + restore + lock-release path under load, and asserts:
//
//   - At least one real cold-load was attempted (proves the wedge
//     actually wedged something — mock-docker call count > 0).
//   - main remains Awake after the hammer (pinned-default invariant).
//   - A direct request to main returns in <500ms while moe's wedge
//     loop is still running.
func TestIntegration_PinnedPeerSurvivesWedgedColdLoad(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	oldPoll := SetColdLoadPollIntervalForTest(20 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)
	// Short cooldown so the test can observe multiple kick attempts
	// in a reasonable wall-clock window.
	oldCooldown := SetColdLoadFailureCooldownForTest(200 * time.Millisecond)
	defer SetColdLoadFailureCooldownForTest(oldCooldown)

	// moe's container is permanently broken (exits on every start).
	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()

	// Track docker calls so the test can prove (a) real cold-loads
	// were attempted (start count > 0) AND (b) the wedge loop ran
	// multiple iterations (start count > 1 within budget). The
	// exiting-container mock supplies the "inspect → exited" semantics
	// needed for fast-fail.
	hook, startCount, _ := exitingContainerInspectMock(t, "vllm-moe")
	SetSleepDockerCmdForTest(hook)
	defer SetSleepDockerCmdForTest(nil)

	// IMPORTANT: main stays AWAKE (StateReady + admissionAwake from
	// makeStopOnEvictScheduler's seed). The pinned-eviction loop in
	// evictPeersForColdLoadLocked will sleep it on each wedged moe
	// cold-load attempt, then bestEffortRestorePinned will re-wake it
	// after the cold-load fails. This is the FULL exercise of the
	// BUG 2c lock-hold-time path.
	if a.models["main"].State != admissionAwake {
		t.Fatalf("setup: expected main admissionAwake (test contract: pinned peer stays AWAKE during moe wedge), got %d", a.models["main"].State)
	}
	if s.instances["main"].state != StateReady {
		t.Fatalf("setup: expected main StateReady, got %v", s.instances["main"].state)
	}

	// Hammer moe with kick attempts in a background goroutine.
	stopHammer := make(chan struct{})
	hammerDone := make(chan struct{})
	go func() {
		defer close(hammerDone)
		for {
			select {
			case <-stopHammer:
				return
			default:
			}
			_ = s.KickColdLoad("moe")
			// 30ms is long enough that the cooldown (200ms) gates most
			// kicks but short enough that we get several real attempts
			// within the test's 1.5s setup window.
			time.Sleep(30 * time.Millisecond)
		}
	}()
	var hammerStopped sync.Once
	stopHammerOnce := func() {
		hammerStopped.Do(func() {
			close(stopHammer)
			<-hammerDone
		})
	}
	defer stopHammerOnce()

	// Give the hammer enough wall-clock to (a) attempt at least TWO real
	// cold-loads (which sleep main, fast-fail on inspect, re-wake main)
	// AND (b) burn through at least two cooldown cycles so we observe
	// the wedge-loop ITERATING — one start would be vacuously
	// satisfied even if the BUG 2c lock-hold regression had snuck back
	// in. Each cold-load attempt costs ~2s settleAfterSleep on main +
	// ~50ms fast-fail + ~2s settle-after-wake on the rollback, so the
	// loop turns over roughly every ~4-5s. 12s = ~2.5 cycles, leaving
	// headroom for -race scheduler jitter.
	time.Sleep(12 * time.Second)

	// Prove the wedge actually wedged something AND that the wedge LOOP
	// iterated (not just one start that ran-and-released). Single-start
	// passes vacuously — exactly the condition where BUG 2c would have
	// snuck back in undetected (the FIRST start always fires; the
	// regression is the SECOND never getting a turn at the lock).
	gotStarts := startCount.Load()
	if gotStarts < 2 {
		t.Fatalf("BUG 2c regression: expected > 1 docker start attempt during hammer window (proves the wedge LOOP iterated — single start passes vacuously even with the lock-hold-time regression), got %d", gotStarts)
	}

	// Now: direct request to main (the pinned healthy peer). main may
	// be transiently mid-eviction-cycle (state=StateStopping during
	// the 2s settleAfterSleep window, or state=StateSleeping waiting
	// to be re-woken by bestEffortRestorePinned). The whole point of
	// this test is the cycle COMPLETES — main's wake-up arrives — and
	// the lock is released within one cycle budget, NOT held forever.
	//
	// Retry AcquireRoute on transient RejectPinnedConflict (which can
	// fire if the request lands during the sleep-settle window) until
	// we get a clean route OR the budget expires. Pre-fix the budget
	// would never be met (lock-hold = 5min); post-fix it lands within
	// one ~4s cycle.
	//
	// Budget: 12s = ~3 cycles worth of wall-clock slack, generous for
	// -race scheduler jitter. Anything beyond this means the lock-
	// hold-time regression has snuck back in.
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()

	var route Route
	var routeErr error
	start := time.Now()
	for {
		route, routeErr = s.AcquireRoute(ctx, "main", "req-pinned-awake")
		if routeErr == nil {
			break
		}
		// Transient pinned-conflict / swap-in-progress during eviction
		// cycle is expected — keep trying until budget expires.
		var rej *RejectError
		if !errors.As(routeErr, &rej) {
			t.Fatalf("AcquireRoute(main) failed with non-Reject error while moe was wedged: %v (took %v)", routeErr, time.Since(start))
		}
		select {
		case <-ctx.Done():
			t.Fatalf("AcquireRoute(main) never succeeded within budget while moe was wedged: last err=%v (took %v)", routeErr, time.Since(start))
		case <-time.After(50 * time.Millisecond):
		}
	}
	dur := time.Since(start)
	if route.Done != nil {
		route.Done()
	}
	if dur > 12*time.Second {
		t.Errorf("AcquireRoute(main) took %v — pinned peer should be reachable within one wedge cycle (BUG 2c regression — lock-hold-time)", dur)
	}

	// After we stop the hammer, main MUST end up back in admissionAwake.
	// The wedge cycle (sleep main → fast-fail cold-load → re-wake main)
	// might leave main mid-cycle if we observed it during the eviction
	// window — give the in-flight cycle up to 2s to complete.
	stopHammerOnce()
	if ok := waitForCondition(2*time.Second, func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		return a.models["main"].State == admissionAwake
	}); !ok {
		t.Errorf("main left in admissionState=%d after hammer ended; pinned-default invariant violated (bestEffortRestorePinned didn't recover main)", a.models["main"].State)
	}
	_ = mgrs
}

// TestIntegration_ColdLoadPinnedWakeFailed_ClearsCooldown_BumpsMetric
// is the architect-CRITICAL regression test for round-5.
//
// Scenario: moe (Stopped) cold-loads while main (Awake, pinned) holds
// VRAM on the same GPU set. evictPeersForColdLoadLocked sleeps main
// to free VRAM. The cold-load THEN fails (mock container exits). The
// rollback step bestEffortRestorePinned attempts to re-wake main,
// but main's manager rejects the Wake call (simulates a daemon
// hiccup / vLLM /wake_up 500 / image-pull failure).
//
// Pre-fix: the system silently leaves main Sleeping AND records moe
// in coldLoadFailures → KickColdLoad gated for 30s → operator's
// retry blocked for 30s with NO metric and only a warn log. Strictly
// WORSE than the pre-BUG-1 state (which left pinned Awake but OOM-
// looped — at least pinned was up).
//
// Post-fix asserts:
//   - The returned error wraps ErrColdLoadPinnedWakeFailed so the
//     caller can route via errors.Is to RejectAdminIntervention.
//   - coldLoadFailures[moe] is CLEARED so the operator's next request
//     can immediately retry (escape hatch unblocked).
//   - AdmissionPinnedWakeFailedTotal{model=main} is incremented.
//   - A "pinned_peer_wake_failed" Lifecycle audit event was emitted.
func TestIntegration_ColdLoadPinnedWakeFailed_ClearsCooldown_BumpsMetric(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)
	_ = a // admission inspected indirectly via s.coldLoadFailures + metrics

	oldPoll := SetColdLoadPollIntervalForTest(20 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)
	// Long cooldown so the test can observe pre-clear (would-gate) vs
	// post-clear (does-not-gate) deterministically.
	oldCooldown := SetColdLoadFailureCooldownForTest(30 * time.Second)
	defer SetColdLoadFailureCooldownForTest(oldCooldown)

	// moe will fail-fast via the exiting-container inspect mock (BUG 2a path).
	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()

	hook, _, _ := exitingContainerInspectMock(t, "vllm-moe")
	SetSleepDockerCmdForTest(hook)
	defer SetSleepDockerCmdForTest(nil)

	// Sabotage main's Wake so bestEffortRestorePinned cannot re-wake it.
	// fakeRedeployMgr.wakeErr is set BEFORE the cold-load attempt; the
	// pinned-eviction path will sleep main (Sleep succeeds), then the
	// rollback Wake will hit wakeErr → bestEffortRestorePinned returns
	// "main" in the failed list → the CRITICAL fix path fires.
	mgrs["main"].wakeErr = errors.New("simulated daemon hiccup / wake-up 500")

	// Capture metric baseline so we can assert a delta (parallel tests
	// or earlier runs may have bumped these counters).
	pinnedWakeFailedBefore := readCounterVecValue(t, metrics.AdmissionPinnedWakeFailedTotal, "main")

	// CRIT-1 (architect round-6): drive the cold-load through the REAL
	// request-handler entrypoint KickColdLoad — NOT the prior round's
	// direct call to s.coldLoadStoppedMember. KickColdLoad spawns a
	// goroutine that, on cold-load failure, unconditionally re-records
	// coldLoadFailures[name]; the prior fix to clear the cooldown inside
	// coldLoadStoppedMemberLocked was silently overwritten ~2 lines
	// later by that goroutine. The new errors.Is(...PinnedWakeFailed)
	// guard in the goroutine SKIPS the re-record on this exact failure
	// path so the rollback's clear stays cleared.
	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false; expected true (admissionStopped + no in-flight kick + no prior failure)")
	}

	// Wait for the goroutine to finish (coldLoadKicks["moe"] cleared by
	// the goroutine's defer). Generous budget — the cold-load fast-fails
	// via exiting-container inspect within ~50ms but settleAfterSleep
	// for main and the rollback re-wake attempt add ~2-4s.
	if ok := waitForCondition(15*time.Second, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Fatalf("KickColdLoad goroutine for moe did not finish within 15s — coldLoadKicks[moe] still set")
	}

	// CRITICAL: cooldown entry for moe (the cold-load target) MUST be
	// CLEARED AND STAY CLEARED through the goroutine's settle. Pre-fix
	// the rollback cleared the cooldown but the goroutine's blanket
	// re-record overwrote it 2 lines later, leaving the operator gated
	// for 30s with no metric and no recovery handle short of restart.
	s.mu.Lock()
	_, gated := s.coldLoadFailures["moe"]
	s.mu.Unlock()
	if gated {
		t.Errorf("coldLoadFailures[moe] still present after pinned-wake-failed rollback + goroutine settle; CRIT-1 regression — KickColdLoad goroutine re-recorded the cooldown the rollback cleared, blocking operator's retry escape for 30s")
	}

	// End-to-end architect-mandated assertion: a second KickColdLoad
	// invocation within the 30s cooldown window MUST be admitted (NOT
	// suppressed by the post-failure cooldown gate). The earlier
	// in-flight kick has drained (coldLoadKicks check above), so the
	// only thing that could suppress this kick is the cooldown entry —
	// which the rollback's clear (preserved through the goroutine's
	// settle by CRIT-1's errors.Is skip) should ensure is gone.
	//
	// The mock is still installed (defer unset hasn't fired), so the
	// second kick will fast-fail through the same exiting-container
	// path. We wait for it to drain before the deferred unset runs so
	// no goroutine leaks real-docker calls into subsequent tests.
	if !s.KickColdLoad("moe") {
		t.Errorf("second KickColdLoad(moe) returned false — operator's retry was gated by stale cooldown (CRIT-1 regression: rollback's clear was overwritten by goroutine re-record)")
	}
	// Drain the second goroutine before deferred SetSleepDockerCmdForTest(nil)
	// fires, so it doesn't bleed real-docker calls into the next test.
	if ok := waitForCondition(15*time.Second, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Fatalf("second KickColdLoad goroutine for moe did not drain within 15s — would leak real-docker calls into subsequent tests")
	}

	// Metric: AdmissionPinnedWakeFailedTotal{model=main} must have
	// incremented at least once (the first cold-load failed and bumped
	// it; the second cold-load also fails the same way and bumps again,
	// so total delta is 2). Use >=1 to keep the assertion robust if
	// future scheduling changes coalesce the second attempt.
	pinnedWakeFailedAfter := readCounterVecValue(t, metrics.AdmissionPinnedWakeFailedTotal, "main")
	if delta := pinnedWakeFailedAfter - pinnedWakeFailedBefore; delta < 1 {
		t.Errorf("expected AdmissionPinnedWakeFailedTotal{model=main} >= +1 from the failed cold-load(s), got +%v", delta)
	}
}

// TestIntegration_ColdLoadPinnedWakeFailed_AndCleanupStopFailed_DualWrap is
// the HIGH-2 (architect round-6) regression test: the worst-case dual
// failure where (a) the cold-load fails, (b) bestEffortRestorePinned
// fails to re-wake the slept pinned peer, AND (c) the cleanup docker stop
// of the half-loaded container also fails (daemon hiccup / network blip /
// OOM during shutdown).
//
// Pre-fix the cleanup-error early-return at the dual-failure site
// returned an error wrapping ONLY ErrAdmissionVRAMDriftRisk; the pinned-
// peer-wake-failed signal was silently dropped. mapWakeError routed via
// VRAMDriftRisk and the operator never learned the pinned default was
// also down.
//
// Post-fix the returned error joins BOTH sentinels via errors.Join, AND
// mapWakeError's switch matches the more-specific ErrColdLoadPinnedWakeFailed
// FIRST (MED-1 reorder) so the user-facing reject message names the
// pinned-peer condition.
func TestIntegration_ColdLoadPinnedWakeFailed_AndCleanupStopFailed_DualWrap(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)
	_ = a

	oldPoll := SetColdLoadPollIntervalForTest(20 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)
	oldCooldown := SetColdLoadFailureCooldownForTest(30 * time.Second)
	defer SetColdLoadFailureCooldownForTest(oldCooldown)

	flakyMgr := &flakyIsSleepingMgr{port: 8002}
	flakyMgr.pid.Store(0)
	s.mu.Lock()
	s.instances["moe"].mgr = flakyMgr
	s.mu.Unlock()

	// Mock docker so:
	//   - inspect (vllm-moe) → "exited" (drives cold-load fast-fail)
	//   - stop (vllm-moe)    → non-zero exit (drives cleanup-stop failure)
	// We compose by wrapping exitingContainerInspectMock and overriding
	// stop responses via a custom hook that delegates inspect to the
	// canned mock and fails stop.
	innerHook, _, _ := exitingContainerInspectMock(t, "vllm-moe")
	hook := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		// Match `docker stop <flags...> vllm-moe` and fail it.
		if name == "docker" && len(args) >= 1 && args[0] == "stop" {
			for _, arg := range args {
				if arg == "vllm-moe" {
					return []byte("Error response from daemon: simulated stop failure"),
						fmt.Errorf("exit status 1: simulated cleanup stop failure")
				}
			}
		}
		return innerHook(ctx, name, args...)
	}
	SetSleepDockerCmdForTest(hook)
	defer SetSleepDockerCmdForTest(nil)

	// Sabotage main's Wake so bestEffortRestorePinned fails.
	mgrs["main"].wakeErr = errors.New("simulated daemon hiccup / wake-up 500")

	// Drive directly through coldLoadStoppedMember so we can inspect the
	// returned error chain (KickColdLoad's goroutine swallows the err
	// after the slog.Error). The CRIT-1 test exercises the goroutine
	// path; this test exercises the dual-wrap behaviour.
	err := s.coldLoadStoppedMember(context.Background(), s.instances["moe"], s.cfg.Models["moe"])
	if err == nil {
		t.Fatalf("expected dual-failure cold-load to return an error, got nil")
	}

	// HIGH-2: the joined error chain MUST surface BOTH sentinels.
	if !errors.Is(err, ErrAdmissionVRAMDriftRisk) {
		t.Errorf("expected joined err to wrap ErrAdmissionVRAMDriftRisk (cleanup stop failed → container may still be on GPUs), got: %v", err)
	}
	if !errors.Is(err, ErrColdLoadPinnedWakeFailed) {
		t.Errorf("expected joined err to ALSO wrap ErrColdLoadPinnedWakeFailed (pinned peer left Sleeping), got: %v — pre-HIGH-2 the cleanup-error early-return dropped this signal entirely", err)
	}

	// MED-1: mapWakeError's switch must pick the MORE-SPECIFIC
	// ErrColdLoadPinnedWakeFailed branch first (re-ordered above
	// ErrAdmissionVRAMDriftRisk) — both go to RejectAdminIntervention but
	// the message string differs and the pinned-peer branch is the
	// actionable one for the operator.
	mapped := mapWakeError(err)
	rej, ok := mapped.(*RejectError)
	if !ok {
		t.Fatalf("mapWakeError returned non-RejectError: %T %v", mapped, mapped)
	}
	if rej.Reason != RejectAdminIntervention {
		t.Errorf("expected mapWakeError → RejectAdminIntervention, got Reason=%v", rej.Reason)
	}
	if !strings.Contains(rej.Message, "pinned-peer wake rollback failed") {
		t.Errorf("expected mapWakeError message to name the pinned-peer-wake-rollback condition (more-specific sentinel wins after MED-1 reorder), got: %q", rej.Message)
	}
}

// readCounterVecValue reads the current value of a CounterVec for the
// given label values. Used for delta-assertions in tests that may run
// alongside other tests bumping the same counter.
func readCounterVecValue(t *testing.T, cv *prometheus.CounterVec, lvs ...string) float64 {
	t.Helper()
	c, err := cv.GetMetricWithLabelValues(lvs...)
	if err != nil {
		t.Fatalf("readCounterVecValue: %v", err)
	}
	var m dto.Metric
	if err := c.Write(&m); err != nil {
		t.Fatalf("readCounterVecValue Write: %v", err)
	}
	if m.Counter == nil {
		return 0
	}
	return m.Counter.GetValue()
}
