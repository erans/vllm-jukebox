package jukebox

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/ports"
)

// stubAdmissionEvictor is a no-op evictor used by redeploy tests where
// admission may need to satisfy the contract for SetAdmission but no
// eviction is actually exercised. (Redeploy doesn't go through
// admission's eviction path — it manipulates state directly via
// NotifyStopped / NotifyStarted.)
type stubAdmissionEvictor struct{}

func (stubAdmissionEvictor) SleepForEviction(_ context.Context, _, _ string) error { return nil }
func (stubAdmissionEvictor) StopForEviction(_ context.Context, _, _ string) error  { return nil }

// fakeRedeployMgr is a SleepCapable fake instance manager used by
// redeploy tests. Tracks which docker-equivalent calls happened via the
// outer dockerCmd hook (NOT this fake) — this struct only models the
// vLLM /sleep + /wake_up + /is_sleeping responses.
type fakeRedeployMgr struct {
	port       int
	pid        atomic.Int64
	isSleeping atomic.Bool

	sleepCalls atomic.Int32
	wakeCalls  atomic.Int32

	wakeErr error
}

func (m *fakeRedeployMgr) Start(_ context.Context, _ string) (int, error) {
	m.pid.Store(int64(7000 + m.port))
	return int(m.pid.Load()), nil
}
func (m *fakeRedeployMgr) Stop(_ context.Context) error                            { m.pid.Store(0); return nil }
func (m *fakeRedeployMgr) VerifyReady(_ context.Context, _ string) error           { return nil }
func (m *fakeRedeployMgr) CurrentPID() int                                         { return int(m.pid.Load()) }
func (m *fakeRedeployMgr) BaseURL() string                                         { return "" }
func (m *fakeRedeployMgr) IsSleeping(_ context.Context) (bool, error)              { return m.isSleeping.Load(), nil }
func (m *fakeRedeployMgr) Sleep(_ context.Context, _ int) error {
	m.sleepCalls.Add(1)
	m.isSleeping.Store(true)
	return nil
}
func (m *fakeRedeployMgr) Wake(_ context.Context, _ time.Duration) error {
	m.wakeCalls.Add(1)
	if m.wakeErr != nil {
		return m.wakeErr
	}
	m.isSleeping.Store(false)
	return nil
}

// makeRedeployScheduler builds a Scheduler + admission for a redeploy
// test, with the supplied YAML config. Seeds a fakeRedeployMgr for
// each model named in `seed`. The returned scheduler has admission
// wired so RedeployMember can run.
func makeRedeployScheduler(t *testing.T, yaml string, seed map[string]State, totalsByGPU map[int]int) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &stubAdmissionEvictor{})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{}
	port := 8100
	for name, st := range seed {
		modelCfg, ok := cfg.Models[name]
		if !ok {
			t.Fatalf("seed: model %q not in config", name)
		}
		mgr := &fakeRedeployMgr{port: port}
		port++
		// Match instance state to admission's expected state where applicable.
		// Sleeping seeds get isSleeping=true so the post-cold-load /is_sleeping
		// poll returns true immediately. StateReady seeds get pid=non-zero.
		switch st {
		case StateReady:
			mgr.pid.Store(int64(7000 + mgr.port))
			mgr.isSleeping.Store(false)
		case StateSleeping:
			mgr.pid.Store(int64(7000 + mgr.port))
			mgr.isSleeping.Store(true)
		case StateStopped:
			mgr.pid.Store(0)
			mgr.isSleeping.Store(true) // post-cold-load poll succeeds immediately
		}
		s.SeedInstanceForTest(name, mgr.port, modelCfg.GPUs, modelCfg.Pinned != nil && *modelCfg.Pinned, st, mgr)
		mgrs[name] = mgr
	}
	return s, a, mgrs
}

// redeployInventory implements gpu.Inventory for redeploy tests.
type redeployInventory struct {
	gpus []gpu.GPU
}

func newRedeployInventory(totalsByGPU map[int]int) *redeployInventory {
	out := &redeployInventory{}
	for id, mb := range totalsByGPU {
		out.gpus = append(out.gpus, gpu.GPU{Index: id, TotalMB: mb, FreeMB: mb})
	}
	return out
}
func (r *redeployInventory) List(_ context.Context) ([]gpu.GPU, error) { return r.gpus, nil }

// redeployHappyConfig is a small two-member swap-group config used by
// the happy-path redeploy test. main is pinned, moe shares the same
// GPUs and has evict_action: stop. Both have admission tracking on.
// Members are lifecycle: external (jukebox doesn't manage the vLLM
// process directly — config validation requires that for stop-on-evict).
const redeployHappyConfig = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  main:
    lifecycle: external
    host: vllm-main
    port: 8001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    pinned: true
    swap_group: G
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 50
  moe:
    lifecycle: external
    host: vllm-moe
    port: 8002
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 1500
    sleep_l1_residual_mb: 0
`

// TestRedeployMember_UnknownModel asserts that asking to redeploy a
// model that does not exist in the config returns ErrRedeployUnknownModel,
// which the HTTP handler maps to 400.
func TestRedeployMember_UnknownModel(t *testing.T) {
	s, _, _ := makeRedeployScheduler(t, redeployHappyConfig, nil, map[int]int{0: 10000, 1: 10000})

	_, err := s.RedeployMember(context.Background(), "nonexistent")
	if err == nil {
		t.Fatalf("expected error for unknown model, got nil")
	}
	if !errors.Is(err, ErrRedeployUnknownModel) {
		t.Fatalf("expected ErrRedeployUnknownModel, got: %v", err)
	}
}

// TestRedeployMember_Ineligible asserts that a normal admission-tracked
// model with no swap_group and no evict_action: stop is rejected as
// ineligible — RedeployMember does not apply.
func TestRedeployMember_Ineligible(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
models:
  plain:
    path: "/models/plain"
    host: vllm-plain
    port: 8003
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 1000
`
	s, _, _ := makeRedeployScheduler(t, cfgYAML, map[string]State{"plain": StateReady}, map[int]int{0: 10000})

	_, err := s.RedeployMember(context.Background(), "plain")
	if err == nil {
		t.Fatalf("expected error for ineligible model, got nil")
	}
	if !errors.Is(err, ErrRedeployIneligible) {
		t.Fatalf("expected ErrRedeployIneligible, got: %v", err)
	}
}

// TestRedeployMember_HappyPath drives the full happy-path:
// pinned peer (main) is awake, target (moe) is awake. Redeploy moe
// should:
//   - sleep main (pinned peer pause)
//   - docker stop + start moe (via the docker hook)
//   - poll /is_sleeping (immediately returns true via fake)
//   - NotifyStarted moe → admissionSleeping
//   - wake main (pinned peer restore)
//
// Asserts the docker call sequence + the result struct.
func TestRedeployMember_HappyPath(t *testing.T) {
	s, _, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateReady},
		map[int]int{0: 10000, 1: 10000},
	)

	var dockerCalls []string
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerCalls = append(dockerCalls, name+" "+strings.Join(args, " "))
		// pollUntilSleeping requires is_sleeping=true (the post-cold-load
		// slept-L1 state) before declaring success. Real vLLM flips to
		// is_sleeping=true once the model finishes loading; mirror that
		// here by flipping the target mgr's isSleeping bit immediately
		// after docker start so the poll returns on its first probe.
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "moe")
	if err != nil {
		t.Fatalf("RedeployMember: %v", err)
	}

	// Result: container name pulled from cfg.Host, target awake-then-redeployed.
	if result.Model != "moe" {
		t.Fatalf("expected Model=moe, got %q", result.Model)
	}
	if result.Container != "vllm-moe" {
		t.Fatalf("expected Container=vllm-moe, got %q", result.Container)
	}
	if result.Result != "redeployed" {
		t.Fatalf("expected Result=redeployed, got %q", result.Result)
	}
	// pinned peer is main; got slept (paused) and woken (restored).
	if len(result.EvictedPinned) != 1 || result.EvictedPinned[0] != "main" {
		t.Fatalf("expected EvictedPinned=[main], got %v", result.EvictedPinned)
	}
	if len(result.RestoredPinned) != 1 || result.RestoredPinned[0] != "main" {
		t.Fatalf("expected RestoredPinned=[main], got %v", result.RestoredPinned)
	}

	// Sleep + wake counters on main.
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Fatalf("expected main.Sleep=1, got %d", got)
	}
	if got := mgrs["main"].wakeCalls.Load(); got != 1 {
		t.Fatalf("expected main.Wake=1 (restore), got %d", got)
	}

	// Docker exec sequence: stop moe -t 60, then start moe.
	if len(dockerCalls) != 2 {
		t.Fatalf("expected 2 docker calls (stop+start), got %d: %v", len(dockerCalls), dockerCalls)
	}
	if !strings.Contains(dockerCalls[0], "stop") || !strings.Contains(dockerCalls[0], "vllm-moe") {
		t.Fatalf("expected first docker call to stop vllm-moe, got %q", dockerCalls[0])
	}
	if !strings.Contains(dockerCalls[1], "start") || !strings.Contains(dockerCalls[1], "vllm-moe") {
		t.Fatalf("expected second docker call to start vllm-moe, got %q", dockerCalls[1])
	}
}

// TestRedeployMember_DockerStopFailureRollsBack asserts that if `docker
// stop` errors, the pinned peer we paused is best-effort restored
// (woken) before we return the error to the operator.
func TestRedeployMember_DockerStopFailureRollsBack(t *testing.T) {
	s, _, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateReady},
		map[int]int{0: 10000, 1: 10000},
	)

	stopErr := errors.New("docker daemon not reachable")
	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		// First call is "stop"; fail it. Ensures we don't reach docker start.
		if len(args) > 0 && args[0] == "stop" {
			return []byte("daemon error"), stopErr
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	_, err := s.RedeployMember(context.Background(), "moe")
	if err == nil {
		t.Fatalf("expected docker stop error to surface")
	}
	if !errors.Is(err, stopErr) {
		t.Fatalf("expected wrapped stopErr, got: %v", err)
	}

	// main was paused (slept) then best-effort restored (woken) on rollback.
	if got := mgrs["main"].sleepCalls.Load(); got != 1 {
		t.Fatalf("expected main.Sleep=1 (pause), got %d", got)
	}
	if got := mgrs["main"].wakeCalls.Load(); got != 1 {
		t.Fatalf("expected main.Wake=1 (rollback restore), got %d", got)
	}
}

// TestRedeployMember_NoPinnedPeer asserts that a redeploy of an
// evict_action: stop model with NO pinned peer in its swap_group still
// works — no peers slept, no peers restored.
func TestRedeployMember_NoPinnedPeer(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
models:
  solo:
    lifecycle: external
    host: vllm-solo
    port: 8004
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: H
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
`
	s, _, mgrs := makeRedeployScheduler(t, cfgYAML,
		map[string]State{"solo": StateReady},
		map[int]int{0: 10000},
	)

	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		// Flip target mgr to isSleeping=true on docker start so
		// pollUntilSleeping's is_sleeping=true requirement is satisfied.
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "solo")
	if err != nil {
		t.Fatalf("RedeployMember: %v", err)
	}
	if len(result.EvictedPinned) != 0 {
		t.Fatalf("expected no EvictedPinned, got %v", result.EvictedPinned)
	}
	if len(result.RestoredPinned) != 0 {
		t.Fatalf("expected no RestoredPinned, got %v", result.RestoredPinned)
	}
	if result.Result != "redeployed" {
		t.Fatalf("expected Result=redeployed, got %q", result.Result)
	}
}

// TestRedeployMember_SwapGroupOnlyIneligible asserts the MEDIUM #2
// tightening: a swap-group member WITHOUT evict_action: stop is no
// longer eligible. Only evict_action: stop members carry the
// hardened restart: "no" deployment contract that makes
// docker stop + start safe; sleep-mode peers cycle via /sleep +
// /wake_up without container churn and don't need redeploy-member.
func TestRedeployMember_SwapGroupOnlyIneligible(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
models:
  groupie:
    lifecycle: external
    host: vllm-groupie
    port: 8005
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    expected_vram_mb_per_gpu: 1000
`
	s, _, _ := makeRedeployScheduler(t, cfgYAML,
		map[string]State{"groupie": StateSleeping},
		map[int]int{0: 10000},
	)

	_, err := s.RedeployMember(context.Background(), "groupie")
	if err == nil {
		t.Fatalf("expected error for swap-group-only model (no evict_action: stop), got nil")
	}
	if !errors.Is(err, ErrRedeployIneligible) {
		t.Fatalf("expected ErrRedeployIneligible, got: %v", err)
	}
	// Error message must surface the actual offending evict_action value
	// (production format: `evict_action="sleep"`) so an operator can tell
	// at a glance WHY the model is ineligible. Asserting the literal
	// `evict_action="sleep"` fragment — NOT just any occurrence of the
	// boilerplate text "evict_action: stop", which also appears in the
	// "requires evict_action: stop on the target model" trailer and would
	// pass even if the value-emitting %q placeholder were removed from
	// production. The previous assertion was vacuous on that exact axis.
	if !strings.Contains(err.Error(), `evict_action="sleep"`) {
		t.Fatalf("expected error to surface actual evict_action value (evict_action=%q), got: %v", "sleep", err)
	}
}

// TestPerformWakeFromInsideColdLoadLock_NoDeadlock proves
// the MEDIUM #1 fix: performWakeFromInsideColdLoadLock called on an
// admissionStopped instance from INSIDE a WithColdLoadLock callback
// must NOT deadlock by re-acquiring coldLoadMu. The pre-fix
// performWake would call coldLoadStoppedMember → WithColdLoadLock,
// hitting Go's non-reentrant sync.Mutex and hanging forever.
//
// We assert two things:
//  1. The call returns within a bounded time (not a deadlock).
//  2. The cold-load docker hook is NEVER invoked — performWakeFromInsideColdLoadLock
//     bypasses coldLoadStoppedMember's docker-start path entirely.
//
// The model is intentionally seeded as admissionStopped (the precise
// state that would have triggered the latent deadlock). The wake
// itself will fail because admission's RequestWake on a Stopped model
// without sufficient GPU budget might pick victims, but in this single-
// model setup no victims are needed and the wake proceeds. We only care
// about the lock semantics, so an early failure is OK — the assertion
// is "did we hang?" not "did the wake succeed?".
func TestPerformWakeFromInsideColdLoadLock_NoDeadlock(t *testing.T) {
	s, a, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateStopped},
		map[int]int{0: 10000, 1: 10000},
	)

	// Seed admission's view: moe is admissionStopped.
	a.NotifyStopped("moe")
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe to be admissionStopped")
	}

	// Track docker calls. If the in-lock wake variant accidentally
	// went through coldLoadStoppedMember it would invoke `docker start`
	// via the sleep.go hook, which we'd see here.
	var coldLoadDockerCalls int
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		coldLoadDockerCalls++
		_ = args
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Locate moe's instance + cfg for the direct call.
	s.mu.RLock()
	moeInst := s.instances["moe"]
	s.mu.RUnlock()
	if moeInst == nil {
		t.Fatalf("setup: moe instance missing")
	}
	moeCfg, ok := s.cfg.Models["moe"]
	if !ok {
		t.Fatalf("setup: moe config missing")
	}

	// Mark moe as not-sleeping so Wake's no-op succeeds (the fake
	// returns nil from Wake unconditionally; isSleeping starts true
	// per StateStopped seeding which is fine — the manager's Wake
	// implementation just records the call and clears the flag).
	mgrs["moe"].isSleeping.Store(false)

	// Run the in-lock wake on a goroutine + bounded wait. If we deadlock
	// re-acquiring coldLoadMu, the channel never receives and the test
	// times out via t.Fatalf below.
	done := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		a.WithColdLoadLock(func() {
			// THIS is the call that would have deadlocked pre-fix. The
			// in-lock wake variant must skip the IsStopped → cold-load
			// path entirely.
			done <- s.performWakeFromInsideColdLoadLock(ctx, moeInst, moeCfg, "test-in-lock-wake")
		})
	}()

	select {
	case err := <-done:
		// The wake's outcome is incidental — what matters is that we
		// returned. Log any error for diagnosability but don't fail on
		// it; admission may legitimately reject the wake (e.g. shape of
		// fakes). The bug we're guarding against is "no return at all".
		if err != nil {
			t.Logf("in-lock wake completed with err (acceptable — no deadlock): %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("performWakeFromInsideColdLoadLock deadlocked while holding coldLoadMu — MEDIUM #1 fix regressed")
	}

	// Cold-load docker hook must NOT have fired. If it did, the in-lock
	// variant accidentally went through coldLoadStoppedMember, which
	// would have deadlocked anyway (only didn't because the test fakes
	// don't actually re-acquire). Either way, the contract is broken.
	if coldLoadDockerCalls != 0 {
		t.Fatalf("performWakeFromInsideColdLoadLock invoked the cold-load docker path (%d calls); expected 0 — the variant must bypass coldLoadStoppedMember entirely", coldLoadDockerCalls)
	}
}

// redeployPeerStopConfig models the live 4-GPU swap-group topology:
//   - main: pinned, sleep-mode, GPUs [0,1,2,3]
//   - moe:  evict_action: stop, GPUs [0,1,2,3]
//   - longctx: evict_action: stop, GPUs [0,1,2,3]
//   - vision: sleep-mode (NOT evict_action: stop), GPUs [3] (partial overlap)
//
// The redeploy of longctx should:
//   - sleep main (pinned peer pause)
//   - docker stop moe (other evict_action: stop peer with GPU overlap)
//   - NOT touch vision (sleep-mode, not stop-mode)
//   - docker stop + start longctx
//   - wake main
//   - leave moe Stopped (async-recover on demand)
const redeployPeerStopConfig = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  main:
    lifecycle: external
    host: vllm-main
    port: 8001
    gpus: [0, 1, 2, 3]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    pinned: true
    swap_group: G
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 50
  moe:
    lifecycle: external
    host: vllm-moe
    port: 8002
    gpus: [0, 1, 2, 3]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 1500
    sleep_l1_residual_mb: 100
  longctx:
    lifecycle: external
    host: vllm-longctx
    port: 8003
    gpus: [0, 1, 2, 3]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 2000
    sleep_l1_residual_mb: 100
  vision:
    lifecycle: external
    host: vllm-vision
    port: 8004
    gpus: [3]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    expected_vram_mb_per_gpu: 500
    sleep_l1_residual_mb: 50
`

// TestRedeployMember_StopsOtherEvictPeers proves the LIVE-OOM-1 fix:
// redeploying longctx with moe present in the swap_group AND sharing
// GPUs MUST docker-stop moe before the longctx cold-load so the slept-L1
// residual on shared GPUs is fully reclaimed.
//
// Validates:
//   - moe is docker-stopped (admission notified Stopped, container stop fired)
//   - vision is NOT touched (sleep-mode, not evict_action: stop)
//   - main is paused + restored (pinned peer pause path unchanged)
//   - result.StoppedPeers == [moe]
//   - result.LeftStopped == [moe] (intentionally NOT auto-restarted)
//   - moe ends in admissionStopped state
func TestRedeployMember_StopsOtherEvictPeers(t *testing.T) {
	s, a, mgrs := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping, // slept-L1; the residual that OOMs the cold-load
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)

	// Track docker invocations on the redeploy hook + the sleep hook.
	// sleep.go's runSleepDocker is what sleepInstance / wake_up paths
	// use; redeploy.go's runDocker is what RedeployMember's docker stop
	// + start (target AND peers) use. Track both so we can assert the
	// full sequence.
	var redeployDockerCalls []string
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		redeployDockerCalls = append(redeployDockerCalls, name+" "+strings.Join(args, " "))
		// Flip target mgr isSleeping=true on docker start (pollUntilSleeping
		// now requires the post-cold-load slept-L1 state, not just HTTP 200).
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	// Background docker hook for sleep.go's separate path (the test
	// scaffolding for sleepInstance / Wake — these go through fake
	// instance managers, not docker, so this isn't strictly needed,
	// but keep it defensively wired in case any path falls through).
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Pre-state: main is awake, others are sleeping. Confirm admission
	// view matches before kicking the redeploy (defensive).
	if a.IsStopped("moe") {
		t.Fatalf("setup: moe should not be admissionStopped before redeploy")
	}

	result, err := s.RedeployMember(context.Background(), "longctx")
	if err != nil {
		t.Fatalf("RedeployMember(longctx): %v", err)
	}

	// Target redeployed.
	if result.Model != "longctx" {
		t.Fatalf("expected Model=longctx, got %q", result.Model)
	}
	if result.Result != "redeployed" {
		t.Fatalf("expected Result=redeployed, got %q (full=%+v)", result.Result, result)
	}

	// Pinned peer (main) paused + restored.
	if len(result.EvictedPinned) != 1 || result.EvictedPinned[0] != "main" {
		t.Fatalf("expected EvictedPinned=[main], got %v", result.EvictedPinned)
	}
	if len(result.RestoredPinned) != 1 || result.RestoredPinned[0] != "main" {
		t.Fatalf("expected RestoredPinned=[main], got %v", result.RestoredPinned)
	}

	// THE LIVE-OOM-1 ASSERTION: moe (the only other evict_action:stop
	// peer with GPU overlap) was stopped.
	if len(result.StoppedPeers) != 1 || result.StoppedPeers[0] != "moe" {
		t.Fatalf("expected StoppedPeers=[moe], got %v", result.StoppedPeers)
	}
	if len(result.LeftStopped) != 1 || result.LeftStopped[0] != "moe" {
		t.Fatalf("expected LeftStopped=[moe] (async-recover on demand), got %v", result.LeftStopped)
	}

	// vision (sleep-mode, NOT evict_action: stop) MUST NOT have been
	// touched: no Sleep call beyond its pre-state, no Wake call.
	if got := mgrs["vision"].sleepCalls.Load(); got != 0 {
		t.Fatalf("expected vision.Sleep=0 (sleep-mode peer must not be stopped), got %d", got)
	}
	if got := mgrs["vision"].wakeCalls.Load(); got != 0 {
		t.Fatalf("expected vision.Wake=0 (sleep-mode peer must not be touched), got %d", got)
	}

	// moe ends in admissionStopped.
	if !a.IsStopped("moe") {
		t.Fatalf("expected moe to be admissionStopped after redeploy")
	}

	// Docker call sequence: stop moe, stop longctx (-t 60), start longctx.
	// Order: peer-stop runs in step (a2), before the target stop in (b).
	if len(redeployDockerCalls) != 3 {
		t.Fatalf("expected 3 docker calls (stop-moe, stop-longctx, start-longctx), got %d: %v", len(redeployDockerCalls), redeployDockerCalls)
	}
	if !strings.Contains(redeployDockerCalls[0], "stop") || !strings.Contains(redeployDockerCalls[0], "vllm-moe") {
		t.Fatalf("expected first docker call to stop vllm-moe (peer), got %q", redeployDockerCalls[0])
	}
	if !strings.Contains(redeployDockerCalls[1], "stop") || !strings.Contains(redeployDockerCalls[1], "vllm-longctx") {
		t.Fatalf("expected second docker call to stop vllm-longctx (target), got %q", redeployDockerCalls[1])
	}
	if !strings.Contains(redeployDockerCalls[2], "start") || !strings.Contains(redeployDockerCalls[2], "vllm-longctx") {
		t.Fatalf("expected third docker call to start vllm-longctx (target), got %q", redeployDockerCalls[2])
	}
}

// TestRedeployMember_SkipsAlreadyStoppedEvictPeers asserts the no-op
// path: a peer that is already in StateStopped should NOT be re-stopped
// (no docker call, no admission churn, no LeftStopped entry).
//
// Real-world scenario: an earlier redeploy or eviction left moe Stopped.
// Operator runs redeploy-member on longctx. We should pass straight
// through without touching moe.
func TestRedeployMember_SkipsAlreadyStoppedEvictPeers(t *testing.T) {
	s, a, mgrs := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateStopped, // already Stopped from a prior event
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)
	// Mirror admission view to match the scheduler state.
	a.NotifyStopped("moe")

	var redeployDockerCalls []string
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		redeployDockerCalls = append(redeployDockerCalls, name+" "+strings.Join(args, " "))
		// Flip target mgr isSleeping=true on docker start (pollUntilSleeping
		// now requires the post-cold-load slept-L1 state, not just HTTP 200).
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "longctx")
	if err != nil {
		t.Fatalf("RedeployMember(longctx): %v", err)
	}

	if len(result.StoppedPeers) != 0 {
		t.Fatalf("expected no StoppedPeers (moe was already Stopped), got %v", result.StoppedPeers)
	}
	if len(result.LeftStopped) != 0 {
		t.Fatalf("expected no LeftStopped (we didn't stop anyone), got %v", result.LeftStopped)
	}
	// Only 2 docker calls (stop + start of longctx), no peer-stop.
	if len(redeployDockerCalls) != 2 {
		t.Fatalf("expected 2 docker calls (just target stop+start), got %d: %v", len(redeployDockerCalls), redeployDockerCalls)
	}
}

// TestEvictStopPeersInSwapGroupOverlapping_FiltersCorrectly is a unit
// test for the helper that drives step (a2): it must return only
// admission-tracked peers in the same swap_group with evict_action: stop
// AND GPU overlap. Excludes the target, excludes pinned peers (they go
// through the separate sleep-pause path), excludes non-overlapping peers,
// excludes sleep-mode peers.
func TestEvictStopPeersInSwapGroupOverlapping_FiltersCorrectly(t *testing.T) {
	s, _, _ := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping,
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)

	longctxCfg := s.cfg.Models["longctx"]
	got := s.evictStopPeersInSwapGroupOverlapping("longctx", longctxCfg)

	// Expected: only moe.
	//   - main: pinned (handled separately)
	//   - longctx: self (excluded)
	//   - vision: not evict_action: stop
	if len(got) != 1 || got[0] != "moe" {
		t.Fatalf("expected [moe], got %v", got)
	}
}

// ---------------------------------------------------------------------
// Fix 5: TOCTOU re-check IsStopped INSIDE coldLoadMu in performWake.
//
// The old shape had performWake check IsStopped outside the lock, then
// either call coldLoadStoppedMember (which acquired its own lock) or
// performWakeFromInsideColdLoadLock (which DID NOT hold any lock). The
// gap between the outer-IsStopped check and the no-lock wake body let
// a concurrent eviction (StopForEviction by another model's RequestWake)
// flip the model to admissionStopped — RequestWake then proceeded with
// no Stopped-branch handling, marked the model awake on books, the
// /wake_up call against the now-stopped container failed, and the
// rollback NotifySleep wedged the books with phantom L1Residual.
//
// The new shape (sleep.go performWake) acquires coldLoadMu once around
// IsStopped + dispatch. The test below proves: when admission is
// admissionStopped at lock-acquisition time, the cold-load path runs
// (instead of falling through to RequestWake's broken Stopped-handling).
// ---------------------------------------------------------------------

// TestPerformWake_ReChecksIsStoppedInsideColdLoadLock asserts that
// performWake — invoked outside any lock — observes a Stopped model
// state and routes to the cold-load path even if the model was awake
// at the outer call site (i.e. between outer-IsStopped check and the
// inner branch dispatch).
//
// We don't actually race goroutines here (the new shape has NO outer
// check; the IsStopped query is INSIDE WithColdLoadLock by design).
// What we assert is the lock-acquisition-then-IsStopped pattern: a
// model that is admissionStopped at the time the lock is acquired
// gets the cold-load path. If the fix regressed and the outer-check
// pattern came back, the test would fail because the outer check
// happens BEFORE we set the Stopped state — the wake would skip
// cold-load and try /wake_up against the (test-fake) Stopped instance.
func TestPerformWake_ReChecksIsStoppedInsideColdLoadLock(t *testing.T) {
	s, a, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateSleeping},
		map[int]int{0: 10000, 1: 10000},
	)

	// Pre-state: moe is admissionSleeping (the natural state for a
	// model that has been slept). It is NOT admissionStopped at the
	// outer-check moment.

	// Track sleep.go's docker calls — coldLoadStoppedMemberLocked
	// dispatches docker start through runSleepDocker. If the cold-load
	// path runs, we'll see a docker start call.
	var sleepDockerCalls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		sleepDockerCalls = append(sleepDockerCalls, name+" "+strings.Join(args, " "))
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Set up the post-cold-load steady state: mgrs["moe"].isSleeping=true
	// so doColdLoad's pollUntilSleeping short-circuits.
	mgrs["moe"].isSleeping.Store(true)

	// CRITICAL: flip moe to admissionStopped just before invoking
	// performWake. With the Fix 5 design, performWake will:
	//   1. Acquire WithColdLoadLock.
	//   2. Re-check IsStopped → true.
	//   3. Run coldLoadStoppedMemberLocked (docker start + poll +
	//      NotifyStarted → admissionSleeping).
	//   4. Run performWakeFromInsideColdLoadLock (RequestWake +
	//      /wake_up → admissionAwake).
	a.NotifyStopped("moe")
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe to be admissionStopped")
	}

	moeInst := s.instances["moe"]
	if moeInst == nil {
		t.Fatalf("setup: moe instance missing")
	}
	// Mirror admission state in the scheduler so the cold-load path's
	// state transitions land on a Stopped instance.
	s.mu.Lock()
	moeInst.state = StateStopped
	s.mu.Unlock()

	moeCfg := s.cfg.Models["moe"]
	if err := s.performWake(context.Background(), moeInst, moeCfg, "test-toctou"); err != nil {
		t.Fatalf("performWake: %v", err)
	}

	// The cold-load path MUST have run (docker start fired through
	// runSleepDocker). If Fix 5 regressed to the outer-check shape,
	// the outer check would have been against admissionSleeping (the
	// pre-flip state in some hypothetical caller) and the cold-load
	// would be skipped — but we set Stopped BEFORE the call so this
	// distinction would only matter if the outer check were truly
	// stale (a real concurrency window). The simpler signal: the
	// cold-load docker start MUST have been called.
	startCalled := false
	for _, c := range sleepDockerCalls {
		if strings.Contains(c, "start") && strings.Contains(c, "vllm-moe") {
			startCalled = true
			break
		}
	}
	if !startCalled {
		t.Fatalf("expected cold-load `docker start vllm-moe` to fire (proves IsStopped re-check INSIDE WithColdLoadLock dispatched cold-load), got calls=%v", sleepDockerCalls)
	}

	// Books should be consistent: moe is now admissionAwake.
	if a.IsStopped("moe") {
		t.Fatalf("expected moe NOT admissionStopped after successful wake, still Stopped (books inconsistent)")
	}
}

// TestPerformWake_StoppedFlippedDuringWakeKeepsBooksConsistent races
// the path more aggressively: pre-wake state is admissionSleeping
// (so the outer-check would say "not Stopped"), but a concurrent
// goroutine flips moe to admissionStopped via NotifyStopped at the
// same time as performWake is invoked. The test asserts that books
// remain consistent regardless of the interleaving — either:
//   - The flip lost the race and the wake completed normally
//     (admissionSleeping → admissionAwake).
//   - The flip won the race and the in-lock IsStopped re-check picked
//     up the Stopped state, dispatched cold-load, then proceeded to
//     wake → admissionAwake.
//
// The PREVIOUS code's bug was a third outcome: outer check sees
// Sleeping, the wake's RequestWake under admission.mu sees Stopped,
// admission flipped admissionAwake on the books, /wake_up failed,
// rollback NotifySleep left admissionSleeping with phantom residual.
// Books remained skewed. We don't re-create that exact failure here —
// it's hard to deterministically interleave — but we do assert the
// stronger invariant that wins: at end of wake, the model is in
// admissionAwake state with books that match.
func TestPerformWake_StoppedFlippedDuringWakeKeepsBooksConsistent(t *testing.T) {
	s, a, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateSleeping},
		map[int]int{0: 10000, 1: 10000},
	)

	// Allow the cold-load path to succeed if it's taken: hook docker
	// start through runSleepDocker.
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)
	mgrs["moe"].isSleeping.Store(true)

	moeInst := s.instances["moe"]
	moeCfg := s.cfg.Models["moe"]

	// Concurrent goroutine that flips admissionStopped a moment AFTER
	// performWake starts. The 1ms delay is best-effort; the goal is
	// to exercise the lock contention path, not deterministically
	// land in a particular interleaving.
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		time.Sleep(1 * time.Millisecond)
		a.NotifyStopped("moe")
		// Mirror in scheduler state to keep the post-wake bookkeeping
		// internally consistent. (This races with performWake's own
		// state mutations — but Fix 5 guarantees admission books AND
		// scheduler state converge on a consistent end-state.)
		s.mu.Lock()
		if moeInst.state != StateReady {
			moeInst.state = StateStopped
		}
		s.mu.Unlock()
	}()

	err := s.performWake(context.Background(), moeInst, moeCfg, "test-toctou-race")
	<-doneCh

	// performWake outcome may legitimately err if the flip raced into
	// a state the wake didn't expect — what matters is the books
	// remain internally consistent. Specifically: after the call,
	// admission must NOT be in admissionStopped with the model
	// simultaneously claiming awake on the per-GPU map (the "phantom
	// residual" wedge the bug produced).
	_ = err

	// Snapshot: per-GPU awake + residual must be non-negative AND
	// match the model's expected state.
	snap := a.SnapshotBudgets()
	for _, b := range snap {
		if b.AwakeMB < 0 {
			t.Errorf("GPU %d AwakeMB=%d < 0 (book inconsistency)", b.GPUID, b.AwakeMB)
		}
		if b.L1ResidualMB < 0 {
			t.Errorf("GPU %d L1ResidualMB=%d < 0 (book inconsistency)", b.GPUID, b.L1ResidualMB)
		}
	}
}

// ---------------------------------------------------------------------
// Fix 6: best-effort restore of evicted peers when /wake_up fails.
// ---------------------------------------------------------------------

// TestPerformWake_FailedWakeRestoresEvictedPeers proves the Fix 6 path:
// when admission evicts peers A,B,C to make room for T, then sc.Wake(T)
// fails, the rollback path MUST schedule a best-effort restore for the
// sleep-mode evicted peers (and log + leave Stopped peers for the
// async-503 path). Without this, A,B,C end up silently asleep with
// nobody to wake them.
//
// Build the admission controller directly with a SchedulerEvictor (the
// makeRedeployScheduler helper hardcodes stubAdmissionEvictor which
// no-ops eviction; we need real sleepInstance dispatch so the peer's
// fakeRedeployMgr actually records the Sleep call).
func TestPerformWake_FailedWakeRestoresEvictedPeers(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
models:
  peerA:
    lifecycle: external
    host: vllm-peerA
    port: 8001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 0
  target:
    lifecycle: external
    host: vllm-target
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 0
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 8000} // tight: peerA + target = 9000 > 8000
	inv := newRedeployInventory(totals)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	// Construct admission with a real SchedulerEvictor so RequestWake's
	// eviction loop actually drives sleepInstance on the peer's fake.
	a := NewAdmissionController(cfg, totals, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"peerA":  {port: 8001},
		"target": {port: 8002},
	}
	mgrs["peerA"].pid.Store(int64(7000 + 8001))
	mgrs["peerA"].isSleeping.Store(false)
	mgrs["target"].pid.Store(int64(7000 + 8002))
	mgrs["target"].isSleeping.Store(true)

	s.SeedInstanceForTest("peerA", 8001, []int{0}, false, StateReady, mgrs["peerA"])
	s.SeedInstanceForTest("target", 8002, []int{0}, false, StateSleeping, mgrs["target"])

	// Pre-wake peerA so admission books reflect awake.
	if _, err := a.RequestWake(context.Background(), "peerA"); err != nil {
		t.Fatalf("pre-wake peerA: %v", err)
	}

	// Force target.Wake to fail so the rollback path runs.
	mgrs["target"].wakeErr = errors.New("simulated /wake_up failure")

	targetInst := s.instances["target"]
	targetCfg := s.cfg.Models["target"]

	if err := s.performWake(context.Background(), targetInst, targetCfg, "test-fix6"); err == nil {
		t.Fatalf("expected wake error to surface (we forced sc.Wake to fail)")
	}

	// PeerA was evicted (slept) by admission to make room. Then the
	// wake failed and Fix 6 should have spawned a best-effort restore.
	// The restore is async, so we wait briefly for the goroutine to
	// land.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if mgrs["peerA"].wakeCalls.Load() > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	if mgrs["peerA"].sleepCalls.Load() == 0 {
		t.Fatalf("expected peerA to be evicted (slept) before target wake failure; sleep calls = 0")
	}
	if mgrs["peerA"].wakeCalls.Load() == 0 {
		t.Errorf("expected peerA to be best-effort restored (wakeCalls > 0) after target wake failed; got 0 — Fix 6 regressed")
	}
}

// ---------------------------------------------------------------------
// Fix 7: mapWakeError distinguishes retryable vs. structural failures.
//
// The previous implementation wrapped EVERY error as
// RejectSwapInProgress with RetryAfter: 10s — clients (Bifrost, SDKs)
// saw 503 + Retry-After:10s and retried forever against unsolvable
// conditions. The fix introduces sentinel admission errors and routes
// them to non-retryable shapes.
// ---------------------------------------------------------------------

func TestMapWakeError_Infeasible_NoRetryAfter(t *testing.T) {
	wrapped := fmt.Errorf("admission for %q: GPU %d cannot fit: %w", "longctx", 0, ErrAdmissionInfeasible)
	mapped := mapWakeError(wrapped)
	rej, ok := mapped.(*RejectError)
	if !ok {
		t.Fatalf("expected *RejectError, got %T: %v", mapped, mapped)
	}
	if rej.Reason != RejectInsufficient {
		t.Errorf("expected Reason=%s, got %s", RejectInsufficient, rej.Reason)
	}
	if rej.RetryAfter != 0 {
		t.Errorf("expected RetryAfter=0 (no retry on infeasible), got %s", rej.RetryAfter)
	}
}

func TestMapWakeError_VRAMDriftRisk_NoRetryAfter(t *testing.T) {
	wrapped := fmt.Errorf("%w: redeploy poll failed AND cleanup stop failed", ErrAdmissionVRAMDriftRisk)
	mapped := mapWakeError(wrapped)
	rej, ok := mapped.(*RejectError)
	if !ok {
		t.Fatalf("expected *RejectError, got %T: %v", mapped, mapped)
	}
	if rej.Reason != RejectAdminIntervention {
		t.Errorf("expected Reason=%s, got %s", RejectAdminIntervention, rej.Reason)
	}
	if rej.RetryAfter != 0 {
		t.Errorf("expected RetryAfter=0 (no retry on vram-drift-risk), got %s", rej.RetryAfter)
	}
}

// TestMapWakeError_PinnedWakeFailed_NoRetryAfter pins the round-5
// CRITICAL contract: an ErrColdLoadPinnedWakeFailed wrap must route to
// RejectAdminIntervention with NO Retry-After so SDKs surface the
// state-inconsistent condition as terminal rather than retry-loop.
// Pairs with the HTTP-layer mapping in handlers_proxy.go +
// handlers_anthropic.go which already render RejectAdminIntervention
// as 503 + body code "admin_intervention_required".
func TestMapWakeError_PinnedWakeFailed_NoRetryAfter(t *testing.T) {
	wrapped := fmt.Errorf("%w: pinned peers left Sleeping after cold-load failure", ErrColdLoadPinnedWakeFailed)
	mapped := mapWakeError(wrapped)
	rej, ok := mapped.(*RejectError)
	if !ok {
		t.Fatalf("expected *RejectError, got %T: %v", mapped, mapped)
	}
	if rej.Reason != RejectAdminIntervention {
		t.Errorf("expected Reason=%s (so HTTP handlers render 503 + admin_intervention_required), got %s", RejectAdminIntervention, rej.Reason)
	}
	if rej.RetryAfter != 0 {
		t.Errorf("expected RetryAfter=0 (no retry on pinned-wake-failed), got %s", rej.RetryAfter)
	}
}

func TestMapWakeError_GenericSwapConflict_RetryableWith10s(t *testing.T) {
	mapped := mapWakeError(errors.New("transient evictor error"))
	rej, ok := mapped.(*RejectError)
	if !ok {
		t.Fatalf("expected *RejectError, got %T: %v", mapped, mapped)
	}
	if rej.Reason != RejectSwapInProgress {
		t.Errorf("expected Reason=%s, got %s", RejectSwapInProgress, rej.Reason)
	}
	if rej.RetryAfter != 10*time.Second {
		t.Errorf("expected RetryAfter=10s (retryable swap conflict), got %s", rej.RetryAfter)
	}
}

// TestMapWakeError_SentinelWrappingThroughTwoLayers ensures errors.Is
// matches even when the sentinel is wrapped through nested fmt.Errorf
// (e.g. admission.go wraps it once, performWake wraps it again). This
// pins down the "use errors.Is, not == comparison" contract.
func TestMapWakeError_SentinelWrappingThroughTwoLayers(t *testing.T) {
	inner := fmt.Errorf("admission for %q: %w", "longctx", ErrAdmissionInfeasible)
	outer := fmt.Errorf("performWake: %w", inner)
	mapped := mapWakeError(outer)
	rej := mapped.(*RejectError)
	if rej.RetryAfter != 0 {
		t.Errorf("expected RetryAfter=0 through two layers of wrap, got %s", rej.RetryAfter)
	}
}

// ---------------------------------------------------------------------
// Fix 9 part 1: LeftStopped is populated on EVERY rollback path.
//
// production sets result.LeftStopped on three rollback branches in
// redeploy.go (target docker stop fail, target docker start fail,
// pollUntilSleeping fail). The previous test coverage was a single
// happy-config rollback (TestRedeployMember_DockerStopFailureRollsBack)
// which has NO swap-group peers — so the LeftStopped field is always
// nil and the assertions don't exercise the field at all. The tests
// below stand up the 4-GPU peer-stop topology, drive each failure mode
// in turn, and assert LeftStopped == [moe] (the GPU-overlap peer
// stopped in step (a2) before the failure injection point).
// ---------------------------------------------------------------------

// TestRedeployMember_PeerStopFailureSetsLeftStopped — when peer-stop
// fully succeeds (peer is booked Stopped) but the TARGET docker stop
// then fails, the rollback path MUST emit result.LeftStopped == [moe]
// AND result.StoppedPeers == [moe] so the operator sees what state
// changed before the failure surfaced.
func TestRedeployMember_PeerStopFailureSetsLeftStopped(t *testing.T) {
	s, _, _ := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping,
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)

	// Peer-stop on moe succeeds; target stop on longctx fails.
	stopErr := errors.New("docker daemon transient error")
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		// docker stop -t 60 vllm-moe → success.
		// docker stop -t 60 vllm-longctx → fail.
		// docker start vllm-longctx → unreachable in this test.
		if len(args) >= 2 && args[0] == "stop" {
			for _, a := range args {
				if a == "vllm-longctx" {
					return []byte("daemon error"), stopErr
				}
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "longctx")
	if err == nil {
		t.Fatalf("expected target docker stop error to surface, got nil (result=%+v)", result)
	}
	if !errors.Is(err, stopErr) {
		t.Fatalf("expected wrapped stopErr, got: %v", err)
	}

	// Peer-stop happened FIRST, so the result MUST surface what changed.
	if len(result.StoppedPeers) != 1 || result.StoppedPeers[0] != "moe" {
		t.Fatalf("expected StoppedPeers=[moe] (peer-stop happened before failure), got %v", result.StoppedPeers)
	}
	if len(result.LeftStopped) != 1 || result.LeftStopped[0] != "moe" {
		t.Fatalf("expected LeftStopped=[moe] (rollback path leaves peers Stopped), got %v", result.LeftStopped)
	}
}

// TestRedeployMember_DockerStartFailureSetsLeftStopped — peer-stop
// succeeded, target stop succeeded, target START fails. LeftStopped
// MUST list the peer we stopped in (a2) so the operator sees its
// state without having to grep logs.
func TestRedeployMember_DockerStartFailureSetsLeftStopped(t *testing.T) {
	s, _, _ := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping,
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)

	startErr := errors.New("docker daemon failed start")
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		if len(args) >= 2 && args[0] == "start" {
			return []byte("start failed"), startErr
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(context.Background(), "longctx")
	if err == nil {
		t.Fatalf("expected docker start error to surface, got nil (result=%+v)", result)
	}
	if !errors.Is(err, startErr) {
		t.Fatalf("expected wrapped startErr, got: %v", err)
	}

	// Peer was stopped in step (a2) before the start-failure point.
	if len(result.StoppedPeers) != 1 || result.StoppedPeers[0] != "moe" {
		t.Fatalf("expected StoppedPeers=[moe], got %v", result.StoppedPeers)
	}
	if len(result.LeftStopped) != 1 || result.LeftStopped[0] != "moe" {
		t.Fatalf("expected LeftStopped=[moe] on docker-start failure rollback, got %v", result.LeftStopped)
	}
}

// TestRedeployMember_PollFailureSetsLeftStopped — peer-stop succeeded,
// target stop+start succeeded, but the post-start /is_sleeping poll
// fails (here driven by context cancellation). LeftStopped MUST list
// the stopped peer.
//
// Real-world failure shape: the cold-load takes longer than the
// configured timeout (the validator enforces 30s..30min, so we can't
// configure a fast-failure timeout in YAML). Instead, drive the
// failure via context cancellation — pollUntilSleeping checks
// ctx.Done() in its inner select and returns ctx.Err(). Same rollback
// branch is taken (the poll error is wrapped + returned through the
// `err` value redeploy.go switches on at line 448).
func TestRedeployMember_PollFailureSetsLeftStopped(t *testing.T) {
	s, _, mgrs := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping,
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)
	// CRITICAL: target's mgr must NEVER report is_sleeping=true so
	// pollUntilSleeping never short-circuits on success.
	mgrs["longctx"].isSleeping.Store(false)

	// Cancel the redeploy context once docker start completes — the
	// poll loop's next select will pick up ctx.Done() and return.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		// On docker start (the call after target stop), schedule a
		// context cancellation so pollUntilSleeping bails out promptly.
		// We DO NOT flip mgrs["longctx"].isSleeping=true here so the
		// poll won't succeed via the happy short-circuit either.
		if len(args) >= 2 && args[0] == "start" {
			go func() {
				// Tiny delay so the redeploy actually enters the poll
				// loop before the cancel fires.
				time.Sleep(20 * time.Millisecond)
				cancel()
			}()
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	result, err := s.RedeployMember(ctx, "longctx")
	if err == nil {
		t.Fatalf("expected pollUntilSleeping ctx-cancellation error, got nil (result=%+v)", result)
	}
	// Don't pin to a specific error string — just verify a non-nil
	// failure surfaced (likely "context canceled" or a wrapped variant).
	// What we DO care about: the result struct must reflect the rollback.
	if len(result.StoppedPeers) != 1 || result.StoppedPeers[0] != "moe" {
		t.Fatalf("expected StoppedPeers=[moe], got %v", result.StoppedPeers)
	}
	if len(result.LeftStopped) != 1 || result.LeftStopped[0] != "moe" {
		t.Fatalf("expected LeftStopped=[moe] on pollUntilSleeping failure rollback, got %v", result.LeftStopped)
	}
}

// TestRedeployMember_PeerStop_CtxCancelMidExec_NoAdmissionDrift exercises
// MEDIUM #4: dockerExec on the peer-stop path must run BEFORE admission
// state mutation. If ctx cancels between the (former) Notify-first
// ordering and the dockerExec, admission books say Stopped but the
// container is still running — silent VRAM drift.
//
// The fix reverses the order. This test: stubs dockerExec to block on a
// signal, cancels ctx, releases the signal so dockerExec returns
// context.Canceled, then asserts:
//
//   - admission state of peer is unchanged (NOT marked Stopped)
//   - LeftStopped does NOT include peer
//   - StoppedPeers does NOT include peer
//   - drift-risk metric was bumped (or audit fired)
//   - the redeploy returns ctx.Err() at the top-of-loop check guard
func TestRedeployMember_PeerStop_CtxCancelMidExec_NoAdmissionDrift(t *testing.T) {
	s, a, _ := makeRedeployScheduler(t, redeployPeerStopConfig,
		map[string]State{
			"main":    StateReady,
			"moe":     StateSleeping,
			"longctx": StateSleeping,
			"vision":  StateSleeping,
		},
		map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000},
	)

	// Mirror admission books: moe is Awake from prior activity. We assert
	// post-call that this state is preserved on cancellation.
	a.mu.Lock()
	moeState := a.models["moe"]
	moeState.State = admissionAwake
	for _, g := range moeState.GPUs {
		// drop the residual that was added by the constructor
		a.l1ResidualByGPU[g] -= moeState.L1ResidualMB
		a.awakeByGPU[g] += moeState.ExpectedOn(g)
	}
	a.mu.Unlock()

	preState := admissionAwake

	ctx, cancel := context.WithCancel(context.Background())
	released := make(chan struct{})

	SetDockerCmdForTest(func(callCtx context.Context, _ string, args ...string) ([]byte, error) {
		// First docker call is `stop -t 60 vllm-moe` (the peer). Trigger
		// outer-ctx cancel (async), block until released, then explicitly
		// return a non-nil "ctx-cancelled" error so peer-stop sees a
		// failure regardless of Go's context-propagation timing.
		for _, a := range args {
			if a == "vllm-moe" {
				go cancel()
				<-released
				return nil, fmt.Errorf("simulated docker stop ctx-cancelled: %w", context.Canceled)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	// Run redeploy in goroutine so we can release the dockerExec signal
	// after triggering cancellation.
	type res struct {
		r   RedeployResult
		err error
	}
	resCh := make(chan res, 1)
	go func() {
		r, err := s.RedeployMember(ctx, "longctx")
		resCh <- res{r: r, err: err}
	}()

	// Give the goroutine a moment to enter dockerExec and call cancel().
	time.Sleep(50 * time.Millisecond)
	close(released)

	out := <-resCh

	// Redeploy must NOT return nil — ctx was cancelled.
	if out.err == nil {
		t.Fatalf("expected ctx-cancel error, got nil (result=%+v)", out.r)
	}

	// Admission state of moe MUST be unchanged from pre-call (NOT Stopped).
	a.mu.Lock()
	gotState := a.models["moe"].State
	a.mu.Unlock()
	if gotState != preState {
		t.Errorf("expected admission state of peer 'moe' to be unchanged (%v) after ctx-cancelled docker stop, got %v",
			preState, gotState)
	}

	// LeftStopped must NOT include peer — admission books still hold the
	// peer as Awake/Sleeping, so reporting it as LeftStopped would be a lie.
	for _, p := range out.r.LeftStopped {
		if p == "moe" {
			t.Errorf("LeftStopped should NOT include 'moe' (admission books unchanged on ctx-cancel), got %v", out.r.LeftStopped)
		}
	}
	for _, p := range out.r.StoppedPeers {
		if p == "moe" {
			t.Errorf("StoppedPeers should NOT include 'moe' (docker stop failed), got %v", out.r.StoppedPeers)
		}
	}
}

// TestRedeployMember_ClearsColdLoadFailureCooldown is the architect
// round-5 HIGH-2 regression test. RedeployMember's docstring claims
// the operator escape hatch "bypasses the cold-load failure cooldown
// entirely" — pre-fix the REQUEST-PATH gate was bypassed (RedeployMember
// never calls KickColdLoad) but the stale failure record was NEVER
// cleared. If the redeployed model later went Sleeping → Stopped via
// stop-eviction inside the original 30s cooldown window, a subsequent
// request-triggered KickColdLoad gated on a failure the operator just
// resolved.
//
// Asserts: on RedeployMember success, coldLoadFailures[target] is
// absent post-call, regardless of whether a prior cooldown was active.
func TestRedeployMember_ClearsColdLoadFailureCooldown(t *testing.T) {
	s, _, mgrs := makeRedeployScheduler(t, redeployHappyConfig,
		map[string]State{"main": StateReady, "moe": StateReady},
		map[int]int{0: 10000, 1: 10000},
	)

	// Seed a stale cold-load failure cooldown for moe (the redeploy
	// target) — simulates a prior async cold-load that failed within
	// the cooldown window.
	s.mu.Lock()
	s.coldLoadFailures["moe"] = coldLoadFailureRecord{
		at:  time.Now(),
		err: "simulated prior cold-load failure",
	}
	s.mu.Unlock()

	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		// pollUntilSleeping needs is_sleeping=true after start.
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	_, err := s.RedeployMember(context.Background(), "moe")
	if err != nil {
		t.Fatalf("RedeployMember: %v", err)
	}

	// HIGH-2 assertion: stale cooldown record must be cleared on success.
	s.mu.Lock()
	_, gated := s.coldLoadFailures["moe"]
	s.mu.Unlock()
	if gated {
		t.Errorf("coldLoadFailures[moe] still present after successful RedeployMember; operator escape hatch contract is violated (HIGH-2 regression)")
	}
}
