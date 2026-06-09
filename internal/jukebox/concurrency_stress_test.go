package jukebox

// Concurrency stress tests for the admission + scheduler + cold-load +
// redeploy paths. Each test hammers a specific code path with N
// goroutines under -race and asserts no deadlock, no leak, no state
// corruption.
//
// Design choices:
//
//   - Deterministic where possible. Tests that depend on goroutine
//     interleaving use explicit handshake channels (not time.Sleep) to
//     guarantee the interleaving happens.
//   - Short wall-clocks where the hardcoded 5s cold-load poll allows.
//     Tests that exercise the full cold-load goroutine path are forced
//     above 5s wall by the poll interval; those use minimal goroutine
//     counts (N=1 or 2) to stay close to the 10s/test budget.
//   - All docker exec is mocked via SetDockerCmdForTest /
//     SetSleepDockerCmdForTest (defined in sleep_export_test.go). All
//     /is_sleeping responses are served by fakeRedeployMgr (defined in
//     redeploy_test.go).
//   - Uses lowercase identifiers from the package directly (this file is
//     package jukebox, not jukebox_test).

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// newStressAdmission builds an AdmissionController with N tracked models
// in a single swap group, each sized to fit on a single 24 GiB GPU.
// Returns the controller plus the recording stub evictor (which
// satisfies both AdmissionEvictor and AdmissionStopper).
func newStressAdmission(t *testing.T, numModels int) (*AdmissionController, *stubEvictor) {
	t.Helper()
	models := map[string]config.ModelConfig{}
	for i := 0; i < numModels; i++ {
		name := fmt.Sprintf("m%02d", i)
		models[name] = config.ModelConfig{
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
			SleepL1ResidualMB:    200,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
		}
	}
	cfg := makeCfg(models)
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)
	return a, ev
}

// stressYAML builds a YAML config for tests that need a real Scheduler.
// numModels models, all evict_action: stop swap-group "g", lifecycle
// external, on GPU 0 only so they all overlap.
func stressYAML(numModels int) string {
	var b strings.Builder
	b.WriteString(`
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
`)
	for i := 0; i < numModels; i++ {
		fmt.Fprintf(&b, `  m%02d:
    lifecycle: external
    host: vllm-m%02d
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: g
    evict_action: stop
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`, i, i, 8100+i)
	}
	return b.String()
}

// newStressScheduler builds a Scheduler + admission for tests that need
// the cold-load path. All models start admissionStopped so KickColdLoad
// can fire on any of them. Each model has a fakeRedeployMgr seeded as
// "post-cold-load resting state" (isSleeping=true) so doColdLoad's poll
// succeeds on the first probe.
func newStressScheduler(t *testing.T, numModels int) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(stressYAML(numModels)))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 24000}
	inv := newRedeployInventory(totals)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totals, &stubAdmissionEvictor{})
	s.SetAdmission(a)
	// Force a 60s cold-load timeout — doColdLoad bumps anything < 60s to
	// 600s; we set 60s explicitly so /is_sleeping is polled fast and
	// the test finishes within bounds. Real wall time per cold-load is
	// dominated by the hardcoded 5s poll interval.
	s.cfg.VLLM.StartupTimeout.Duration = 60 * time.Second

	mgrs := map[string]*fakeRedeployMgr{}
	for i := 0; i < numModels; i++ {
		name := fmt.Sprintf("m%02d", i)
		port := 8100 + i
		mgr := &fakeRedeployMgr{port: port}
		mgr.pid.Store(0)
		mgr.isSleeping.Store(true) // first /is_sleeping poll succeeds
		s.SeedInstanceForTest(name, port, []int{0}, false, StateStopped, mgr)
		mgrs[name] = mgr
		// Drive admission state to Stopped.
		a.mu.Lock()
		a.markStoppedLocked(a.models[name])
		a.mu.Unlock()
	}
	return s, a, mgrs
}

// waitGoroutineCount blocks until runtime.NumGoroutine() drops to ≤ target
// (or timeout elapses). Returns the final count.
func waitGoroutineCount(target int, timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		n := runtime.NumGoroutine()
		if n <= target {
			return n
		}
		time.Sleep(10 * time.Millisecond)
	}
	return runtime.NumGoroutine()
}

// ---------------------------------------------------------------------------
// TestStress_ConcurrentRequestWakes
//
// 10 goroutines all call RequestWake for different models in the same
// swap group. After all complete, books are consistent: the per-GPU
// awake VRAM equals the sum of awake-model footprints, and the per-GPU
// L1 residual equals the sum of sleeping-model residuals. No goroutine
// leaks past test exit.
//
// Each model is sized at 4000 MB; the GPU is 24000 MB; up to 6 can be
// awake simultaneously. With 10 concurrent wakes some will need to
// evict same-group peers. The admission mutex serializes everything, so
// after wg.Wait() the books MUST be consistent.
// ---------------------------------------------------------------------------

func TestStress_ConcurrentRequestWakes(t *testing.T) {
	const N = 10
	a, ev := newStressAdmission(t, N)

	gorBefore := runtime.NumGoroutine()

	var wg sync.WaitGroup
	errs := make([]error, N)
	wg.Add(N)
	for i := 0; i < N; i++ {
		i := i
		go func() {
			defer wg.Done()
			_, err := a.RequestWake(context.Background(), fmt.Sprintf("m%02d", i))
			errs[i] = err
		}()
	}
	wg.Wait()

	// Tally awake models vs. their expected VRAM contribution.
	a.mu.Lock()
	expectedAwakeMB := 0
	expectedResidualMB := 0
	for _, m := range a.models {
		switch m.State {
		case admissionAwake:
			expectedAwakeMB += m.ExpectedVRAMMB
		case admissionSleeping:
			expectedResidualMB += m.L1ResidualMB
		}
	}
	gotAwake := a.awakeByGPU[0]
	gotResidual := a.l1ResidualByGPU[0]
	a.mu.Unlock()

	if gotAwake != expectedAwakeMB {
		t.Errorf("awakeByGPU[0]=%d but sum of awake-model footprints=%d (book/state inconsistent)", gotAwake, expectedAwakeMB)
	}
	if gotResidual != expectedResidualMB {
		t.Errorf("l1ResidualByGPU[0]=%d but sum of sleeping residuals=%d (book/state inconsistent)", gotResidual, expectedResidualMB)
	}

	// Each victim recorded by the evictor must correspond to a model
	// that is now NOT admissionAwake (it was evicted).
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, c := range ev.Calls() {
		parts := strings.SplitN(c, ":", 2)
		if len(parts) != 2 {
			t.Errorf("malformed evictor call: %q", c)
			continue
		}
		victim := parts[1]
		m, ok := a.models[victim]
		if !ok {
			t.Errorf("evictor call %q references unknown model", c)
			continue
		}
		if m.State == admissionAwake {
			// Each goroutine wakes a unique model exactly once, so a
			// victim that's now Awake is a bookkeeping bug.
			t.Errorf("evicted victim %q is currently admissionAwake (book/state divergence)", victim)
		}
	}

	// Goroutine leak check. RequestWake itself spawns no goroutines, so
	// numGoroutine after wg.Wait should drop back to baseline.
	if n := waitGoroutineCount(gorBefore+1, 2*time.Second); n > gorBefore+1 {
		t.Errorf("goroutine leak: before=%d after=%d (expected ≤ before+1 slack)", gorBefore, n)
	}
}

// ---------------------------------------------------------------------------
// TestStress_ConcurrentKickColdLoad_SameModel
//
// 100 goroutines call s.KickColdLoad("m00") on the same Stopped model
// simultaneously. The dedupe in KickColdLoad means exactly ONE returns
// true; the others bail without spawning a goroutine. Even if dedupe
// were buggy, the global cold-load mutex + TOCTOU re-check inside
// coldLoadStoppedMember guarantees at most ONE docker start hits the
// mock.
//
// Wall-time floor: 5s (the hardcoded poll interval inside doColdLoad).
// ---------------------------------------------------------------------------

func TestStress_ConcurrentKickColdLoad_SameModel(t *testing.T) {
	s, _, _ := newStressScheduler(t, 1)

	var startCount atomic.Int32
	startEntered := make(chan struct{})
	releaseStart := make(chan struct{})
	var startOnce sync.Once
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			n := startCount.Add(1)
			if n == 1 {
				startOnce.Do(func() { close(startEntered) })
				<-releaseStart
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	const N = 100
	var kickedOK atomic.Int32
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			if s.KickColdLoad("m00") {
				kickedOK.Add(1)
			}
		}()
	}
	wg.Wait()

	// Wait for the single winner to enter the docker mock, then release.
	select {
	case <-startEntered:
	case <-time.After(3 * time.Second):
		close(releaseStart)
		t.Fatalf("no docker start hit the mock within 3s")
	}
	close(releaseStart)

	// Wait for the cold-load goroutine to complete — admission flips to
	// Sleeping AND the kicks-map entry is cleared.
	if ok := waitForCondition(8*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["m00"]
	}); !ok {
		t.Fatalf("cold-load kicks map not cleared within 8s (goroutine leaked or hung)")
	}

	if got := kickedOK.Load(); got != 1 {
		t.Errorf("expected exactly 1 KickColdLoad to return true (dedupe), got %d/%d", got, N)
	}
	if got := startCount.Load(); got != 1 {
		t.Errorf("expected EXACTLY 1 docker start (single-flight), got %d", got)
	}
}

// ---------------------------------------------------------------------------
// TestStress_ConcurrentKickColdLoad_DifferentModels
//
// N goroutines kick cold-loads on N different Stopped models. They must
// serialize through coldLoadMu (so they don't contend on the shared GPU).
// All N must EVENTUALLY succeed (admissionSleeping, not Stopped).
//
// Wall-time control: each cold-load includes a hardcoded 5s poll inside
// the lock. N serialized cold-loads → ~N × 5s. To stay inside the
// 10s/test budget, N=2.
//
// Serialization is observed via an in-flight counter recorded in the
// docker mock — peak in-flight should be exactly 1.
// ---------------------------------------------------------------------------

func TestStress_ConcurrentKickColdLoad_DifferentModels(t *testing.T) {
	const N = 2
	s, a, _ := newStressScheduler(t, N)

	var (
		inflight     atomic.Int32
		peakInflight atomic.Int32
		startsSeen   atomic.Int32
	)
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			cur := inflight.Add(1)
			defer inflight.Add(-1)
			for {
				p := peakInflight.Load()
				if cur <= p || peakInflight.CompareAndSwap(p, cur) {
					break
				}
			}
			startsSeen.Add(1)
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	for i := 0; i < N; i++ {
		s.KickColdLoad(fmt.Sprintf("m%02d", i))
	}

	if ok := waitForCondition(15*time.Second, func() bool {
		for i := 0; i < N; i++ {
			if a.IsStopped(fmt.Sprintf("m%02d", i)) {
				return false
			}
		}
		return true
	}); !ok {
		t.Fatalf("not all %d cold-loads completed within 15s; inflight=%d starts=%d",
			N, inflight.Load(), startsSeen.Load())
	}

	if got := startsSeen.Load(); int(got) != N {
		t.Errorf("expected %d docker start calls, got %d", N, got)
	}
	if got := peakInflight.Load(); got > 1 {
		t.Errorf("expected peak in-flight docker start = 1 (serial via coldLoadMu), got %d", got)
	}
	for i := 0; i < N; i++ {
		name := fmt.Sprintf("m%02d", i)
		if a.models[name].State != admissionSleeping {
			t.Errorf("expected %q admissionSleeping after cold-load, got state %d", name, a.models[name].State)
		}
	}
}

// ---------------------------------------------------------------------------
// TestStress_RequestWake_During_Redeploy
//
// A redeploy is in flight on model X (holding coldLoadMu via a long
// WithColdLoadLock callback). N concurrent goroutines call RequestWake
// on a DIFFERENT model Y in the same swap group.
//
// RequestWake itself acquires only admission.mu, NOT coldLoadMu — so it
// MUST NOT block on the cold-load lock. This test pins down the
// "two locks are independent" invariant for the wake path.
//
// No deadlock = the RequestWake goroutines complete within a tight
// timeout while the cold-load lock is still held.
// ---------------------------------------------------------------------------

func TestStress_RequestWake_During_Redeploy(t *testing.T) {
	const N = 5
	a, _ := newStressAdmission(t, N+1)

	// Pre-wake the wakers' models so the RequestWake calls below
	// short-circuit on "already awake" — this is what we want to test:
	// admission.mu acquisition is independent of coldLoadMu.
	for i := 0; i < N; i++ {
		if _, err := a.RequestWake(context.Background(), fmt.Sprintf("m%02d", i)); err != nil {
			t.Fatalf("pre-wake m%02d: %v", i, err)
		}
	}

	// Simulate "redeploy in flight" via a goroutine that holds coldLoadMu.
	heldEnough := make(chan struct{})
	releaseHold := make(chan struct{})
	go func() {
		a.WithColdLoadLock(func() {
			close(heldEnough)
			<-releaseHold
		})
	}()
	<-heldEnough

	// Fire N RequestWake calls. They MUST complete promptly — they only
	// touch admission.mu, never coldLoadMu.
	done := make(chan struct{})
	go func() {
		defer close(done)
		var wg sync.WaitGroup
		wg.Add(N)
		for i := 0; i < N; i++ {
			i := i
			go func() {
				defer wg.Done()
				_, _ = a.RequestWake(context.Background(), fmt.Sprintf("m%02d", i))
			}()
		}
		wg.Wait()
	}()

	select {
	case <-done:
		// Good — RequestWake calls completed without waiting on coldLoadMu.
	case <-time.After(2 * time.Second):
		close(releaseHold)
		t.Fatal("RequestWake blocked on coldLoadMu (lock independence regressed)")
	}
	close(releaseHold)
}

// ---------------------------------------------------------------------------
// TestStress_RandomOpsRace
//
// 50 goroutines, each running 100 random ops on a small model set.
// Asserts: no panic, no negative books, no deadlock, books stay
// consistent.
//
// Operations exercised (uniform mix):
//   - RequestWake
//   - NotifySleep
//   - NotifyStopped
//   - NotifyStarted
//   - IsStopped
//   - SnapshotBudgets
//
// A separate watchdog goroutine continuously checks per-GPU books for
// non-negativity (the canonical double-decrement bug).
// ---------------------------------------------------------------------------

func TestStress_RandomOpsRace(t *testing.T) {
	const (
		numWorkers   = 50
		opsPerWorker = 100
		numModels    = 5
		wallBudget   = 5 * time.Second
	)
	a, _ := newStressAdmission(t, numModels)

	ctx, cancel := context.WithTimeout(context.Background(), wallBudget)
	defer cancel()

	var (
		bookViolation atomic.Bool
		opsRun        atomic.Int64
	)

	// Watchdog: snapshot budgets periodically and verify non-negative.
	watchdogDone := make(chan struct{})
	go func() {
		defer close(watchdogDone)
		ticker := time.NewTicker(2 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				snap := a.SnapshotBudgets()
				for _, s := range snap {
					if s.AwakeMB < 0 || s.L1ResidualMB < 0 || s.AvailableMB > s.TotalMB {
						bookViolation.Store(true)
						return
					}
				}
			}
		}
	}()

	// Workers — each picks a model and an op via a deterministic LCG
	// derived from the worker id, so the test is reproducible.
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		w := w
		go func() {
			defer wg.Done()
			seed := uint64(w)
			for i := 0; i < opsPerWorker; i++ {
				if ctx.Err() != nil {
					return
				}
				seed = seed*6364136223846793005 + 1442695040888963407 // LCG
				name := fmt.Sprintf("m%02d", int(seed>>32)%numModels)
				op := int(seed>>16) % 6
				switch op {
				case 0:
					_, _ = a.RequestWake(ctx, name)
				case 1:
					a.NotifySleep(name, "manual")
				case 2:
					a.NotifyStopped(name)
				case 3:
					a.NotifyStarted(name)
				case 4:
					_ = a.IsStopped(name)
				case 5:
					_ = a.SnapshotBudgets()
				}
				opsRun.Add(1)
			}
		}()
	}

	// Bounded overall timeout — if anything deadlocks, fail here instead
	// of hanging the suite.
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()
	select {
	case <-doneCh:
	case <-time.After(wallBudget + 5*time.Second):
		t.Fatalf("random-ops workers did not finish (deadlock?). ops_run=%d", opsRun.Load())
	}
	cancel()
	<-watchdogDone

	if bookViolation.Load() {
		snap := a.SnapshotBudgets()
		t.Errorf("watchdog observed negative/inconsistent books: %+v", snap)
	}
	if opsRun.Load() == 0 {
		t.Errorf("no ops ran (test setup bug)")
	}

	// Final book consistency: per-GPU awake must equal sum of awake
	// models' expected VRAM, residual must equal sum of sleeping
	// residuals.
	a.mu.Lock()
	expectedAwake := 0
	expectedResidual := 0
	for _, m := range a.models {
		switch m.State {
		case admissionAwake:
			expectedAwake += m.ExpectedVRAMMB
		case admissionSleeping:
			expectedResidual += m.L1ResidualMB
		}
	}
	gotAwake := a.awakeByGPU[0]
	gotResidual := a.l1ResidualByGPU[0]
	a.mu.Unlock()
	if gotAwake != expectedAwake {
		t.Errorf("final awakeByGPU[0]=%d but sum of awake-model VRAM=%d (books drift)", gotAwake, expectedAwake)
	}
	if gotResidual != expectedResidual {
		t.Errorf("final l1ResidualByGPU[0]=%d but sum of sleeping residuals=%d (books drift)", gotResidual, expectedResidual)
	}
}

// ---------------------------------------------------------------------------
// TestStress_ColdLoadLock_NoStarvation
//
// 20 goroutines all want the cold-load lock for a brief critical section.
// Every goroutine must acquire within a bounded time — no starvation.
//
// We don't assert ordering (sync.Mutex doesn't guarantee strict FIFO),
// just that no goroutine waits indefinitely.
// ---------------------------------------------------------------------------

func TestStress_ColdLoadLock_NoStarvation(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {GPUs: []int{0}, ExpectedVRAMMBPerGPU: 1000, Priority: config.PriorityNormal},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	const N = 20
	const holdTime = 5 * time.Millisecond
	// Each goroutine waits at most ~(N-1)*holdTime + scheduling slack.
	// 20 × 5ms = 100ms; bound at 2s for race-overhead headroom.
	const perGoroutineBudget = 2 * time.Second

	type result struct {
		id       int
		acquired bool
		waited   time.Duration
	}
	results := make(chan result, N)

	start := time.Now()
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		i := i
		go func() {
			defer wg.Done()
			launchedAt := time.Now()
			done := make(chan struct{})
			go func() {
				a.WithColdLoadLock(func() {
					time.Sleep(holdTime)
				})
				close(done)
			}()
			select {
			case <-done:
				results <- result{id: i, acquired: true, waited: time.Since(launchedAt)}
			case <-time.After(perGoroutineBudget):
				results <- result{id: i, acquired: false, waited: time.Since(launchedAt)}
			}
		}()
	}
	wg.Wait()
	close(results)

	totalAcquired := 0
	var maxWait time.Duration
	for r := range results {
		if r.acquired {
			totalAcquired++
			if r.waited > maxWait {
				maxWait = r.waited
			}
		} else {
			t.Errorf("goroutine %d failed to acquire coldLoadMu within %v (starvation)", r.id, perGoroutineBudget)
		}
	}
	if totalAcquired != N {
		t.Errorf("only %d/%d goroutines acquired coldLoadMu", totalAcquired, N)
	}
	t.Logf("starvation stats: N=%d holdTime=%v totalWall=%v maxWait=%v", N, holdTime, time.Since(start), maxWait)
}

// ---------------------------------------------------------------------------
// TestStress_GoroutineLeak_KickColdLoad
//
// Kick N cold-loads on N different Stopped models. After all complete,
// verify the runtime goroutine count drops back to baseline AND the
// per-model dedupe map (coldLoadKicks) is empty.
//
// Wall-time floor: N × 5s (cold-loads serialize, each costs one 5s poll).
// N=2 keeps the test under the 10s/test budget.
// ---------------------------------------------------------------------------

func TestStress_GoroutineLeak_KickColdLoad(t *testing.T) {
	const N = 2
	s, a, _ := newStressScheduler(t, N)

	var startCount atomic.Int32
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			startCount.Add(1)
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	gorBefore := runtime.NumGoroutine()

	for i := 0; i < N; i++ {
		s.KickColdLoad(fmt.Sprintf("m%02d", i))
	}

	// Wait for all cold-loads to complete (admissionSleeping) and the
	// dedupe map to clear.
	if ok := waitForCondition(20*time.Second, func() bool {
		s.mu.RLock()
		kicksLen := len(s.coldLoadKicks)
		s.mu.RUnlock()
		if kicksLen != 0 {
			return false
		}
		for i := 0; i < N; i++ {
			if a.IsStopped(fmt.Sprintf("m%02d", i)) {
				return false
			}
		}
		return true
	}); !ok {
		s.mu.RLock()
		t.Fatalf("not all cold-loads completed within 20s; coldLoadKicks=%v startCount=%d",
			s.coldLoadKicks, startCount.Load())
	}

	if got := startCount.Load(); int(got) != N {
		t.Errorf("expected %d docker start calls (one per kick), got %d", N, got)
	}

	// Goroutine count should return to baseline + small slack.
	if n := waitGoroutineCount(gorBefore+2, 3*time.Second); n > gorBefore+2 {
		t.Errorf("goroutine leak: before=%d after=%d (kicked %d cold-loads)", gorBefore, n, N)
	}

	// Dedupe map must be empty.
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.coldLoadKicks) != 0 {
		t.Errorf("coldLoadKicks map not empty after all cold-loads completed: %v", s.coldLoadKicks)
	}
}
