package jukebox

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// reprobeFakeMgr is a controllable InstanceManager whose VerifyReady
// (/health probe) result can be toggled, so the Bug #3 re-probe tests
// can simulate "container recreated, now healthy" vs "container still
// down". Mirrors fakeRedeployMgr but with a settable health error.
type reprobeFakeMgr struct {
	port       int
	pid        atomic.Int64
	isSleeping atomic.Bool
	verifyErr  atomic.Pointer[error]
	verifyN    atomic.Int32
	wakeN      atomic.Int32
}

func (m *reprobeFakeMgr) Start(_ context.Context, _ string) (int, error) {
	m.pid.Store(int64(7000 + m.port))
	return int(m.pid.Load()), nil
}
func (m *reprobeFakeMgr) Stop(_ context.Context) error { m.pid.Store(0); return nil }
func (m *reprobeFakeMgr) VerifyReady(_ context.Context, _ string) error {
	m.verifyN.Add(1)
	if p := m.verifyErr.Load(); p != nil {
		return *p
	}
	return nil
}
func (m *reprobeFakeMgr) CurrentPID() int                            { return int(m.pid.Load()) }
func (m *reprobeFakeMgr) BaseURL() string                            { return "http://vllm-reprobe:9001" }
func (m *reprobeFakeMgr) IsSleeping(_ context.Context) (bool, error) { return m.isSleeping.Load(), nil }
func (m *reprobeFakeMgr) Sleep(_ context.Context, _ int) error       { m.isSleeping.Store(true); return nil }
func (m *reprobeFakeMgr) Wake(_ context.Context, _ time.Duration) error {
	m.wakeN.Add(1)
	m.isSleeping.Store(false)
	return nil
}

func (m *reprobeFakeMgr) setHealthy(healthy bool) {
	if healthy {
		m.verifyErr.Store(nil)
		return
	}
	e := error(errors.New("connection refused"))
	m.verifyErr.Store(&e)
}

const reprobeConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  ocr:
    lifecycle: external
    host: vllm-ocr
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    evict_action: stop
    swap_group: ocr-group
    expected_vram_mb_per_gpu: 8000
    wake_timeout: 1s
`

// makeReprobeScheduler builds a scheduler with model "ocr" seeded as
// StateStopped and admission marked Stopped — the exact stale state that
// makes IsModelColdLoading return true (and the proxy 503 "warming_up").
func makeReprobeScheduler(t *testing.T) (*Scheduler, *reprobeFakeMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(reprobeConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgr := &reprobeFakeMgr{port: 9001}
	mgr.pid.Store(0)
	mgr.isSleeping.Store(true)
	s.SeedInstanceForTest("ocr", 9001, []int{0}, false, StateStopped, mgr)

	// Put admission into Stopped so IsModelColdLoading/IsStopped is true.
	a.mu.Lock()
	a.markStoppedLocked(a.models["ocr"])
	a.mu.Unlock()
	return s, mgr
}

// TestReprobe_ClearsWarmingWhenContainerHealthy is the Bug #3
// regression: after the external container is recreated and reaches
// /health=200, a request must NOT keep getting the stale "warming_up"
// 503. ReprobeStoppedExternal observes the healthy container, reconciles
// admission Stopped→Sleeping (clearing IsModelColdLoading), flips the
// instance to StateSleeping, and returns true so the handler routes
// through the wake path.
func TestReprobe_ClearsWarmingWhenContainerHealthy(t *testing.T) {
	s, mgr := makeReprobeScheduler(t)
	old := SetReprobeStoppedHealthBudgetForTest(200 * time.Millisecond)
	defer SetReprobeStoppedHealthBudgetForTest(old)

	// Precondition: the model is in the warming_up gate.
	if !s.IsModelColdLoading("ocr") {
		t.Fatalf("precondition: expected IsModelColdLoading(ocr)=true (admission Stopped)")
	}

	// Container is now healthy (recreated out-of-band).
	mgr.setHealthy(true)

	got := s.ReprobeStoppedExternal(context.Background(), "ocr")
	if !got {
		t.Fatalf("Bug #3: ReprobeStoppedExternal returned false for a healthy container (should clear warming_up)")
	}
	if mgr.verifyN.Load() == 0 {
		t.Fatalf("Bug #3: expected a /health re-probe to have fired (duration_ms=1, no re-probe was the bug)")
	}
	// warming_up must now be cleared.
	if s.IsModelColdLoading("ocr") {
		t.Fatalf("Bug #3: IsModelColdLoading(ocr) still true after successful re-probe — warming_up not cleared")
	}
	// Instance flipped Stopped→Sleeping so the wake path can run.
	s.mu.RLock()
	st := s.instances["ocr"].state
	s.mu.RUnlock()
	if st != StateSleeping {
		t.Fatalf("Bug #3: expected instance state Sleeping after re-probe, got %v", st)
	}
}

// TestReprobe_NoActionWhenContainerStillDown proves the re-probe is
// safe: a genuinely-down container must NOT be reconciled — the async-503
// + KickColdLoad contract stays intact, so the caller still 503s.
func TestReprobe_NoActionWhenContainerStillDown(t *testing.T) {
	s, mgr := makeReprobeScheduler(t)
	old := SetReprobeStoppedHealthBudgetForTest(200 * time.Millisecond)
	defer SetReprobeStoppedHealthBudgetForTest(old)

	mgr.setHealthy(false) // container still down

	got := s.ReprobeStoppedExternal(context.Background(), "ocr")
	if got {
		t.Fatalf("Bug #3: ReprobeStoppedExternal returned true for a DOWN container (must not clear warming_up)")
	}
	if !s.IsModelColdLoading("ocr") {
		t.Fatalf("Bug #3: a down container must remain in the warming_up/cold-loading gate")
	}
	s.mu.RLock()
	st := s.instances["ocr"].state
	s.mu.RUnlock()
	if st != StateStopped {
		t.Fatalf("Bug #3: down container instance state must stay Stopped, got %v", st)
	}
}

// nonSleepReprobeConfig: an external, pinned model with sleep_mode OMITTED
// (→ false). The exact shape that surfaces the HIGH bug: a non-sleep-mode
// external model that is unreachable at boot (seeded Stopped) then becomes
// healthy. ReprobeStoppedExternal must reconcile it to Ready (full VRAM,
// just "up") and must NOT take the wake path — there is no
// --enable-sleep-mode, so POSTing /wake_up would 404/error and perma-wedge.
const nonSleepReprobeConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  ocr:
    lifecycle: external
    host: vllm-ocr
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    pinned: true
    expected_vram_mb_per_gpu: 8000
    wake_timeout: 1s
`

// makeNonSleepReprobeScheduler builds a scheduler with the non-sleep-mode
// "ocr" model seeded StateStopped + admission Stopped — same stale-Stopped
// starting point as makeReprobeScheduler but for a sleep_mode:false model.
func makeNonSleepReprobeScheduler(t *testing.T) (*Scheduler, *reprobeFakeMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(nonSleepReprobeConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgr := &reprobeFakeMgr{port: 9001}
	mgr.pid.Store(0)
	mgr.isSleeping.Store(false) // non-sleep model is awake-or-down, never slept
	s.SeedInstanceForTest("ocr", 9001, []int{0}, true /* pinned */, StateStopped, mgr)

	a.mu.Lock()
	a.markStoppedLocked(a.models["ocr"])
	a.mu.Unlock()
	return s, mgr
}

// TestReprobe_NonSleepModeMarksReadyNotSleeping is the HIGH regression: a
// non-sleep-mode external model that comes back healthy must be reconciled
// to Ready (full VRAM booked, no wake path) — NOT to Sleeping. Sleeping
// would route the next request through performWake → POST /wake_up to a
// vLLM with no --enable-sleep-mode → error → perma-wedge until restart.
//
// Pre-fix this FAILS: ReprobeStoppedExternal unconditionally set
// StateSleeping (and NotifyStarted's L1-residual booking) regardless of
// sleep_mode.
func TestReprobe_NonSleepModeMarksReadyNotSleeping(t *testing.T) {
	s, mgr := makeNonSleepReprobeScheduler(t)
	old := SetReprobeStoppedHealthBudgetForTest(200 * time.Millisecond)
	defer SetReprobeStoppedHealthBudgetForTest(old)

	if !s.IsModelColdLoading("ocr") {
		t.Fatalf("precondition: expected IsModelColdLoading(ocr)=true (admission Stopped)")
	}

	// Container is now healthy (recreated / reachable after boot).
	mgr.setHealthy(true)

	got := s.ReprobeStoppedExternal(context.Background(), "ocr")
	if !got {
		t.Fatalf("ReprobeStoppedExternal returned false for a healthy non-sleep container (should reconcile + clear warming_up)")
	}
	if mgr.verifyN.Load() == 0 {
		t.Fatalf("expected a /health re-probe to have fired")
	}
	// warming_up cleared.
	if s.IsModelColdLoading("ocr") {
		t.Fatalf("IsModelColdLoading(ocr) still true after successful re-probe — warming_up not cleared")
	}
	// THE FIX: non-sleep-mode model reconciles to Ready, not Sleeping.
	s.mu.RLock()
	st := s.instances["ocr"].state
	s.mu.RUnlock()
	if st == StateSleeping {
		t.Fatalf("HIGH bug: non-sleep-mode model reconciled to Sleeping — next request would POST /wake_up to a vLLM with no --enable-sleep-mode and perma-wedge")
	}
	if st != StateReady {
		t.Fatalf("expected non-sleep-mode model reconciled to Ready, got %v", st)
	}
	// No /wake_up must ever be attempted for a non-sleep-mode model.
	if mgr.wakeN.Load() != 0 {
		t.Fatalf("HIGH bug: %d /wake_up attempt(s) on a non-sleep-mode model (must be 0)", mgr.wakeN.Load())
	}
	// Admission must reflect a fully-awake (not slept-L1) model.
	if s.admission.IsStopped("ocr") {
		t.Fatalf("admission still reports Stopped after reconcile")
	}
}

// TestReprobe_NoActionWhenNotStopped proves the re-probe is a no-op for a
// model admission does not believe is Stopped (it must not interfere with
// healthy/Sleeping models).
func TestReprobe_NoActionWhenNotStopped(t *testing.T) {
	s, mgr := makeReprobeScheduler(t)
	// Move admission off Stopped: NotifyStarted → Sleeping.
	s.admission.NotifyStarted("ocr")
	if s.IsModelColdLoading("ocr") {
		t.Fatalf("precondition: NotifyStarted should clear Stopped")
	}
	if got := s.ReprobeStoppedExternal(context.Background(), "ocr"); got {
		t.Fatalf("Bug #3: re-probe must be a no-op (false) for a non-Stopped model")
	}
	if mgr.verifyN.Load() != 0 {
		t.Fatalf("Bug #3: re-probe must NOT health-probe a non-Stopped model, fired %d times", mgr.verifyN.Load())
	}
}

// TestReprobe_SplitBrainAdmissionStoppedInstanceSleeping is the FIX
// HIGH-1 regression: the split-brain window where admission.IsStopped(name)
// is still true (so the function gets past its early admission gate) BUT
// the schedInstance has ALREADY been reconciled to StateSleeping by a
// concurrent path (KickColdLoad/doColdLoad, the 30s state reconciler, a
// peer restore). In that case ReprobeStoppedExternal must:
//
//   - NOT fire a redundant /health probe (verifyN stays 0),
//   - NOT re-call NotifyStarted (no double-book of L1 residual / no
//     redundant Stopped→Sleeping admission transition),
//   - return true so the caller falls through to AcquireRoute →
//     tryRouteFromSleep, which already handles a Sleeping instance.
//
// Against the pre-FIX-1 code this FAILS: that code probed unconditionally
// (verifyN==1), then called NotifyStarted unconditionally and returned a
// stale true, triggering a redundant wake attempt.
func TestReprobe_SplitBrainAdmissionStoppedInstanceSleeping(t *testing.T) {
	s, mgr := makeReprobeScheduler(t)
	old := SetReprobeStoppedHealthBudgetForTest(200 * time.Millisecond)
	defer SetReprobeStoppedHealthBudgetForTest(old)

	// Split-brain: admission still believes Stopped (so IsModelColdLoading
	// is true and the function clears its early !IsStopped gate)...
	if !s.admission.IsStopped("ocr") {
		t.Fatalf("precondition: expected admission Stopped for split-brain setup")
	}
	// ...but a concurrent path already reconciled the schedInstance to
	// StateSleeping (the exact state tryRouteFromSleep expects).
	s.mu.Lock()
	s.instances["ocr"].state = StateSleeping
	s.mu.Unlock()

	// Container is healthy — but that's irrelevant; the point is the probe
	// must never even fire because the instance is already reconciled.
	mgr.setHealthy(true)

	got := s.ReprobeStoppedExternal(context.Background(), "ocr")
	if !got {
		t.Fatalf("FIX HIGH-1: split-brain (admission Stopped, instance Sleeping) must return true (already reconciled → fall through to AcquireRoute), got false")
	}
	if mgr.verifyN.Load() != 0 {
		t.Fatalf("FIX HIGH-1: split-brain must NOT fire a redundant /health probe (instance already Sleeping), fired %d times", mgr.verifyN.Load())
	}
	// Admission must NOT have been touched by a redundant NotifyStarted.
	// (It was Stopped going in; a real reconcile happens via the wake path,
	// not here.) The instance must remain Sleeping.
	s.mu.RLock()
	st := s.instances["ocr"].state
	s.mu.RUnlock()
	if st != StateSleeping {
		t.Fatalf("FIX HIGH-1: split-brain must leave instance Sleeping, got %v", st)
	}
}
