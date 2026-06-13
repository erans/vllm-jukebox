package jukebox

// Tests for the cumem PP-broadcast race mitigations — two jukebox-side
// guards against upstream vllm bugs:
//
//   - Guard A (cumem_drain_timeout_seconds) — defends
//     vllm-project/vllm#45520. /sleep with in-flight decode corrupts
//     cumem state. We refuse to /sleep while requests are mid-flight.
//
//   - Guard B (wake_settle_seconds) — defends
//     vllm-project/vllm#45519. /wake_up returns 200 before PP workers
//     settle. We refuse a /sleep that lands inside the settle window.
//
// Both guards are opt-in via per-model knobs (zero default = no-op so
// existing configs keep their pre-guard behavior).

import (
	"context"
	"errors"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
)

// stubSleepableMgr is a minimal SleepCapable implementation for guard
// tests — we only need to observe whether Sleep was called and to
// hold state in a way that lets the test drive the schedInstance
// state machine through sleepInstance.
type stubSleepableMgr struct {
	sleepCalls int
	wakeCalls  int
	sleepErr   error
	wakeErr    error
	sleeping   bool
}

func (m *stubSleepableMgr) Start(_ context.Context, _ string) (int, error) { return 1, nil }
func (m *stubSleepableMgr) Stop(_ context.Context) error                   { return nil }
func (m *stubSleepableMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *stubSleepableMgr) CurrentPID() int                                { return 1 }
func (m *stubSleepableMgr) BaseURL() string                                { return "" }
func (m *stubSleepableMgr) Sleep(_ context.Context, _ int) error {
	m.sleepCalls++
	if m.sleepErr != nil {
		return m.sleepErr
	}
	m.sleeping = true
	return nil
}
func (m *stubSleepableMgr) Wake(_ context.Context, _ time.Duration) error {
	m.wakeCalls++
	if m.wakeErr != nil {
		return m.wakeErr
	}
	m.sleeping = false
	return nil
}
func (m *stubSleepableMgr) IsSleeping(_ context.Context) (bool, error) {
	return m.sleeping, nil
}

// guardTestCfg builds a minimal scheduler config with one model and
// per-model overrides for the cumem-drain + wake-settle knobs.
func guardTestCfg(t *testing.T, modelName string, cumemDrainSec, wakeSettleSec int) *config.Config {
	t.Helper()
	cfg := &config.Config{
		Scheduler: &config.SchedulerConfig{
			PortRangeStart: 8200,
			PortRangeEnd:   8299,
		},
		VLLM: config.VLLMConfig{
			Port:           8000,
			StartupTimeout: config.Duration{Duration: 1 * time.Second},
			DrainTimeout:   config.Duration{Duration: 50 * time.Millisecond},
		},
		Models: map[string]config.ModelConfig{
			modelName: {
				Path:                     "/models/" + modelName,
				GPUs:                     []int{0},
				SleepMode:                true,
				CumemDrainTimeoutSeconds: cumemDrainSec,
				WakeSettleSeconds:        wakeSettleSec,
			},
		},
	}
	return cfg
}

// TestSleepRejectsOnInflightCumemSafety — Guard A. Seed an instance,
// leak an inflight.Track without calling Done, then invoke
// sleepInstance. Expect:
//   - error wrapping ErrSleepRejectedInflight
//   - SleepRejectedInflightTotal{model,trigger_reason} bumped by 1
//   - Mgr.Sleep NOT called (we refused before reaching the upstream)
//   - schedInstance.state restored to prevState (StateReady)
//   - schedInstance.draining cleared
func TestSleepRejectsOnInflightCumemSafety(t *testing.T) {
	const model = "guard-a-target"
	const reason = "test-cumem-guard"
	cfg := guardTestCfg(t, model, 1 /* 1s drain */, 0 /* settle off */)

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8202, []int{0}, false, StateReady, mgr)

	inst := s.instances[model]
	// Leak one in-flight tracker — never call done(). This simulates an
	// active streaming/decoding request that has NOT completed by the
	// time admission decides to evict.
	_ = inst.inflight.Track(context.Background())

	pre := counterValue(metrics.SleepRejectedInflightTotal, model, reason)

	t0 := time.Now()
	err := s.sleepInstance(context.Background(), inst, 1, reason)
	elapsed := time.Since(t0)

	if err == nil {
		t.Fatalf("expected ErrSleepRejectedInflight, got nil")
	}
	if !errors.Is(err, ErrSleepRejectedInflight) {
		t.Fatalf("expected error to wrap ErrSleepRejectedInflight, got: %v", err)
	}
	// Drain budget is 1s; we expect to wait ~at most 1s + a little
	// slop. Catching a regression where the budget isn't respected
	// (e.g. accidental WaitForDrain on the default 60s).
	if elapsed > 3*time.Second {
		t.Errorf("guard A waited too long (%s) — expected ~drain budget (1s)", elapsed)
	}
	if mgr.sleepCalls != 0 {
		t.Errorf("expected Mgr.Sleep to NOT be called when guard refuses; got %d call(s)", mgr.sleepCalls)
	}

	post := counterValue(metrics.SleepRejectedInflightTotal, model, reason)
	if got := post - pre; got != 1 {
		t.Errorf("expected SleepRejectedInflightTotal{%q,%q} += 1, got delta %v", model, reason, got)
	}

	// State must be restored.
	s.mu.RLock()
	gotState := inst.state
	gotDraining := inst.draining
	s.mu.RUnlock()
	if gotState != StateReady {
		t.Errorf("expected state restored to StateReady, got %s", gotState)
	}
	if gotDraining {
		t.Errorf("expected draining cleared after guard refusal")
	}
}

// TestSleepSkippedByWakeSettleWindow — Guard B. Mark the instance
// recently-woken (lastWakeAt = now) then immediately invoke
// sleepInstance. Expect:
//   - error wrapping ErrSleepSkippedWakeSettle
//   - SleepSkippedCooldownTotal{model,"wake_settle"} bumped by 1
//   - Mgr.Sleep NOT called
//   - state UNCHANGED — guard B fires BEFORE the state transition,
//     so there's nothing to roll back.
//   - After settle elapses, a fresh sleepInstance succeeds.
func TestSleepSkippedByWakeSettleWindow(t *testing.T) {
	const model = "guard-b-target"
	const reason = "test-wake-settle"
	// 1s settle, no cumem-drain guard — we want only Guard B to fire.
	cfg := guardTestCfg(t, model, 0, 1)

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8203, []int{0}, false, StateReady, mgr)

	inst := s.instances[model]

	// Mark the instance as "just woken" (well within the 1s settle).
	s.mu.Lock()
	inst.lastWakeAt = s.now()
	s.mu.Unlock()

	pre := counterValue(metrics.SleepSkippedCooldownTotal, model, "wake_settle")

	err := s.sleepInstance(context.Background(), inst, 1, reason)

	if err == nil {
		t.Fatalf("expected ErrSleepSkippedWakeSettle, got nil")
	}
	if !errors.Is(err, ErrSleepSkippedWakeSettle) {
		t.Fatalf("expected error to wrap ErrSleepSkippedWakeSettle, got: %v", err)
	}
	if mgr.sleepCalls != 0 {
		t.Errorf("expected Mgr.Sleep to NOT be called when settle blocks; got %d call(s)", mgr.sleepCalls)
	}

	post := counterValue(metrics.SleepSkippedCooldownTotal, model, "wake_settle")
	if got := post - pre; got != 1 {
		t.Errorf("expected SleepSkippedCooldownTotal{%q,wake_settle} += 1, got delta %v", model, got)
	}

	// State must be untouched (guard B runs BEFORE the state flip).
	s.mu.RLock()
	gotState := inst.state
	gotDraining := inst.draining
	s.mu.RUnlock()
	if gotState != StateReady {
		t.Errorf("expected state untouched at StateReady, got %s", gotState)
	}
	if gotDraining {
		t.Errorf("expected draining unset before/after guard B refusal")
	}

	// Backdate lastWakeAt so the settle window has elapsed; the next
	// sleep call must succeed.
	s.mu.Lock()
	inst.lastWakeAt = s.now().Add(-5 * time.Second)
	s.mu.Unlock()

	if err := s.sleepInstance(context.Background(), inst, 1, reason); err != nil {
		t.Fatalf("expected sleep to succeed after settle elapses, got: %v", err)
	}
	if mgr.sleepCalls != 1 {
		t.Errorf("expected Mgr.Sleep to be called exactly once after settle elapses, got %d", mgr.sleepCalls)
	}
}

// TestSleepGuardsDisabledByDefault — sanity: when both knobs are 0
// (the default), sleepInstance behaves exactly as it did pre-guards.
// In particular, a leaked in-flight tracker does NOT block the sleep
// (only the existing best-effort WaitForDrain still runs).
func TestSleepGuardsDisabledByDefault(t *testing.T) {
	const model = "guards-default-off"
	const reason = "test-default-noop"
	cfg := guardTestCfg(t, model, 0, 0) // both off

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8204, []int{0}, false, StateReady, mgr)
	inst := s.instances[model]

	// Mark "just woken" — should NOT block when wake_settle = 0.
	s.mu.Lock()
	inst.lastWakeAt = s.now()
	s.mu.Unlock()

	// Leak an in-flight — should NOT block when cumem_drain = 0.
	_ = inst.inflight.Track(context.Background())

	err := s.sleepInstance(context.Background(), inst, 1, reason)
	if err != nil {
		t.Fatalf("expected sleep to succeed with both guards disabled, got: %v", err)
	}
	if mgr.sleepCalls != 1 {
		t.Errorf("expected Mgr.Sleep called once, got %d", mgr.sleepCalls)
	}
}
