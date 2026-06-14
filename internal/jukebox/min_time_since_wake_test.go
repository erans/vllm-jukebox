package jukebox

// Tests for the rapid-cycle anti-race gate (Fix A) —
// min_time_since_wake_seconds. Distinct from the existing wake_settle
// gate (which defends the narrow PP-broadcast window from vllm#45519).
// This gate defends the broader cycle-23 stress race documented in
// project_jukebox_rapid_sleep_wake_race: 2-3s sleep-after-wake
// produces cudaErrorIllegalAddress in cumem.py:202; /wake_up returns
// 200 but the engine is silently wedged.
//
// Three test cases cover the spec:
//
//   1. RefusedWithinMinTimeSinceWake — fresh wake (1s ago), 5s gate;
//      expect ErrSleepTooSoonAfterWake, no Mgr.Sleep call, state
//      untouched.
//   2. AllowedAfterCooldown — wake 6s ago, 5s gate; expect success.
//   3. FirstSleepHasNoWakeRecord — brand-new peer (lastWakeAt zero
//      value); expect success (no rate-limit on the first sleep).
//
// Fix B (softer recovery on /wake_up=500) is NOT covered here — vLLM
// does not expose an engine-restart endpoint and the Fix B half is
// deferred to a pending-user-decision item. See CLAUDE.md handoff
// notes.

import (
	"context"
	"errors"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
)

// minTimeSinceWakeTestCfg builds a scheduler config with the
// rapid-cycle anti-race gate explicitly configured (in seconds), and
// the cumem-drain + wake-settle gates disabled so the new gate is
// the only thing in play. minSec=0 means "use the package default
// (5s)"; minSec=-1 disables.
func minTimeSinceWakeTestCfg(t *testing.T, modelName string, minSec int) *config.Config {
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
				CumemDrainTimeoutSeconds: 0, // off (opt-in default)
				WakeSettleSeconds:        0, // off (opt-in default)
				MinTimeSinceWakeSeconds:  minSec,
			},
		},
	}
	return cfg
}

// TestSleep_RefusedWithinMinTimeSinceWake — Fix A's primary case.
// Mark the instance as woken 1s ago, with a 5s gate; sleepInstance
// must refuse with ErrSleepTooSoonAfterWake, NOT call Mgr.Sleep, and
// leave instance state untouched (the gate fires before the state
// transition, like guard B).
func TestSleep_RefusedWithinMinTimeSinceWake(t *testing.T) {
	const model = "rapid-cycle-target"
	const reason = "test-rapid-cycle"
	cfg := minTimeSinceWakeTestCfg(t, model, 5) // 5s gate

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8210, []int{0}, false, StateReady, mgr)
	inst := s.instances[model]

	// Mark "woken 1s ago" — well inside the 5s gate.
	s.mu.Lock()
	inst.lastWakeAt = s.now().Add(-1 * time.Second)
	s.mu.Unlock()

	pre := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")

	err := s.sleepInstance(context.Background(), inst, 1, reason)

	if err == nil {
		t.Fatalf("expected ErrSleepTooSoonAfterWake, got nil")
	}
	if !errors.Is(err, ErrSleepTooSoonAfterWake) {
		t.Fatalf("expected error to wrap ErrSleepTooSoonAfterWake, got: %v", err)
	}
	if mgr.sleepCalls != 0 {
		t.Errorf("expected Mgr.Sleep to NOT be called when rapid-cycle gate fires; got %d call(s)", mgr.sleepCalls)
	}

	post := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")
	if got := post - pre; got != 1 {
		t.Errorf("expected SleepSkippedCooldownTotal{%q,min_time_since_wake} += 1, got delta %v", model, got)
	}

	// State must be untouched (gate runs before state flip).
	s.mu.RLock()
	gotState := inst.state
	gotDraining := inst.draining
	s.mu.RUnlock()
	if gotState != StateReady {
		t.Errorf("expected state untouched at StateReady, got %s", gotState)
	}
	if gotDraining {
		t.Errorf("expected draining unset before/after rapid-cycle gate refusal")
	}
}

// TestSleep_AllowedAfterCooldown — once the gate window has passed,
// sleepInstance proceeds normally. We backdate lastWakeAt to 6s ago
// against a 5s gate, then expect Mgr.Sleep to be called exactly
// once.
func TestSleep_AllowedAfterCooldown(t *testing.T) {
	const model = "rapid-cycle-elapsed"
	const reason = "test-rapid-cycle-elapsed"
	cfg := minTimeSinceWakeTestCfg(t, model, 5) // 5s gate

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8211, []int{0}, false, StateReady, mgr)
	inst := s.instances[model]

	// 6s ago — past the 5s gate.
	s.mu.Lock()
	inst.lastWakeAt = s.now().Add(-6 * time.Second)
	s.mu.Unlock()

	preSkip := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")

	if err := s.sleepInstance(context.Background(), inst, 1, reason); err != nil {
		t.Fatalf("expected sleep to succeed past the gate, got: %v", err)
	}
	if mgr.sleepCalls != 1 {
		t.Errorf("expected Mgr.Sleep called exactly once, got %d", mgr.sleepCalls)
	}

	postSkip := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")
	if got := postSkip - preSkip; got != 0 {
		t.Errorf("expected SleepSkippedCooldownTotal{%q,min_time_since_wake} unchanged after success, got delta %v",
			model, got)
	}
}

// TestSleep_FirstSleepHasNoWakeRecord — for a brand-new peer that
// has never been woken (lastWakeAt is zero value), the gate must NOT
// fire. The first sleep is allowed regardless of how recently the
// scheduler started.
func TestSleep_FirstSleepHasNoWakeRecord(t *testing.T) {
	const model = "rapid-cycle-virgin"
	const reason = "test-rapid-cycle-virgin"
	cfg := minTimeSinceWakeTestCfg(t, model, 5) // 5s gate

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &stubSleepableMgr{}
	s.SeedInstanceForTest(model, 8212, []int{0}, false, StateReady, mgr)
	inst := s.instances[model]

	// Sanity: lastWakeAt is the zero value (never woken).
	s.mu.RLock()
	if !inst.lastWakeAt.IsZero() {
		t.Fatalf("test setup invariant violated: expected lastWakeAt zero value, got %v", inst.lastWakeAt)
	}
	s.mu.RUnlock()

	preSkip := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")

	if err := s.sleepInstance(context.Background(), inst, 1, reason); err != nil {
		t.Fatalf("expected sleep to succeed for never-woken peer, got: %v", err)
	}
	if mgr.sleepCalls != 1 {
		t.Errorf("expected Mgr.Sleep called exactly once, got %d", mgr.sleepCalls)
	}

	postSkip := counterValue(metrics.SleepSkippedCooldownTotal, model, "min_time_since_wake")
	if got := postSkip - preSkip; got != 0 {
		t.Errorf("expected SleepSkippedCooldownTotal{%q,min_time_since_wake} unchanged on first-sleep, got delta %v",
			model, got)
	}
}
