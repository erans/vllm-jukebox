package jukebox

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// stateReconcilerTestConfig matches the cross-group layout used by
// external_start_test.go but tunes it for state-reconciler scenarios:
// `holder` is the typical Ready peer; `target` is the peer whose
// admission state we'll drive into drift cases.
const stateReconcilerTestConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  holder:
    lifecycle: external
    host: vllm-holder
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: holder-group
    pinned: true
    expected_vram_mb_per_gpu: 12000
    sleep_l1_residual_mb: 1000
    wake_timeout: 1s
  target:
    lifecycle: external
    host: vllm-target
    port: 9002
    gpus: [1, 2]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: target-group
    expected_vram_mb_per_gpu: 11000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// makeStateReconcilerScheduler builds a scheduler with `holder` and
// `target` seeded into the requested states. Mirrors
// makeExternalStartScheduler — distinct because reconciler tests
// commonly seed holder Ready and target Ready/Sleeping (not Stopped).
func makeStateReconcilerScheduler(t *testing.T, holderState, targetState State) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(stateReconcilerTestConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000, 1: 24000, 2: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"holder": {port: 9001},
		"target": {port: 9002},
	}
	for name, m := range mgrs {
		modelCfg := cfg.Models[name]
		var st State
		switch name {
		case "holder":
			st = holderState
		case "target":
			st = targetState
		}
		switch st {
		case StateReady:
			m.pid.Store(int64(7000 + m.port))
			m.isSleeping.Store(false)
		case StateSleeping:
			m.pid.Store(int64(7000 + m.port))
			m.isSleeping.Store(true)
		case StateStopped:
			m.pid.Store(0)
			m.isSleeping.Store(true)
		}
		s.SeedInstanceForTest(name, m.port, modelCfg.GPUs, modelCfg.Pinned != nil && *modelCfg.Pinned, st, m)
	}

	a.mu.Lock()
	if targetState == StateStopped {
		a.markStoppedLocked(a.models["target"])
	}
	if holderState == StateStopped {
		a.markStoppedLocked(a.models["holder"])
	}
	a.mu.Unlock()

	return s, a, mgrs
}

// TestStateReconciler_FlipsReadyToStoppedWhenContainerExited is the
// load-bearing Issue #8 regression. Admission records `target` as
// Ready (admission is holding ~11 GiB of VRAM budget for it), but
// docker reports `vllm-target` as exited (someone ran `docker stop
// vllm-target` out-of-band, or the container OOM-died after init).
//
// Without the reconciler, admission keeps over-counting that VRAM
// indefinitely — every subsequent admission decision is wrong until
// some path (e.g. the next request through KickColdLoad) happens to
// notice. The reconciler MUST:
//   1. Call admission.NotifyStopped("target") to release the budget.
//   2. Flip schedInstance.state from Ready to Stopped.
//
// Pre-fix behavior (when this file did not exist): admission stayed at
// Ready, GPU budget showed ~11 GiB consumed by a peer that wasn't
// actually present. Post-fix: admission and instance state both move
// to Stopped on the next tick.
func TestStateReconciler_FlipsReadyToStoppedWhenContainerExited(t *testing.T) {
	s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateReady)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running") // holder legitimately Ready
	rec.setStatus("vllm-target", "exited")  // <-- container is dead!
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	// Pre-condition: target is admission-Ready and instance-Ready.
	if st, ok := s.InstanceStateForTest("target"); !ok || st != StateReady {
		t.Fatalf("setup: expected target instance state Ready; got %q ok=%v", st, ok)
	}

	s.ReconcileStateForTest(context.Background())

	// Post-condition: target must have been flipped to Stopped because
	// docker reported it as exited.
	st, ok := s.InstanceStateForTest("target")
	if !ok {
		t.Fatalf("target instance vanished after reconcile")
	}
	if st != StateStopped {
		t.Fatalf("expected target state Stopped after reconcile; got %q", st)
	}

	// Holder is legitimately Ready (docker says running, admission says
	// Ready). Must NOT have been touched.
	if hst, ok := s.InstanceStateForTest("holder"); !ok || hst != StateReady {
		t.Fatalf("holder must remain Ready (no drift); got %q ok=%v", hst, ok)
	}
}

// TestStateReconciler_FlipsSleepingToStoppedWhenContainerExited covers
// the second drift class: admission records `target` as Sleeping
// (vLLM /sleep was called, VRAM dropped to L1 residual ~0), but the
// container itself is gone (operator `docker stop` after the peer
// went to sleep, or sleep-from-stopped raced badly). Admission is
// over-counting L1 residual budget. Reconciler MUST flip to Stopped.
func TestStateReconciler_FlipsSleepingToStoppedWhenContainerExited(t *testing.T) {
	s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateSleeping)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "exited")
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.ReconcileStateForTest(context.Background())

	st, ok := s.InstanceStateForTest("target")
	if !ok {
		t.Fatalf("target instance vanished after reconcile")
	}
	if st != StateStopped {
		t.Fatalf("expected target Sleeping→Stopped after reconcile (container exited); got %q", st)
	}
}

// TestStateReconciler_NoOpWhenInSync asserts the steady-state path:
// admission says Ready, docker says running → no action. Critical
// regression test for any future "be aggressive" refactor that might
// start mutating state on every tick.
func TestStateReconciler_NoOpWhenInSync(t *testing.T) {
	s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateReady)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running") // matches admission Ready
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.ReconcileStateForTest(context.Background())

	for _, name := range []string{"holder", "target"} {
		st, ok := s.InstanceStateForTest(name)
		if !ok {
			t.Fatalf("%s instance vanished after no-op reconcile", name)
		}
		if st != StateReady {
			t.Fatalf("%s must remain Ready (in-sync, no drift); got %q", name, st)
		}
	}

	// No stop / start docker calls should have fired (the reconciler
	// only inspects, then no-ops).
	if len(rec.stops) != 0 {
		t.Fatalf("reconciler must not issue docker stop on in-sync state; got stops=%v", rec.stops)
	}
	if len(rec.starts) != 0 {
		t.Fatalf("reconciler must not issue docker start on in-sync state; got starts=%v", rec.starts)
	}
}

// TestStateReconciler_NoOpForAdmissionDisabledModel: a model with no
// expected_vram_mb_per_gpu is invisible to admission. The reconciler
// must NOT touch it — the AdmissionEnabled gate is the selection filter
// and a future "tighten the filter" refactor must preserve it.
func TestStateReconciler_NoOpForAdmissionDisabledModel(t *testing.T) {
	const cfg = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  untracked:
    lifecycle: external
    host: vllm-untracked
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    wake_timeout: 1s
`
	parsed, err := config.Load([]byte(cfg))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(parsed, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(parsed, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)
	mgr := &fakeRedeployMgr{port: 9001}
	mgr.pid.Store(int64(7000 + mgr.port))
	mgr.isSleeping.Store(false)
	modelCfg := parsed.Models["untracked"]
	s.SeedInstanceForTest("untracked", mgr.port, modelCfg.GPUs, false, StateReady, mgr)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-untracked", "exited") // would normally be drift, but admission-disabled
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.ReconcileStateForTest(context.Background())

	// The reconciler must NOT even inspect an admission-disabled model.
	for _, c := range rec.calls {
		if strings.Contains(c, "inspect") && strings.Contains(c, "vllm-untracked") {
			t.Fatalf("admission-disabled model must NOT be inspected; got call %q", c)
		}
	}
	// And of course state must be unchanged.
	if st, _ := s.InstanceStateForTest("untracked"); st != StateReady {
		t.Fatalf("admission-disabled model state must be untouched; got %q", st)
	}
}

// TestStateReconciler_MonitorLoopExitsOnContextCancel exercises the
// production goroutine path: start the monitor, cancel the ctx, and
// verify it returns within a bounded window. Guards against a missing
// <-ctx.Done() branch in a future refactor.
func TestStateReconciler_MonitorLoopExitsOnContextCancel(t *testing.T) {
	s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateReady)

	// Very tight cadence so the test doesn't pay 5s per tick.
	old := SetStateReconcilePollIntervalForTest(5 * time.Millisecond)
	defer SetStateReconcilePollIntervalForTest(old)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var ranAtLeastOnce atomic.Bool
	inner := rec.handler()
	SetSleepDockerCmdForTest(func(ctx context.Context, name string, args ...string) ([]byte, error) {
		ranAtLeastOnce.Store(true)
		return inner(ctx, name, args...)
	})
	defer SetSleepDockerCmdForTest(nil)

	go func() {
		s.StateReconciler(ctx)
		close(done)
	}()

	// Give the ticker enough wall-clock to fire a few times.
	time.Sleep(50 * time.Millisecond)
	if !ranAtLeastOnce.Load() {
		t.Fatalf("StateReconciler did not run any ticks in 50ms")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("StateReconciler did not return within 500ms of ctx cancel")
	}
}

// TestStateReconciler_DoesNotDuplicateExternalStartRemediation: when
// admission says Stopped and docker says running, ExternalStartMonitor
// owns that case (it issues docker stop + KickColdLoad). The
// reconciler must NOT also fire — two racing remediations would log
// spam and double-kick. This test verifies the reconciler stays silent.
func TestStateReconciler_DoesNotDuplicateExternalStartRemediation(t *testing.T) {
	s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateStopped)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running") // ExternalStartMonitor's territory
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.ReconcileStateForTest(context.Background())

	// Reconciler must not have issued any docker stop / start — that's
	// ExternalStartMonitor's job, not ours.
	if len(rec.stops) != 0 {
		t.Fatalf("reconciler must not stop containers on admission-Stopped + docker-running (that's ExternalStartMonitor's job); got stops=%v", rec.stops)
	}
	if len(rec.starts) != 0 {
		t.Fatalf("reconciler must not start containers; got starts=%v", rec.starts)
	}
	// target instance state must be unchanged — left for ExternalStartMonitor.
	if st, _ := s.InstanceStateForTest("target"); st != StateStopped {
		t.Fatalf("reconciler must leave Stopped+running state alone; got %q", st)
	}
}

// TestStateReconciler_SkipsPeerMidColdLoad is the same dedupe-race
// regression as TestExternalStart_DoesNotRefireOnOwnInFlightColdLoad,
// applied to reconcileOne. The cold-load path leaves inst.state at
// StateStopped while the container is `running` and admission may
// still report Sleeping (for a pinned peer being woken). Without the
// guard, the reconciler's Sleeping+running branch is a no-op, but the
// Stopped branch could re-introduce reconciliation actions if future
// refactors change the matrix. The guard is belt-and-suspenders to
// keep the reconciler from ever competing with an in-flight cold-load,
// matching the architect's diagnosis 2026-06-14.
//
// We assert the strongest possible property: the guard short-circuits
// BEFORE `docker inspect` runs, so the inspect-call recorder shows no
// entry for the mid-cold-load peer.
func TestStateReconciler_SkipsPeerMidColdLoad(t *testing.T) {
	t.Run("guarded_by_coldLoadKicks", func(t *testing.T) {
		s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateSleeping)

		rec := newDockerInspectRecorder()
		rec.setStatus("vllm-holder", "running")
		rec.setStatus("vllm-target", "running")
		SetSleepDockerCmdForTest(rec.handler())
		defer SetSleepDockerCmdForTest(nil)

		s.mu.Lock()
		if s.coldLoadKicks == nil {
			s.coldLoadKicks = make(map[string]bool)
		}
		s.coldLoadKicks["target"] = true
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			delete(s.coldLoadKicks, "target")
			s.mu.Unlock()
		}()

		s.ReconcileStateForTest(context.Background())

		// Guard short-circuits BEFORE inspect → no inspect call for vllm-target.
		rec.mu.Lock()
		defer rec.mu.Unlock()
		for _, c := range rec.calls {
			if strings.Contains(c, "inspect") && strings.Contains(c, "vllm-target") {
				t.Fatalf("reconciler inspected vllm-target despite coldLoadKicks guard; call=%q (all=%v)", c, rec.calls)
			}
		}
		// And of course no stop/start.
		if len(rec.stops) != 0 || len(rec.starts) != 0 {
			t.Fatalf("reconciler took action on mid-cold-load peer; stops=%v starts=%v", rec.stops, rec.starts)
		}
	})

	t.Run("guarded_by_coldLoadEviction", func(t *testing.T) {
		s, _, _ := makeStateReconcilerScheduler(t, StateReady, StateSleeping)

		rec := newDockerInspectRecorder()
		rec.setStatus("vllm-holder", "running")
		rec.setStatus("vllm-target", "running")
		SetSleepDockerCmdForTest(rec.handler())
		defer SetSleepDockerCmdForTest(nil)

		s.markColdLoadEviction("target")
		defer s.unmarkColdLoadEviction("target")

		s.ReconcileStateForTest(context.Background())

		rec.mu.Lock()
		defer rec.mu.Unlock()
		for _, c := range rec.calls {
			if strings.Contains(c, "inspect") && strings.Contains(c, "vllm-target") {
				t.Fatalf("reconciler inspected vllm-target despite IsInColdLoadEviction guard; call=%q (all=%v)", c, rec.calls)
			}
		}
		if len(rec.stops) != 0 || len(rec.starts) != 0 {
			t.Fatalf("reconciler took action on mid-cold-load peer; stops=%v starts=%v", rec.stops, rec.starts)
		}
	})
}
