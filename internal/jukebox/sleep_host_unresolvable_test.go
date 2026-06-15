package jukebox

// Regression test for the permanently-absent-member state-machine bug
// (forensics A4).
//
// THE BUG: when a sleep-mode external member's backend host is
// unresolvable (NXDOMAIN — container never deployed, DNS name gone), the
// auto-suspend path's wedge-probe "rolled back optimistically" and left
// the member in StateReady forever. Consequences:
//
//   - the member kept appearing in /v1/models (StateReady is routable);
//   - the IdleMonitor (auto-suspend) re-selected it every 30s tick
//     (its candidate filter is `state == StateReady && !pinned && idle`),
//     called sleepInstance, failed the same NXDOMAIN way, rolled back to
//     StateReady, and WARN-spammed `auto_suspend_failed` indefinitely
//     (1397 lines / session observed).
//
// THE FIX: in sleepInstance's sleep-API-error path, when the sleep error
// AND every wedge-probe error are *net.DNSError "no such host", the
// backend is DEFINITIVELY absent — reconcile DOWN to StateStopped +
// admissionStopped instead of optimistic-rollback-to-Ready. StateStopped
// drops the member out of /v1/models AND out of the auto-suspend
// candidate set, so the hammering loop stops.
//
// This test asserts BOTH halves:
//   1. after one checkIdle pass, the member is NOT StateReady (it is
//      StateStopped) + admission agrees (IsStopped) — pre-fix this FAILED
//      (stayed StateReady).
//   2. a SECOND checkIdle pass does NOT call Sleep again — pre-fix this
//      FAILED (StateReady member re-selected → repeated auto-suspend
//      attempt → the WARN-spam loop).
//
// The complementary TestSleepTransientProbeError_StillOptimisticRollback
// proves we did NOT over-rotate: a TRANSIENT (non-NXDOMAIN) probe failure
// must still roll back to StateReady so the backend (which exists) gets
// retried.

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// nxdomainMgr is a SleepCapable whose Sleep and IsSleeping both fail the
// way an unresolvable host fails: a *net.DNSError with IsNotFound (the
// canonical NXDOMAIN shape), wrapped the same way the real vllmcli.Sleep /
// IsSleeping wrap their http.DefaultClient.Do errors (fmt.Errorf "%w").
type nxdomainMgr struct {
	sleepCalls      int
	isSleepingCalls int
}

func newDNSNotFoundErr(host string) error {
	// Mirror the *url.Error{ *net.OpError{ *net.DNSError } } chain that
	// http.DefaultClient.Do produces against an unresolvable host. We use a
	// bare *net.DNSError wrapped once — errors.As walks the chain either
	// way, and the production code wraps with fmt.Errorf("...: %w", err).
	return &net.DNSError{
		Err:        "no such host",
		Name:       host,
		IsNotFound: true,
	}
}

func (m *nxdomainMgr) Start(_ context.Context, _ string) (int, error) { return 1, nil }
func (m *nxdomainMgr) Stop(_ context.Context) error                   { return nil }
func (m *nxdomainMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *nxdomainMgr) CurrentPID() int                                { return 1 }
func (m *nxdomainMgr) BaseURL() string                                { return "http://vllm-ghost.mesh:8000" }
func (m *nxdomainMgr) Sleep(_ context.Context, _ int) error {
	m.sleepCalls++
	return newDNSNotFoundErr("vllm-ghost.mesh")
}
func (m *nxdomainMgr) Wake(_ context.Context, _ time.Duration) error { return nil }
func (m *nxdomainMgr) IsSleeping(_ context.Context) (bool, error) {
	m.isSleepingCalls++
	return false, newDNSNotFoundErr("vllm-ghost.mesh")
}

// transientMgr is a SleepCapable whose Sleep + IsSleeping fail with a
// NON-NXDOMAIN error (the backend EXISTS but is momentarily unanswerable).
// Used by the complementary test to prove the optimistic rollback is
// preserved for transient failures.
type transientMgr struct {
	sleepCalls int
}

func (m *transientMgr) Start(_ context.Context, _ string) (int, error) { return 1, nil }
func (m *transientMgr) Stop(_ context.Context) error                   { return nil }
func (m *transientMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *transientMgr) CurrentPID() int                                { return 1 }
func (m *transientMgr) BaseURL() string                                { return "http://vllm-real.mesh:8000" }
func (m *transientMgr) Sleep(_ context.Context, _ int) error {
	m.sleepCalls++
	return errors.New("POST /sleep: read tcp 198.51.100.5:8000: i/o timeout")
}
func (m *transientMgr) Wake(_ context.Context, _ time.Duration) error { return nil }
func (m *transientMgr) IsSleeping(_ context.Context) (bool, error) {
	return false, errors.New("Get \"http://vllm-real.mesh:8000/is_sleeping\": context deadline exceeded")
}

// hostUnresolvableTestCfg builds a one-model scheduler config with a short
// idle_timeout so the auto-suspend candidate filter selects the seeded
// instance immediately. The cumem/wake-settle guards are disabled so the
// test exercises ONLY the sleep-API-error → wedge-probe → reconcile path.
func hostUnresolvableTestCfg(modelName string) *config.Config {
	return &config.Config{
		Scheduler: &config.SchedulerConfig{
			PortRangeStart: 8200,
			PortRangeEnd:   8299,
		},
		VLLM: config.VLLMConfig{
			Port:           8000,
			StartupTimeout: config.Duration{Duration: 1 * time.Second},
			DrainTimeout:   config.Duration{Duration: 10 * time.Millisecond},
		},
		Models: map[string]config.ModelConfig{
			modelName: {
				Path:                    "/models/" + modelName,
				GPUs:                    []int{0},
				SleepMode:               true,
				IdleTimeout:             config.Duration{Duration: 1 * time.Millisecond},
				MinTimeSinceWakeSeconds: -1,   // disable rapid-cycle gate
				ExpectedVRAMMBPerGPU:    8000, // admission-tracked (so NotifyStopped/IsStopped apply)
			},
		},
	}
}

// TestAutoSuspend_UnresolvableHost_ReconcilesStoppedAndStopsHammering is
// the load-bearing regression. FAILS pre-fix (member stays StateReady;
// second tick calls Sleep again), PASSES post-fix.
func TestAutoSuspend_UnresolvableHost_ReconcilesStoppedAndStopsHammering(t *testing.T) {
	const model = "vllm-ghost"

	cfg := hostUnresolvableTestCfg(model)
	inv := newRedeployInventory(map[int]int{0: 24000})
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	// Wire admission and seed the model as adopted-awake so it mirrors a
	// live Ready member whose VRAM is booked. The fix must flip admission
	// to Stopped, which we assert via IsStopped.
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &SchedulerEvictor{S: s})
	s.SetAdmission(a)
	a.NotifyAdopted(model, true /* awake */)

	mgr := &nxdomainMgr{}
	s.SeedInstanceForTest(model, 8202, []int{0}, false /* pinned */, StateReady, mgr)
	// Make it idle past the 1ms idle_timeout so the auto-suspend candidate
	// filter selects it.
	s.SetInstanceLastUsedForTest(model, time.Now().Add(-time.Hour))

	// --- First auto-suspend pass: the bug's trigger. ---
	s.CheckIdleForTest(context.Background())

	if mgr.sleepCalls != 1 {
		t.Fatalf("expected exactly 1 Sleep attempt on first pass, got %d", mgr.sleepCalls)
	}
	// The wedge-probe must have actually run (3 bounded retries) — that is
	// what classifies the failure as host-unresolvable rather than guessing.
	if mgr.isSleepingCalls == 0 {
		t.Fatalf("expected the wedge-probe (IsSleeping) to run on the sleep-API error; got 0 calls")
	}

	// HALF 1: member must NOT be StateReady — it must reconcile to Stopped.
	gotState, ok := s.InstanceStateForTest(model)
	if !ok {
		t.Fatalf("instance %q vanished from scheduler", model)
	}
	if gotState == StateReady {
		t.Fatalf("BUG: unresolvable-host member rolled back to StateReady (optimistic) — it must reconcile DOWN; got %s", gotState)
	}
	if gotState != StateStopped {
		t.Fatalf("expected StateStopped after unresolvable-host reconcile, got %s", gotState)
	}
	// Admission must agree the member is Stopped (out of the VRAM books +
	// the wake path's TOCTOU gate).
	if !a.IsStopped(model) {
		t.Fatalf("expected admission.IsStopped(%q)=true after reconcile; auto-suspend hammering will not stop otherwise", model)
	}

	// --- Second auto-suspend pass: the bug's symptom (the 30s hammer). ---
	sleepCallsBefore := mgr.sleepCalls
	s.CheckIdleForTest(context.Background())

	// HALF 2: a Stopped member is NOT an auto-suspend candidate, so Sleep
	// must NOT be attempted again. Pre-fix the StateReady member would be
	// re-selected → another Sleep attempt → the indefinite WARN loop.
	if mgr.sleepCalls != sleepCallsBefore {
		t.Fatalf("BUG: auto-suspend re-attempted Sleep on a permanently-absent member (sleepCalls %d → %d) — this is the 30s WARN-spam hammer",
			sleepCallsBefore, mgr.sleepCalls)
	}
}

// TestSleepTransientProbeError_StillOptimisticRollback proves the fix is
// SCOPED: a transient (non-NXDOMAIN) failure must still roll back to
// StateReady so the (existing) backend is retried, exactly as before.
// Guards against over-rotating the fix into "any probe failure stops the
// member".
func TestSleepTransientProbeError_StillOptimisticRollback(t *testing.T) {
	const model = "vllm-real"

	cfg := hostUnresolvableTestCfg(model)
	inv := newRedeployInventory(map[int]int{0: 24000})
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &SchedulerEvictor{S: s})
	s.SetAdmission(a)
	a.NotifyAdopted(model, true)

	mgr := &transientMgr{}
	s.SeedInstanceForTest(model, 8203, []int{0}, false, StateReady, mgr)
	s.SetInstanceLastUsedForTest(model, time.Now().Add(-time.Hour))

	s.CheckIdleForTest(context.Background())

	if mgr.sleepCalls != 1 {
		t.Fatalf("expected 1 Sleep attempt, got %d", mgr.sleepCalls)
	}
	gotState, ok := s.InstanceStateForTest(model)
	if !ok {
		t.Fatalf("instance %q vanished", model)
	}
	// Transient failure: the backend exists, keep the optimistic rollback.
	if gotState != StateReady {
		t.Fatalf("transient probe failure must roll back to StateReady (backend exists, retry next tick); got %s", gotState)
	}
	if a.IsStopped(model) {
		t.Fatalf("transient probe failure must NOT mark admission Stopped; the backend exists")
	}
}
