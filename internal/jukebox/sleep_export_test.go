package jukebox

import (
	"context"
	"time"
)

// CheckIdleForTest is a test-only export that triggers a single idle-check
// pass synchronously. Lets tests validate auto-suspend without waiting for
// the 30s ticker. The _test.go suffix means this symbol is only present
// during `go test` builds.
func (s *Scheduler) CheckIdleForTest(ctx context.Context) { s.checkIdle(ctx) }

// SetDockerCmdForTest swaps the package-level docker exec hook used by
// RedeployMember. Tests use this to exercise the redeploy flow without
// spawning a real docker process. nil restores the default real-exec
// behavior. Only present in test builds. Tests should defer a
// SetDockerCmdForTest(nil) so they don't leak the hook into other tests.
//
// Implementation: stored via atomic.Pointer so concurrent test goroutines
// reading the hook (production code under test) and the test goroutine
// writing it are race-detector clean.
func SetDockerCmdForTest(fn func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	if fn == nil {
		redeployDockerCmd.Store(nil)
		return
	}
	wrapped := dockerCmdFn(fn)
	redeployDockerCmd.Store(&wrapped)
}

// SetSleepDockerCmdForTest swaps the package-level docker exec hook used
// by sleep.go (StopForEviction's docker stop, doColdLoad's docker start,
// coldLoadStoppedMember's best-effort cleanup stop). Tests use this to
// exercise the wake-from-Stopped + stop-on-evict flows without spawning
// a real docker process. nil restores the default real-exec behavior.
// Tests should defer a SetSleepDockerCmdForTest(nil) so they don't leak
// the hook into other tests.
//
// Implementation: stored via atomic.Pointer so concurrent test goroutines
// reading the hook (production code under test) and the test goroutine
// writing it are race-detector clean.
func SetSleepDockerCmdForTest(fn func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	if fn == nil {
		sleepDockerCmd.Store(nil)
		return
	}
	wrapped := sleepDockerCmdFn(fn)
	sleepDockerCmd.Store(&wrapped)
}

// SeedInstanceForTest is a test-only helper that registers a pre-built
// fake instance manager under `model` without going through
// AcquireRoute. Used by RedeployMember tests that need to assert
// behavior against an already-up Awake or Sleeping instance without
// the noise of the normal scheduling/start path. `state` controls the
// seeded schedInstance.state (typically StateReady or StateSleeping).
// `pinned` is recorded on the instance.
func (s *Scheduler) SeedInstanceForTest(model string, port int, gpus []int, pinned bool, state State, mgr InstanceManager) {
	inst := &schedInstance{
		model:      model,
		port:       port,
		gpus:       append([]int(nil), gpus...),
		pinned:     pinned,
		state:      state,
		mgr:        mgr,
		startedAt:  s.now(),
		lastUsedAt: s.now(),
	}
	s.mu.Lock()
	s.instances[model] = inst
	s.mu.Unlock()
}

// SetInstanceLastUsedForTest overrides the lastUsedAt timestamp of a
// seeded instance so tests can deterministically express "this model was
// used more recently than that one" without driving the clock through
// real request routing. Used by the Bug #4 current_model tests. No-op if
// the model isn't registered.
func (s *Scheduler) SetInstanceLastUsedForTest(model string, t time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if inst := s.instances[model]; inst != nil {
		inst.lastUsedAt = t
	}
}

// SetInstanceStateForTest overrides the schedInstance.state of a seeded
// instance. Used by tests that need to flip a registered instance's
// admission/lifecycle state directly (e.g. Bug #3 warming_up re-probe).
func (s *Scheduler) SetInstanceStateForTest(model string, state State) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if inst := s.instances[model]; inst != nil {
		inst.state = state
	}
}

// SetReprobeStoppedHealthBudgetForTest overrides the per-request /health
// probe budget used by ReprobeStoppedExternal (Bug #3). Returns the prior
// value so tests can defer-restore. Lets re-probe tests run fast without
// the 2s production budget.
func SetReprobeStoppedHealthBudgetForTest(d time.Duration) time.Duration {
	old := time.Duration(reprobeStoppedHealthBudgetNS.Load())
	reprobeStoppedHealthBudgetNS.Store(int64(d))
	return old
}
// /is_sleeping poll interval (production default: 5s). Returns the
// previous value so tests can defer a restore. Used by tests that
// exercise the cold-load loop and would otherwise pay 5s wall-clock
// per iteration.
//
// LOW #9: backed by atomic.Int64 (nanoseconds) so concurrent reads
// from production code under test and writes from a parallel test
// goroutine are race-detector clean.
//
// Usage:
//
//	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
//	defer SetColdLoadPollIntervalForTest(old)
func SetColdLoadPollIntervalForTest(d time.Duration) time.Duration {
	old := time.Duration(coldLoadPollInterval.Load())
	coldLoadPollInterval.Store(int64(d))
	return old
}

// SetColdLoadFailureCooldownForTest overrides the cold-load failure
// cooldown that KickColdLoad consults to suppress immediate re-kicks
// after a failure (production default: 30s). Returns the previous value
// so tests can defer a restore. Used by tests that need to exercise
// the cooldown gate without paying 30s wall-clock OR to disable the
// cooldown entirely (set to 0) when the test wants to fire repeated
// kicks back-to-back.
//
// Backed by atomic.Int64 (nanoseconds) so concurrent reads from
// production code under test and writes from a parallel test goroutine
// are race-detector clean.
//
// Usage:
//
//	old := SetColdLoadFailureCooldownForTest(50 * time.Millisecond)
//	defer SetColdLoadFailureCooldownForTest(old)
func SetColdLoadFailureCooldownForTest(d time.Duration) time.Duration {
	old := time.Duration(coldLoadFailureCooldown.Load())
	coldLoadFailureCooldown.Store(int64(d))
	return old
}

// SetInstanceLastWakeForTest sets the inst.lastWakeAt timestamp for the
// named instance, so tests can exercise the wake_settle / min_time_since_wake
// gates without waiting wall-clock for /wake_up to land. Returns false if
// the instance is not seeded.
func (s *Scheduler) SetInstanceLastWakeForTest(model string, t time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	inst, ok := s.instances[model]
	if !ok {
		return false
	}
	inst.lastWakeAt = t
	return true
}

// InstanceStateForTest returns the live schedInstance.state under the
// scheduler's RLock. Lets tests assert the TOCTOU / state-revert paths
// (MEDIUM #2 + MEDIUM #5) without reaching into private fields.
func (s *Scheduler) InstanceStateForTest(model string) (State, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	inst, ok := s.instances[model]
	if !ok {
		return "", false
	}
	return inst.state, true
}

// SetLifecycleAuditSinkForTest installs a hook that observes every
// LifecycleEvent emitted by LogLifecycleTransition. Returns the
// previous hook (or nil) so tests can defer-restore. nil clears any
// previously-installed sink.
//
// Used by Fix 11's regression test to count emitted audits per
// (peer, action, reason) tuple — the previous implementation double-
// emitted the redeploy-member-peer-stop audit on stop-failure paths,
// and a counting assertion on the audit stream is the cleanest way to
// guarantee that doesn't regress.
func SetLifecycleAuditSinkForTest(fn func(ev LifecycleEvent)) func(ev LifecycleEvent) {
	prev := lifecycleAuditSink.Load()
	if fn == nil {
		lifecycleAuditSink.Store(nil)
	} else {
		wrapped := lifecycleAuditSinkFn(fn)
		lifecycleAuditSink.Store(&wrapped)
	}
	if prev == nil {
		return nil
	}
	return *prev
}

// CheckExternalStartsForTest triggers one ExternalStartMonitor scan
// pass synchronously. Tests use this to validate the detector without
// driving the production 2s ticker (which would slow the suite by
// orders of magnitude). The _test.go suffix scopes the symbol to test
// builds only.
func (s *Scheduler) CheckExternalStartsForTest(ctx context.Context) {
	s.checkExternalStarts(ctx)
}

// SetExternalStartPollIntervalForTest overrides the ExternalStartMonitor
// tick cadence (production default: 2s). Returns the previous value so
// tests can defer a restore. Used by tests that exercise the live
// monitor goroutine and would otherwise pay 2s wall-clock per iteration.
func SetExternalStartPollIntervalForTest(d time.Duration) time.Duration {
	old := time.Duration(externalStartPollInterval.Load())
	externalStartPollInterval.Store(int64(d))
	return old
}

// ReconcileStateForTest triggers one StateReconciler scan pass
// synchronously. Tests use this to validate the reconciler without
// driving the production 5s ticker. The _test.go suffix scopes the
// symbol to test builds only.
func (s *Scheduler) ReconcileStateForTest(ctx context.Context) {
	s.reconcileState(ctx)
}

// SetStateReconcilePollIntervalForTest overrides the StateReconciler
// tick cadence (production default: 5s). Returns the previous value
// so tests can defer a restore.
func SetStateReconcilePollIntervalForTest(d time.Duration) time.Duration {
	old := time.Duration(stateReconcilePollInterval.Load())
	stateReconcilePollInterval.Store(int64(d))
	return old
}

// SetEvictionLoopPreActionHookForTest installs a hook that fires inside
// evictCrossGroupGPUContendersLocked AFTER the bulk-snapshot RUnlock
// and BEFORE the per-peer fresh RLock recheck. Lets a test mutate live
// state in that exact window to prove the MEDIUM #2 TOCTOU recheck
// actually catches the mutation.
//
// Returns a restore func; deferring it pins the hook lifetime to the
// test and prevents leakage across tests.
//
// Implementation: stored via atomic.Pointer so concurrent reads from
// production code under test and writes from this setter are
// race-detector clean.
func SetEvictionLoopPreActionHookForTest(h func(peerName string)) func() {
	old := evictionLoopPreActionHook.Swap(&h)
	return func() { evictionLoopPreActionHook.Store(old) }
}

// SetWakeVerifyProbeForTest installs a hook that replaces
// vllmcli.VerifyWakeWithProbe in the wake-success path. Tests use this
// to inject deterministic phantom-vs-healthy probe outcomes without
// spinning up an httptest server inside every wake test. Pass nil to
// restore the production probe.
//
// Returns a restore func; deferring it scopes the hook to a single test.
//
// Implementation: stored via atomic.Pointer so concurrent reads (the
// wake path under test) and writes (this setter) are race-detector clean.
func SetWakeVerifyProbeForTest(fn func(ctx context.Context, baseURL, model string, timeout time.Duration) error) func() {
	var old *wakeVerifyProbeFn
	if fn == nil {
		old = wakeVerifyProbe.Swap(nil)
	} else {
		wrapped := wakeVerifyProbeFn(fn)
		old = wakeVerifyProbe.Swap(&wrapped)
	}
	return func() { wakeVerifyProbe.Store(old) }
}

// SetRedeployMemberAsyncForTest installs a hook that replaces the
// background RedeployMember goroutine spawned by the phantom-wake
// recovery path. Tests use this to deterministically observe that a
// recreate WAS scheduled for a given model — without spinning a real
// docker process or paying the goroutine-timing variance. Pass nil to
// restore the production async-goroutine behavior.
//
// Returns a restore func; deferring it scopes the hook to a single test.
func SetRedeployMemberAsyncForTest(fn func(model string)) func() {
	var old *redeployAsyncFn
	if fn == nil {
		old = redeployMemberAsyncForTest.Swap(nil)
	} else {
		wrapped := redeployAsyncFn(fn)
		old = redeployMemberAsyncForTest.Swap(&wrapped)
	}
	return func() { redeployMemberAsyncForTest.Store(old) }
}
