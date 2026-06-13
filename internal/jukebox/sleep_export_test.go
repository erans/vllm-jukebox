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

// SetColdLoadPollIntervalForTest overrides the cold-load + redeploy
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
