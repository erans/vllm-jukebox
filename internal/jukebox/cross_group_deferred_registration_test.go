package jukebox

// Cross-group contender deferred-registration regression test for
// PR #27 architect-consensus fix (a983 + a3872 + afef, 2026-06-13).
//
// BUG: coldLoadStoppedMember pre-registered cross-group GPU contenders
// (e.g. vllm-main vs vllm-vision via swap_group=gpu-4-evict) in the
// coldLoadEviction gate BEFORE acquiring coldLoadMu. When the mutex was
// held by a prior cold-load (typical wait 0-5min, pathological 10+min),
// every inbound request to those healthy cross-group peers fast-503'd
// at handlers_proxy.go IsInColdLoadEviction with duration_ms=0 — a peer
// that was awake, healthy, serving traffic, with no eviction in flight
// against it would be artificially gated for the entire upstream
// queue-wait window.
//
// FIX: cross-group contenders are now registered INSIDE the
// WithColdLoadLock callback (only during the actual eviction window
// where evictCrossGroupGPUContendersLocked is about to mutate them).
// Same-group peers (shared swap_group invariant) remain pre-lock
// gated unconditionally.
//
// TEST: latch a first cold-load to hold coldLoadMu, fire a second
// cold-load targeting a peer with cross-group GPU overlap to the
// first, assert the cross-group contender is NOT in the eviction
// gate while the second cold-load is queued behind the mutex.

import (
	"context"
	"testing"
	"time"
)

// TestColdLoadStoppedMember_CrossGroupContenderNotGatedDuringLockWait
// is the architect-consensus regression test for the deferred
// cross-group eviction-gate registration in coldLoadStoppedMember.
//
// Topology (from crossGroupIntegrationConfig):
//   - holder:    GPUs [0,1], swap_group=holder-group, pinned, StateReady
//   - target:    GPUs [1,2], swap_group=target-group, StateStopped
//   - bystander: GPU  [3],   swap_group=bystander-group, StateReady
//
// holder overlaps with target on GPU 1 → holder is a cross-group GPU
// contender for a target cold-load.
//
// Strategy:
//
//  1. Goroutine A acquires coldLoadMu directly via
//     s.admission.WithColdLoadLock and holds it for a known duration
//     (300ms) without registering ANY eviction gate of its own —
//     this simulates a prior cold-load occupying the lock without
//     contaminating the gate state we're observing.
//
//  2. While A holds the lock, goroutine B fires coldLoadStoppedMember
//     for target. B will register the same-group eviction gate
//     unconditionally (pre-lock — correct invariant), emit
//     cold_load_queued_behind_lock, and block on Lock() inside
//     WithColdLoadLock.
//
//  3. Observation point: while B is queued behind A on the mutex,
//     assert IsInColdLoadEviction(holder) returns FALSE. Pre-fix this
//     returned TRUE (the bug). Post-fix it returns FALSE because the
//     cross-group registration is deferred until inside B's
//     WithColdLoadLock callback (which hasn't run yet).
//
//  4. A releases the lock. B acquires, registers cross-group
//     contenders inside the callback, and runs the (latched) docker
//     start. While docker start is latched, observe
//     IsInColdLoadEviction(holder) is now TRUE (the gate is
//     ACTIVE during the actual eviction window).
//
//  5. Release the docker-start latch, let B complete, assert
//     IsInColdLoadEviction(holder) is FALSE again (defer cleared it).
func TestColdLoadStoppedMember_CrossGroupContenderNotGatedDuringLockWait(t *testing.T) {
	s, _, _ := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateReady)

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)

	// Latch B's docker start so we can observe the in-lock gate state
	// at step 4 below before letting B's cold-load complete.
	bDockerReleased := make(chan struct{})
	SetSleepDockerCmdForTest(func(ctx context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			select {
			case <-bDockerReleased:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Step 1: Goroutine A acquires coldLoadMu for a known duration.
	// We use WithColdLoadLock directly (no cold-load body) so A's
	// callback never registers an eviction gate of its own — the
	// only gate-state we observe must be from B's invocation.
	aHoldDur := 300 * time.Millisecond
	aAcquired := make(chan struct{})
	aDone := make(chan struct{})
	go func() {
		s.admission.WithColdLoadLock(func() {
			close(aAcquired)
			time.Sleep(aHoldDur)
		})
		close(aDone)
	}()

	// Wait for A to hold the lock before kicking B (so B definitely
	// queues behind A rather than racing through).
	select {
	case <-aAcquired:
	case <-time.After(2 * time.Second):
		t.Fatalf("goroutine A failed to acquire coldLoadMu in 2s")
	}

	// Step 2: Goroutine B fires coldLoadStoppedMember for target.
	// B will block on Lock() inside WithColdLoadLock after registering
	// the same-group gate and emitting cold_load_queued_behind_lock.
	bDone := make(chan error, 1)
	go func() {
		bDone <- s.coldLoadStoppedMember(context.Background(), s.instances["target"], s.cfg.Models["target"])
	}()

	// Brief settle: B's pre-lock registration + Lock() entry is a few
	// nanoseconds; 20ms is generous. After this sleep, B is provably
	// queued behind A (A is sleeping for 300ms, B can't pass A's lock
	// hold). The 50ms safety margin (20ms settle + 250ms < aHoldDur)
	// guarantees we observe the queued-but-not-acquired state.
	time.Sleep(20 * time.Millisecond)

	// Step 3: THE KEY ASSERTION — while B is queued behind A on
	// coldLoadMu, holder (cross-group GPU contender) must NOT be in
	// the eviction gate. Pre-fix this returned true (the bug); post-
	// fix it returns false (cross-group registration is deferred
	// until inside B's WithColdLoadLock callback).
	if s.IsInColdLoadEviction("holder") {
		// Clean up before failing.
		close(bDockerReleased)
		<-aDone
		<-bDone
		t.Fatalf("REGRESSION: IsInColdLoadEviction(holder) returned true while B is queued behind A on coldLoadMu — cross-group contender was pre-registered before lock acquire (the bug this PR fixes)")
	}

	// Sanity: target itself IS in the gate (same-group registration
	// is intentionally pre-lock for the swap_group invariant).
	if !s.IsInColdLoadEviction("target") {
		close(bDockerReleased)
		<-aDone
		<-bDone
		t.Fatalf("IsInColdLoadEviction(target) returned false while B is queued — same-group pre-lock registration regressed")
	}

	// Sanity: bystander (no GPU overlap) must never be in the gate.
	if s.IsInColdLoadEviction("bystander") {
		close(bDockerReleased)
		<-aDone
		<-bDone
		t.Fatalf("IsInColdLoadEviction(bystander) returned true (no GPU overlap with target) — over-broad gate")
	}

	// Step 4: Let A complete; B should acquire the lock and proceed
	// into its callback, registering cross-group contenders inside
	// the lock, then block on the docker-start latch.
	<-aDone

	// Wait for B to advance into docker start (which means the
	// cross-group registration inside the callback has run and the
	// gate is now ACTIVE during the actual eviction window).
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if s.IsInColdLoadEviction("holder") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !s.IsInColdLoadEviction("holder") {
		close(bDockerReleased)
		<-bDone
		t.Fatalf("IsInColdLoadEviction(holder) returned false during B's in-lock eviction window — cross-group in-lock registration regressed")
	}

	// Step 5: Release docker-start latch, drain B, confirm gate clears.
	close(bDockerReleased)
	if err := <-bDone; err != nil {
		t.Logf("B cold-load completed with err=%v (acceptable — gate behaviour was the assertion)", err)
	}
	if s.IsInColdLoadEviction("holder") {
		t.Errorf("IsInColdLoadEviction(holder) still true after B completed — defer in callback failed to clear cross-group registration")
	}
	if s.IsInColdLoadEviction("target") {
		t.Errorf("IsInColdLoadEviction(target) still true after B completed — outer defer failed to clear same-group registration")
	}
}
