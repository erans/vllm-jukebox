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

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
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

// crossGroupNonPinnedConfig: same shape as crossGroupIntegrationConfig
// but holder is NON-PINNED (pinned: false) with sleep_mode=true.
// This is the more common PRODUCTION topology (vllm-vision /
// vllm-moe vs vllm-main): the cross-group GPU contender is a regular
// swap-group peer, not a pinned holder.
//
// Deferred-registration semantics are IDENTICAL regardless of pinned
// flag — both pinned and non-pinned cross-group GPU contenders are
// admission-evictable via their swap_group and therefore both must be
// gated INSIDE the WithColdLoadLock callback (not pre-lock). This test
// proves that explicitly so the production topology has direct coverage
// alongside the existing pinned-holder case.
const crossGroupNonPinnedConfig = `
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
    pinned: false
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
  bystander:
    lifecycle: external
    host: vllm-bystander
    port: 9003
    gpus: [3]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: bystander-group
    expected_vram_mb_per_gpu: 8000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// makeCrossGroupNonPinnedScheduler mirrors
// makeCrossGroupIntegrationScheduler but uses crossGroupNonPinnedConfig
// (holder.pinned=false). Same seed mechanics, same admission wiring.
func makeCrossGroupNonPinnedScheduler(t *testing.T, holderState, targetState, bystanderState State) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(crossGroupNonPinnedConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"holder":    {port: 9001},
		"target":    {port: 9002},
		"bystander": {port: 9003},
	}
	for name, m := range mgrs {
		modelCfg, ok := cfg.Models[name]
		if !ok {
			continue
		}
		var st State
		switch name {
		case "holder":
			st = holderState
		case "target":
			st = targetState
		case "bystander":
			st = bystanderState
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
	a.mu.Unlock()

	return s, a, mgrs
}

// TestColdLoadStoppedMember_CrossGroupContenderNotGatedDuringLockWait_NonPinned
// is the NON-PINNED variant of the deferred-registration regression
// test. Production topology coverage: vllm-vision / vllm-moe / vllm-main
// peers are non-pinned swap-group members, NOT pinned holders. The
// deferred-registration semantics must hold identically:
//
//  1. While a prior cold-load (goroutine A) holds coldLoadMu, a second
//     cold-load (goroutine B) targeting `target` must NOT pre-register
//     the non-pinned cross-group contender (`holder`) in the eviction
//     gate. Pre-fix this fast-503'd every request to holder for the
//     full mutex-wait window even though nothing was evicting it.
//
//  2. Once B acquires the lock and runs its WithColdLoadLock callback,
//     cross-group contenders are registered INSIDE the lock and the
//     gate is ACTIVE during the actual eviction window (docker start).
//
//  3. After B completes, the deferred defer clears the cross-group
//     registration.
//
// Mirrors TestColdLoadStoppedMember_CrossGroupContenderNotGatedDuringLockWait
// exactly except for the holder.pinned topology.
func TestColdLoadStoppedMember_CrossGroupContenderNotGatedDuringLockWait_NonPinned(t *testing.T) {
	s, _, _ := makeCrossGroupNonPinnedScheduler(t, StateReady, StateStopped, StateReady)

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)

	// Latch B's docker start so we can observe the in-lock gate state
	// before letting B's cold-load complete.
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

	// Step 1: Goroutine A holds coldLoadMu for a known duration without
	// registering any eviction gate of its own.
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

	select {
	case <-aAcquired:
	case <-time.After(2 * time.Second):
		t.Fatalf("goroutine A failed to acquire coldLoadMu in 2s")
	}

	// Step 2: Goroutine B fires coldLoadStoppedMember for target.
	bDone := make(chan error, 1)
	go func() {
		bDone <- s.coldLoadStoppedMember(context.Background(), s.instances["target"], s.cfg.Models["target"])
	}()

	// Brief settle so B is provably queued behind A.
	time.Sleep(20 * time.Millisecond)

	// Step 3: THE KEY ASSERTION — holder (NON-PINNED cross-group GPU
	// contender) must NOT be in the eviction gate while B is queued.
	if s.IsInColdLoadEviction("holder") {
		close(bDockerReleased)
		<-aDone
		<-bDone
		t.Fatalf("REGRESSION: IsInColdLoadEviction(holder) returned true while B is queued behind A on coldLoadMu — non-pinned cross-group contender was pre-registered before lock acquire (the bug this PR fixes)")
	}

	// Sanity: target itself IS in the gate (same-group pre-lock).
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

	// Step 4: Let A complete; B acquires lock, registers cross-group
	// contenders inside the callback, then blocks on docker-start latch.
	<-aDone

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
		t.Fatalf("IsInColdLoadEviction(holder) returned false during B's in-lock eviction window — non-pinned cross-group in-lock registration regressed")
	}

	// Step 5: Release docker-start latch, drain B, confirm gate clears.
	close(bDockerReleased)
	if err := <-bDone; err != nil {
		t.Logf("B cold-load completed with err=%v (acceptable — gate behaviour was the assertion)", err)
	}
	if s.IsInColdLoadEviction("holder") {
		t.Errorf("IsInColdLoadEviction(holder) still true after B completed — defer in callback failed to clear non-pinned cross-group registration")
	}
	if s.IsInColdLoadEviction("target") {
		t.Errorf("IsInColdLoadEviction(target) still true after B completed — outer defer failed to clear same-group registration")
	}
}
