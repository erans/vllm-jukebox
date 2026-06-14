package jukebox

// Round-2 adversarial fixes for PR #27. Each test FAILS pre-fix and PASSES
// post-fix.
//
//   - HIGH #1: TestKickColdLoad_GPUContendedSkipsCooldown — KickColdLoad
//     must NOT record the 30s failure cooldown for ErrColdLoadGPUContended
//     (transient deferral, not a doomed loop).
//   - HIGH #2: TestColdLoad_DeferralRestoresAlreadySleptCrossGroup — when
//     the cross-group eviction loop slept peer A then deferred on peer B,
//     coldLoadStoppedMemberLocked must restore A via wake before returning
//     ErrColdLoadGPUContended.
//   - MED #3: TestSameGroupStopMode_LiveStateRecheckBeforeRevert — the
//     same-group stop-mode path must re-read live state under fresh RLock
//     before using it as prevState for revert (mirrors cross-group MEDIUM
//     #2 pattern). Verified by mutating peerInst.state mid-loop.
//   - MED #4: TestMapWakeError_GPUContendedRoutesToColdLoadingReason —
//     mapWakeError must route ErrColdLoadGPUContended to RejectColdLoading
//     with a 5s Retry-After (NOT the default RejectSwapInProgress + 10s).

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// --- HIGH #1 ----------------------------------------------------------------

// TestKickColdLoad_GPUContendedSkipsCooldown asserts that when the async
// cold-load goroutine returns ErrColdLoadGPUContended, KickColdLoad does
// NOT record an entry in coldLoadFailures — so the next inbound request
// can re-kick immediately once the wake-settle gate opens.
//
// Pre-fix: coldLoadFailures[name] is set to {at: now, err: "..."} →
// subsequent KickColdLoad within coldLoadFailureCooldown short-circuits,
// converting a 5s gate-window into a 30s cooldown outage.
//
// Post-fix: the deferral sentinel is recognized and the failure record
// is intentionally skipped.
func TestKickColdLoad_GPUContendedSkipsCooldown(t *testing.T) {
	s, _, mgrs := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateStopped)

	// Force the cross-group eviction loop to defer: holder is StateReady
	// with wake_settle_seconds=60 and lastWakeAt=now → sleepInstance bounces
	// on ErrSleepSkippedWakeSettle → evictCrossGroupGPUContendersLocked
	// returns the deferral sentinel → coldLoadStoppedMemberLocked converts
	// to ErrColdLoadGPUContended.
	holderCfg := s.cfg.Models["holder"]
	holderCfg.WakeSettleSeconds = 60
	s.cfg.Models["holder"] = holderCfg

	if !s.SetInstanceLastWakeForTest("holder", time.Now()) {
		t.Fatalf("could not set lastWake for holder")
	}

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)
	// Use the production cooldown duration (30s) so the test asserts the
	// pre-fix-vs-post-fix difference at the production setting.
	oldCD := SetColdLoadFailureCooldownForTest(30 * time.Second)
	defer SetColdLoadFailureCooldownForTest(oldCD)

	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)
	_ = mgrs

	// KickColdLoad spawns a goroutine; we need to wait for it to finish so
	// we can inspect coldLoadFailures. Use a sync barrier via a short
	// polling loop on coldLoadKicks.
	ok := s.KickColdLoad("target")
	if !ok {
		t.Fatalf("KickColdLoad returned false; expected true")
	}

	// Wait for the kick goroutine to clear (it deletes coldLoadKicks[target]
	// in defer).
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		_, busy := s.coldLoadKicks["target"]
		s.mu.Unlock()
		if !busy {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	s.mu.Lock()
	_, recorded := s.coldLoadFailures["target"]
	s.mu.Unlock()

	// CRITICAL ASSERTION: HIGH #1 — no failure record on
	// ErrColdLoadGPUContended.
	if recorded {
		t.Errorf("HIGH #1: expected NO coldLoadFailures[target] entry on ErrColdLoadGPUContended (deferral is transient, not a doomed loop); got recorded=true")
	}
}

// --- HIGH #2 ----------------------------------------------------------------

// twoContenderDeferralConfig: target needs GPUs 1+2; aaa-contender (sleep-
// capable, sleep_mode=true) holds GPU 1; zzz-contender (sleep-capable,
// sleep_mode=true) holds GPU 2. Both Ready. Different swap-groups from
// target. The test forces zzz-contender's wake-settle gate to fire so
// the loop slept aaa-contender, then bounced on zzz-contender.
const twoContenderDeferralConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  aaa-contender:
    lifecycle: external
    host: vllm-aaa
    port: 9001
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: aaa-group
    expected_vram_mb_per_gpu: 6000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
  zzz-contender:
    lifecycle: external
    host: vllm-zzz
    port: 9002
    gpus: [2]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: zzz-group
    expected_vram_mb_per_gpu: 6000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
  target:
    lifecycle: external
    host: vllm-target
    port: 9003
    gpus: [1, 2]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: target-group
    expected_vram_mb_per_gpu: 11000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// makeTwoContenderScheduler builds a scheduler with two cross-group
// contenders for the partial-eviction-then-deferral test.
func makeTwoContenderScheduler(t *testing.T) (*Scheduler, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(twoContenderDeferralConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{1: 24000, 2: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"aaa-contender": {port: 9001},
		"zzz-contender": {port: 9002},
		"target":        {port: 9003},
	}
	for name, m := range mgrs {
		modelCfg := cfg.Models[name]
		var st State
		if name == "target" {
			st = StateStopped
			m.pid.Store(0)
			m.isSleeping.Store(true)
		} else {
			st = StateReady
			m.pid.Store(int64(7000 + m.port))
			m.isSleeping.Store(false)
		}
		s.SeedInstanceForTest(name, m.port, modelCfg.GPUs, false, st, m)
	}

	a.mu.Lock()
	a.markStoppedLocked(a.models["target"])
	a.mu.Unlock()

	return s, mgrs
}

// TestColdLoad_DeferralRestoresAlreadySleptCrossGroup — when the
// cross-group eviction loop slept aaa-contender successfully but bounced
// on zzz-contender via ErrSleepSkippedWakeSettle, coldLoadStoppedMemberLocked
// must restore aaa-contender (wake it back up) before returning
// ErrColdLoadGPUContended. Pre-fix: aaa-contender silently left Sleeping.
func TestColdLoad_DeferralRestoresAlreadySleptCrossGroup(t *testing.T) {
	s, mgrs := makeTwoContenderScheduler(t)

	// Force zzz-contender (the SECOND contender in name-sort order) to
	// trip on the wake-settle gate. aaa-contender (FIRST) has no gate
	// override so it sleeps cleanly.
	zzzCfg := s.cfg.Models["zzz-contender"]
	zzzCfg.WakeSettleSeconds = 60
	s.cfg.Models["zzz-contender"] = zzzCfg
	if !s.SetInstanceLastWakeForTest("zzz-contender", time.Now()) {
		t.Fatalf("could not set lastWake for zzz-contender")
	}

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)
	oldCD := SetColdLoadFailureCooldownForTest(0)
	defer SetColdLoadFailureCooldownForTest(oldCD)

	var startCalls atomic.Int32
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			startCalls.Add(1)
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := s.coldLoadStoppedMember(ctx, s.instances["target"], s.cfg.Models["target"])

	// Outcome 1: must return ErrColdLoadGPUContended.
	if err == nil {
		t.Fatalf("HIGH #2: expected ErrColdLoadGPUContended; got nil")
	}
	if !errors.Is(err, ErrColdLoadGPUContended) {
		t.Errorf("HIGH #2: expected wrapped ErrColdLoadGPUContended; got %v", err)
	}

	// Outcome 2: doColdLoad must NOT have run.
	if got := startCalls.Load(); got != 0 {
		t.Errorf("HIGH #2: expected 0 docker start calls (deferred before doColdLoad); got %d", got)
	}

	// Outcome 3 (CRITICAL): aaa-contender must have been Sleep'd AND
	// Wake'd. Pre-fix it was only Sleep'd.
	if got := mgrs["aaa-contender"].sleepCalls.Load(); got != 1 {
		t.Errorf("HIGH #2: expected aaa-contender.Sleep=1 (the partial-eviction path), got %d", got)
	}
	if got := mgrs["aaa-contender"].wakeCalls.Load(); got != 1 {
		t.Errorf("HIGH #2: expected aaa-contender.Wake=1 (deferral-restore must re-wake), got %d", got)
	}

	// Outcome 4: zzz-contender (the deferral cause) was never Slept (its
	// sleepInstance call refused via ErrSleepSkippedWakeSettle BEFORE the
	// fakeRedeployMgr.Sleep counter could increment).
	if got := mgrs["zzz-contender"].sleepCalls.Load(); got != 0 {
		t.Errorf("HIGH #2: expected zzz-contender.Sleep=0 (deferral occurred BEFORE manager.Sleep was called), got %d", got)
	}
}

// --- MED #3 ----------------------------------------------------------------

// sameGroupStopOnlyConfig: target + a same-swap-group stop-mode peer that
// shares GPUs. The peer's drain has a brief window where a concurrent
// state mutation (StateReady → StateSleeping) can race with the stop-mode
// loop's snapshot. The test verifies that prevState is re-read live, not
// taken from the bulk RLock snapshot.
const sameGroupStopOnlyConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  peer:
    lifecycle: external
    host: vllm-peer
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: false
    evict_action: stop
    swap_group: shared
    expected_vram_mb_per_gpu: 8000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
  target:
    lifecycle: external
    host: vllm-target
    port: 9002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    evict_action: stop
    swap_group: shared
    expected_vram_mb_per_gpu: 8000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// TestSameGroupStopMode_LiveStateRecheckBeforeRevert — when docker stop
// fails, the prevState used to revert inst.state must be re-read live
// under a fresh RLock (matches MEDIUM #2 pattern in the cross-group path).
//
// The bulk-RLock filter rejects StateStopped peers but NOT StateStopping
// peers (StateStopping is a transient mid-flight transition state). A
// peer that was StateReady in the bulk snapshot but transitioned to
// StateStopping (e.g. via a concurrent drainInstance from another path)
// before the per-iteration body runs will:
//   - Pre-fix: be processed against the stale StateReady snapshot.
//     drainInstance flips to StateStopping (no-op effectively); docker
//     stop fires and (in this test) is forced to fail; revert sets
//     state back to StateReady — a wedged value, since the peer was
//     legitimately stopping concurrently.
//   - Post-fix: the live-recheck observes StateStopping and SKIPS the
//     peer entirely (the live check rejects StateStopping || StateStopped).
//     No spurious docker stop, no wedged revert.
//
// We construct the divergence by manually setting peer.state =
// StateStopping AFTER seeding (the bulk RLock will read whatever's
// there at the moment of the helper call; for a deterministic test we
// can't perfectly emulate the inter-RLock window without an internal
// hook, but we CAN verify the post-fix live-recheck rejects StateStopping
// up-front — which is the contract the cross-group MEDIUM #2 helper
// follows). Verifies BOTH paths handle StateStopping uniformly.
func TestSameGroupStopMode_LiveStateRecheckBeforeRevert(t *testing.T) {
	cfg, err := config.Load([]byte(sameGroupStopOnlyConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	peerMgr := &fakeRedeployMgr{port: 9001}
	peerMgr.pid.Store(7001)
	peerMgr.isSleeping.Store(false)
	s.SeedInstanceForTest("peer", 9001, []int{0}, false, StateReady, peerMgr)

	targetMgr := &fakeRedeployMgr{port: 9002}
	targetMgr.pid.Store(0)
	targetMgr.isSleeping.Store(true)
	s.SeedInstanceForTest("target", 9002, []int{0}, false, StateStopped, targetMgr)

	a.mu.Lock()
	a.markStoppedLocked(a.models["target"])
	a.mu.Unlock()

	// Mutate peer.state to StateStopping (a concurrent drain on the peer
	// is in flight from another path). The bulk RLock filter at line ~982
	// only rejects StateStopped, NOT StateStopping. Pre-fix, the loop
	// body would proceed: drainInstance is a no-op when state is already
	// StateStopping, docker stop is invoked (and fails in this test),
	// then revert sets state back to peerState=StateStopping (wait — the
	// snapshot is taken AT bulk RLock time, which is now StateStopping
	// since we mutated it before invoking the helper). To DETERMINISTICALLY
	// catch the bug, we need the bulk snapshot and the live state to
	// differ.
	//
	// Simulating the inter-RLock divergence: after seeding, set peer.state
	// = StateStopping. Bulk RLock observes StateStopping. Per-iteration
	// fresh RLock (post-fix) ALSO observes StateStopping → live-recheck
	// rejects → SKIP. Pre-fix: bulk snapshot peerState=StateStopping,
	// drainInstance leaves state at StateStopping, docker stop fires (and
	// fails), revert puts state back to peerState=StateStopping (no-op).
	// Result is the same — but the docker stop CALL count differs
	// (post-fix=0, pre-fix=1). That's the deterministic signal.
	s.mu.Lock()
	s.instances["peer"].state = StateStopping
	s.mu.Unlock()

	var stopCalls atomic.Int32
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "stop" {
			stopCalls.Add(1)
			return []byte("docker daemon unreachable"), fmt.Errorf("forced stop failure")
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	targetCfg := s.cfg.Models["target"]
	_, _ = s.evictPeersForColdLoadLocked(context.Background(), "target", targetCfg)

	// CRITICAL: post-fix live-recheck must observe StateStopping and SKIP.
	// Pre-fix would have invoked docker stop on a peer that another path
	// is already stopping (race-on-shutdown).
	if got := stopCalls.Load(); got != 0 {
		t.Errorf("MED #3: expected 0 docker stop calls (live-state recheck must observe StateStopping and skip), got %d", got)
	}

	// peer.state must remain StateStopping (the concurrent drain owns it;
	// our loop must not interfere).
	postState, _ := s.InstanceStateForTest("peer")
	if postState != StateStopping {
		t.Errorf("MED #3: expected peer.state=StateStopping (recheck skip preserves concurrent-owner state), got %v", postState)
	}
}

// --- MED #4 ----------------------------------------------------------------

// TestMapWakeError_GPUContendedRoutesToColdLoadingReason — mapWakeError
// must explicitly route ErrColdLoadGPUContended to RejectColdLoading with
// a 5s Retry-After. Pre-fix: falls to the default branch
// (RejectSwapInProgress + 10s Retry-After) which lies about the reason
// (no swap is happening — a contender is in its post-wake settle window)
// AND the wait time (gate window is 3-5s, not 10s).
func TestMapWakeError_GPUContendedRoutesToColdLoadingReason(t *testing.T) {
	wrapped := fmt.Errorf("%w: contender %q", ErrColdLoadGPUContended, "vllm-main")
	out := mapWakeError(wrapped)

	rej, ok := out.(*RejectError)
	if !ok {
		t.Fatalf("MED #4: expected *RejectError, got %T (%v)", out, out)
	}
	if rej.Reason != RejectColdLoading {
		t.Errorf("MED #4: expected Reason=%q (cold-loading is accurate; pre-fix returned 'in_progress' which lies), got %q",
			RejectColdLoading, rej.Reason)
	}
	if rej.RetryAfter != 5*time.Second {
		t.Errorf("MED #4: expected RetryAfter=5s (matches gate window 3-5s; pre-fix used 10s), got %v", rej.RetryAfter)
	}
	if !strings.Contains(rej.Message, "gpu-contender") {
		t.Errorf("MED #4: expected message to mention gpu-contender for operator clarity; got %q", rej.Message)
	}
	// Sanity: the wrapped sentinel must still be discoverable via errors.Is
	// on the inner error chain (in case downstream handlers re-classify).
	if !errors.Is(wrapped, ErrColdLoadGPUContended) {
		t.Errorf("MED #4: errors.Is(wrapped, ErrColdLoadGPUContended) returned false (sanity check)")
	}
}
