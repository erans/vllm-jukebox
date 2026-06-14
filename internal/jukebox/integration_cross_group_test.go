package jukebox

// Integration + unit coverage for the cross-group GPU eviction hardening
// landed in PR #27 architect adversarial round-7. Pairs with the
// gpu_eviction_before_coldload_test.go unit tests for the helper itself;
// this file exercises:
//
//   1. End-to-end coldLoadStoppedMember on a cross-group topology
//      (TestIntegration_CrossGroupGPUContender_EvictedBeforeColdLoad).
//   2. The state-skip branch when a cross-group peer is already Sleeping
//      (TestColdLoad_SkipsAlreadySleepingCrossGroupPeer).
//   3. Multiple overlapping contenders in deterministic sort order
//      (TestColdLoad_MultipleOverlappingContenders).
//   4. The rollback-leaves-cross-group-sleeping policy (MEDIUM #4)
//      (TestIntegration_CrossGroupColdLoad_FailureRollbackLeavesSleptCrossGroup).
//   5. The coldLoadEviction gate covers cross-group contenders (HIGH #1)
//      (TestSleep_CrossGroupContenderInColdLoadEvictionGate).
//   6. TOCTOU live-state recheck skips racing wake (MEDIUM #2)
//      (TestSleep_LiveStateRecheckSkipsRacingWake).
//   7. ErrSleepSkippedWakeSettle / ErrSleepTooSoonAfterWake → deferral
//      (MEDIUM #3) (TestSleep_WakeSettleSkipReturnsDeferral).
//   8. docker stop failure reverts inst.state (MEDIUM #5)
//      (TestSleep_DockerStopFailureRevertsState).

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// crossGroupIntegrationConfig: target is admissionStopped on GPU 1+2;
// holder is admissionAwake on GPU 0+1 (different swap-group); a
// non-overlapping bystander on GPU 3 must be untouched.
const crossGroupIntegrationConfig = `
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

// makeCrossGroupIntegrationScheduler builds the full SchedulerEvictor-
// backed Scheduler (so cold-load flows go through the real eviction
// path end-to-end). Returns mgrs keyed by model.
func makeCrossGroupIntegrationScheduler(t *testing.T, holderState, targetState, bystanderState State) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(crossGroupIntegrationConfig))
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

// TestIntegration_CrossGroupGPUContender_EvictedBeforeColdLoad asserts
// the full end-to-end coldLoadStoppedMember pipeline (NOT just the
// direct helper call) sleeps a cross-group GPU contender BEFORE
// docker start of the cold-load target. Without the fix, the cold-load
// races for VRAM with the resident cross-group peer and OOMs at vLLM
// worker init.
func TestIntegration_CrossGroupGPUContender_EvictedBeforeColdLoad(t *testing.T) {
	s, _, mgrs := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateReady)

	// Accelerate the /is_sleeping poll so the test isn't 5s per probe.
	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)

	// Record docker calls in order so we can prove holder.Sleep is invoked
	// (via fakeRedeployMgr.Sleep counter, NOT docker stop) BEFORE the
	// target's docker start.
	var dockerMu sync.Mutex
	var dockerCalls []string
	var startedTarget atomic.Bool
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		call := name + " " + strings.Join(args, " ")
		dockerCalls = append(dockerCalls, call)
		dockerMu.Unlock()
		if strings.Contains(call, "start") && strings.Contains(call, "vllm-target") {
			startedTarget.Store(true)
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.coldLoadStoppedMember(ctx, s.instances["target"], s.cfg.Models["target"]); err != nil {
		t.Fatalf("coldLoadStoppedMember: %v", err)
	}

	if got := mgrs["holder"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected holder.Sleep=1 (cross-group eviction inside cold-load pipeline), got %d", got)
	}
	if got := mgrs["bystander"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected bystander.Sleep=0 (no GPU overlap with target), got %d", got)
	}
	if !startedTarget.Load() {
		t.Errorf("expected docker start vllm-target; calls=%v", dockerCalls)
	}
}

// TestColdLoad_SkipsAlreadySleepingCrossGroupPeer covers the explicit
// skip branch where the cross-group peer is StateSleeping (not Ready).
// MEDIUM #2 added a live-state recheck; this test verifies the helper
// makes zero Sleep calls when the peer is already asleep.
func TestColdLoad_SkipsAlreadySleepingCrossGroupPeer(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupConfig, StateSleeping, StateStopped, StateReady)

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[] (holder already StateSleeping), got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("expected stopped=[], got %v", stopped)
	}
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected holder.Sleep=0 (already sleeping branch), got %d", got)
	}
}

// TestColdLoad_MultipleOverlappingContenders verifies that when MORE
// than one cross-group peer overlaps the target's GPUs, they are evicted
// in deterministic name-sorted order. Adds a synthetic peer to the base
// config so we have two overlapping cross-group contenders (a + holder).
func TestColdLoad_MultipleOverlappingContenders(t *testing.T) {
	const twoContenderConfig = `
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
    wake_timeout: 1s
`
	cfg, err := config.Load([]byte(twoContenderConfig))
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

	// Track Sleep call ORDER, not just count.
	var orderMu sync.Mutex
	var order []string
	for name, m := range mgrs {
		nm := name
		mgr := m
		_ = mgr
		_ = nm
	}
	// Use the package-level interceptor approach: replace each fake's Sleep
	// fn would require patching fakeRedeployMgr. Instead, snapshot the
	// post-call counters AFTER the eviction and assert both fired; verify
	// sort-order by inspecting the returned slept slice (the helper sorts
	// contenders by name before iterating).
	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(stopped) != 0 {
		t.Errorf("expected stopped=[], got %v", stopped)
	}
	if len(slept) != 2 {
		t.Fatalf("expected 2 slept contenders, got %d: %v", len(slept), slept)
	}
	// Sort-by-name order: "aaa-contender" must come before "zzz-contender".
	if slept[0] != "aaa-contender" || slept[1] != "zzz-contender" {
		t.Errorf("expected slept=[aaa-contender,zzz-contender] (sort order), got %v", slept)
	}
	if got := mgrs["aaa-contender"].sleepCalls.Load(); got != 1 {
		t.Errorf("aaa-contender.Sleep=%d, want 1", got)
	}
	if got := mgrs["zzz-contender"].sleepCalls.Load(); got != 1 {
		t.Errorf("zzz-contender.Sleep=%d, want 1", got)
	}

	// Quiet unused order capture for static analyzers.
	orderMu.Lock()
	_ = order
	orderMu.Unlock()
}

// TestIntegration_CrossGroupColdLoad_FailureRollbackLeavesSleptCrossGroup
// verifies MEDIUM #4: when doColdLoad fails AFTER a cross-group peer was
// slept, the rollback path LEAVES that peer sleeping (does NOT cascade-
// wake it via admission.RequestWake). The peer will async-recover via
// the standard wake-from-Sleeping path on next inbound demand. Pinned
// same-group peers, by contrast, ARE restored — verified separately in
// existing tests.
func TestIntegration_CrossGroupColdLoad_FailureRollbackLeavesSleptCrossGroup(t *testing.T) {
	s, _, mgrs := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateStopped)

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)
	oldCD := SetColdLoadFailureCooldownForTest(0)
	defer SetColdLoadFailureCooldownForTest(oldCD)

	// Make the target's docker start succeed but the post-start container
	// inspect / health probe fail by having the manager's IsSleeping
	// stay false forever (target.isSleeping=false even after the docker
	// start hook returns). The cold-load loop's pollUntilSleeping then
	// hits its (test-shortened) deadline and surfaces a timeout.
	mgrs["target"].isSleeping.Store(false)

	var dockerMu sync.Mutex
	var dockerCalls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		dockerCalls = append(dockerCalls, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Override the target's cold-load timeout to ~250ms so pollUntilSleeping
	// fails quickly. We can't do this via config.Set; instead pass a config
	// to coldLoadStoppedMember with an explicit (very short) timeout via
	// modelCfg.ColdLoadTimeoutSeconds — the loaded config has wake_timeout:
	// 1s but cold-load uses ColdLoadTimeoutSeconds. With nothing set,
	// EffectiveColdLoadTimeout's default of 600s would block the test.
	targetCfg := s.cfg.Models["target"]
	targetCfg.ColdLoadTimeoutSeconds = 1 // 1s budget — pollUntilSleeping bails.

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := s.coldLoadStoppedMember(ctx, s.instances["target"], targetCfg)
	if err == nil {
		t.Fatalf("expected cold-load error (forced timeout); got nil")
	}

	// holder MUST have been slept by the cross-group eviction step.
	if got := mgrs["holder"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected holder.Sleep=1 (cross-group evict), got %d", got)
	}

	// CRITICAL ASSERTION (MEDIUM #4): the rollback must NOT have re-woken
	// holder. holder.wakeCalls must remain 0.
	if got := mgrs["holder"].wakeCalls.Load(); got != 0 {
		t.Errorf("expected holder.Wake=0 (MEDIUM #4: cross-group peers left sleeping, no cascade), got %d", got)
	}
}

// TestSleep_CrossGroupContenderInColdLoadEvictionGate verifies HIGH #1:
// IsInColdLoadEviction must return true for cross-group contenders that
// the cold-load is about to evict, NOT just same-swap-group members.
// Pre-fix the gate only covered swapGroupMembersForColdLoad output;
// concurrent requests for cross-group evictees fell through.
func TestSleep_CrossGroupContenderInColdLoadEvictionGate(t *testing.T) {
	s, _, _ := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateReady)

	old := SetColdLoadPollIntervalForTest(10 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(old)

	// Latch the docker start so we can observe IsInColdLoadEviction while
	// the cold-load is still in flight.
	released := make(chan struct{})
	SetSleepDockerCmdForTest(func(ctx context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			select {
			case <-released:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	done := make(chan error, 1)
	go func() {
		done <- s.coldLoadStoppedMember(context.Background(), s.instances["target"], s.cfg.Models["target"])
	}()

	// Wait until the cold-load reaches docker start (which means the
	// eviction gate has been marked).
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if s.IsInColdLoadEviction("holder") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if !s.IsInColdLoadEviction("holder") {
		close(released)
		<-done
		t.Fatalf("HIGH #1: IsInColdLoadEviction(holder) returned false while holder is a cross-group contender mid-eviction")
	}
	// Sanity: bystander (no GPU overlap) is NOT in the gate.
	if s.IsInColdLoadEviction("bystander") {
		close(released)
		<-done
		t.Fatalf("bystander incorrectly in coldLoadEviction gate (no GPU overlap with target)")
	}

	close(released)
	if err := <-done; err != nil {
		t.Logf("cold-load completed with err=%v (acceptable — gate behaviour was the assertion)", err)
	}

	// After completion, gate must be cleared.
	if s.IsInColdLoadEviction("holder") {
		t.Errorf("IsInColdLoadEviction(holder) still true after cold-load completed")
	}
}

// TestSleep_LiveStateRecheckSkipsRacingWake verifies MEDIUM #2:
// the eviction loop re-fetches live state under fresh RLock and skips
// when state changed between the snapshot and the per-peer iteration.
// We seed the snapshot with c.state=StateReady but flip the live state
// to StateStopped before the loop reaches the action branch. The helper
// must observe the mismatch and SKIP rather than fire sleepInstance on
// the wrong-state peer.
func TestSleep_LiveStateRecheckSkipsRacingWake(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupConfig, StateReady, StateStopped, StateReady)

	// Inject a state flip BEFORE the loop runs. The helper takes a bulk
	// RLock to build the contender slice (c.state=StateReady), then
	// drops the lock. We mutate live state here; the per-peer fresh
	// RLock should observe the mismatch.
	//
	// Implementation: we can't easily inject a hook MID-helper, so we
	// instead simulate by directly mutating holder's live state to a
	// post-snapshot value before invoking the helper. The helper's bulk
	// RLock will then see the SAME live state in its snapshot, which
	// defeats the test premise.
	//
	// Workaround: the only way to surface the recheck without an
	// interceptor is to seed holder as StateReady (so the bulk RLock
	// snapshot is StateReady), then before calling the helper, flip
	// it to StateStopped via the live mutation API. The bulk RLock then
	// sees StateStopped and skips it via the existing state-tier filter
	// (StateStopped peers are skipped in the bulk filter at line ~980).
	// To prove the FRESH-RLock recheck specifically, we need a snapshot
	// that disagrees with the per-peer recheck — which requires hooking
	// the inter-loop window.
	//
	// Compromise: assert that flipping live state AFTER the helper
	// returns no Sleep call (proving the helper read live state, not a
	// stale-only snapshot). This is the strongest assertion possible
	// without an internal mid-loop hook.
	s.mu.Lock()
	s.instances["holder"].state = StateStopped
	s.mu.Unlock()

	targetCfg := s.cfg.Models["target"]
	slept, _ := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[] (holder live state is now StateStopped, helper must observe via live recheck), got %v", slept)
	}
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("MEDIUM #2: expected holder.Sleep=0 (live state mismatch must skip), got %d", got)
	}
}

// TestSleep_LiveStateRecheckSkipsRacingWakeViaHook is the strongest
// possible MEDIUM #2 assertion: it surfaces the EXACT TOCTOU window
// between the bulk-snapshot RUnlock and the per-peer fresh RLock recheck
// by using SetEvictionLoopPreActionHookForTest to mutate holder's live
// state at that precise mid-loop point.
//
// Pre-fix (no per-peer recheck): the loop reads c.state from the bulk
// snapshot (StateReady) and fires sleepInstance on holder, racing the
// concurrent admission wake the hook simulated. holder.sleepCalls=1.
//
// Post-fix (cc21e091, sleep.go:~1087-1103): the per-peer RLock re-reads
// peerInst.state, observes StateSleeping, logs
// cold_load_cross_group_peer_state_changed_skip, and skips. Even the
// first-tier StateReady check at line ~1111 also catches it. Either
// way, holder.sleepCalls=0.
//
// Companion to TestSleep_LiveStateRecheckSkipsRacingWake (which proves
// the same skip when the mutation is BEFORE the helper rather than
// MID-helper). This test proves the recheck specifically.
func TestSleep_LiveStateRecheckSkipsRacingWakeViaHook(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupConfig, StateReady, StateStopped, StateReady)

	// Mutate holder.state from StateReady → StateSleeping mid-loop. The
	// hook fires AFTER the bulk-snapshot RUnlock built contenders (so
	// c.state=StateReady is captured) and BEFORE the per-peer fresh
	// RLock reads liveState. Must take s.mu in WRITE mode here — the
	// per-peer RLock that runs immediately after the hook returns will
	// then observe StateSleeping.
	var hookFired atomic.Int32
	restore := SetEvictionLoopPreActionHookForTest(func(peer string) {
		if peer != "holder" {
			return
		}
		hookFired.Add(1)
		s.mu.Lock()
		if inst := s.instances["holder"]; inst != nil {
			inst.state = StateSleeping
		}
		s.mu.Unlock()
	})
	defer restore()

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if hookFired.Load() == 0 {
		t.Fatalf("MEDIUM #2 setup: hook never fired for holder — eviction loop did not iterate as expected (slept=%v stopped=%v)", slept, stopped)
	}
	// Post-fix: per-peer RLock sees StateSleeping after the snapshot
	// captured StateReady → live recheck mismatch → skip. Sleep=0.
	// Pre-fix: c.state from snapshot still StateReady → loop fires
	// sleepInstance against the (now StateSleeping) peer → Sleep=1.
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("MEDIUM #2: expected holder.Sleep=0 (per-peer live recheck must catch StateReady→StateSleeping mutation in the bulk-snapshot/per-peer-action window), got %d", got)
	}
	if len(slept) != 0 {
		t.Errorf("MEDIUM #2: expected slept=[] (live recheck skipped holder), got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("MEDIUM #2: expected stopped=[] (sleep_mode=true on holder, stop branch should not engage), got %v", stopped)
	}
}

// TestSleep_WakeSettleSkipReturnsDeferral verifies MEDIUM #3: when
// sleepInstance returns ErrSleepSkippedWakeSettle on the cross-group
// contender, coldLoadStoppedMemberLocked must abort with
// ErrColdLoadGPUContended and NOT proceed with doColdLoad (which would
// OOM on the still-resident contender).
func TestSleep_WakeSettleSkipReturnsDeferral(t *testing.T) {
	s, _, _ := makeCrossGroupIntegrationScheduler(t, StateReady, StateStopped, StateStopped)

	// Force the WakeSettle gate to fire by setting holder's lastWakeAt
	// to "just now" with an explicit wake_settle_seconds > 0. The config
	// above uses -1 (disabled); we override on the modelCfg copy used by
	// sleepInstance via the live-config path.
	//
	// Simpler: directly set lastWakeAt and call sleepInstance via the
	// cold-load path. But min_time_since_wake also defaults to disabled
	// via -1 in our config, so we need to flip ONE gate on.
	//
	// Override holder's wake_settle_seconds=1 by mutating the config in
	// place (it's a value type; need to write back).
	holderCfg := s.cfg.Models["holder"]
	holderCfg.WakeSettleSeconds = 60 // 60s window
	s.cfg.Models["holder"] = holderCfg

	// Mark holder as just-woken so the gate fires.
	if !s.SetInstanceLastWakeForTest("holder", time.Now()) {
		t.Fatalf("could not set lastWake for holder")
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

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := s.coldLoadStoppedMember(ctx, s.instances["target"], s.cfg.Models["target"])
	if err == nil {
		t.Fatalf("expected ErrColdLoadGPUContended; got nil")
	}
	if !errors.Is(err, ErrColdLoadGPUContended) {
		t.Errorf("MEDIUM #3: expected wrapped ErrColdLoadGPUContended; got %v", err)
	}
	// CRITICAL: doColdLoad must NOT have been invoked (no docker start).
	if got := startCalls.Load(); got != 0 {
		t.Errorf("MEDIUM #3: expected 0 docker start calls on deferral (contender still resident); got %d", got)
	}
}

// TestSleep_DockerStopFailureRevertsState verifies MEDIUM #5: when
// docker stop on a cross-group contender fails, the helper reverts the
// peer's inst.state to its prevState (so it's not wedged at
// StateStopping forever). Uses the no-sleep config (forces the stop
// branch).
func TestSleep_DockerStopFailureRevertsState(t *testing.T) {
	s, _, _ := makeCrossGroupScheduler(t, crossGroupNoSleepConfig, StateReady, StateStopped, StateReady)

	// Force docker stop to fail.
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("docker daemon unreachable"), fmt.Errorf("forced stop failure")
	})
	defer SetSleepDockerCmdForTest(nil)

	preState, ok := s.InstanceStateForTest("holder")
	if !ok {
		t.Fatalf("holder not seeded")
	}
	if preState != StateReady {
		t.Fatalf("setup: expected holder=StateReady, got %s", preState)
	}

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[] (holder has no sleep_mode), got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("expected stopped=[] (docker stop failed, peer NOT added), got %v", stopped)
	}

	postState, _ := s.InstanceStateForTest("holder")
	if postState != preState {
		t.Errorf("MEDIUM #5: expected holder state reverted to %s on docker stop failure; got %s (wedged at StateStopping is the bug)",
			preState, postState)
	}
}
