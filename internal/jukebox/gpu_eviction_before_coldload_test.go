package jukebox

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// crossGroupConfig: two models in DIFFERENT swap-groups whose GPU sets
// overlap on GPU 1. `target` is the cold-loading peer (vision-like);
// `holder` is the resident peer (main-like) that must be evicted before
// the cold-load proceeds. Without the cross-group eviction, the
// existing same-group helper skips `holder` entirely (different
// swap-group) and the cold-load OOMs.
//
// NOTE (Bug #1/#2): holder is intentionally NON-pinned and normal
// priority here so it is a LEGITIMATELY evictable cross-group contender
// (same priority as target → not protected; both have a swap_group →
// both are swap-eligible). The protected-resident invariant (pinned /
// higher-priority resident must NOT be evicted for an equal/lower
// cold-load) is covered by TestColdLoad_DoesNotEvictPinnedResident and
// TestColdLoad_DoesNotEvictHigherPriorityResident below.
const crossGroupConfig = `
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

// crossGroupNoSleepConfig: same shape, but `holder` has sleep_mode:false
// so the cross-group eviction must take the docker-stop branch instead
// of the sleepInstance branch.
const crossGroupNoSleepConfig = `
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
    swap_group: holder-group
    expected_vram_mb_per_gpu: 12000
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
    wake_timeout: 1s
`

// makeCrossGroupScheduler constructs a Scheduler + admission + manager
// map suitable for direct evictCrossGroupGPUContendersLocked invocation.
// All managers start as the seeded state (holder=Ready, target=Stopped,
// bystander=Ready by default — bystander is on a non-overlapping GPU
// and must NOT be evicted).
func makeCrossGroupScheduler(t *testing.T, yaml string, holderState, targetState, bystanderState State) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(yaml))
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

	// Sync admission state for the target so cold-load path engages.
	a.mu.Lock()
	if targetState == StateStopped {
		a.markStoppedLocked(a.models["target"])
	}
	a.mu.Unlock()

	return s, a, mgrs
}

// TestColdLoad_SleepsCrossGroupGPUContender: the load-bearing scenario.
// holder (different swap_group, sleep_mode:true) is awake on GPU 0+1;
// target needs GPU 1+2. Cross-group eviction MUST sleep holder before
// the cold-load proceeds.
func TestColdLoad_SleepsCrossGroupGPUContender(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupConfig, StateReady, StateStopped, StateReady)

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 1 || slept[0] != "holder" {
		t.Errorf("expected slept=[holder], got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("expected stopped=[], got %v", stopped)
	}
	if got := mgrs["holder"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected holder.Sleep=1, got %d", got)
	}
	// bystander is on GPU 3 (no overlap with target's [1,2]) — must NOT
	// be evicted even though it's a cross-group peer.
	if got := mgrs["bystander"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected bystander.Sleep=0 (no GPU overlap), got %d", got)
	}
}

// TestColdLoad_NoEvictionWhenGPUsAvailable: target's GPUs are unused
// (holder is Stopped). Cross-group eviction returns empty slices and
// makes zero Sleep / Stop calls.
func TestColdLoad_NoEvictionWhenGPUsAvailable(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupConfig, StateStopped, StateStopped, StateReady)

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[], got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("expected stopped=[], got %v", stopped)
	}
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected holder.Sleep=0 (Stopped, no VRAM held), got %d", got)
	}
	if got := mgrs["bystander"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected bystander.Sleep=0, got %d", got)
	}
}

// TestColdLoad_StopsContenderWhenNoSleepMode: holder has sleep_mode:false
// and evict_action:stop. Cross-group eviction must use docker stop, not
// sleepInstance.
func TestColdLoad_StopsContenderWhenNoSleepMode(t *testing.T) {
	s, _, mgrs := makeCrossGroupScheduler(t, crossGroupNoSleepConfig, StateReady, StateStopped, StateReady)

	// Hook docker so the stop branch can succeed without shelling out.
	var dockerMu sync.Mutex
	var dockerCalls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		dockerCalls = append(dockerCalls, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[] (holder has no sleep_mode), got %v", slept)
	}
	if len(stopped) != 1 || stopped[0] != "holder" {
		t.Errorf("expected stopped=[holder], got %v", stopped)
	}
	// holder.Sleep must NOT be called (no sleep_mode).
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected holder.Sleep=0 (no sleep_mode), got %d", got)
	}
	// docker stop must have been called against holder's container.
	dockerMu.Lock()
	defer dockerMu.Unlock()
	stopCalled := false
	for _, c := range dockerCalls {
		if strings.Contains(c, "stop") && strings.Contains(c, "vllm-holder") {
			stopCalled = true
			break
		}
	}
	if !stopCalled {
		t.Errorf("expected docker stop vllm-holder, got calls: %v", dockerCalls)
	}
}

// TestColdLoad_SwapGroupEvictionUsesExistingPath: when contender is in
// the SAME swap-group as the target, the cross-group helper must skip
// it (delegating to the existing evictPeersForColdLoadLocked). The
// helper returns empty slices — proof that same-group peers are not
// double-evicted.
func TestColdLoad_SwapGroupEvictionUsesExistingPath(t *testing.T) {
	// Same-swap-group config: holder + target share swap_group "shared-group"
	// but evict_action:sleep (so we don't have to set up stop-mode externals).
	const sameGroupConfig = `
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
    swap_group: shared-group
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
    swap_group: shared-group
    expected_vram_mb_per_gpu: 11000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`
	s, _, mgrs := makeCrossGroupScheduler(t, sameGroupConfig, StateReady, StateStopped, StateReady)

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("expected slept=[] (holder is same-swap-group, must be skipped), got %v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("expected stopped=[], got %v", stopped)
	}
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected holder.Sleep=0 (cross-group helper must not double-evict same-group peer), got %d", got)
	}
}
