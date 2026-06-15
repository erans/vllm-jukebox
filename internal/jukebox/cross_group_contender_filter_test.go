package jukebox

import (
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// pinnedNoSwapGroupConfig: target wants GPU 1+2; pinnedHolder is on GPU
// 0+1 BUT is `pinned: true` with NO swap_group declared — which means
// admission.pickVictims (admission.go ~line 1246) will refuse to evict
// it. evictCrossGroupGPUContendersLocked similarly leaves it alone.
// Pre-registering pinnedHolder in coldLoadEviction is therefore
// guaranteed-wrong: nothing will actually touch it, but every chat
// request to it would 503 for the full cold-load wall-clock.
//
// Also includes `evictableHolder` (cross-group, has swap_group, on GPU
// 0+1) as a regression-positive — it SHOULD still be included after
// the filter tightens, because pickVictims CAN evict it.
const pinnedNoSwapGroupConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  pinnedHolder:
    lifecycle: external
    host: vllm-pinned
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    pinned: true
    expected_vram_mb_per_gpu: 12000
    sleep_l1_residual_mb: 1000
    wake_timeout: 1s
  evictableHolder:
    lifecycle: external
    host: vllm-evictable
    port: 9002
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: evictable-group
    pinned: true
    expected_vram_mb_per_gpu: 12000
    sleep_l1_residual_mb: 1000
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

// makeContenderFilterScheduler builds a scheduler from the
// pinnedNoSwapGroupConfig and lets the caller dial each peer's seeded
// state independently. Mirrors makeCrossGroupScheduler's pattern but
// scoped to the contender-filter tests.
func makeContenderFilterScheduler(t *testing.T, pinnedState, evictableState State) (*Scheduler, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(pinnedNoSwapGroupConfig))
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
		"pinnedHolder":    {port: 9001},
		"evictableHolder": {port: 9002},
		"target":          {port: 9003},
	}
	stateByName := map[string]State{
		"pinnedHolder":    pinnedState,
		"evictableHolder": evictableState,
		"target":          StateStopped,
	}
	for name, m := range mgrs {
		modelCfg, ok := cfg.Models[name]
		if !ok {
			continue
		}
		st := stateByName[name]
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
	return s, mgrs
}

// TestCrossGroupGPUContenderNamesForColdLoad_ExcludesPinnedNonSwapGroup
// is the load-bearing regression test for fix #2: a pinned peer with no
// swap_group declared and GPU overlap with the target MUST NOT appear
// in the contender list, because admission.pickVictims will never
// evict it. Including it caused live cold-loads to fast-503 every chat
// request to Qwen3.6-27B / Sparx for the full 5-12 minute cold-load
// wall-clock even though nothing was actually touching it.
func TestCrossGroupGPUContenderNamesForColdLoad_ExcludesPinnedNonSwapGroup(t *testing.T) {
	s, _ := makeContenderFilterScheduler(t, StateReady, StateReady)
	targetCfg := s.cfg.Models["target"]
	got := s.crossGroupGPUContenderNamesForColdLoad("target", targetCfg)
	for _, name := range got {
		if name == "pinnedHolder" {
			t.Errorf("pinnedHolder (pinned, no swap_group) must be excluded — pickVictims refuses to evict it, pre-registration would dead-503 it. got list=%v", got)
		}
	}
}

// TestCrossGroupGPUContenderNamesForColdLoad_ExcludesStoppedPeers:
// a peer in StateStopped holds no VRAM and is not an eviction
// candidate (admission state is admissionStopped, pickVictims
// requires admissionAwake). Pre-registering it would dead-503 any
// concurrent kick-cold-load for that peer for the full cold-load
// window even though nothing would actually happen to it.
func TestCrossGroupGPUContenderNamesForColdLoad_ExcludesStoppedPeers(t *testing.T) {
	// Stop the cross-group evictable peer (still has swap-group, still
	// non-target swap-group, still has GPU overlap) — but it's Stopped,
	// so eviction can't act on it.
	s, _ := makeContenderFilterScheduler(t, StateReady, StateStopped)
	targetCfg := s.cfg.Models["target"]
	got := s.crossGroupGPUContenderNamesForColdLoad("target", targetCfg)
	for _, name := range got {
		if name == "evictableHolder" {
			t.Errorf("evictableHolder seeded as StateStopped must be excluded — no VRAM to free, eviction can't act on Stopped peers. got list=%v", got)
		}
	}
}

// TestCrossGroupGPUContenderNamesForColdLoad_IncludesReadySwapGroupPeer
// is the regression-positive: a cross-group peer that IS evictable
// (has a swap_group declared, in StateReady, GPU overlap with target)
// MUST STILL appear in the contender list. The filter tightening
// must not over-correct.
func TestCrossGroupGPUContenderNamesForColdLoad_IncludesReadySwapGroupPeer(t *testing.T) {
	s, _ := makeContenderFilterScheduler(t, StateStopped, StateReady)
	targetCfg := s.cfg.Models["target"]
	got := s.crossGroupGPUContenderNamesForColdLoad("target", targetCfg)
	found := false
	for _, name := range got {
		if name == "evictableHolder" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("evictableHolder (cross-group swap-evictable, StateReady, GPU overlap) MUST be included — pickVictims CAN evict it via crossGroupSwapEvictable tier. got list=%v", got)
	}
}

// TestCrossGroupGPUContenderNamesForColdLoad_IncludesSleepingSwapGroupPeer
// belt-and-suspenders: StateSleeping peers also hold residual VRAM
// (Level 1 sleep) and ARE valid eviction targets (sleep-to-stop
// upgrade or just-sleep), so they must remain in the contender list.
func TestCrossGroupGPUContenderNamesForColdLoad_IncludesSleepingSwapGroupPeer(t *testing.T) {
	s, _ := makeContenderFilterScheduler(t, StateStopped, StateSleeping)
	targetCfg := s.cfg.Models["target"]
	got := s.crossGroupGPUContenderNamesForColdLoad("target", targetCfg)
	found := false
	for _, name := range got {
		if name == "evictableHolder" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("evictableHolder (cross-group swap-evictable, StateSleeping, GPU overlap) MUST be included — sleeping peers still hold residual VRAM and are valid eviction targets. got list=%v", got)
	}
}
