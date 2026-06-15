package jukebox

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// These tests pin the cross-group eviction invariant that Bug #1
// (phantom eviction of a no-swap_group model) regresses against:
//
//   INVARIANT A — a model with NO swap_group is NEVER a cross-group
//                 eviction victim. A peer that HAS a swap_group (even if
//                 pinned or priority: critical) made the explicit operator
//                 opt-in to having its VRAM reclaimed for a swap, so it IS
//                 an eligible cross-group victim. (P1 added a pinned/
//                 critical "never evict" clause that reversed this
//                 documented invariant and broke 3 integration tests; the
//                 rework removed it — see TestColdLoad_EvictsPinnedSwapGroupResident.)
//
// They drive evictCrossGroupGPUContendersLocked directly (the same
// entry point gpu_eviction_before_coldload_test.go exercises) so the
// assertions are deterministic and need no fleet / docker.

// makeInvariantScheduler builds a Scheduler from `yaml`, seeds each named
// model's instance to the supplied state, and returns the scheduler plus
// the fake managers keyed by model name. Mirrors makeCrossGroupScheduler
// but is parameterized over an arbitrary model set so each invariant test
// can express exactly the topology it needs.
func makeInvariantScheduler(t *testing.T, yaml string, states map[string]State) (*Scheduler, map[string]*fakeRedeployMgr) {
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

	mgrs := map[string]*fakeRedeployMgr{}
	for name, modelCfg := range cfg.Models {
		st, ok := states[name]
		if !ok {
			st = StateReady
		}
		m := &fakeRedeployMgr{port: modelCfg.Port}
		switch st {
		case StateReady:
			m.pid.Store(int64(7000 + modelCfg.Port))
			m.isSleeping.Store(false)
		case StateSleeping:
			m.pid.Store(int64(7000 + modelCfg.Port))
			m.isSleeping.Store(true)
		case StateStopped:
			m.pid.Store(0)
			m.isSleeping.Store(true)
		}
		s.SeedInstanceForTest(name, modelCfg.Port, modelCfg.GPUs, modelCfg.Pinned != nil && *modelCfg.Pinned, st, m)
		mgrs[name] = m
	}
	return s, mgrs
}

// hookDockerStops installs a SetSleepDockerCmdForTest hook that records
// every docker invocation and succeeds, returning a func that reports
// whether `docker stop <container>` was ever called. Caller defers the
// restore via the returned cleanup.
func hookDockerStops(t *testing.T) (stopCalledFor func(container string) bool, cleanup func()) {
	t.Helper()
	var mu sync.Mutex
	var calls []string
	SetSleepDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		mu.Lock()
		calls = append(calls, name+" "+strings.Join(args, " "))
		mu.Unlock()
		return []byte("ok"), nil
	})
	stopCalledFor = func(container string) bool {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range calls {
			if strings.Contains(c, "stop") && strings.Contains(c, container) {
				return true
			}
		}
		return false
	}
	cleanup = func() { SetSleepDockerCmdForTest(nil) }
	return stopCalledFor, cleanup
}

// noSwapGroupConfig: `resident` has NO swap_group at all (the exact
// topology of the 2026-06-13 phantom-eviction bug — vllm-main with its
// swap_group removed from config). It overlaps GPU 1 with the cold-
// loading `target`. INVARIANT A: resident must never be touched.
const noSwapGroupConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  resident:
    lifecycle: external
    host: vllm-resident
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
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

// TestColdLoad_NeverEvictsNoSwapGroupPeer is the load-bearing Bug #1
// regression: a resident WITHOUT a swap_group must NEVER be selected as
// a cross-group eviction victim, even when its GPUs overlap the cold-
// load target and it is awake. Pre-fix the loop only skipped peers whose
// SwapGroup matched the target's (and "" == "" only when the TARGET also
// had no group), so a no-group resident fell straight through to
// sleepInstance / docker stop. Post-fix the empty-SwapGroup guard skips
// it cleanly.
func TestColdLoad_NeverEvictsNoSwapGroupPeer(t *testing.T) {
	s, mgrs := makeInvariantScheduler(t, noSwapGroupConfig, map[string]State{
		"resident": StateReady,
		"target":   StateStopped,
	})
	stopCalledFor, cleanup := hookDockerStops(t)
	defer cleanup()

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 0 {
		t.Errorf("INVARIANT A violated: no-swap_group resident slept: slept=%v", slept)
	}
	if len(stopped) != 0 {
		t.Errorf("INVARIANT A violated: no-swap_group resident stopped: stopped=%v", stopped)
	}
	if got := mgrs["resident"].sleepCalls.Load(); got != 0 {
		t.Errorf("INVARIANT A violated: resident.Sleep=%d (must be 0 — no swap_group)", got)
	}
	if stopCalledFor("vllm-resident") {
		t.Errorf("INVARIANT A violated: docker stop called on no-swap_group resident")
	}
}

// pinnedResidentConfig: `resident` is pinned and in its own swap_group;
// `target` is normal priority in a different swap_group, GPUs overlap on
// GPU 1.
//
// DOCUMENTED INVARIANT (the one P1's over-reach reversed): a pinned model
// that HAS a swap_group IS evictable — pinned-in-swap_group peers
// (vllm-vision / vllm-moe) are expected to yield their VRAM to a
// cross-group swap. The ONLY peer cross-group eviction must never touch is
// one with NO swap_group at all (see TestColdLoad_NeverEvictsNoSwapGroupPeer,
// INVARIANT A). So with a swap_group present + a GPU overlap, `resident`
// IS a legitimate contender and DOES get slept.
const pinnedResidentConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  resident:
    lifecycle: external
    host: vllm-resident
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: resident-group
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

// TestColdLoad_EvictsPinnedSwapGroupResident pins the DOCUMENTED invariant
// that P1's cross-group guard wrongly reversed: a pinned resident that has
// a swap_group and overlaps the cold-load target's GPU IS a legitimate
// cross-group eviction victim (it gets slept), because declaring a
// swap_group is the operator's explicit opt-in to having that model's VRAM
// reclaimed for a swap. (Only a NO-swap_group peer is protected — see
// TestColdLoad_NeverEvictsNoSwapGroupPeer.) Pre-rework this test would have
// been the inverse assertion (DoesNotEvict); that assertion contradicted
// the invariant and broke 3 integration tests, so it was removed.
func TestColdLoad_EvictsPinnedSwapGroupResident(t *testing.T) {
	s, mgrs := makeInvariantScheduler(t, pinnedResidentConfig, map[string]State{
		"resident": StateReady,
		"target":   StateStopped,
	})
	cleanup := func() { SetSleepDockerCmdForTest(nil) }
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer cleanup()

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 1 || slept[0] != "resident" {
		t.Errorf("documented invariant: pinned swap_group resident must be slept by a cross-group cold-load: slept=%v stopped=%v", slept, stopped)
	}
	if got := mgrs["resident"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected resident.Sleep=1 (pinned-in-swap_group is evictable), got %d", got)
	}
}

// criticalTargetConfig: this time the COLD-LOAD TARGET is critical and
// the resident is a NORMAL, non-pinned cross-group peer. The guard must
// NOT block here — a critical target IS allowed to displace a strictly
// lower-priority cross-group peer. Proves the guard is priority-aware,
// not a blanket "never evict" that would defeat the helper's purpose.
const criticalTargetConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  resident:
    lifecycle: external
    host: vllm-resident
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: resident-group
    priority: normal
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
    priority: critical
    expected_vram_mb_per_gpu: 11000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// TestColdLoad_CriticalTargetMayEvictLowerPriorityPeer proves the guard
// permits a strictly-outranking cold-load to evict a lower-priority
// cross-group peer (so the eviction helper still does its job).
func TestColdLoad_CriticalTargetMayEvictLowerPriorityPeer(t *testing.T) {
	s, mgrs := makeInvariantScheduler(t, criticalTargetConfig, map[string]State{
		"resident": StateReady,
		"target":   StateStopped,
	})
	cleanup := func() { SetSleepDockerCmdForTest(nil) }
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer cleanup()

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if len(slept) != 1 || slept[0] != "resident" {
		t.Errorf("expected critical target to sleep the normal cross-group resident: slept=%v stopped=%v", slept, stopped)
	}
	if got := mgrs["resident"].sleepCalls.Load(); got != 1 {
		t.Errorf("expected resident.Sleep=1 (critical target outranks normal peer), got %d", got)
	}
}

// hotReloadStopConfig: `holder` is a normal-priority, NON-sleep_mode
// cross-group contender (sleep_mode omitted → false), so cross-group
// eviction must take the docker-STOP branch (not sleepInstance). The
// docker stop target is sourced from the peer's config Host. `target`
// is the cold-loading peer overlapping on GPU 1. This is the topology
// that surfaces the MEDIUM-2/HIGH hot-reload TOCTOU: the stop branch
// must `docker stop` the container we picked under the first-pass RLock
// snapshot (c.cfg.Host), NOT a Host that a concurrent hot-reload mutated
// into s.cfg between the snapshot and the action.
const hotReloadStopConfig = `
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
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// TestColdLoad_HotReloadConfigTOCTOUUsesSnapshot is the load-bearing
// MEDIUM-2/HIGH regression: evictCrossGroupGPUContendersLocked must act
// on the per-peer config SNAPSHOT captured under the first-pass RLock
// (the by-value c.cfg), NOT a live re-fetch of s.cfg. A hot-reload that
// changes a peer's Host (container name) mid-eviction must NOT redirect
// the `docker stop` at the freshly-renamed (wrong) container.
//
// We surface the exact window via SetEvictionLoopPreActionHookForTest,
// which fires AFTER the bulk-snapshot RUnlock (so c.cfg.Host=="vllm-holder"
// is already captured) and BEFORE the per-peer action. In the hook we
// simulate a concurrent hot-reload by repointing s.cfg.Models["holder"].Host
// to "wrong-container".
//
// Post-fix (sleep.go ~line 1574: `peerCfg := c.cfg`): the stop branch
// uses the snapshot Host → docker stop targets "vllm-holder".
// Pre-fix (the reverted `liveModelCfg(s.cfg, ...)` form): the stop branch
// re-fetches the mutated live config → docker stop targets "wrong-container".
//
// holder is sleep_mode:false so the docker-STOP branch (which uses
// peerCfg.Host directly) engages — this is where the Host TOCTOU is
// observable.
func TestColdLoad_HotReloadConfigTOCTOUUsesSnapshot(t *testing.T) {
	s, mgrs := makeInvariantScheduler(t, hotReloadStopConfig, map[string]State{
		"holder": StateReady,
		"target": StateStopped,
	})

	stopCalledFor, cleanup := hookDockerStops(t)
	defer cleanup()

	// Install the mid-loop hook: simulate a hot-reload mutating holder's
	// Host between the bulk-snapshot RUnlock and the per-peer stop action.
	// If the production code re-fetches s.cfg here (the bug), it will
	// docker-stop "wrong-container"; if it uses the c.cfg snapshot (the
	// fix), it will docker-stop "vllm-holder".
	var hookFired atomic.Int32
	restore := SetEvictionLoopPreActionHookForTest(func(peer string) {
		if peer != "holder" {
			return
		}
		hookFired.Add(1)
		s.mu.Lock()
		hc := s.cfg.Models["holder"]
		hc.Host = "wrong-container"
		s.cfg.Models["holder"] = hc
		s.mu.Unlock()
	})
	defer restore()

	targetCfg := s.cfg.Models["target"]
	slept, stopped := s.evictCrossGroupGPUContendersLocked(context.Background(), "target", targetCfg)

	if hookFired.Load() == 0 {
		t.Fatalf("setup: pre-action hook never fired for holder — eviction loop did not iterate as expected (slept=%v stopped=%v)", slept, stopped)
	}

	// The stop must have targeted the SNAPSHOT container name.
	if !stopCalledFor("vllm-holder") {
		t.Errorf("TOCTOU: expected docker stop on snapshot container %q, but it was never stopped (slept=%v stopped=%v)", "vllm-holder", slept, stopped)
	}
	// And must NOT have hit the hot-reloaded (wrong) container name.
	if stopCalledFor("wrong-container") {
		t.Errorf("TOCTOU violated: docker stop hit the hot-reloaded container %q instead of the snapshot %q — eviction re-fetched live s.cfg instead of using the c.cfg snapshot", "wrong-container", "vllm-holder")
	}
	// holder is the only overlapping cross-group contender → exactly one stop.
	if len(stopped) != 1 || stopped[0] != "holder" {
		t.Errorf("expected stopped=[holder], got %v (slept=%v)", stopped, slept)
	}
	// sleep_mode:false → the sleepInstance branch must NOT have engaged.
	if got := mgrs["holder"].sleepCalls.Load(); got != 0 {
		t.Errorf("expected holder.Sleep=0 (sleep_mode false → docker-stop branch), got %d", got)
	}
}
