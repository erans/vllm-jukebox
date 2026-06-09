package jukebox

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// stubEvictor records SleepForEviction / StopForEviction calls and
// (optionally) returns errors for specific victims. Adheres to the
// evictor contract: it does NOT call back into the AdmissionController.
type stubEvictor struct {
	mu        sync.Mutex
	calls     []string // sleep+stop calls in order, prefixed "sleep:" / "stop:"
	failWith  map[string]error
	stopFails map[string]error
}

func (s *stubEvictor) SleepForEviction(_ context.Context, victim, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, "sleep:"+victim)
	if err, ok := s.failWith[victim]; ok {
		return err
	}
	return nil
}

func (s *stubEvictor) StopForEviction(_ context.Context, victim, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, "stop:"+victim)
	if err, ok := s.stopFails[victim]; ok {
		return err
	}
	return nil
}

func (s *stubEvictor) Calls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.calls))
	copy(out, s.calls)
	return out
}

// SleepCalls returns just the victim names from sleep-style calls,
// preserving the pre-existing test API used by tests written before
// the stop-on-evict split.
func (s *stubEvictor) SleepCalls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []string{}
	for _, c := range s.calls {
		if strings.HasPrefix(c, "sleep:") {
			out = append(out, strings.TrimPrefix(c, "sleep:"))
		}
	}
	return out
}

// boolPtr is a tiny helper for the optional *bool fields.
func boolPtr(b bool) *bool { return &b }

// makeCfg builds a minimal scheduler-mode config with the supplied
// models. All models are vllm runtime + sleep_mode unless explicitly
// overridden. The scheduler block is non-nil so AdmissionEnabled and
// validateAdmission both work.
func makeCfg(models map[string]config.ModelConfig) *config.Config {
	return &config.Config{
		Scheduler: &config.SchedulerConfig{},
		Models:    models,
	}
}

// TestAdmissionUntrackedNoop: a wake for a model with no
// expected_vram_mb_per_gpu is a no-op (returns nil, nil) and never
// touches the evictor.
func TestAdmissionUntrackedNoop(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"plain": {GPUs: []int{0}},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	victims, err := a.RequestWake(context.Background(), "plain")
	if err != nil {
		t.Fatalf("RequestWake error: %v", err)
	}
	if len(victims) != 0 {
		t.Fatalf("expected no victims, got %v", victims)
	}
	if len(ev.SleepCalls()) != 0 {
		t.Fatalf("evictor should not have been called, got %v", ev.SleepCalls())
	}
}

// TestAdmissionFitsWithoutEviction: a model that fits in the
// pinned-adjusted budget admits immediately, no victims.
func TestAdmissionFitsWithoutEviction(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"small": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 3000,
			SleepL1ResidualMB:    500,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	victims, err := a.RequestWake(context.Background(), "small")
	if err != nil {
		t.Fatalf("RequestWake error: %v", err)
	}
	if len(victims) != 0 {
		t.Fatalf("expected no victims, got %v", victims)
	}
	if len(ev.SleepCalls()) != 0 {
		t.Fatalf("evictor should not have been called, got %v", ev.SleepCalls())
	}

	// Budget reflects the awake model.
	snap := a.SnapshotBudgets()
	if len(snap) != 1 {
		t.Fatalf("expected 1 GPU in snapshot, got %d", len(snap))
	}
	if snap[0].AwakeMB != 3000 {
		t.Errorf("expected awake=3000, got %d", snap[0].AwakeMB)
	}
	if snap[0].AvailableMB != 21000 {
		t.Errorf("expected available=21000, got %d", snap[0].AvailableMB)
	}
}

// TestAdmissionEvictsBestEffortFirst: when eviction is needed,
// best-effort models are evicted before normal-priority peers, even
// when the normal-priority peer was used longer ago.
func TestAdmissionEvictsBestEffortFirst(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big-newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
		},
		"normal-old": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
		},
		"besteffort-new": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityBestEffort,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wake both peers so the GPU is at 16000/24000.
	if _, err := a.RequestWake(context.Background(), "normal-old"); err != nil {
		t.Fatalf("RequestWake normal-old: %v", err)
	}
	// Backdate normal-old so it's the LRU candidate among "normal" peers.
	a.models["normal-old"].LastRequestTime = time.Now().Add(-1 * time.Hour)
	if _, err := a.RequestWake(context.Background(), "besteffort-new"); err != nil {
		t.Fatalf("RequestWake besteffort-new: %v", err)
	}

	// Now wake the big newcomer. 18000 needed, 8000 available → must
	// evict 10000+. besteffort-new alone frees 8000 (not enough),
	// normal-old frees 8000. Picker should grab besteffort FIRST
	// (priority), then normal-old (still short).
	victims, err := a.RequestWake(context.Background(), "big-newcomer")
	if err != nil {
		t.Fatalf("RequestWake big-newcomer: %v", err)
	}
	if len(victims) != 2 {
		t.Fatalf("expected 2 victims, got %v", victims)
	}
	if victims[0] != "besteffort-new" {
		t.Errorf("expected besteffort-new evicted first, got %v", victims)
	}
	if victims[1] != "normal-old" {
		t.Errorf("expected normal-old evicted second, got %v", victims)
	}
}

// TestAdmissionEvictsLRUWithinPriority: with multiple normal-priority
// candidates, the least-recently-used is evicted first.
func TestAdmissionEvictsLRUWithinPriority(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
		},
		"old": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
		},
		"new": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "old"); err != nil {
		t.Fatalf("RequestWake old: %v", err)
	}
	a.models["old"].LastRequestTime = time.Now().Add(-1 * time.Hour)
	if _, err := a.RequestWake(context.Background(), "new"); err != nil {
		t.Fatalf("RequestWake new: %v", err)
	}

	victims, err := a.RequestWake(context.Background(), "big")
	if err != nil {
		t.Fatalf("RequestWake big: %v", err)
	}
	if len(victims) == 0 || victims[0] != "old" {
		t.Errorf("expected oldest peer evicted first, got %v", victims)
	}
}

// TestAdmissionPinnedNeverEvicted: a pinned (forced-critical) model is
// never evicted, even when its eviction would solve the budget shortfall.
func TestAdmissionPinnedNeverEvicted(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"pinned": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			Pinned:               boolPtr(true),
		},
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 6000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)
	// Pinned eats 20000. Available = 4000. Newcomer needs 6000 → infeasible.
	_, err := a.RequestWake(context.Background(), "newcomer")
	if err == nil {
		t.Fatal("expected admission error (pinned uneverevictable), got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected infeasibility error, got %v", err)
	}
	if len(ev.SleepCalls()) != 0 {
		t.Errorf("pinned model should not have been evicted, got calls %v", ev.SleepCalls())
	}
}

// TestAdmissionCriticalNeverEvicted: a non-pinned but explicitly
// critical model is also unevictable.
func TestAdmissionCriticalNeverEvicted(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"critical": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			Priority:             config.PriorityCritical,
		},
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 6000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "critical"); err != nil {
		t.Fatalf("wake critical: %v", err)
	}
	_, err := a.RequestWake(context.Background(), "newcomer")
	if err == nil {
		t.Fatal("expected admission error, got nil")
	}
	if len(ev.SleepCalls()) != 0 {
		t.Errorf("critical should not have been evicted, got %v", ev.SleepCalls())
	}
}

// TestAdmissionTPModelNeedsBothGPUs: a model declared on GPUs [0,3]
// needs budget on BOTH gpus — admission must reject if either is short.
func TestAdmissionTPModelNeedsBothGPUs(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"hog-on-3": {
			GPUs:                 []int{3},
			ExpectedVRAMMBPerGPU: 22000,
			Priority:             config.PriorityNormal,
		},
		"tp-pair": {
			GPUs:                 []int{0, 3},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000, 3: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "hog-on-3"); err != nil {
		t.Fatalf("wake hog-on-3: %v", err)
	}
	// GPU 0 has 24000 free, GPU 3 has 2000 free. tp-pair needs 10000 on
	// each. GPU 0 is fine; GPU 3 is short by 8000. hog-on-3 is the only
	// eligible peer (it's on GPU 3); evicting it frees 10000, enough.
	victims, err := a.RequestWake(context.Background(), "tp-pair")
	if err != nil {
		t.Fatalf("wake tp-pair: %v", err)
	}
	if len(victims) != 1 || victims[0] != "hog-on-3" {
		t.Errorf("expected hog-on-3 evicted, got %v", victims)
	}
}

// TestAdmissionRespectsResidual: a model returning from sleep needs
// only (Expected - L1Residual) additional VRAM. Verify the math.
func TestAdmissionRespectsResidual(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"sleeper": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			SleepL1ResidualMB:    5000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wake then sleep; the model is now in admissionSleeping state with
	// 5000 reserved as residual.
	if _, err := a.RequestWake(context.Background(), "sleeper"); err != nil {
		t.Fatalf("first wake: %v", err)
	}
	a.NotifySleep("sleeper", "manual")

	snap := a.SnapshotBudgets()
	if snap[0].L1ResidualMB != 5000 {
		t.Errorf("expected residual=5000, got %d", snap[0].L1ResidualMB)
	}
	if snap[0].AvailableMB != 19000 {
		t.Errorf("expected available=19000 after sleep, got %d", snap[0].AvailableMB)
	}

	// Re-wake: needs (20000 - 5000) = 15000, available is 19000, fits.
	victims, err := a.RequestWake(context.Background(), "sleeper")
	if err != nil {
		t.Fatalf("re-wake: %v", err)
	}
	if len(victims) != 0 {
		t.Errorf("expected no victims on re-wake, got %v", victims)
	}
	snap = a.SnapshotBudgets()
	if snap[0].AwakeMB != 20000 {
		t.Errorf("expected awake=20000 after re-wake, got %d", snap[0].AwakeMB)
	}
	if snap[0].L1ResidualMB != 0 {
		t.Errorf("expected residual=0 after re-wake, got %d", snap[0].L1ResidualMB)
	}
}

// TestAdmissionEvictorFailureSurfacesError: if the evictor fails on a
// victim, RequestWake returns an error and does NOT mark the requested
// model as awake.
func TestAdmissionEvictorFailureSurfacesError(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
		},
		"peer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityBestEffort,
		},
	})
	ev := &stubEvictor{
		failWith: map[string]error{
			"peer": errors.New("simulated sleep failure"),
		},
	}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "peer"); err != nil {
		t.Fatalf("wake peer: %v", err)
	}
	_, err := a.RequestWake(context.Background(), "big")
	if err == nil {
		t.Fatal("expected error from failed eviction, got nil")
	}
	if !strings.Contains(err.Error(), "simulated sleep failure") {
		t.Errorf("expected error to wrap evictor's error, got %v", err)
	}
	// big should NOT be marked awake.
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB >= 18000 {
		t.Errorf("big should not have been marked awake after evict failure, got awake=%d", snap[0].AwakeMB)
	}
}

// TestAdmissionConcurrentWakesSerialize: two goroutines requesting wake
// for two different models targeting the same GPU should serialize
// through the admission lock — no double-eviction of the same victim.
func TestAdmissionConcurrentWakesSerialize(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big-a": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			Priority:             config.PriorityNormal,
		},
		"big-b": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			Priority:             config.PriorityNormal,
		},
		"victim": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityBestEffort,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "victim"); err != nil {
		t.Fatalf("wake victim: %v", err)
	}

	// Run two wakes in parallel. Total demand is 24000 awake (a+b);
	// budget is 24000; victim must be evicted. Both goroutines must
	// see admission as gating; only ONE should succeed in evicting,
	// and the other should either fit afterward or fail.
	var wg sync.WaitGroup
	var aErr, bErr error
	wg.Add(2)
	go func() { defer wg.Done(); _, aErr = a.RequestWake(context.Background(), "big-a") }()
	go func() { defer wg.Done(); _, bErr = a.RequestWake(context.Background(), "big-b") }()
	wg.Wait()

	// At least one of the wakes must succeed (victim freed 18000;
	// 24000 - 18000 = 6000 reclaimed for awake set; big-a + big-b = 24000;
	// after evicting victim, both fit at 24000/24000). Both should
	// succeed.
	if aErr != nil {
		t.Errorf("big-a failed: %v", aErr)
	}
	if bErr != nil {
		t.Errorf("big-b failed: %v", bErr)
	}

	// Victim should have been evicted exactly once across both wakes.
	count := 0
	for _, c := range ev.SleepCalls() {
		if c == "victim" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("victim should have been evicted exactly once, got %d calls: %v", count, ev.SleepCalls())
	}
}

// TestAdmissionNotifySleepIdempotent: double-NotifySleep doesn't
// double-decrement budgets.
func TestAdmissionNotifySleepIdempotent(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 5000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	a.NotifySleep("m", "manual")
	a.NotifySleep("m", "manual")
	a.NotifySleep("m", "manual")

	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 0 {
		t.Errorf("expected awake=0 after sleep, got %d", snap[0].AwakeMB)
	}
	if snap[0].L1ResidualMB != 1000 {
		t.Errorf("expected residual=1000 (not 3000), got %d", snap[0].L1ResidualMB)
	}
}

// TestAdmissionIdempotentWake: re-wake of an already-awake model is a
// no-op (no eviction, no double-count).
func TestAdmissionIdempotentWake(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 5000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("first wake: %v", err)
	}
	victims, err := a.RequestWake(context.Background(), "m")
	if err != nil {
		t.Fatalf("second wake: %v", err)
	}
	if len(victims) != 0 {
		t.Errorf("expected no victims on re-wake, got %v", victims)
	}
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 5000 {
		t.Errorf("expected awake=5000 (not 10000), got %d", snap[0].AwakeMB)
	}
}

// TestAdmissionPinnedSubtractedFromBudget: a pinned model's footprint
// is permanently subtracted from the per-GPU budget — peers must fit
// in the remainder.
func TestAdmissionPinnedSubtractedFromBudget(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"pinned-27b": {
			GPUs:                 []int{0, 1, 2, 3},
			ExpectedVRAMMBPerGPU: 21000,
			Pinned:               boolPtr(true),
		},
		"small": {
			GPUs:                 []int{1},
			ExpectedVRAMMBPerGPU: 2500,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}, ev)

	snap := a.SnapshotBudgets()
	for _, s := range snap {
		if s.PinnedMB != 21000 {
			t.Errorf("GPU %d: expected pinned=21000, got %d", s.GPUID, s.PinnedMB)
		}
		if s.AvailableMB != 3000 {
			t.Errorf("GPU %d: expected available=3000 after pinned, got %d", s.GPUID, s.AvailableMB)
		}
	}

	// Small fits in the 3000 remainder.
	if _, err := a.RequestWake(context.Background(), "small"); err != nil {
		t.Fatalf("wake small: %v", err)
	}
	snap = a.SnapshotBudgets()
	// Verify GPU 1's available dropped to 500.
	for _, s := range snap {
		if s.GPUID == 1 && s.AvailableMB != 500 {
			t.Errorf("GPU 1: expected available=500 after small wake, got %d", s.AvailableMB)
		}
	}
}

// TestAdmissionInfeasibleEvenWithAllEvictions: a model bigger than the
// total non-pinned budget can never fit, no matter who we evict.
func TestAdmissionInfeasibleEvenWithAllEvictions(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"too-big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 30000, // exceeds 24000 even with no peers
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	_, err := a.RequestWake(context.Background(), "too-big")
	if err == nil {
		t.Fatal("expected infeasibility error, got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected 'cannot free' in error, got %v", err)
	}
}

// TestSwapGroup_MemberCanEvictPinnedSameGroup: a non-pinned swap-group
// member CAN evict a pinned/critical member of the same group — the
// declared swap relationship overrides the pinned-is-sacred rule, but
// ONLY within the group.
func TestSwapGroup_MemberCanEvictPinnedSameGroup(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"group-pinned": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			Pinned:               boolPtr(true),
			SwapGroup:            "g1",
		},
		"group-transient": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// group-pinned is awake at boot eating 20000; available=4000.
	// group-transient needs 18000 on GPU 0. Without swap-group, this
	// would fail with "cannot free" because pinned is sacred. With
	// swap-group, group-pinned is evictable.
	victims, err := a.RequestWake(context.Background(), "group-transient")
	if err != nil {
		t.Fatalf("RequestWake group-transient: %v (swap-group should let pinned be evicted)", err)
	}
	if len(victims) != 1 || victims[0] != "group-pinned" {
		t.Errorf("expected group-pinned evicted via swap-group, got %v", victims)
	}
	if len(ev.SleepCalls()) != 1 || ev.SleepCalls()[0] != "group-pinned" {
		t.Errorf("expected evictor called once for group-pinned, got %v", ev.SleepCalls())
	}
}

// TestSwapGroup_MemberCannotEvictCriticalNonGroupMember: regression
// negative — when NEITHER model declares a swap_group, a normal model
// cannot evict a critical one (existing behavior preserved).
func TestSwapGroup_MemberCannotEvictCriticalNonGroupMember(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"plain-critical": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			Priority:             config.PriorityCritical,
		},
		"plain-normal": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "plain-critical"); err != nil {
		t.Fatalf("wake plain-critical: %v", err)
	}
	_, err := a.RequestWake(context.Background(), "plain-normal")
	if err == nil {
		t.Fatal("expected admission error (critical sacred outside swap-group), got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected infeasibility error, got %v", err)
	}
	if len(ev.SleepCalls()) != 0 {
		t.Errorf("critical (no swap-group) must not be evicted, got %v", ev.SleepCalls())
	}
}

// TestSwapGroup_PrefersNonGroupVictimWhenBothExist: when both a non-group
// evictable peer AND a same-group pinned peer could each satisfy the
// shortfall, the picker prefers the non-group peer (less disruptive).
func TestSwapGroup_PrefersNonGroupVictimWhenBothExist(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"group-pinned": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Pinned:               boolPtr(true),
			SwapGroup:            "g1",
		},
		"plain-normal": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
		},
		"group-newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// group-pinned eats 10000 from boot. Wake plain-normal too → 20000
	// awake, 4000 available. group-newcomer needs 8000 → shortfall=4000.
	// Both group-pinned (in-group, normally untouchable) and plain-normal
	// (non-group, normal-priority) could satisfy. Picker should pick the
	// non-group peer first.
	if _, err := a.RequestWake(context.Background(), "plain-normal"); err != nil {
		t.Fatalf("wake plain-normal: %v", err)
	}

	victims, err := a.RequestWake(context.Background(), "group-newcomer")
	if err != nil {
		t.Fatalf("wake group-newcomer: %v", err)
	}
	if len(victims) != 1 || victims[0] != "plain-normal" {
		t.Errorf("expected non-group peer plain-normal evicted (less disruptive), got %v", victims)
	}
}

// TestSwapGroup_FallsBackToGroupWhenNonGroupInsufficient: when non-group
// candidates don't free enough VRAM, the picker dips into the same-group
// tier (even pinned members).
func TestSwapGroup_FallsBackToGroupWhenNonGroupInsufficient(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"group-pinned-big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Pinned:               boolPtr(true),
			SwapGroup:            "g1",
		},
		"plain-small": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 3000,
			Priority:             config.PriorityBestEffort,
		},
		"group-newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// group-pinned-big at 18000; available=6000. Wake plain-small (3000)
	// → 21000 awake, 3000 available. group-newcomer needs 18000 →
	// shortfall=15000. plain-small frees only 3000; need to fall back
	// into group tier and evict group-pinned-big (frees 18000).
	if _, err := a.RequestWake(context.Background(), "plain-small"); err != nil {
		t.Fatalf("wake plain-small: %v", err)
	}

	victims, err := a.RequestWake(context.Background(), "group-newcomer")
	if err != nil {
		t.Fatalf("wake group-newcomer: %v", err)
	}
	// Expect BOTH evicted: plain-small first (non-group, tier 1), then
	// group-pinned-big (same-group, tier 2 fallback).
	if len(victims) != 2 {
		t.Fatalf("expected 2 victims (plain-small then group-pinned-big), got %v", victims)
	}
	if victims[0] != "plain-small" {
		t.Errorf("expected plain-small evicted first (non-group tier), got %v", victims)
	}
	if victims[1] != "group-pinned-big" {
		t.Errorf("expected group-pinned-big evicted second (same-group fallback), got %v", victims)
	}
}

// TestSwapGroup_CrossGroupIsolation: a member of group g1 cannot evict a
// pinned/critical member of group g2 — swap-group permission is strictly
// in-group.
func TestSwapGroup_CrossGroupIsolation(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"g2-critical": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 20000,
			Priority:             config.PriorityCritical,
			SwapGroup:            "g2",
		},
		"g1-newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wake g2-critical (20000 of 24000). g1-newcomer wants 18000;
	// shortfall=14000. The only awake peer is g2-critical — different
	// group, critical priority → out of bounds. Expect infeasibility.
	if _, err := a.RequestWake(context.Background(), "g2-critical"); err != nil {
		t.Fatalf("wake g2-critical: %v", err)
	}
	_, err := a.RequestWake(context.Background(), "g1-newcomer")
	if err == nil {
		t.Fatal("expected admission error (cross-group critical should be sacred), got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected infeasibility error, got %v", err)
	}
	if len(ev.SleepCalls()) != 0 {
		t.Errorf("cross-group critical must not be evicted, got %v", ev.SleepCalls())
	}
}

// TestSwapGroup_AutoRestoreOnIdleSleep: when an idle-sleep fires
// NotifySleep with reason="idle" for a swap-group member, the controller
// dispatches a wake on the highest-priority sleeping peer in the same
// group via the registered callback.
func TestSwapGroup_AutoRestoreOnIdleSleep(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"g-default": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityCritical,
			SwapGroup:            "g1",
		},
		"g-transient": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wire the callback so we can observe the auto-restore dispatch.
	var restoreMu sync.Mutex
	var restored []string
	done := make(chan struct{}, 4)
	a.SetAutoRestoreWake(func(name, reason string) {
		restoreMu.Lock()
		restored = append(restored, name+":"+reason)
		restoreMu.Unlock()
		done <- struct{}{}
	})

	// Both awake.
	if _, err := a.RequestWake(context.Background(), "g-default"); err != nil {
		t.Fatalf("wake g-default: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "g-transient"); err != nil {
		t.Fatalf("wake g-transient: %v", err)
	}
	// Simulate the operator's pattern: g-default is the sleeping default,
	// g-transient is the awake transient peer. Manually flip state to set
	// up the world before idle-sleep.
	a.mu.Lock()
	a.markSleepingLocked(a.models["g-default"])
	a.mu.Unlock()

	// Now g-transient idles out — should trigger auto-restore of g-default
	// (highest-priority sleeping peer in same group).
	a.NotifySleep("g-transient", "idle")

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("auto-restore callback did not fire within 2s")
	}

	restoreMu.Lock()
	defer restoreMu.Unlock()
	if len(restored) != 1 || restored[0] != "g-default:swap-group-auto-restore" {
		t.Errorf("expected exactly one g-default auto-restore, got %v", restored)
	}
}

// TestSwapGroup_NoAutoRestoreOnManualSleep: a sleep with reason !=
// "idle" (e.g. "manual", "evict", "wake-rollback") must NOT trigger
// auto-restore.
func TestSwapGroup_NoAutoRestoreOnManualSleep(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"g-default": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityCritical,
			SwapGroup:            "g1",
		},
		"g-transient": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g1",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	var fired atomic.Bool
	a.SetAutoRestoreWake(func(string, string) { fired.Store(true) })

	if _, err := a.RequestWake(context.Background(), "g-transient"); err != nil {
		t.Fatalf("wake g-transient: %v", err)
	}
	// Mark g-default sleeping by hand so it's a restore candidate.
	a.mu.Lock()
	a.models["g-default"].State = admissionSleeping
	a.mu.Unlock()

	for _, reason := range []string{"manual", "evict", "wake-rollback", "boot-probe"} {
		fired.Store(false)
		a.NotifySleep("g-transient", reason)
		// Give a goroutine a beat to fire if it was going to.
		time.Sleep(50 * time.Millisecond)
		if fired.Load() {
			t.Errorf("reason %q must not trigger auto-restore", reason)
		}
		// Reset g-transient to awake for the next round.
		a.mu.Lock()
		a.models["g-transient"].State = admissionAwake
		a.mu.Unlock()
	}
}

// TestAdmissionTouchActivityUpdatesLRU: TouchActivity changes the LRU
// ordering used by pickVictims.
func TestAdmissionTouchActivityUpdatesLRU(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
		},
		"a": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
		},
		"b": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			Priority:             config.PriorityNormal,
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "a"); err != nil {
		t.Fatalf("wake a: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "b"); err != nil {
		t.Fatalf("wake b: %v", err)
	}
	// Backdate both so we control LRU explicitly.
	a.models["a"].LastRequestTime = time.Now().Add(-2 * time.Hour)
	a.models["b"].LastRequestTime = time.Now().Add(-1 * time.Hour)

	// Touch a, making it the MORE-recent.
	a.TouchActivity("a")

	// Now wake big; b should be the LRU and get evicted first.
	victims, err := a.RequestWake(context.Background(), "big")
	if err != nil {
		t.Fatalf("wake big: %v", err)
	}
	if len(victims) == 0 || victims[0] != "b" {
		t.Errorf("expected b evicted first after a was touched, got %v", victims)
	}
}

// TestStopOnEvict_DispatchesStopForEvictActionStop: when a swap-group
// member with EvictAction=stop is picked as a victim, the controller
// calls StopForEviction (not SleepForEviction) and marks the peer
// admissionStopped. Both awake AND L1 residual contributions vanish
// from the per-GPU books.
func TestStopOnEvict_DispatchesStopForEvictActionStop(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
		},
		"victim-stop": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			SleepL1ResidualMB:    2000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-victim-stop",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "victim-stop"); err != nil {
		t.Fatalf("wake victim-stop: %v", err)
	}
	// Now wake newcomer — it can't fit alongside victim, so admission
	// must evict it. Because EvictAction=stop, the stop path fires.
	victims, err := a.RequestWake(context.Background(), "newcomer")
	if err != nil {
		t.Fatalf("wake newcomer: %v", err)
	}
	if len(victims) != 1 || victims[0] != "victim-stop" {
		t.Fatalf("expected victim-stop evicted, got %v", victims)
	}

	calls := ev.Calls()
	if len(calls) != 1 || calls[0] != "stop:victim-stop" {
		t.Errorf("expected stop:victim-stop dispatch, got %v", calls)
	}
	if a.models["victim-stop"].State != admissionStopped {
		t.Errorf("expected victim-stop in admissionStopped state, got %d", a.models["victim-stop"].State)
	}

	// After stop, the victim's full footprint (12000 expected + 0 residual
	// since stop reclaims everything) is gone from the books. Newcomer's
	// 18000 takes its place.
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 18000 {
		t.Errorf("expected awake=18000 (newcomer), got %d", snap[0].AwakeMB)
	}
	if snap[0].AvailableMB != 6000 {
		t.Errorf("expected available=6000 (24000-18000), got %d", snap[0].AvailableMB)
	}
}

// TestStopOnEvict_NotifyStartedRestoresResidual: NotifyStarted on a
// previously-Stopped peer flips it to admissionSleeping and re-adds
// the L1 residual to the per-GPU books (matching the post-cold-load
// state where the container is up + slept-L1).
func TestStopOnEvict_NotifyStartedRestoresResidual(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			SleepL1ResidualMB:    2000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-m",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	a.markStoppedLocked(a.models["m"]) // simulate admission stop
	if got := a.SnapshotBudgets()[0].AvailableMB; got != 24000 {
		t.Errorf("post-stop available should be full 24000, got %d", got)
	}

	a.NotifyStarted("m")
	if a.models["m"].State != admissionSleeping {
		t.Errorf("post-NotifyStarted state should be Sleeping, got %d", a.models["m"].State)
	}
	if got := a.SnapshotBudgets()[0].AvailableMB; got != 22000 {
		t.Errorf("post-NotifyStarted available should be 24000-2000=22000, got %d", got)
	}
}

// TestIsStopped_TransitionsCorrectly: IsStopped reports the model's
// current state through the full lifecycle — fresh → wake → stop →
// re-start. Untracked models return false (no panic).
func TestIsStopped_TransitionsCorrectly(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-m",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Fresh — never woken, state == admissionUnknown.
	if a.IsStopped("m") {
		t.Errorf("fresh model should not be Stopped, got IsStopped=true")
	}

	// Wake → Awake.
	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	if a.IsStopped("m") {
		t.Errorf("awake model should not be Stopped, got IsStopped=true")
	}

	// Stop → Stopped.
	a.mu.Lock()
	a.markStoppedLocked(a.models["m"])
	a.mu.Unlock()
	if !a.IsStopped("m") {
		t.Errorf("stopped model should be Stopped, got IsStopped=false")
	}

	// NotifyStarted → Sleeping (not Stopped).
	a.NotifyStarted("m")
	if a.IsStopped("m") {
		t.Errorf("re-started model should not be Stopped (now Sleeping), got IsStopped=true")
	}

	// Untracked → false, NOT panic.
	if a.IsStopped("does-not-exist") {
		t.Errorf("untracked model should report IsStopped=false")
	}
}

// TestNotifyStartFailed_KeepsModelStopped: a model already in
// admissionStopped that fails its cold-load must stay Stopped — the
// books must remain at zero on the model's GPUs (no spurious residual
// or awake contribution).
func TestNotifyStartFailed_KeepsModelStopped(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0, 1},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-m",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000, 1: 24000}, ev)

	// Drive the model into admissionStopped directly.
	a.mu.Lock()
	a.markStoppedLocked(a.models["m"])
	a.mu.Unlock()
	if !a.IsStopped("m") {
		t.Fatalf("setup: expected Stopped, got IsStopped=false")
	}

	a.NotifyStartFailed("m")

	if !a.IsStopped("m") {
		t.Errorf("NotifyStartFailed must keep model Stopped, got IsStopped=false")
	}
	// Books must be zero on both of m's GPUs — no residual was added
	// (NotifyStartFailed must NOT route through markSleepingLocked which
	// would add residual books).
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, g := range []int{0, 1} {
		if got := a.awakeByGPU[g]; got != 0 {
			t.Errorf("GPU %d: expected awakeByGPU=0 after NotifyStartFailed, got %d", g, got)
		}
		if got := a.l1ResidualByGPU[g]; got != 0 {
			t.Errorf("GPU %d: expected l1ResidualByGPU=0 after NotifyStartFailed, got %d", g, got)
		}
	}
}

// TestNotifyStartFailed_FromAwakeAlsoCollapsesToStopped: edge case —
// if a model was somehow left Awake when its cold-load fails (e.g.
// caller's bookkeeping race), NotifyStartFailed must still collapse to
// Stopped + zero books. Guards against an Awake-wedge.
func TestNotifyStartFailed_FromAwakeAlsoCollapsesToStopped(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-m",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wake — model in Awake, awakeByGPU[0] = 8000.
	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	if a.models["m"].State != admissionAwake {
		t.Fatalf("setup: expected Awake, got %d", a.models["m"].State)
	}

	// Cold-load "failed" while we were somehow Awake — must collapse.
	a.NotifyStartFailed("m")

	if !a.IsStopped("m") {
		t.Errorf("NotifyStartFailed must collapse Awake → Stopped, got state %d", a.models["m"].State)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if got := a.awakeByGPU[0]; got != 0 {
		t.Errorf("awakeByGPU[0] should be 0 (Awake book released), got %d", got)
	}
	if got := a.l1ResidualByGPU[0]; got != 0 {
		t.Errorf("l1ResidualByGPU[0] should be 0 (no residual added), got %d", got)
	}
}

// TestWithColdLoadLock_SerializesConcurrentCalls: N concurrent
// goroutines each call WithColdLoadLock(fn) where fn brackets a brief
// sleep with entry/exit timestamps. Verify no two intervals overlap.
func TestWithColdLoadLock_SerializesConcurrentCalls(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {GPUs: []int{0}, ExpectedVRAMMBPerGPU: 1000, Priority: config.PriorityNormal},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	const N = 8
	type interval struct {
		start, end time.Time
	}
	var (
		mu        sync.Mutex
		intervals []interval
	)
	var inside atomic.Int32
	var maxInside atomic.Int32

	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			a.WithColdLoadLock(func() {
				start := time.Now()
				cur := inside.Add(1)
				// Track peak concurrency — must be exactly 1 if lock works.
				for {
					m := maxInside.Load()
					if cur <= m || maxInside.CompareAndSwap(m, cur) {
						break
					}
				}
				time.Sleep(5 * time.Millisecond)
				inside.Add(-1)
				end := time.Now()
				mu.Lock()
				intervals = append(intervals, interval{start, end})
				mu.Unlock()
			})
		}()
	}
	wg.Wait()

	if got := maxInside.Load(); got != 1 {
		t.Errorf("expected peak concurrent fn count = 1 (serial), got %d", got)
	}

	// Sort intervals by start; verify each interval ends before the
	// next begins (strict non-overlap).
	mu.Lock()
	defer mu.Unlock()
	if len(intervals) != N {
		t.Fatalf("expected %d intervals, got %d", N, len(intervals))
	}
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].start.Before(intervals[j].start) })
	for i := 1; i < len(intervals); i++ {
		if intervals[i].start.Before(intervals[i-1].end) {
			t.Errorf("intervals %d and %d overlap: prev=[%v..%v] next=[%v..%v]",
				i-1, i, intervals[i-1].start, intervals[i-1].end, intervals[i].start, intervals[i].end)
		}
	}
}

// TestWithColdLoadLock_NonReentrant: WithColdLoadLock provides mutual
// exclusion via a non-reentrant sync.Mutex. The contract (see
// admission.go's comment on coldLoadMu) is that fn must NOT recursively
// call WithColdLoadLock — a recursive call would deadlock.
//
// We verify the underlying mutual-exclusion property without leaking a
// goroutine: hold the lock from a controlled fn, observe that a second
// concurrent acquire blocks, then release the outer fn so the second
// acquire can complete. Both goroutines join cleanly, so repeated
// `-count=N` runs do not accumulate stuck goroutines or held mutexes.
func TestWithColdLoadLock_NonReentrant(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {GPUs: []int{0}, ExpectedVRAMMBPerGPU: 1000, Priority: config.PriorityNormal},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	acquired := make(chan struct{})
	releaseOuter := make(chan struct{})
	outerDone := make(chan struct{})
	go func() {
		a.WithColdLoadLock(func() {
			close(acquired)
			<-releaseOuter
		})
		close(outerDone)
	}()

	// Wait until the outer fn is actually running (lock held).
	select {
	case <-acquired:
	case <-time.After(1 * time.Second):
		close(releaseOuter)
		<-outerDone
		t.Fatal("outer WithColdLoadLock never started; setup failure")
	}

	// A second concurrent acquire MUST block — the mutex is non-reentrant
	// and held by the outer fn.
	innerDone := make(chan struct{})
	go func() {
		a.WithColdLoadLock(func() {})
		close(innerDone)
	}()

	select {
	case <-innerDone:
		// Inner returned while outer still holds the lock — contract
		// broken (mutex is reentrant or not actually held).
		close(releaseOuter)
		<-outerDone
		t.Fatal("second WithColdLoadLock returned while outer holds the lock; expected blocking per non-reentrant contract")
	case <-time.After(200 * time.Millisecond):
		// Expected: inner is blocked on the mutex.
	}

	// Release the outer fn — inner should now acquire and complete.
	close(releaseOuter)
	select {
	case <-innerDone:
	case <-time.After(1 * time.Second):
		t.Fatal("inner WithColdLoadLock did not complete after outer released; possible deadlock")
	}
	select {
	case <-outerDone:
	case <-time.After(1 * time.Second):
		t.Fatal("outer WithColdLoadLock did not return after fn signaled completion")
	}
}

// TestStopOnEvict_VictimMustBeAwake_NotStopped: pickVictims must never
// pick an admissionStopped peer (which holds zero VRAM — evicting it
// frees nothing and would mark it "stopped" twice). A Stopped peer plus
// an unmet shortfall must surface as infeasibility, not a no-op pick.
func TestStopOnEvict_VictimMustBeAwake_NotStopped(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"a": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-a",
		},
		"b": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-b",
		},
	})
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	// Wake A so it's Awake at 18000/24000.
	if _, err := a.RequestWake(context.Background(), "a"); err != nil {
		t.Fatalf("wake a: %v", err)
	}
	// Forcibly mark A Stopped (simulate: an out-of-band stop happened,
	// books reflect it). After this, A holds zero VRAM on GPU 0.
	a.mu.Lock()
	a.markStoppedLocked(a.models["a"])
	a.mu.Unlock()
	if !a.IsStopped("a") {
		t.Fatalf("setup: expected a Stopped")
	}
	// Now GPU 0 should be fully free (24000 available).
	if got := a.SnapshotBudgets()[0].AvailableMB; got != 24000 {
		t.Fatalf("setup: expected 24000 free after stopping a, got %d", got)
	}

	// Wake B — needs 18000, has 24000 free. Should succeed WITHOUT picking
	// the Stopped A as a victim. Crucially: no SleepForEviction/StopForEviction
	// call for a, since a is Stopped (already zero-budget).
	victims, err := a.RequestWake(context.Background(), "b")
	if err != nil {
		t.Fatalf("wake b: %v", err)
	}
	if len(victims) != 0 {
		t.Errorf("expected zero victims (a is Stopped, holds nothing); got %v", victims)
	}
	for _, c := range ev.Calls() {
		if c == "stop:a" || c == "sleep:a" {
			t.Errorf("Stopped peer 'a' must NEVER be picked as victim, but evictor was called: %v", ev.Calls())
		}
	}
}

// TestColdLoadLock_DoesNotBlockAdmissionMu: coldLoadMu and admission.mu
// are separate locks. A goroutine holding coldLoadMu (via a slow fn)
// must NOT block other goroutines from calling IsStopped, NotifySleep,
// NotifyWakeComplete (all of which acquire admission.mu).
func TestColdLoadLock_DoesNotBlockAdmissionMu(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"m": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 5000,
			SleepL1ResidualMB:    500,
			Priority:             config.PriorityNormal,
		},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})
	if _, err := a.RequestWake(context.Background(), "m"); err != nil {
		t.Fatalf("wake: %v", err)
	}

	// Goroutine 1: hold coldLoadMu for a chunky duration.
	heldEnough := make(chan struct{})
	releaseHold := make(chan struct{})
	go func() {
		a.WithColdLoadLock(func() {
			close(heldEnough)
			<-releaseHold
		})
	}()
	<-heldEnough // confirm coldLoadMu is held

	// Goroutine 2: must be able to call admission-mu methods without
	// blocking. Run them in a goroutine guarded by a tight timeout.
	doneAdmissionWork := make(chan struct{})
	go func() {
		_ = a.IsStopped("m")
		a.NotifySleep("m", "manual")
		a.NotifyWakeComplete("m")
		close(doneAdmissionWork)
	}()
	select {
	case <-doneAdmissionWork:
		// Good — admission ops completed despite coldLoadMu being held.
	case <-time.After(500 * time.Millisecond):
		close(releaseHold) // let the goroutine exit before failing
		t.Fatal("admission-mu operations blocked on coldLoadMu (should be independent locks)")
	}
	close(releaseHold)
}

// sleepOnlyEvictor is a stub that implements ONLY the required
// AdmissionEvictor interface — NOT the optional AdmissionStopper. Used
// to exercise the source-compatibility fallback path: when the
// controller picks an evict_action: stop victim but the evictor doesn't
// know how to stop, the controller falls back to SleepForEviction +
// sleep accounting (with a warning log line).
type sleepOnlyEvictor struct {
	mu         sync.Mutex
	sleepCalls []string
}

func (s *sleepOnlyEvictor) SleepForEviction(_ context.Context, victim, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sleepCalls = append(s.sleepCalls, victim)
	return nil
}

// Intentionally does NOT implement StopForEviction.

func (s *sleepOnlyEvictor) Calls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.sleepCalls))
	copy(out, s.sleepCalls)
	return out
}

// TestAdmissionStopper_FallbackWhenEvictorMissingStop asserts the
// source-compatibility fallback: an evictor that only implements
// AdmissionEvictor (not AdmissionStopper) processing an evict_action:
// stop victim must NOT crash or wedge. The controller falls back to
// SleepForEviction and uses sleep accounting (markSleepingLocked —
// residual stays on the books), preserving the existing interface
// contract for any external implementers that pre-date stop-on-evict.
func TestAdmissionStopper_FallbackWhenEvictorMissingStop(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 18000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
		},
		"victim-stop": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			SleepL1ResidualMB:    2000,
			Priority:             config.PriorityNormal,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Lifecycle:            config.LifecycleExternal,
			Host:                 "vllm-victim-stop",
		},
	})
	// CRITICAL: this evictor implements ONLY AdmissionEvictor, NOT the
	// optional AdmissionStopper. The controller must type-assert + fall
	// back to SleepForEviction.
	ev := &sleepOnlyEvictor{}

	// Compile-time assertion: ev satisfies AdmissionEvictor but NOT
	// AdmissionStopper. If a future refactor accidentally re-adds a
	// StopForEviction method to sleepOnlyEvictor, this test loses its
	// teeth — guard with an inverted type-assert.
	var _ AdmissionEvictor = ev
	if _, ok := any(ev).(AdmissionStopper); ok {
		t.Fatalf("test setup bug: sleepOnlyEvictor should NOT implement AdmissionStopper")
	}

	a := NewAdmissionController(cfg, map[int]int{0: 24000}, ev)

	if _, err := a.RequestWake(context.Background(), "victim-stop"); err != nil {
		t.Fatalf("wake victim-stop: %v", err)
	}

	// Now wake newcomer — it can't fit alongside victim, so admission
	// must evict. The catalog says EvictAction=stop on victim, but the
	// evictor can't stop → fall back to sleep.
	victims, err := a.RequestWake(context.Background(), "newcomer")
	if err != nil {
		t.Fatalf("wake newcomer (expected fallback to sleep): %v", err)
	}
	if len(victims) != 1 || victims[0] != "victim-stop" {
		t.Fatalf("expected victim-stop evicted, got %v", victims)
	}

	// The evictor must have received SleepForEviction (the fallback),
	// NOT StopForEviction (which it doesn't implement).
	if got := ev.Calls(); len(got) != 1 || got[0] != "victim-stop" {
		t.Errorf("expected fallback SleepForEviction(victim-stop), got %v", got)
	}

	// CRITICAL: the victim's state must reflect that a sleep happened
	// (admissionSleeping), NOT a stop (admissionStopped). The L1 residual
	// (2000 MB) must remain on the books — sleep accounting, not stop
	// accounting — because we performed a sleep, not a stop.
	if a.models["victim-stop"].State != admissionSleeping {
		t.Errorf("expected victim-stop in admissionSleeping (fallback used sleep), got state %d", a.models["victim-stop"].State)
	}
	snap := a.SnapshotBudgets()
	if snap[0].L1ResidualMB != 2000 {
		t.Errorf("expected L1 residual=2000 (sleep accounting), got %d", snap[0].L1ResidualMB)
	}
	// Newcomer ate 18000. Awake = 18000. Residual = 2000. Available = 24000-18000-2000 = 4000.
	if snap[0].AwakeMB != 18000 {
		t.Errorf("expected awake=18000 (newcomer), got %d", snap[0].AwakeMB)
	}
	if snap[0].AvailableMB != 4000 {
		t.Errorf("expected available=4000 (24000-18000-2000), got %d", snap[0].AvailableMB)
	}
}

// TestIsTracked_BasicSemantics asserts the gate-method semantics of
// AdmissionController.IsTracked: tracked models return true; models
// configured but not admission-enabled (expected_vram_mb_per_gpu == 0)
// return false; never-registered names return false; the empty string
// returns false.
//
// IsTracked is the single gate that decides whether admission runs for
// a given model — a regression making it return false would silently
// disable admission system-wide. Direct coverage is cheap insurance.
func TestIsTracked_BasicSemantics(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"tracked-a": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
		},
		"tracked-b": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
		},
		"untracked-zero-vram": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 0,
		},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	if !a.IsTracked("tracked-a") {
		t.Errorf("IsTracked(tracked-a) = false, want true")
	}
	if !a.IsTracked("tracked-b") {
		t.Errorf("IsTracked(tracked-b) = false, want true")
	}
	if a.IsTracked("untracked-zero-vram") {
		t.Errorf("IsTracked(untracked-zero-vram) = true, want false (expected_vram=0 disables admission)")
	}
	if a.IsTracked("never-registered") {
		t.Errorf("IsTracked(never-registered) = true, want false")
	}
	if a.IsTracked("") {
		t.Errorf("IsTracked(\"\") = true, want false")
	}
}

// TestTrackedModels_ListsAllAdmissionEnabled asserts TrackedModels
// returns exactly the admission-enabled models (expected_vram > 0) and
// excludes the rest. Sorted output for determinism.
//
// Same regression-bait as IsTracked: a bug here that drops or duplicates
// names would corrupt every caller that iterates the model set (e.g.
// metrics gauges, status endpoints, eviction-candidate filters).
func TestTrackedModels_ListsAllAdmissionEnabled(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"alpha": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
		},
		"beta": {
			GPUs:                 []int{1},
			ExpectedVRAMMBPerGPU: 4000,
		},
		"gamma-untracked": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 0,
		},
	})
	a := NewAdmissionController(cfg, map[int]int{0: 24000, 1: 24000}, &stubEvictor{})

	got := a.TrackedModels()
	want := []string{"alpha", "beta"}
	if len(got) != len(want) {
		t.Fatalf("TrackedModels() = %v, want %v (len mismatch)", got, want)
	}
	for i, n := range want {
		if got[i] != n {
			t.Errorf("TrackedModels()[%d] = %q, want %q (full got=%v)", i, got[i], n, got)
		}
	}
	// Defense-in-depth: untracked must not appear regardless of order.
	for _, n := range got {
		if n == "gamma-untracked" {
			t.Errorf("TrackedModels() included gamma-untracked (expected_vram=0 should be excluded), got %v", got)
		}
	}
}

