package jukebox

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// stubEvictor records SleepForEviction calls and (optionally) returns
// errors for specific victims. Adheres to the evictor contract: it
// does NOT call back into the AdmissionController.
type stubEvictor struct {
	mu       sync.Mutex
	calls    []string
	failWith map[string]error
}

func (s *stubEvictor) SleepForEviction(_ context.Context, victim, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, victim)
	if err, ok := s.failWith[victim]; ok {
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
	if len(ev.Calls()) != 0 {
		t.Fatalf("evictor should not have been called, got %v", ev.Calls())
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
	if len(ev.Calls()) != 0 {
		t.Fatalf("evictor should not have been called, got %v", ev.Calls())
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
	if len(ev.Calls()) != 0 {
		t.Errorf("pinned model should not have been evicted, got calls %v", ev.Calls())
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
	if len(ev.Calls()) != 0 {
		t.Errorf("critical should not have been evicted, got %v", ev.Calls())
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
	a.NotifySleep("sleeper")

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
	for _, c := range ev.Calls() {
		if c == "victim" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("victim should have been evicted exactly once, got %d calls: %v", count, ev.Calls())
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
	a.NotifySleep("m")
	a.NotifySleep("m")
	a.NotifySleep("m")

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
