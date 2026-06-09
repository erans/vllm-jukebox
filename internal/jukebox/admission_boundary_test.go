package jukebox

import (
	"context"
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// Boundary / property-based tests for AdmissionController.
//
// Each test exercises an arithmetic edge or topology corner case that
// the per-GPU subtraction math in pickVictims / availableMB / state
// transitions could plausibly mishandle. Tests are deterministic and
// fast (< 1s each); they share no state with the regular admission
// tests.

// newBoundaryAdmission is a flexible factory that builds a controller
// with the supplied models + totalsByGPU. Returns the controller and a
// stub evictor whose calls can be inspected via SleepCalls() / Calls().
func newBoundaryAdmission(_ *testing.T, cfg *config.Config, totalsByGPU map[int]int) (*AdmissionController, *stubEvictor) {
	ev := &stubEvictor{}
	a := NewAdmissionController(cfg, totalsByGPU, ev)
	return a, ev
}

// assertNoPanic runs fn and fails the test if it panics. Used for
// tests whose primary property is "doesn't crash".
func assertNoPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	fn()
}

// ---------------------------------------------------------------------
// 1. Zero VRAM — model is not admission-tracked at all.
// ---------------------------------------------------------------------

func TestBoundary_ZeroVRAM_NotTracked(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"plain": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 0, // disables admission tracking
		},
	})
	a, ev := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	if a.IsTracked("plain") {
		t.Fatal("plain has expected_vram=0 → must not be admission-tracked")
	}
	victims, err := a.RequestWake(context.Background(), "plain")
	if err != nil {
		t.Fatalf("RequestWake: %v", err)
	}
	if len(victims) != 0 {
		t.Fatalf("expected no victims, got %v", victims)
	}
	if len(ev.Calls()) != 0 {
		t.Fatalf("evictor not called; got %v", ev.Calls())
	}
}

// ---------------------------------------------------------------------
// 2. Validator rejects residual > expected.
//    (Validator-level test lives in config; this test confirms the
//    admission controller never sees a config that violates that
//    invariant. We round-trip through Validate() to keep the fence
//    tight.)
// ---------------------------------------------------------------------

func TestBoundary_ResidualExceedsAwake_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"bad": {
				Path:                 "/m/bad",
				GPUs:                 []int{0},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 1000,
				SleepL1ResidualMB:    2000, // > expected → invalid
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validator to reject sleep_l1_residual_mb > expected_vram_mb_per_gpu, got nil")
	}
	if !strings.Contains(err.Error(), "sleep_l1_residual_mb") || !strings.Contains(err.Error(), "expected_vram_mb_per_gpu") {
		t.Errorf("expected error to cite both fields, got: %v", err)
	}
}

// intPtr is a local helper to keep the validator-shape tests compact.
func intPtr(i int) *int { return &i }

// ---------------------------------------------------------------------
// 3. Residual == expected — sleep frees zero. Degenerate but legal.
// ---------------------------------------------------------------------

func TestBoundary_ResidualEqualsAwake_Allowed(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"degen": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    8000, // residual == expected (legal)
			Priority:             config.PriorityNormal,
		},
	})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	if _, err := a.RequestWake(context.Background(), "degen"); err != nil {
		t.Fatalf("first wake: %v", err)
	}
	a.NotifySleep("degen", "manual")

	// After sleep: awake=0, residual=8000. Available = 24000 - 8000 = 16000.
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 0 {
		t.Errorf("expected awake=0, got %d", snap[0].AwakeMB)
	}
	if snap[0].L1ResidualMB != 8000 {
		t.Errorf("expected residual=8000, got %d", snap[0].L1ResidualMB)
	}
	if snap[0].AvailableMB != 16000 {
		t.Errorf("expected available=16000, got %d", snap[0].AvailableMB)
	}

	// Re-wake: needs (8000 - 8000) = 0 additional. Should fit trivially.
	victims, err := a.RequestWake(context.Background(), "degen")
	if err != nil {
		t.Fatalf("re-wake: %v", err)
	}
	if len(victims) != 0 {
		t.Errorf("expected 0 victims, got %v", victims)
	}
	snap = a.SnapshotBudgets()
	if snap[0].AwakeMB != 8000 {
		t.Errorf("expected awake=8000, got %d", snap[0].AwakeMB)
	}
}

// ---------------------------------------------------------------------
// 4. Overcommitted expected — model bigger than the GPU.
// ---------------------------------------------------------------------

func TestBoundary_OvercommittedExpected_AdmissionRejects(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"too-big": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 30000, // bigger than the 24000 GPU
			Priority:             config.PriorityNormal,
		},
	})
	a, ev := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	_, err := a.RequestWake(context.Background(), "too-big")
	if err == nil {
		t.Fatal("expected admission rejection for overcommitted model, got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected infeasibility error, got: %v", err)
	}
	// No state corruption — model was never marked awake.
	snap := a.SnapshotBudgets()
	if snap[0].AwakeMB != 0 {
		t.Errorf("expected awake=0 after rejected wake, got %d", snap[0].AwakeMB)
	}
	if len(ev.Calls()) != 0 {
		t.Errorf("evictor must not be called on infeasible wake, got %v", ev.Calls())
	}
}

// ---------------------------------------------------------------------
// 5. Disjoint GPU sets — wakes on one model don't bleed into the other.
// ---------------------------------------------------------------------

func TestBoundary_GPUSetDisjoint_NoCrosstalk(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"left": {
			GPUs:                 []int{0, 1},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
		},
		"right": {
			GPUs:                 []int{2, 3},
			ExpectedVRAMMBPerGPU: 10000,
			Priority:             config.PriorityNormal,
		},
	})
	totals := map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}
	a, ev := newBoundaryAdmission(t, cfg, totals)

	if _, err := a.RequestWake(context.Background(), "left"); err != nil {
		t.Fatalf("wake left: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "right"); err != nil {
		t.Fatalf("wake right: %v", err)
	}
	if len(ev.Calls()) != 0 {
		t.Errorf("disjoint wakes should not evict each other, got %v", ev.Calls())
	}
	// Both models should be Awake, books should reflect ONLY their own
	// GPUs.
	snap := a.SnapshotBudgets()
	byGPU := map[int]GPUBudgetSnapshot{}
	for _, s := range snap {
		byGPU[s.GPUID] = s
	}
	for _, g := range []int{0, 1} {
		if byGPU[g].AwakeMB != 10000 {
			t.Errorf("GPU %d expected awake=10000 (left), got %d", g, byGPU[g].AwakeMB)
		}
	}
	for _, g := range []int{2, 3} {
		if byGPU[g].AwakeMB != 10000 {
			t.Errorf("GPU %d expected awake=10000 (right), got %d", g, byGPU[g].AwakeMB)
		}
	}
}

// ---------------------------------------------------------------------
// 6. Subset GPU sets — single-GPU member shares one GPU with TP+PP peer.
// ---------------------------------------------------------------------

func TestBoundary_GPUSetSubset(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big-pp": {
			GPUs:                 []int{0, 1, 2, 3},
			ExpectedVRAMMBPerGPU: 18000,
			SwapGroup:            "vision-or-moe",
			Priority:             config.PriorityNormal,
		},
		"vision": {
			GPUs:                 []int{3},
			ExpectedVRAMMBPerGPU: 8000,
			SwapGroup:            "vision-or-moe",
			Priority:             config.PriorityNormal,
		},
	})
	totals := map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}
	a, _ := newBoundaryAdmission(t, cfg, totals)

	if _, err := a.RequestWake(context.Background(), "big-pp"); err != nil {
		t.Fatalf("wake big-pp: %v", err)
	}
	// GPU 3 now has 18000 awake, 6000 free. vision needs 8000 → must
	// evict big-pp (the only group peer overlapping GPU 3).
	victims, err := a.RequestWake(context.Background(), "vision")
	if err != nil {
		t.Fatalf("wake vision: %v", err)
	}
	if len(victims) != 1 || victims[0] != "big-pp" {
		t.Errorf("expected big-pp evicted, got %v", victims)
	}
	// Verify books: big-pp must no longer be awake on ANY of its GPUs.
	snap := a.SnapshotBudgets()
	byGPU := map[int]GPUBudgetSnapshot{}
	for _, s := range snap {
		byGPU[s.GPUID] = s
	}
	for _, g := range []int{0, 1, 2} {
		if byGPU[g].AwakeMB != 0 {
			t.Errorf("GPU %d expected awake=0 (big-pp evicted), got %d", g, byGPU[g].AwakeMB)
		}
	}
	if byGPU[3].AwakeMB != 8000 {
		t.Errorf("GPU 3 expected awake=8000 (vision), got %d", byGPU[3].AwakeMB)
	}
}

// ---------------------------------------------------------------------
// 7. 1-GPU machine.
// ---------------------------------------------------------------------

func TestBoundary_SingleGPUMachine(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"only-model": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 12000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
		},
	})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	if _, err := a.RequestWake(context.Background(), "only-model"); err != nil {
		t.Fatalf("wake: %v", err)
	}
	a.NotifySleep("only-model", "manual")
	if _, err := a.RequestWake(context.Background(), "only-model"); err != nil {
		t.Fatalf("re-wake: %v", err)
	}

	snap := a.SnapshotBudgets()
	if len(snap) != 1 || snap[0].GPUID != 0 {
		t.Fatalf("expected single GPU 0 in snapshot, got %+v", snap)
	}
	if snap[0].AwakeMB != 12000 {
		t.Errorf("expected awake=12000, got %d", snap[0].AwakeMB)
	}
}

// ---------------------------------------------------------------------
// 8. 8-GPU machine.
// ---------------------------------------------------------------------

func TestBoundary_EightGPUMachine(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"big-tp": {
			GPUs:                 []int{0, 1, 2, 3, 4, 5, 6, 7},
			ExpectedVRAMMBPerGPU: 16000,
			Priority:             config.PriorityNormal,
		},
	})
	totals := map[int]int{}
	for g := 0; g < 8; g++ {
		totals[g] = 80000 // pretend H100s
	}
	a, _ := newBoundaryAdmission(t, cfg, totals)

	if _, err := a.RequestWake(context.Background(), "big-tp"); err != nil {
		t.Fatalf("wake big-tp on 8gpu: %v", err)
	}
	snap := a.SnapshotBudgets()
	if len(snap) != 8 {
		t.Fatalf("expected 8 GPUs in snapshot, got %d", len(snap))
	}
	for _, s := range snap {
		if s.AwakeMB != 16000 {
			t.Errorf("GPU %d expected awake=16000, got %d", s.GPUID, s.AwakeMB)
		}
		if s.AvailableMB != 80000-16000 {
			t.Errorf("GPU %d expected available=64000, got %d", s.GPUID, s.AvailableMB)
		}
	}
}

// ---------------------------------------------------------------------
// 9. Empty model set — controller is a no-op factory.
// ---------------------------------------------------------------------

func TestBoundary_EmptyModelSet(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	if a.IsTracked("anything") {
		t.Error("untracked model should report IsTracked=false")
	}
	assertNoPanic(t, func() {
		victims, err := a.RequestWake(context.Background(), "anything")
		if err != nil {
			t.Errorf("RequestWake on empty controller: %v", err)
		}
		if len(victims) != 0 {
			t.Errorf("expected no victims, got %v", victims)
		}
		a.NotifySleep("anything", "manual")
		a.NotifyWakeComplete("anything")
		a.TouchActivity("anything")
		a.NotifyStopped("anything")
		a.NotifyStarted("anything")
		a.NotifyStartFailed("anything")
	})
}

// ---------------------------------------------------------------------
// 10. Pinned non-group: NotifySleep is a no-op (books stay in pinnedByGPU).
// ---------------------------------------------------------------------

func TestBoundary_RequestWakeOfPinnedNonGroup(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"pin": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 10000,
			Pinned:               boolPtr(true),
			// no swap_group
		},
	})
	a, ev := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	// Initial budgets: pinned footprint is in pinnedByGPU.
	snap := a.SnapshotBudgets()
	if snap[0].PinnedMB != 10000 {
		t.Errorf("expected pinnedMB=10000, got %d", snap[0].PinnedMB)
	}
	if snap[0].AwakeMB != 0 {
		t.Errorf("non-group pinned must have awakeMB=0 (footprint in pinnedByGPU), got %d", snap[0].AwakeMB)
	}

	// RequestWake on pinned/awake model is a no-op.
	victims, err := a.RequestWake(context.Background(), "pin")
	if err != nil {
		t.Fatalf("RequestWake on pinned awake model: %v", err)
	}
	if len(victims) != 0 {
		t.Fatalf("expected no victims, got %v", victims)
	}
	if len(ev.Calls()) != 0 {
		t.Errorf("evictor must not be called, got %v", ev.Calls())
	}

	// NotifySleep on a pinned non-group model must not change books
	// (the footprint lives in pinnedByGPU, not awakeByGPU).
	a.NotifySleep("pin", "manual")
	snap = a.SnapshotBudgets()
	if snap[0].PinnedMB != 10000 {
		t.Errorf("pinnedMB must be unchanged after spurious NotifySleep, got %d", snap[0].PinnedMB)
	}
	if snap[0].AwakeMB != 0 {
		t.Errorf("awakeMB must remain 0 (no underflow), got %d", snap[0].AwakeMB)
	}
	if snap[0].L1ResidualMB != 0 {
		t.Errorf("residual must remain 0 (sleep should not have applied), got %d", snap[0].L1ResidualMB)
	}
}

// ---------------------------------------------------------------------
// 11. Pickvictims: shortfall == one peer's freed amount exactly.
// ---------------------------------------------------------------------

func TestBoundary_PickVictims_ExactlyEnough(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 16000,
			Priority:             config.PriorityNormal,
		},
		"victim-a": {
			// Freed = expected - residual = 8000 - 0 = 8000
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    0,
			Priority:             config.PriorityBestEffort,
		},
		"victim-b": {
			// Same shape. Picker should NOT need both.
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    0,
			Priority:             config.PriorityBestEffort,
		},
	})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	if _, err := a.RequestWake(context.Background(), "victim-a"); err != nil {
		t.Fatalf("wake victim-a: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "victim-b"); err != nil {
		t.Fatalf("wake victim-b: %v", err)
	}
	// 16000 awake, 8000 free. Newcomer needs 16000 → shortfall 8000.
	// One victim freeing 8000 exactly satisfies. Picker must pick ONE,
	// not both.
	victims, err := a.RequestWake(context.Background(), "newcomer")
	if err != nil {
		t.Fatalf("wake newcomer: %v", err)
	}
	if len(victims) != 1 {
		t.Errorf("expected exactly 1 victim (shortfall == one peer's freed), got %v", victims)
	}
}

// ---------------------------------------------------------------------
// 12. Mixed sleep + stop members in same swap group — pickVictims
//     correctly accounts freed amounts (sleep frees expected - residual,
//     stop frees full expected).
// ---------------------------------------------------------------------

func TestBoundary_PickVictims_StopAndSleepDispatch(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 22000,
			SwapGroup:            "g",
			Priority:             config.PriorityNormal,
		},
		"sleepy": {
			// Freed via sleep = 8000 - 3000 = 5000.
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    3000,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionSleep,
			Priority:             config.PriorityNormal,
		},
		"stoppy": {
			// Freed via stop = full 8000 (residual irrelevant).
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    3000,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Priority:             config.PriorityNormal,
		},
	})
	a, ev := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	// Wake both peers. GPU 0: awake 16000, available 8000.
	if _, err := a.RequestWake(context.Background(), "sleepy"); err != nil {
		t.Fatalf("wake sleepy: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "stoppy"); err != nil {
		t.Fatalf("wake stoppy: %v", err)
	}

	// Newcomer needs 22000. Available = 8000 → shortfall = 14000.
	// To satisfy: we need freed >= 14000.
	//   - Both peers freed = 5000 + 8000 = 13000 (NOT enough)
	//   - That'd reject. So bump newcomer to 21000 instead.
	// Adjust expected demand within the test by re-creating with smaller need.
	// (We want a case where BOTH need to be picked AND the freed math
	// distinguishes sleep vs stop.)
	// Actually let's verify: 5000 + 8000 = 13000. Newcomer needs 22000,
	// available 8000, shortfall 14000. 13000 < 14000 → reject. Adjust.

	// Re-setup with a need that BOTH peers together can satisfy:
	// shortfall 13000. Need = available + shortfall = 8000 + 13000 = 21000.
	cfg2 := makeCfg(map[string]config.ModelConfig{
		"newcomer": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 21000,
			SwapGroup:            "g",
			Priority:             config.PriorityNormal,
		},
		"sleepy": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    3000,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionSleep,
			Priority:             config.PriorityNormal,
		},
		"stoppy": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 8000,
			SleepL1ResidualMB:    3000,
			SwapGroup:            "g",
			EvictAction:          config.EvictActionStop,
			Priority:             config.PriorityNormal,
		},
	})
	a, ev = newBoundaryAdmission(t, cfg2, map[int]int{0: 24000})
	if _, err := a.RequestWake(context.Background(), "sleepy"); err != nil {
		t.Fatalf("wake sleepy: %v", err)
	}
	if _, err := a.RequestWake(context.Background(), "stoppy"); err != nil {
		t.Fatalf("wake stoppy: %v", err)
	}

	victims, err := a.RequestWake(context.Background(), "newcomer")
	if err != nil {
		t.Fatalf("wake newcomer: %v", err)
	}
	if len(victims) != 2 {
		t.Errorf("expected both peers picked, got %v", victims)
	}

	// Confirm each was dispatched via its correct evictor verb.
	calls := ev.Calls()
	sawSleep := false
	sawStop := false
	for _, c := range calls {
		if c == "sleep:sleepy" {
			sawSleep = true
		}
		if c == "stop:stoppy" {
			sawStop = true
		}
	}
	if !sawSleep {
		t.Errorf("expected sleep dispatch for sleepy, calls=%v", calls)
	}
	if !sawStop {
		t.Errorf("expected stop dispatch for stoppy, calls=%v", calls)
	}

	// Books: sleepy left 3000 residual, stoppy left zero (full reclaim).
	snap := a.SnapshotBudgets()
	if snap[0].L1ResidualMB != 3000 {
		t.Errorf("expected residual=3000 (sleepy only), got %d", snap[0].L1ResidualMB)
	}
}

// ---------------------------------------------------------------------
// 13. Model declares a GPU not in the inventory.
//
//	availableMB returns 0 for unknown GPUs (defensive fallback) — so
//	a model declared on GPU 5 with no totalsByGPU[5] will see
//	available=0 and either reject (no peers) or require eviction.
//	Either way: no panic, useful error.
// ---------------------------------------------------------------------

func TestBoundary_GPUMissingFromInventory(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"orphan": {
			GPUs:                 []int{5}, // not in inventory
			ExpectedVRAMMBPerGPU: 1000,
			Priority:             config.PriorityNormal,
		},
	})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000, 1: 24000})

	assertNoPanic(t, func() {
		_, err := a.RequestWake(context.Background(), "orphan")
		if err == nil {
			t.Fatal("expected admission rejection for GPU 5 not in inventory, got nil")
		}
		if !strings.Contains(err.Error(), "GPU 5") && !strings.Contains(err.Error(), "cannot free") {
			t.Errorf("expected error to identify the orphan GPU/infeasibility, got: %v", err)
		}
	})
}

// ---------------------------------------------------------------------
// 14. Validator rejects duplicate GPU IDs in a model's gpus list.
//     (Verifies validateScheduler catches it before admission ever sees
//     the duplicates.)
// ---------------------------------------------------------------------

func TestBoundary_GPUMixedDuplicate(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"dup": {
				Path:                 "/m/dup",
				GPUs:                 []int{0, 1, 0, 2}, // duplicate 0
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 1000,
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validator to reject duplicate GPU IDs, got nil")
	}
	if !strings.Contains(err.Error(), "duplicates") {
		t.Errorf("expected error to mention duplicates, got: %v", err)
	}
}

// ---------------------------------------------------------------------
// 15. 1 GPU, 1 model that fills it. Self-eviction path: a duplicate
//     wake is a no-op; a different-model wake sees no eligible peers
//     (the only awake model IS too-big to evict — it's also critical
//     by way of being pinned, so picker correctly skips it).
// ---------------------------------------------------------------------

func TestBoundary_OneGPUOneSinglemodel(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"hog": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 24000, // exactly the GPU
			Pinned:               boolPtr(true),
		},
		"interloper": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 1000,
			Priority:             config.PriorityNormal,
		},
	})
	a, ev := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	// hog is pinned-non-group; its 24000 lives in pinnedByGPU. GPU has
	// 0 free. interloper needs 1000 → shortfall 1000, no eligible
	// peers (hog is pinned-non-group so not in candidate pool).
	_, err := a.RequestWake(context.Background(), "interloper")
	if err == nil {
		t.Fatal("expected admission rejection (only peer is pinned), got nil")
	}
	if !strings.Contains(err.Error(), "cannot free") {
		t.Errorf("expected infeasibility error, got: %v", err)
	}
	if len(ev.Calls()) != 0 {
		t.Errorf("evictor must not have been called, got %v", ev.Calls())
	}

	// And: a duplicate wake of hog itself is a no-op (already awake).
	victims, err := a.RequestWake(context.Background(), "hog")
	if err != nil {
		t.Fatalf("dup wake of pinned awake: %v", err)
	}
	if len(victims) != 0 {
		t.Errorf("expected no victims on dup wake, got %v", victims)
	}
}

// ---------------------------------------------------------------------
// Property test: state-transition arithmetic stays balanced under a
// shuffled wake/sleep sequence. Iterate a small fixed shuffled sequence
// and assert the snapshot books always sum to GPU total (no underflow,
// no leak).
// ---------------------------------------------------------------------

func TestBoundary_BookConservation(t *testing.T) {
	cfg := makeCfg(map[string]config.ModelConfig{
		"a": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 6000,
			SleepL1ResidualMB:    1000,
			Priority:             config.PriorityNormal,
		},
		"b": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 5000,
			SleepL1ResidualMB:    500,
			Priority:             config.PriorityNormal,
		},
		"c": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
			SleepL1ResidualMB:    0,
			Priority:             config.PriorityBestEffort,
		},
	})
	a, _ := newBoundaryAdmission(t, cfg, map[int]int{0: 24000})

	// Backdate so LRU order is deterministic: a oldest, b middle, c newest.
	now := time.Now()
	a.models["a"].LastRequestTime = now.Add(-3 * time.Hour)
	a.models["b"].LastRequestTime = now.Add(-2 * time.Hour)
	a.models["c"].LastRequestTime = now.Add(-1 * time.Hour)

	ops := []struct {
		op   string
		name string
	}{
		{"wake", "a"},
		{"wake", "b"},
		{"sleep", "a"},
		{"wake", "c"},
		{"sleep", "b"},
		{"sleep", "c"},
		{"wake", "b"},
	}

	for _, step := range ops {
		switch step.op {
		case "wake":
			if _, err := a.RequestWake(context.Background(), step.name); err != nil {
				t.Fatalf("wake %s: %v", step.name, err)
			}
		case "sleep":
			a.NotifySleep(step.name, "manual")
		}
		snap := a.SnapshotBudgets()[0]
		// Conservation: pinned + awake + residual + available == total.
		sum := snap.PinnedMB + snap.AwakeMB + snap.L1ResidualMB + snap.AvailableMB
		if sum != snap.TotalMB {
			t.Errorf("after %s %s: books not conserved: %d+%d+%d+%d=%d != total %d",
				step.op, step.name, snap.PinnedMB, snap.AwakeMB, snap.L1ResidualMB, snap.AvailableMB, sum, snap.TotalMB)
		}
		// No underflow: nothing should go negative (would manifest as a
		// huge AvailableMB exceeding TotalMB or a negative summand).
		if snap.PinnedMB < 0 || snap.AwakeMB < 0 || snap.L1ResidualMB < 0 || snap.AvailableMB < 0 {
			t.Errorf("after %s %s: negative book entry: %+v", step.op, step.name, snap)
		}
		if snap.AvailableMB > snap.TotalMB {
			t.Errorf("after %s %s: available (%d) > total (%d) — underflow somewhere",
				step.op, step.name, snap.AvailableMB, snap.TotalMB)
		}
	}
}
