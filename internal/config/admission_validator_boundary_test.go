package config_test

import (
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
)

// Boundary tests for the admission-config validator. These tests round-
// trip a synthetic config through Validate() and assert the validator
// catches (or correctly allows) the boundary condition.
//
// They live in the config package so the validator's behavior is fenced
// against the same arithmetic invariants the admission controller
// relies on.

func intPtr(i int) *int { return &i }

// TestBoundary_Validator_ResidualEqualsAwake_Allowed: equal residual
// and expected is legal (degenerate but valid — sleep frees zero).
func TestBoundary_Validator_ResidualEqualsAwake_Allowed(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"degen": {
				Path:                 "/m/degen",
				GPUs:                 []int{0},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 5000,
				SleepL1ResidualMB:    5000, // == expected, legal
			},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected validator to allow residual == expected, got: %v", err)
	}
}

// TestBoundary_Validator_ResidualExceedsAwake_Rejected: residual >
// expected is structurally nonsense (sleeping holds more than awake).
func TestBoundary_Validator_ResidualExceedsAwake_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"bad": {
				Path:                 "/m/bad",
				GPUs:                 []int{0},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 4000,
				SleepL1ResidualMB:    5000,
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for residual > expected, got nil")
	}
	if !strings.Contains(err.Error(), "sleep_l1_residual_mb") {
		t.Errorf("expected error to mention the field, got: %v", err)
	}
}

// TestBoundary_Validator_NegativeResidual_Rejected: negative residual
// is rejected with a clean message (not allowed to underflow books).
func TestBoundary_Validator_NegativeResidual_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"neg": {
				Path:                 "/m/neg",
				GPUs:                 []int{0},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 4000,
				SleepL1ResidualMB:    -1,
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for negative residual, got nil")
	}
	if !strings.Contains(err.Error(), "sleep_l1_residual_mb") {
		t.Errorf("expected error to cite the field, got: %v", err)
	}
}

// TestBoundary_Validator_DuplicateGPUs_Rejected: gpus: [0, 1, 0]
// is structurally invalid (admission would double-count the GPU).
func TestBoundary_Validator_DuplicateGPUs_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"dup": {
				Path:                 "/m/dup",
				GPUs:                 []int{0, 1, 0, 2},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 1000,
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for duplicate gpu IDs, got nil")
	}
	if !strings.Contains(err.Error(), "duplicates") {
		t.Errorf("expected error to mention duplicates, got: %v", err)
	}
}

// TestBoundary_Validator_AdmissionWithEmptyGPUs_Rejected: admission
// requires gpus to be set (it tracks per-GPU budget).
func TestBoundary_Validator_AdmissionWithEmptyGPUs_Rejected(t *testing.T) {
	// scheduler mode already rejects empty gpus for non-alias models,
	// so this is layered defense — both validators should fire. We
	// rely on EITHER firing as long as we get a useful error.
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"nogpus": {
				Path:                 "/m/nogpus",
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 1000,
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for admission-tracked model with no gpus, got nil")
	}
	if !strings.Contains(err.Error(), "gpus") {
		t.Errorf("expected error to cite gpus, got: %v", err)
	}
}

// TestBoundary_Validator_PinnedSwapGroupStop_Rejected: phase 4 invariant
// — pinned + swap_group + evict_action: stop is rejected by the validator.
// The combination is structurally nonsense (operator should drop pinned
// first if they really want stop-on-evict).
func TestBoundary_Validator_PinnedSwapGroupStop_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"m": {
				Lifecycle:            config.LifecycleExternal,
				Host:                 "vllm-m",
				Port:                 9000,
				GPUs:                 []int{0},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 8000,
				SwapGroup:            "g",
				EvictAction:          config.EvictActionStop,
				Pinned:               func() *bool { b := true; return &b }(),
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validator to reject pinned + swap_group + evict_action: stop, got nil")
	}
	if !strings.Contains(err.Error(), "pinned") {
		t.Errorf("expected error to mention pinned, got: %v", err)
	}
}

// ---------------------------------------------------------------------
// Per-GPU expected VRAM (expected_vram_mb_by_gpu) — task #231.
// ---------------------------------------------------------------------

// TestBoundary_Validator_PerGPU_Allowed: per-GPU map alone (no flat
// scalar) with one entry per declared GPU, all > 0 → legal.
func TestBoundary_Validator_PerGPU_Allowed(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"asym": {
				Path:               "/m/asym",
				GPUs:               []int{0, 1},
				MinFreeMemMBPerGPU: intPtr(100),
				ExpectedVRAMMBByGPU: map[int]int{
					0: 16500,
					1: 18000,
				},
				SleepL1ResidualMB: 1000, // < min(per-gpu) is fine
			},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected per-GPU-only config to be allowed, got: %v", err)
	}
	// Effective lookups should return the per-GPU values.
	m := cfg.Models["asym"]
	if got := m.EffectiveExpectedVRAMMB(0); got != 16500 {
		t.Errorf("EffectiveExpectedVRAMMB(0) = %d, want 16500", got)
	}
	if got := m.EffectiveExpectedVRAMMB(1); got != 18000 {
		t.Errorf("EffectiveExpectedVRAMMB(1) = %d, want 18000", got)
	}
	if !m.AdmissionEnabled() {
		t.Error("AdmissionEnabled() = false, want true with per-GPU map set")
	}
}

// TestBoundary_Validator_FlatScalar_StillAllowed: legacy flat scalar
// remains valid; EffectiveExpectedVRAMMB returns the scalar for every
// GPU.
func TestBoundary_Validator_FlatScalar_StillAllowed(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"flat": {
				Path:                 "/m/flat",
				GPUs:                 []int{0, 1},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 18000,
			},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("flat scalar should still validate, got: %v", err)
	}
	m := cfg.Models["flat"]
	if got := m.EffectiveExpectedVRAMMB(0); got != 18000 {
		t.Errorf("EffectiveExpectedVRAMMB(0) = %d, want 18000", got)
	}
	if got := m.EffectiveExpectedVRAMMB(1); got != 18000 {
		t.Errorf("EffectiveExpectedVRAMMB(1) = %d, want 18000", got)
	}
}

// TestBoundary_Validator_BothSet_Rejected: setting both the flat scalar
// AND the per-GPU map is ambiguous and rejected.
func TestBoundary_Validator_BothSet_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"both": {
				Path:                 "/m/both",
				GPUs:                 []int{0, 1},
				MinFreeMemMBPerGPU:   intPtr(100),
				ExpectedVRAMMBPerGPU: 18000,
				ExpectedVRAMMBByGPU: map[int]int{
					0: 16500,
					1: 18000,
				},
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected rejection when both flat scalar and per-GPU map set")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected error to mention 'mutually exclusive', got: %v", err)
	}
}

// TestBoundary_Validator_PerGPU_MissingEntry_Rejected: a per-GPU map
// must cover EVERY GPU listed in the model's gpus list.
func TestBoundary_Validator_PerGPU_MissingEntry_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"missing": {
				Path:               "/m/missing",
				GPUs:               []int{0, 1, 2},
				MinFreeMemMBPerGPU: intPtr(100),
				ExpectedVRAMMBByGPU: map[int]int{
					0: 16500,
					1: 18000,
					// 2 missing
				},
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected rejection when per-GPU map missing an entry for a declared GPU")
	}
	if !strings.Contains(err.Error(), "GPU 2") {
		t.Errorf("expected error to cite GPU 2, got: %v", err)
	}
}

// TestBoundary_Validator_PerGPU_ZeroValue_Rejected: every per-GPU value
// must be > 0 (zero is meaningless and would silently disable budget
// tracking on that GPU).
func TestBoundary_Validator_PerGPU_ZeroValue_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"zero": {
				Path:               "/m/zero",
				GPUs:               []int{0, 1},
				MinFreeMemMBPerGPU: intPtr(100),
				ExpectedVRAMMBByGPU: map[int]int{
					0: 16500,
					1: 0,
				},
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected rejection when a per-GPU value is 0")
	}
	if !strings.Contains(err.Error(), "must be > 0") {
		t.Errorf("expected error to demand > 0, got: %v", err)
	}
}

// TestBoundary_Validator_PerGPU_StrayEntry_Rejected: per-GPU map must
// not have entries for GPUs the model doesn't touch.
func TestBoundary_Validator_PerGPU_StrayEntry_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"stray": {
				Path:               "/m/stray",
				GPUs:               []int{0, 1},
				MinFreeMemMBPerGPU: intPtr(100),
				ExpectedVRAMMBByGPU: map[int]int{
					0: 16500,
					1: 18000,
					3: 12000, // not in gpus list
				},
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected rejection when per-GPU map has an entry for a non-declared GPU")
	}
	if !strings.Contains(err.Error(), "GPU 3") {
		t.Errorf("expected error to cite stray GPU 3, got: %v", err)
	}
}

// TestBoundary_Validator_PerGPU_ResidualExceedsMin_Rejected: residual
// must not exceed the SMALLEST per-GPU expected (binding constraint).
func TestBoundary_Validator_PerGPU_ResidualExceedsMin_Rejected(t *testing.T) {
	cfg := &config.Config{
		Server:    config.ServerConfig{Port: 8000},
		VLLM:      config.VLLMConfig{Port: 8001},
		Scheduler: &config.SchedulerConfig{PortRangeStart: 8100, PortRangeEnd: 8200},
		Models: map[string]config.ModelConfig{
			"big-residual": {
				Path:               "/m/big-residual",
				GPUs:               []int{0, 1},
				MinFreeMemMBPerGPU: intPtr(100),
				ExpectedVRAMMBByGPU: map[int]int{
					0: 1000,
					1: 5000,
				},
				SleepL1ResidualMB: 2000, // > min (1000)
			},
		},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected rejection when residual exceeds smallest per-GPU expected")
	}
	if !strings.Contains(err.Error(), "sleep_l1_residual_mb") {
		t.Errorf("expected error to mention sleep_l1_residual_mb, got: %v", err)
	}
}
