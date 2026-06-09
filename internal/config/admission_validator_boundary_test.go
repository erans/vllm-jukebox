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
