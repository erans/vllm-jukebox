package config_test

import (
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
)

// TestEvictAction_ValidatorRejectsStopOnPinned: evict_action: stop is
// incompatible with pinned: true. The operator must drop pinned first
// before opting into stop-on-evict for the daily-driver.
func TestEvictAction_ValidatorRejectsStopOnPinned(t *testing.T) {
	_, err := config.Load([]byte(`
scheduler:
  port_range_start: 8100
  port_range_end: 8200
vllm:
  port: 8000
models:
  m:
    pinned: true
    swap_group: g
    evict_action: stop
    lifecycle: external
    host: vllm-m
    port: 9000
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    expected_vram_mb_per_gpu: 8000
`))
	if err == nil {
		t.Fatal("expected validation error for pinned + evict_action: stop, got nil")
	}
	if !strings.Contains(err.Error(), "pinned") {
		t.Errorf("expected error to mention pinned conflict, got: %v", err)
	}
}

// TestEvictAction_ValidatorRejectsStopOnNonExternalLifecycle: managed
// lifecycle (the default) is incompatible with evict_action: stop —
// stopping a jukebox-managed process makes no sense; jukebox would
// re-spawn it. Must be lifecycle: external.
func TestEvictAction_ValidatorRejectsStopOnNonExternalLifecycle(t *testing.T) {
	_, err := config.Load([]byte(`
scheduler:
  port_range_start: 8100
  port_range_end: 8200
vllm:
  port: 8000
models:
  m:
    swap_group: g
    evict_action: stop
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    expected_vram_mb_per_gpu: 8000
`))
	if err == nil {
		t.Fatal("expected validation error for managed lifecycle + evict_action: stop, got nil")
	}
	if !strings.Contains(err.Error(), "lifecycle") {
		t.Errorf("expected error to mention lifecycle conflict, got: %v", err)
	}
}

// TestEvictAction_ValidatorRejectsStopWithoutSwapGroup: evict_action:
// stop requires swap_group membership — admission only triggers eviction
// within a group, so stop-on-evict outside any group is dead config.
func TestEvictAction_ValidatorRejectsStopWithoutSwapGroup(t *testing.T) {
	_, err := config.Load([]byte(`
scheduler:
  port_range_start: 8100
  port_range_end: 8200
vllm:
  port: 8000
models:
  m:
    evict_action: stop
    lifecycle: external
    host: vllm-m
    port: 9000
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    expected_vram_mb_per_gpu: 8000
`))
	if err == nil {
		t.Fatal("expected validation error for evict_action: stop without swap_group, got nil")
	}
	if !strings.Contains(err.Error(), "swap_group") {
		t.Errorf("expected error to mention swap_group requirement, got: %v", err)
	}
}
