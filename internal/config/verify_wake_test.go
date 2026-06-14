package config_test

import (
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// TestEffectiveWakeVerifyTimeout_DefaultWhenZero proves that unset (0)
// uses the package default — defends the phantom-wake probe from
// silently disabling when an operator forgets to set the knob.
func TestEffectiveWakeVerifyTimeout_DefaultWhenZero(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["m"]
	if got := m.EffectiveWakeVerifyTimeout(); got != config.DefaultWakeVerifyTimeout {
		t.Errorf("expected default %s; got %s", config.DefaultWakeVerifyTimeout, got)
	}
}

// TestEffectiveWakeVerifyTimeout_ExplicitDisableReturnsZero proves the
// -1 sentinel surfaces as 0 (the convention the wake path uses to skip
// the probe). This is the only way to opt OUT of phantom-wake defense.
func TestEffectiveWakeVerifyTimeout_ExplicitDisableReturnsZero(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    wake_verify_timeout_ms: -1
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["m"]
	if got := m.EffectiveWakeVerifyTimeout(); got != 0 {
		t.Errorf("expected explicit disable -> 0; got %s", got)
	}
}

// TestEffectiveWakeVerifyTimeout_ExplicitValueHonored proves a positive
// value is converted from ms → Duration as expected.
func TestEffectiveWakeVerifyTimeout_ExplicitValueHonored(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    wake_verify_timeout_ms: 3500
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["m"]
	if got, want := m.EffectiveWakeVerifyTimeout(), 3500*time.Millisecond; got != want {
		t.Errorf("expected %s; got %s", want, got)
	}
}

// TestValidate_WakeVerifyTimeoutBelowMinusOneRejected proves the
// validator catches obvious config bugs (anything < -1 makes no sense
// in the {-1=disable, 0=default, positive=ms} encoding).
func TestValidate_WakeVerifyTimeoutBelowMinusOneRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    wake_verify_timeout_ms: -2
`)
	if err == nil {
		t.Fatalf("expected validator to reject wake_verify_timeout_ms=-2")
	}
	if !strings.Contains(err.Error(), "wake_verify_timeout_ms") {
		t.Errorf("expected error to mention wake_verify_timeout_ms; got %v", err)
	}
}
