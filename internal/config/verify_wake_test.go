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

// TestEffectiveServedModelName_FallbackWhenUnset proves the post-wake
// probe falls back to the jukebox config key when served_model_name
// is unset (matches the common case where vLLM uses the default
// served name = path basename and the operator wired the config key
// to match).
func TestEffectiveServedModelName_FallbackWhenUnset(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  vllm-main:
    path: "/models/vllm-main"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["vllm-main"]
	if got, want := m.EffectiveServedModelName("vllm-main"), "vllm-main"; got != want {
		t.Errorf("expected fallback %q; got %q", want, got)
	}
}

// TestEffectiveServedModelName_OverrideHonored proves the operator's
// served_model_name takes precedence over the jukebox config key. This
// is the HIGH-3 fix path: vLLM was launched with --served-model-name
// X but jukebox knows the model as Y; the probe MUST send X, not Y.
func TestEffectiveServedModelName_OverrideHonored(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  vllm-main:
    path: "/models/whatever"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    served_model_name: "Qwen/Qwen3-32B-AWQ"
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["vllm-main"]
	if got, want := m.EffectiveServedModelName("vllm-main"), "Qwen/Qwen3-32B-AWQ"; got != want {
		t.Errorf("expected override %q to win over fallback; got %q", want, got)
	}
}

// TestEffectiveServedModelName_WhitespaceTreatedAsUnset proves that
// a whitespace-only served_model_name behaves the same as unset
// (defense against YAML quoting accidents that produce " ").
func TestEffectiveServedModelName_WhitespaceTreatedAsUnset(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  vllm-main:
    path: "/models/x"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    served_model_name: "   "
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["vllm-main"]
	if got, want := m.EffectiveServedModelName("fallback-key"), "fallback-key"; got != want {
		t.Errorf("expected whitespace served_model_name to be treated as unset (fallback %q); got %q", want, got)
	}
}

// TestValidate_ExtraArgsServedModelNameWithoutConfigStillLoads proves
// the validator EMITS A WARNING but does NOT reject when extra_args
// overrides --served-model-name and served_model_name is unset. We
// can't easily assert the log line in a unit test (slog routing is
// global), but we can prove the config loads cleanly — the warning
// is non-fatal so operators can iterate. The negative path (validator
// rejects the bad combo) would be a behavior change that breaks
// existing in-production configs that happen to have this shape and
// just-luck-out by having a matching default served name.
func TestValidate_ExtraArgsServedModelNameWithoutConfigStillLoads(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  vllm-main:
    path: "/models/x"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    extra_args: ["--served-model-name", "Qwen/Qwen3-32B-AWQ"]
`)
	if err != nil {
		t.Fatalf("expected config with --served-model-name + missing served_model_name to LOAD (warn only); got reject: %v", err)
	}
	if _, ok := cfg.Models["vllm-main"]; !ok {
		t.Fatalf("model missing after load")
	}
}

// TestValidate_ExtraArgsServedModelNameEqualsFormLoads covers the
// `--served-model-name=X` (single arg) form of the override.
func TestValidate_ExtraArgsServedModelNameEqualsFormLoads(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  vllm-main:
    path: "/models/x"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    sleep_mode: true
    extra_args: ["--served-model-name=Qwen/Qwen3-32B-AWQ"]
`)
	if err != nil {
		t.Fatalf("expected --served-model-name=X form to LOAD; got reject: %v", err)
	}
	if _, ok := cfg.Models["vllm-main"]; !ok {
		t.Fatalf("model missing after load")
	}
}
