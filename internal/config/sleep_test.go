package config_test

import (
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

func TestSleepMode_ParsesFully(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  bigmodel:
    path: "/models/big"
    gpus: [0, 3]
    min_free_mem_mb_per_gpu: 40000
    sleep_mode: true
    sleep_level: 2
    wake_timeout: 90s
    idle_timeout: 300s
    pinned: false
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["bigmodel"]
	if !m.SleepMode {
		t.Fatalf("expected sleep_mode=true")
	}
	if m.SleepLevel != 2 {
		t.Fatalf("expected sleep_level=2, got %d", m.SleepLevel)
	}
	if m.WakeTimeout.Duration != 90*time.Second {
		t.Fatalf("expected wake_timeout=90s, got %s", m.WakeTimeout.Duration)
	}
	if m.IdleTimeout.Duration != 300*time.Second {
		t.Fatalf("expected idle_timeout=300s, got %s", m.IdleTimeout.Duration)
	}
}

func TestSleepMode_DefaultsApplied(t *testing.T) {
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
	if m.EffectiveSleepLevel() != config.DefaultSleepLevel {
		t.Fatalf("expected effective sleep_level=%d, got %d", config.DefaultSleepLevel, m.EffectiveSleepLevel())
	}
	if m.EffectiveWakeTimeout() != config.DefaultWakeTimeout {
		t.Fatalf("expected effective wake_timeout=%s, got %s", config.DefaultWakeTimeout, m.EffectiveWakeTimeout())
	}
	if m.EffectiveLifecycle() != config.LifecycleManaged {
		t.Fatalf("expected effective lifecycle=%s, got %s", config.LifecycleManaged, m.EffectiveLifecycle())
	}
	if m.IdleTimeout.Duration != 0 {
		t.Fatalf("expected idle_timeout default = 0 (never auto-suspend), got %s", m.IdleTimeout.Duration)
	}
}

func TestSleepMode_ZeroBehaviorWhenUnset(t *testing.T) {
	cfg, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["m"]
	if m.SleepMode {
		t.Fatalf("expected sleep_mode default false")
	}
	if m.Lifecycle != "" {
		t.Fatalf("expected lifecycle default empty, got %q", m.Lifecycle)
	}
	if m.EffectiveLifecycle() != config.LifecycleManaged {
		t.Fatalf("expected effective lifecycle to fall back to managed")
	}
}

func TestSleepMode_RejectedWithLlamaCpp(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
    sleep_mode: true
`)
	if err == nil {
		t.Fatalf("expected sleep_mode + llama_cpp to be rejected")
	}
	if !strings.Contains(err.Error(), "sleep_mode") || !strings.Contains(err.Error(), "llama_cpp") {
		t.Fatalf("expected error to mention sleep_mode + llama_cpp, got: %v", err)
	}
}

func TestSleepMode_InvalidSleepLevelRejected(t *testing.T) {
	for _, level := range []int{-1, 3, 99} {
		t.Run("level="+itoa(level), func(t *testing.T) {
			_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    sleep_mode: true
    sleep_level: `+itoa(level)+`
`)
			if err == nil {
				t.Fatalf("expected invalid sleep_level=%d to be rejected", level)
			}
			if !strings.Contains(err.Error(), "sleep_level") {
				t.Fatalf("expected error to mention sleep_level, got: %v", err)
			}
		})
	}
}

func TestLifecycle_ExternalParsesFully(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  emb:
    lifecycle: external
    host: "vllm-embeddings"
    port: 8003
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
    sleep_mode: true
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m := cfg.Models["emb"]
	if m.EffectiveLifecycle() != config.LifecycleExternal {
		t.Fatalf("expected external lifecycle")
	}
	if m.Host != "vllm-embeddings" {
		t.Fatalf("expected host vllm-embeddings, got %q", m.Host)
	}
	if m.Port != 8003 {
		t.Fatalf("expected port 8003, got %d", m.Port)
	}
}

func TestLifecycle_ExternalDefaultsHostTo127001(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  emb:
    lifecycle: external
    port: 8003
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Models["emb"].Host != "127.0.0.1" {
		t.Fatalf("expected host default 127.0.0.1, got %q", cfg.Models["emb"].Host)
	}
}

func TestLifecycle_ExternalMissingPortRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  emb:
    lifecycle: external
    host: "vllm-embeddings"
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
`)
	if err == nil || !strings.Contains(err.Error(), "port") {
		t.Fatalf("expected missing-port error, got: %v", err)
	}
}

func TestLifecycle_ExternalWithLlamaCppRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    lifecycle: external
    host: "x"
    port: 9000
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil || !strings.Contains(err.Error(), "external") {
		t.Fatalf("expected external+llama_cpp rejection, got: %v", err)
	}
}

func TestLifecycle_ExternalRequiresSchedulerMode(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    lifecycle: external
    host: "x"
    port: 9000
`)
	if err == nil || !strings.Contains(err.Error(), "scheduler") {
		t.Fatalf("expected scheduler-mode-required error, got: %v", err)
	}
}

func TestLifecycle_UnknownValueRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    lifecycle: hybrid
`)
	if err == nil || !strings.Contains(err.Error(), "lifecycle") {
		t.Fatalf("expected unknown-lifecycle error, got: %v", err)
	}
}

func TestLifecycle_ExternalWithAliasRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  real:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
  fake:
    alias: real
    lifecycle: external
    host: "x"
    port: 9000
`)
	if err == nil || !strings.Contains(err.Error(), "alias") {
		t.Fatalf("expected external+alias rejection, got: %v", err)
	}
}

func TestLifecycle_ExternalOmitsPathRequirement(t *testing.T) {
	// External lifecycle doesn't need 'path' (the path lives on the external
	// service's own command line).
	if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  emb:
    lifecycle: external
    host: "x"
    port: 9000
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
`); err != nil {
		t.Fatalf("expected external without path to parse, got: %v", err)
	}
}

func itoa(n int) string {
	// inline strconv to avoid importing it for a single call
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
