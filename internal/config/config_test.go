package config_test

import (
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
)

func loadFromYAML(t *testing.T, s string) (*config.Config, error) {
	t.Helper()
	return config.Load([]byte(s))
}

func TestLoad_RejectsUnknownTopLevelField(t *testing.T) {
	_, err := loadFromYAML(t, `
server:
  host: "0.0.0.0"
  port: 8080
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
typo_field: 123
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestLoad_RejectsUnknownModelField(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    typo_field: 123
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_EmptyModelsRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models: {}
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_AliasToMissingModelRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  gpt-4:
    alias: llama-3-70b
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_AliasChainRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  a:
    alias: b
  b:
    alias: c
  c:
    path: "/models/c"
`)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "alias") {
		t.Fatalf("expected alias-related error, got: %v", err)
	}
}

func TestValidate_AliasLoopRejected(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  a:
    alias: b
  b:
    alias: a
`)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "alias") {
		t.Fatalf("expected alias-related error, got: %v", err)
	}
}

func TestValidate_DefaultModelMustExist(t *testing.T) {
	_, err := loadFromYAML(t, `
behavior:
  default_model: nope
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_GPUUtilRange(t *testing.T) {
	_, err := loadFromYAML(t, `
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpu_memory_utilization: 1.1
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

