package config_test

import (
	"strings"
	"testing"
	"time"

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

func TestLoad_DefaultsVLLMBinaryToUVX(t *testing.T) {
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
	if cfg.VLLM.Binary != "uvx" {
		t.Fatalf("expected vllm.binary default uvx, got %q", cfg.VLLM.Binary)
	}
}

func TestLoad_Scheduler_AllowsNull(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler: null
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
}

func TestValidate_Scheduler_PortRangeRequiredWhenSchedulerEnabled(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler: {}
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_Scheduler_PortRangeInvalid(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8200
  port_range_end: 8100
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_Scheduler_MaxInstancesMustFitPortRange(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8101
  max_instances: 3
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_Scheduler_ModelsRequireGPUsAndMinFreeWhenSchedulerEnabled(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8101
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

func TestValidate_Scheduler_RejectsCUDAVisibleDevicesInModelEnv(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8101
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
    env:
      CUDA_VISIBLE_DEVICES: "0"
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_Scheduler_RejectsCUDAVisibleDevicesInDefaultEnv(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8101
vllm:
  port: 8000
  default_env:
    CUDA_VISIBLE_DEVICES: "0"
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestValidate_Scheduler_ModelGPUsMustBeUniqueAndNonNegative(t *testing.T) {
	cases := []string{
		`
scheduler:
  port_range_start: 8100
  port_range_end: 8101
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0,0]
    min_free_mem_mb_per_gpu: 100
`,
		`
scheduler:
  port_range_start: 8100
  port_range_end: 8101
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [-1]
    min_free_mem_mb_per_gpu: 100
`,
	}

	for _, tc := range cases {
		_, err := loadFromYAML(t, tc)
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestLoad_Scheduler_MinInstanceUptimeNullDefaultsTo30s(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8101
  min_instance_uptime: null
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if cfg.Scheduler == nil || cfg.Scheduler.MinInstanceUptime == nil {
		t.Fatalf("expected scheduler.min_instance_uptime to be defaulted")
	}
	if cfg.Scheduler.MinInstanceUptime.Duration != 30*time.Second {
		t.Fatalf("expected default 30s, got %s", cfg.Scheduler.MinInstanceUptime.Duration)
	}
}
