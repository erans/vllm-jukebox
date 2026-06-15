package config_test

import (
	"os"
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

func TestConfig_GPUPowerLimits(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
gpu_power_limits:
  0: 250
  1: 300
models:
  test:
    path: "/models/test"
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.GPUPowerLimits == nil {
		t.Fatal("expected GPUPowerLimits to be set")
	}
	if cfg.GPUPowerLimits[0] != 250 {
		t.Errorf("expected GPU 0 power limit 250, got %d", cfg.GPUPowerLimits[0])
	}
	if cfg.GPUPowerLimits[1] != 300 {
		t.Errorf("expected GPU 1 power limit 300, got %d", cfg.GPUPowerLimits[1])
	}
}

func TestConfig_ModelPowerLimit(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limit: 300
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Models["test"].PowerLimit == nil {
		t.Fatal("expected PowerLimit to be set")
	}
	if *cfg.Models["test"].PowerLimit != 300 {
		t.Errorf("expected 300, got %d", *cfg.Models["test"].PowerLimit)
	}
}

func TestConfig_ModelPowerLimitsMap(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limits:
      0: 250
      1: 300
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Models["test"].PowerLimits == nil {
		t.Fatal("expected PowerLimits to be set")
	}
	if cfg.Models["test"].PowerLimits[0] != 250 {
		t.Errorf("expected GPU 0 = 250, got %d", cfg.Models["test"].PowerLimits[0])
	}
}

func TestConfig_GPUPowerLimitsAndDefaultMutuallyExclusive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
gpu_power_limits:
  0: 250
default_power_limit: 200
models:
  test:
    path: "/models/test"
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for mutually exclusive fields")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected 'mutually exclusive' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitAndPowerLimitsMutuallyExclusive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limit: 300
    power_limits:
      0: 250
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for mutually exclusive fields")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected 'mutually exclusive' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitsGPUsMustBeSubset(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
scheduler:
  port_range_start: 8100
  port_range_end: 8199
models:
  test:
    path: "/models/test"
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 1000
    power_limits:
      0: 250
      2: 300
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for GPU not in gpus list")
	}
	if !strings.Contains(err.Error(), "not in gpus list") {
		t.Errorf("expected 'not in gpus list' error, got: %v", err)
	}
}

func TestConfig_GPUPowerLimitsMustBePositive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
gpu_power_limits:
  0: 0
models:
  test:
    path: "/models/test"
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for non-positive power limit")
	}
	if !strings.Contains(err.Error(), "must be > 0") {
		t.Errorf("expected 'must be > 0' error, got: %v", err)
	}
}

func TestConfig_DefaultPowerLimitMustBePositive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
default_power_limit: -100
models:
  test:
    path: "/models/test"
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for non-positive power limit")
	}
	if !strings.Contains(err.Error(), "must be > 0") {
		t.Errorf("expected 'must be > 0' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitMustBePositive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limit: 0
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for non-positive power limit")
	}
	if !strings.Contains(err.Error(), "must be > 0") {
		t.Errorf("expected 'must be > 0' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitsMapMustBePositive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limits:
      0: -50
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for non-positive power limit")
	}
	if !strings.Contains(err.Error(), "must be > 0") {
		t.Errorf("expected 'must be > 0' error, got: %v", err)
	}
}

func TestValidate_LogSettings(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "valid log settings",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_dir: /var/log/vllm
  log_max_size_mb: 100
  log_max_files: 10
`,
			wantErr: "",
		},
		{
			name: "negative log_max_size_mb",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_max_size_mb: -1
`,
			wantErr: "log_max_size_mb must be >= 0",
		},
		{
			name: "negative log_max_files",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_max_files: -1
`,
			wantErr: "log_max_files must be >= 0",
		},
		{
			name: "per-model log_file",
			yaml: `
models:
  test:
    path: /models/test
    log_file: /var/log/test.log
`,
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := config.Load([]byte(tt.yaml))
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected error containing %q", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
			}
		})
	}
}

func TestLoad_ModelRuntimeDefaultsToVLLM(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := cfg.Models["m"].Runtime; got != "" && got != "vllm" {
		t.Fatalf("expected runtime to default to \"\" or \"vllm\", got %q", got)
	}
}

func TestLoad_LlamaCppBlockParses(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "/opt/llama.cpp/llama-server"
  default_args: ["--jinja"]
models:
  m:
    runtime: llama_cpp
    path: "unsloth/foo-GGUF:Q4_K_M"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.LlamaCpp == nil {
		t.Fatalf("expected llama_cpp block to parse, got nil")
	}
	if cfg.LlamaCpp.Binary != "/opt/llama.cpp/llama-server" {
		t.Fatalf("binary: %q", cfg.LlamaCpp.Binary)
	}
	if got := cfg.Models["m"].Runtime; got != "llama_cpp" {
		t.Fatalf("expected runtime=llama_cpp, got %q", got)
	}
	if len(cfg.LlamaCpp.DefaultArgs) != 1 || cfg.LlamaCpp.DefaultArgs[0] != "--jinja" {
		t.Fatalf("default_args: %v", cfg.LlamaCpp.DefaultArgs)
	}
}

func TestValidate_UnknownRuntimeRejected(t *testing.T) {
	_, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    runtime: "tgi"
    path: "/models/m"
`))
	if err == nil {
		t.Fatalf("expected unknown runtime to be rejected")
	}
	if !strings.Contains(err.Error(), "runtime") {
		t.Fatalf("expected error to mention runtime, got: %v", err)
	}
}

func TestValidate_LlamaCppRequiresBinary(t *testing.T) {
	_, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
`))
	if err == nil {
		t.Fatalf("expected missing llama_cpp.binary to be rejected")
	}
	if !strings.Contains(err.Error(), "llama_cpp.binary") {
		t.Fatalf("expected error to mention llama_cpp.binary, got: %v", err)
	}
}

func TestValidate_LlamaCppOK(t *testing.T) {
	if _, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
`)); err != nil {
		t.Fatalf("expected llama_cpp config with binary to load, got: %v", err)
	}
}

func TestValidate_LlamaCppRejectedInSchedulerMode(t *testing.T) {
	_, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
scheduler:
  port_range_start: 9000
  port_range_end: 9009
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
`))
	if err == nil {
		t.Fatalf("expected llama_cpp under scheduler mode to be rejected")
	}
	if !strings.Contains(err.Error(), "scheduler") {
		t.Fatalf("expected error to mention scheduler, got: %v", err)
	}
}

func TestValidate_LlamaCppSchedulerErrorWinsOverMissingGPUs(t *testing.T) {
	_, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
scheduler:
  port_range_start: 9000
  port_range_end: 9009
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
`))
	if err == nil {
		t.Fatalf("expected validation error, got nil")
	}
	if !strings.Contains(err.Error(), "scheduler") {
		t.Fatalf("expected scheduler-mode error, got: %v", err)
	}
	if strings.Contains(err.Error(), "requires 'gpus'") {
		t.Fatalf("scheduler missing-gpus error should not surface first, got: %v", err)
	}
}

func TestLoad_QwenLlamaCppExampleParses(t *testing.T) {
	data, err := os.ReadFile("../../configs/qwen36-35b-a3b-llamacpp.yaml")
	if err != nil {
		t.Skipf("example config not readable: %v", err)
	}
	if _, err := config.Load(data); err != nil {
		t.Fatalf("expected example to parse, got: %v", err)
	}
}

func TestModelConfig_IsGenerative(t *testing.T) {
	cases := map[string]bool{
		"":              true, // default = generative (backward compatible)
		"generate":      true,
		"GENERATE":      true, // case-insensitive
		" generate ":    true, // trimmed
		"transcription": true,
		"embed":         false,
		"embedding":     false,
		"rerank":        false,
		"reranker":      false,
		"classify":      false,
		"score":         false,
		"reward":        false,
		"pooling":       false,
	}
	for task, want := range cases {
		got := config.ModelConfig{Task: task}.IsGenerative()
		if got != want {
			t.Fatalf("IsGenerative(task=%q)=%v, want %v", task, got, want)
		}
	}
}

func TestLoad_VerifyForwardPassDefaults(t *testing.T) {
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
	if cfg.VLLM.VerifyForwardPass == nil || !*cfg.VLLM.VerifyForwardPass {
		t.Fatalf("VerifyForwardPass should default to true")
	}
	if cfg.VLLM.VerifyForwardPassTimeout.Duration != 30*time.Second {
		t.Fatalf("VerifyForwardPassTimeout should default to 30s, got %v", cfg.VLLM.VerifyForwardPassTimeout.Duration)
	}
}
