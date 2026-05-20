package runtime_test

import (
	"reflect"
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/runtime"
	"vllm-jukebox/internal/vllm"
)

func TestVLLMRuntime_BuildArgsMatchesLegacy(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
  defaults:
    gpu_memory_utilization: 0.9
    dtype: auto
models:
  m:
    path: "/models/m"
    tensor_parallel_size: 4
    max_model_len: 8192
    extra_args: ["--enforce-eager"]
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	rt, err := runtime.For(cfg.Models["m"].Runtime)
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	if rt.Name() != "vllm" {
		t.Fatalf("expected vllm, got %q", rt.Name())
	}
	if got := rt.Binary(cfg); got != "vllm" {
		t.Fatalf("Binary: %q", got)
	}

	legacy, err := vllm.BuildServeArgsForPort(cfg, "m", 8000)
	if err != nil {
		t.Fatalf("legacy: %v", err)
	}
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	if !reflect.DeepEqual(legacy, got) {
		t.Fatalf("vllm runtime BuildArgs diverges from legacy.\n  legacy: %v\n     got: %v", legacy, got)
	}
}

func TestRuntimeFor_UnknownName(t *testing.T) {
	_, err := runtime.For("tgi")
	if err == nil {
		t.Fatalf("expected unknown runtime to error")
	}
	if !strings.Contains(err.Error(), "tgi") {
		t.Fatalf("expected error to mention %q, got: %v", "tgi", err)
	}
}
