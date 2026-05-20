package runtime_test

import (
	"reflect"
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/runtime"
)

func TestVLLMRuntime_BuildArgs_FullShape(t *testing.T) {
	// Set every leg that BuildServeArgsForPort touches to a non-default value
	// so any future refactor that drops or reorders a flag will fail this test.
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
  defaults:
    gpu_memory_utilization: 0.5
    dtype: float16
    max_model_len: 4096
models:
  m:
    path: "/models/m"
    tensor_parallel_size: 4
    pipeline_parallel_size: 2
    max_model_len: 32768
    gpu_memory_utilization: 0.9
    dtype: bfloat16
    quantization: awq
    extra_args:
      - "--enforce-eager"
      - "--seed"
      - "42"
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

	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}

	want := []string{
		"serve", "/models/m",
		"--host", "127.0.0.1",
		"--port", "8000",
		"--tensor-parallel-size", "4",
		"--pipeline-parallel-size", "2",
		"--gpu-memory-utilization", "0.9",
		"--max-model-len", "32768",
		"--dtype", "bfloat16",
		"--quantization", "awq",
		"--enforce-eager", "--seed", "42",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("vllm runtime args don't match expected shape\n  got: %v\n want: %v", got, want)
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
