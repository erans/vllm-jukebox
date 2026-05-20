package runtime_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/runtime"
)

func loadLlamaCfg(t *testing.T, body string) *config.Config {
	t.Helper()
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return cfg
}

func TestLlamaCppRuntime_BuildArgs_HFRepoPath(t *testing.T) {
	cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
    max_model_len: 65536
    extra_args: ["--jinja", "--n-gpu-layers", "all"]
`)
	rt, _ := runtime.For("llama_cpp")
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	want := []string{
		"--host", "127.0.0.1",
		"--port", "8000",
		"--hf-repo", "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
		"--ctx-size", "65536",
		"--jinja", "--n-gpu-layers", "all",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args mismatch\n  got: %v\n want: %v", got, want)
	}
}

func TestLlamaCppRuntime_BuildArgs_LocalGGUFPath(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(gguf, []byte("x"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "`+gguf+`"
`)
	rt, _ := runtime.For("llama_cpp")
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	want := []string{"--host", "127.0.0.1", "--port", "8000", "-m", gguf}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args mismatch\n  got: %v\n want: %v", got, want)
	}
}

func TestLlamaCppRuntime_BuildArgs_DotGGUFSuffixUsesLocal(t *testing.T) {
	// File doesn't exist on disk, but `.gguf` suffix should still pick `-m`.
	cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "model-q4.gguf"
`)
	rt, _ := runtime.For("llama_cpp")
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	if got[len(got)-2] != "-m" || got[len(got)-1] != "model-q4.gguf" {
		t.Fatalf("expected `-m model-q4.gguf` tail, got: %v", got)
	}
}

func TestLlamaCppRuntime_BuildArgs_DefaultArgsPrependExtraArgs(t *testing.T) {
	cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
  default_args: ["--metrics", "--no-webui"]
models:
  m:
    runtime: llama_cpp
    path: "org/repo-GGUF"
    extra_args: ["--jinja"]
`)
	rt, _ := runtime.For("llama_cpp")
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	// default_args must appear before extra_args, both after path flags.
	wantTail := []string{"--metrics", "--no-webui", "--jinja"}
	if !reflect.DeepEqual(got[len(got)-len(wantTail):], wantTail) {
		t.Fatalf("expected tail %v, got %v (full: %v)", wantTail, got[len(got)-len(wantTail):], got)
	}
}

func TestLlamaCppRuntime_BuildArgs_IgnoresVLLMOnlyFields(t *testing.T) {
	tp := 4
	gmu := 0.9
	cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "org/repo-GGUF"
    dtype: "auto"
    quantization: "awq"
`)
	// Mutate in-memory to set the pointer fields too (yaml doesn't carry them above).
	mc := cfg.Models["m"]
	mc.TensorParallelSize = &tp
	mc.GPUMemoryUtilization = &gmu
	cfg.Models["m"] = mc

	rt, _ := runtime.For("llama_cpp")
	got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
	if err != nil {
		t.Fatalf("BuildArgs: %v", err)
	}
	for _, banned := range []string{"--tensor-parallel-size", "--gpu-memory-utilization", "--dtype", "--quantization"} {
		for _, a := range got {
			if a == banned {
				t.Fatalf("vLLM-only flag %q leaked into llama.cpp args: %v", banned, got)
			}
		}
	}
}
