package vllm_test

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/vllm"
)

func TestBuildServeArgs_IncludesDefaultsAndOverrides(t *testing.T) {
	gpuDefault := 0.9
	maxLen := 8192
	tp := 4

	cfg := &config.Config{
		VLLM: config.VLLMConfig{
			Port:   8000,
			Binary: "vllm",
			Defaults: config.VLLMDefaults{
				GPUMemoryUtilization: &gpuDefault,
				DType:                "auto",
			},
		},
		Models: map[string]config.ModelConfig{
			"m": {
				Path:               "/models/m",
				TensorParallelSize: &tp,
				MaxModelLen:        &maxLen,
				ExtraArgs:          []string{"--enforce-eager"},
			},
		},
	}

	args, err := vllm.BuildServeArgs(cfg, "m")
	if err != nil {
		t.Fatalf("BuildServeArgs: %v", err)
	}

	wantContains := [][]string{
		{"serve", "/models/m"},
		{"--host", "127.0.0.1"},
		{"--port", "8000"},
		{"--tensor-parallel-size", "4"},
		{"--gpu-memory-utilization", "0.9"},
		{"--max-model-len", "8192"},
		{"--dtype", "auto"},
		{"--enforce-eager"},
	}

	for _, want := range wantContains {
		if !containsSubsequence(args, want) {
			t.Fatalf("expected args to contain %v; args=%v", want, args)
		}
	}
}

func TestBuildEnv_MergesWithPrecedence(t *testing.T) {
	base := []string{"FOO=1", "BAR=old"}
	defaultEnv := map[string]string{"BAR": "default", "BAZ": "default"}
	modelEnv := map[string]string{"BAR": "model"}

	got := vllm.BuildEnv(base, defaultEnv, modelEnv)

	if envGet(got, "FOO") != "1" {
		t.Fatalf("expected FOO=1, got %q", envGet(got, "FOO"))
	}
	if envGet(got, "BAZ") != "default" {
		t.Fatalf("expected BAZ=default, got %q", envGet(got, "BAZ"))
	}
	if envGet(got, "BAR") != "model" {
		t.Fatalf("expected BAR=model, got %q", envGet(got, "BAR"))
	}
}

func containsSubsequence(haystack []string, needle []string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		match := true
		for j := range needle {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func envGet(env []string, key string) string {
	prefix := key + "="
	for _, kv := range env {
		if len(kv) >= len(prefix) && kv[:len(prefix)] == prefix {
			return kv[len(prefix):]
		}
	}
	return ""
}

func TestManager_StopKillsProcessGroup(t *testing.T) {
	tmp := t.TempDir()
	scriptPath := filepath.Join(tmp, "dummy.sh")
	childPIDFile := filepath.Join(tmp, "child.pid")

	if err := os.WriteFile(scriptPath, []byte(`#!/bin/sh
set -e
(sleep 1000) &
echo $! > "$CHILD_PID_FILE"
while true; do sleep 1; done
`), 0o755); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := &config.Config{
		VLLM: config.VLLMConfig{
			Port:   8000,
			Binary: scriptPath,
		},
		Models: map[string]config.ModelConfig{
			"m": {
				Path: "/models/m",
				Env:  map[string]string{"CHILD_PID_FILE": childPIDFile},
			},
		},
	}
	cfgLoaded, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "` + scriptPath + `"
models:
  m:
    path: "/models/m"
    env:
      CHILD_PID_FILE: "` + childPIDFile + `"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	mgr := vllm.NewManager(cfgLoaded)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := mgr.Start(ctx, "m"); err != nil {
		t.Fatalf("start: %v", err)
	}

	var childPID int
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		b, err := os.ReadFile(childPIDFile)
		if err == nil {
			n, err := strconv.Atoi(string(bytesTrimSpace(b)))
			if err != nil {
				t.Fatalf("parse child pid: %v", err)
			}
			childPID = n
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for child pid file")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := syscall.Kill(childPID, 0); err != nil {
		t.Fatalf("expected child alive, got %v", err)
	}

	stopCtx, stopCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer stopCancel()
	if err := mgr.Stop(stopCtx); err != nil {
		t.Fatalf("stop: %v", err)
	}

	// Child should be gone (or at least no longer visible to kill(0)); poll to avoid races/zombies.
	deadline = time.Now().Add(750 * time.Millisecond)
	for {
		err := syscall.Kill(childPID, 0)
		if err != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected child to be terminated")
		}
		time.Sleep(10 * time.Millisecond)
	}

	_ = cfg // silence unused in case we refactor later
}

func bytesTrimSpace(b []byte) []byte {
	i := 0
	j := len(b)
	for i < j && (b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r') {
		i++
	}
	for j > i && (b[j-1] == ' ' || b[j-1] == '\n' || b[j-1] == '\t' || b[j-1] == '\r') {
		j--
	}
	return b[i:j]
}
