package vllmcli_test

import (
	"testing"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/vllmcli"
)

func TestBuildServeArgsForPort_RunnerPoolingAppendsTaskEmbed(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  bge:
    path: "BAAI/bge-m3"
    runner: pooling
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	args, err := vllmcli.BuildServeArgsForPort(cfg, "bge", 8001)
	if err != nil {
		t.Fatalf("BuildServeArgsForPort: %v", err)
	}

	if !containsSubsequence(args, []string{"--task", "embed"}) {
		t.Fatalf("expected --task embed in args, got: %v", args)
	}
}

func TestBuildServeArgsForPort_RunnerGenerateOmitsTaskFlag(t *testing.T) {
	for _, runner := range []string{"", "generate"} {
		t.Run("runner="+runner, func(t *testing.T) {
			cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    runner: "` + runner + `"
`))
			if err != nil {
				t.Fatalf("load: %v", err)
			}

			args, err := vllmcli.BuildServeArgsForPort(cfg, "m", 8001)
			if err != nil {
				t.Fatalf("BuildServeArgsForPort: %v", err)
			}

			for i, a := range args {
				if a == "--task" {
					t.Fatalf("did not expect --task flag for runner=%q, got args: %v (at index %d)", runner, args, i)
				}
			}
		})
	}
}

func containsSubsequence(haystack, needle []string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		match := true
		for j, n := range needle {
			if haystack[i+j] != n {
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
