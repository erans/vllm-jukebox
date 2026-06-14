package config_test

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

// captureSlogWarn redirects the default slog handler to an in-memory buffer
// for the duration of fn, then returns whatever was written.
func captureSlogWarn(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	fn()
	return buf.String()
}

func TestNVLinkPairs_RejectedWhenPairWrongArity(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2, 4]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0, 3]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil || !strings.Contains(err.Error(), "exactly 2") {
		t.Fatalf("expected pair-arity error, got: %v", err)
	}
}

func TestNVLinkPairs_RejectedWhenPairHasDuplicateGPUs(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 0]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil || !strings.Contains(err.Error(), "distinct") {
		t.Fatalf("expected distinct-GPU error, got: %v", err)
	}
}

func TestNVLinkPairs_RejectedWhenGPUInMultiplePairs(t *testing.T) {
	_, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [3, 1]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 100
`)
	if err == nil || !strings.Contains(err.Error(), "appears in pair") {
		t.Fatalf("expected overlap error, got: %v", err)
	}
}

func TestNVLinkPairs_NoWarningWhenUnconfigured(t *testing.T) {
	out := captureSlogWarn(t, func() {
		// No nvlink_pairs declared. Model gpus could be anything; no warnings expected.
		if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  arbitrary:
    path: "/models/x"
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 100
`); err != nil {
			t.Fatalf("load: %v", err)
		}
	})
	if strings.Contains(out, "NVLink") {
		t.Fatalf("did not expect NVLink warning when pairs unconfigured, got: %s", out)
	}
}

func TestNVLinkPairs_NoWarningWhenTwoGPUModelMatchesPair(t *testing.T) {
	for _, gpus := range []string{"[0, 3]", "[3, 0]"} {
		t.Run("gpus="+gpus, func(t *testing.T) {
			out := captureSlogWarn(t, func() {
				if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: `+gpus+`
    min_free_mem_mb_per_gpu: 100
`); err != nil {
					t.Fatalf("load: %v", err)
				}
			})
			if strings.Contains(out, "NVLink") {
				t.Fatalf("did not expect NVLink warning for matching pair, got: %s", out)
			}
		})
	}
}

func TestNVLinkPairs_WarnsWhenTwoGPUModelDoesNotMatchPair(t *testing.T) {
	out := captureSlogWarn(t, func() {
		if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 100
`); err != nil {
			t.Fatalf("load: %v", err)
		}
	})
	if !strings.Contains(out, "NVLink") {
		t.Fatalf("expected NVLink warning for non-matching pair, got: %s", out)
	}
}

func TestNVLinkPairs_NoWarningWhenFourGPUModelSpansTwoPairs(t *testing.T) {
	for _, gpus := range []string{"[0, 3, 1, 2]", "[1, 2, 0, 3]", "[0, 1, 2, 3]"} {
		t.Run("gpus="+gpus, func(t *testing.T) {
			out := captureSlogWarn(t, func() {
				if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: `+gpus+`
    min_free_mem_mb_per_gpu: 100
`); err != nil {
					t.Fatalf("load: %v", err)
				}
			})
			if strings.Contains(out, "NVLink") {
				t.Fatalf("did not expect warning for clean 2-pair span, got: %s", out)
			}
		})
	}
}

func TestNVLinkPairs_WarnsWhenFourGPUModelDoesNotSpanTwoPairs(t *testing.T) {
	// Only one of the configured pairs ([0,3]) is fully contained; the other two
	// GPUs (1, 4) aren't a configured pair.
	out := captureSlogWarn(t, func() {
		if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0, 3, 1, 4]
    min_free_mem_mb_per_gpu: 100
`); err != nil {
			t.Fatalf("load: %v", err)
		}
	})
	if !strings.Contains(out, "NVLink") {
		t.Fatalf("expected NVLink warning for 4-GPU model that doesn't span two pairs, got: %s", out)
	}
}

func TestNVLinkPairs_NoWarningForSingleGPUModel(t *testing.T) {
	out := captureSlogWarn(t, func() {
		if _, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [1]
    min_free_mem_mb_per_gpu: 100
`); err != nil {
			t.Fatalf("load: %v", err)
		}
	})
	if strings.Contains(out, "NVLink") {
		t.Fatalf("did not expect warning for single-GPU model, got: %s", out)
	}
}

// Sanity check on the exported config field shape.
func TestNVLinkPairs_ParsedIntoStruct(t *testing.T) {
	cfg, err := loadFromYAML(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  nvlink_pairs:
    - [0, 3]
    - [1, 2]
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
    gpus: [0, 3]
    min_free_mem_mb_per_gpu: 100
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Scheduler == nil {
		t.Fatalf("expected scheduler config")
	}
	if len(cfg.Scheduler.NVLinkPairs) != 2 {
		t.Fatalf("expected 2 pairs, got %d", len(cfg.Scheduler.NVLinkPairs))
	}
	wantPairs := [][]int{{0, 3}, {1, 2}}
	for i, want := range wantPairs {
		got := cfg.Scheduler.NVLinkPairs[i]
		if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("pair[%d]: got %v, want %v", i, got, want)
		}
	}
}
