package gpu_test

import (
	"context"
	"errors"
	"os/exec"
	"testing"

	"vllm-jukebox/internal/gpu"
)

func TestParseInventoryCSV_ParsesAndSorts(t *testing.T) {
	out, err := gpu.ParseInventoryCSV("1, 100, 90\n0,100,80\n")
	if err != nil {
		t.Fatalf("ParseInventoryCSV: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 gpus, got %d", len(out))
	}
	if out[0].Index != 0 || out[0].TotalMB != 100 || out[0].FreeMB != 80 {
		t.Fatalf("unexpected gpu0: %+v", out[0])
	}
	if out[1].Index != 1 || out[1].TotalMB != 100 || out[1].FreeMB != 90 {
		t.Fatalf("unexpected gpu1: %+v", out[1])
	}
}

func TestParseInventoryCSV_AllowsEmpty(t *testing.T) {
	out, err := gpu.ParseInventoryCSV("")
	if err != nil {
		t.Fatalf("ParseInventoryCSV: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("expected empty, got %v", out)
	}
}

func TestParseInventoryCSV_RejectsMalformed(t *testing.T) {
	cases := []string{
		"0,1\n",
		"nope, 1, 2\n",
		"0, nope, 2\n",
		"0, 1, nope\n",
	}
	for _, tc := range cases {
		_, err := gpu.ParseInventoryCSV(tc)
		if err == nil {
			t.Fatalf("expected error for %q", tc)
		}
	}
}

// TestNvidiaSMIInventory_MissingBinary_ReturnsSentinel — when the
// configured nvidia-smi binary doesn't exist (PATH lookup miss OR
// absolute path that doesn't exist), List() must wrap the underlying
// error so callers can detect "this environment has no nvidia-smi" via
// errors.Is(err, gpu.ErrBinaryNotFound) and degrade gracefully instead
// of crash-looping. Regression test: scheduler-mode tolerance for
// missing nvidia-smi (production blocker on `llm` host, 2026-06-09).
func TestNvidiaSMIInventory_MissingBinary_ReturnsSentinel(t *testing.T) {
	t.Run("absolute path that does not exist", func(t *testing.T) {
		inv := gpu.NvidiaSMIInventory{Binary: "/definitely/not/a/real/path/nvidia-smi"}
		_, err := inv.List(context.Background())
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if !gpu.IsBinaryNotFound(err) {
			t.Fatalf("expected IsBinaryNotFound(err) == true, got err=%v", err)
		}
		if !errors.Is(err, gpu.ErrBinaryNotFound) {
			t.Fatalf("expected errors.Is(err, ErrBinaryNotFound) == true, got err=%v", err)
		}
	})

	t.Run("PATH lookup miss", func(t *testing.T) {
		// A binary name unlikely to exist anywhere on PATH.
		inv := gpu.NvidiaSMIInventory{Binary: "nvidia-smi-does-not-exist-zzzzz"}
		_, err := inv.List(context.Background())
		if err == nil {
			t.Skip("a binary named 'nvidia-smi-does-not-exist-zzzzz' exists on PATH; cannot exercise this case")
		}
		if !gpu.IsBinaryNotFound(err) {
			t.Fatalf("expected IsBinaryNotFound(err) == true, got err=%v", err)
		}
	})
}

func TestIsBinaryNotFound(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"sentinel", gpu.ErrBinaryNotFound, true},
		{"exec.ErrNotFound", exec.ErrNotFound, true},
		{"unrelated", errors.New("some other failure"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := gpu.IsBinaryNotFound(tc.err); got != tc.want {
				t.Fatalf("IsBinaryNotFound(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}
