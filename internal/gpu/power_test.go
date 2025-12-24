package gpu_test

import (
	"testing"

	"vllm-jukebox/internal/gpu"
)

func TestNewPowerManager(t *testing.T) {
	defaults := map[int]int{0: 250, 1: 300}
	pm := gpu.NewPowerManager("nvidia-smi", defaults, false)
	if pm == nil {
		t.Fatal("expected non-nil PowerManager")
	}
}
