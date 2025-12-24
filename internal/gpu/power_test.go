package gpu_test

import (
	"context"
	"os"
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

func TestPowerManager_SetLimit_CommandFormat(t *testing.T) {
	// Create a script that logs the command it receives
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-calls.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-power.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-calls.txt")
	defer os.Remove("/tmp/nvidia-smi-calls.txt")

	pm := gpu.NewPowerManager(scriptPath, nil, false)
	err := pm.SetLimit(context.Background(), 0, 250)
	if err != nil {
		t.Fatalf("SetLimit: %v", err)
	}

	data, err := os.ReadFile("/tmp/nvidia-smi-calls.txt")
	if err != nil {
		t.Fatalf("failed to read calls: %v", err)
	}
	expected := "-i 0 -pl 250\n"
	if string(data) != expected {
		t.Errorf("expected %q, got %q", expected, string(data))
	}
}
