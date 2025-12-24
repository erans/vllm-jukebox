package gpu_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"vllm-jukebox/internal/gpu"
)

func TestPowerManager_FullLifecycle(t *testing.T) {
	// Create fake nvidia-smi that logs all calls
	script := `#!/bin/bash
echo "$(date +%s) $@" >> /tmp/nvidia-smi-lifecycle.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-lifecycle.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	logPath := "/tmp/nvidia-smi-lifecycle.txt"
	os.Remove(logPath)
	defer os.Remove(logPath)

	// Startup defaults
	defaults := map[int]int{0: 200, 1: 200, 2: 200, 3: 200}
	pm := gpu.NewPowerManager(scriptPath, defaults, false)

	ctx := context.Background()

	// 1. Apply startup limits
	if err := pm.ApplyStartupLimits(ctx); err != nil {
		t.Fatalf("ApplyStartupLimits: %v", err)
	}

	// 2. Model load with override
	gpus := []int{0, 1, 2, 3}
	powerLimit := 300
	if err := pm.ApplyModelLimits(ctx, gpus, &powerLimit, nil); err != nil {
		t.Fatalf("ApplyModelLimits: %v", err)
	}

	// 3. Model unload - revert to defaults
	if err := pm.RevertModelLimits(ctx, gpus); err != nil {
		t.Fatalf("RevertModelLimits: %v", err)
	}

	// Verify all calls were made
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log: %v", err)
	}
	output := string(data)

	// Should have: 4 startup + 4 model load + 4 revert = 12 calls
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) != 12 {
		t.Errorf("expected 12 nvidia-smi calls, got %d:\n%s", len(lines), output)
	}
}
