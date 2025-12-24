package gpu_test

import (
	"context"
	"os"
	"strings"
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

func TestPowerManager_RevertToDefault(t *testing.T) {
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-revert.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-revert.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-revert.txt")
	defer os.Remove("/tmp/nvidia-smi-revert.txt")

	defaults := map[int]int{0: 200}
	pm := gpu.NewPowerManager(scriptPath, defaults, false)

	// First set a different limit
	_ = pm.SetLimit(context.Background(), 0, 300)
	os.Remove("/tmp/nvidia-smi-revert.txt") // Clear the set call

	// Now revert
	err := pm.RevertToDefault(context.Background(), 0)
	if err != nil {
		t.Fatalf("RevertToDefault: %v", err)
	}

	data, err := os.ReadFile("/tmp/nvidia-smi-revert.txt")
	if err != nil {
		t.Fatalf("failed to read calls: %v", err)
	}
	expected := "-i 0 -pl 200\n"
	if string(data) != expected {
		t.Errorf("expected %q, got %q", expected, string(data))
	}
}

func TestPowerManager_RevertToDefault_NoDefault(t *testing.T) {
	pm := gpu.NewPowerManager("nvidia-smi", nil, false)
	// Should not error if no default exists - just no-op
	err := pm.RevertToDefault(context.Background(), 0)
	if err != nil {
		t.Fatalf("RevertToDefault with no default should not error: %v", err)
	}
}

func TestPowerManager_ApplyStartupLimits(t *testing.T) {
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-startup.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-startup.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-startup.txt")
	defer os.Remove("/tmp/nvidia-smi-startup.txt")

	defaults := map[int]int{0: 200, 1: 250}
	pm := gpu.NewPowerManager(scriptPath, defaults, false)

	err := pm.ApplyStartupLimits(context.Background())
	if err != nil {
		t.Fatalf("ApplyStartupLimits: %v", err)
	}

	data, err := os.ReadFile("/tmp/nvidia-smi-startup.txt")
	if err != nil {
		t.Fatalf("failed to read calls: %v", err)
	}
	// Order might vary, so check both are present
	output := string(data)
	if !strings.Contains(output, "-i 0 -pl 200") {
		t.Errorf("expected GPU 0 call, got: %s", output)
	}
	if !strings.Contains(output, "-i 1 -pl 250") {
		t.Errorf("expected GPU 1 call, got: %s", output)
	}
}

func TestPowerManager_ApplyStartupLimits_RequiredFailure(t *testing.T) {
	script := `#!/bin/bash
exit 1
`
	scriptPath := "/tmp/fake-nvidia-smi-fail.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)

	defaults := map[int]int{0: 200}
	pm := gpu.NewPowerManager(scriptPath, defaults, true) // required=true

	err := pm.ApplyStartupLimits(context.Background())
	if err == nil {
		t.Fatal("expected error when required=true and command fails")
	}
}

func TestPowerManager_ApplyStartupLimits_WarnOnFailure(t *testing.T) {
	script := `#!/bin/bash
exit 1
`
	scriptPath := "/tmp/fake-nvidia-smi-warn.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)

	defaults := map[int]int{0: 200}
	pm := gpu.NewPowerManager(scriptPath, defaults, false) // required=false

	err := pm.ApplyStartupLimits(context.Background())
	if err != nil {
		t.Fatalf("expected no error when required=false, got: %v", err)
	}
}
