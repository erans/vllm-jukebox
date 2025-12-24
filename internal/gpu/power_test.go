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

func TestPowerManager_ApplyModelLimits_SingleValue(t *testing.T) {
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-model.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-model.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-model.txt")
	defer os.Remove("/tmp/nvidia-smi-model.txt")

	pm := gpu.NewPowerManager(scriptPath, nil, false)

	gpus := []int{0, 1}
	powerLimit := 300
	err := pm.ApplyModelLimits(context.Background(), gpus, &powerLimit, nil)
	if err != nil {
		t.Fatalf("ApplyModelLimits: %v", err)
	}

	data, _ := os.ReadFile("/tmp/nvidia-smi-model.txt")
	output := string(data)
	if !strings.Contains(output, "-i 0 -pl 300") {
		t.Errorf("expected GPU 0 at 300W, got: %s", output)
	}
	if !strings.Contains(output, "-i 1 -pl 300") {
		t.Errorf("expected GPU 1 at 300W, got: %s", output)
	}
}

func TestPowerManager_ApplyModelLimits_PerGPU(t *testing.T) {
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-pergpu.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-pergpu.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-pergpu.txt")
	defer os.Remove("/tmp/nvidia-smi-pergpu.txt")

	pm := gpu.NewPowerManager(scriptPath, nil, false)

	gpus := []int{0, 1}
	powerLimits := map[int]int{0: 250, 1: 350}
	err := pm.ApplyModelLimits(context.Background(), gpus, nil, powerLimits)
	if err != nil {
		t.Fatalf("ApplyModelLimits: %v", err)
	}

	data, _ := os.ReadFile("/tmp/nvidia-smi-pergpu.txt")
	output := string(data)
	if !strings.Contains(output, "-i 0 -pl 250") {
		t.Errorf("expected GPU 0 at 250W, got: %s", output)
	}
	if !strings.Contains(output, "-i 1 -pl 350") {
		t.Errorf("expected GPU 1 at 350W, got: %s", output)
	}
}

func TestPowerManager_RevertModelLimits(t *testing.T) {
	script := `#!/bin/bash
echo "$@" >> /tmp/nvidia-smi-revert-model.txt
exit 0
`
	scriptPath := "/tmp/fake-nvidia-smi-revert-model.sh"
	if err := os.WriteFile(scriptPath, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write script: %v", err)
	}
	defer os.Remove(scriptPath)
	os.Remove("/tmp/nvidia-smi-revert-model.txt")
	defer os.Remove("/tmp/nvidia-smi-revert-model.txt")

	defaults := map[int]int{0: 200, 1: 200}
	pm := gpu.NewPowerManager(scriptPath, defaults, false)

	// First apply model limits to trigger override tracking
	powerLimit := 300
	_ = pm.ApplyModelLimits(context.Background(), []int{0, 1}, &powerLimit, nil)
	os.Remove("/tmp/nvidia-smi-revert-model.txt") // Clear apply calls

	// Now revert
	gpus := []int{0, 1}
	err := pm.RevertModelLimits(context.Background(), gpus)
	if err != nil {
		t.Fatalf("RevertModelLimits: %v", err)
	}

	data, _ := os.ReadFile("/tmp/nvidia-smi-revert-model.txt")
	output := string(data)
	if !strings.Contains(output, "-i 0 -pl 200") {
		t.Errorf("expected GPU 0 revert to 200W, got: %s", output)
	}
	if !strings.Contains(output, "-i 1 -pl 200") {
		t.Errorf("expected GPU 1 revert to 200W, got: %s", output)
	}
}
