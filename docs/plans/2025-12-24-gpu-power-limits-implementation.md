# GPU Power Limits Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU power limit configuration via nvidia-smi with startup defaults and per-model overrides.

**Architecture:** New `PowerManager` in `internal/gpu/power.go` wraps nvidia-smi calls. Config validation ensures mutual exclusivity. Power limits applied at startup and on model load/unload.

**Tech Stack:** Go, nvidia-smi CLI, existing config/gpu packages

---

## Task 1: Add Config Fields for Power Limits

**Files:**
- Modify: `internal/config/config.go:31-93`
- Test: `internal/config/config_test.go`

**Step 1: Write failing test for gpu_power_limits parsing**

```go
func TestConfig_GPUPowerLimits(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
gpu_power_limits:
  0: 250
  1: 300
models:
  test:
    path: "/models/test"
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.GPUPowerLimits == nil {
		t.Fatal("expected GPUPowerLimits to be set")
	}
	if cfg.GPUPowerLimits[0] != 250 {
		t.Errorf("expected GPU 0 power limit 250, got %d", cfg.GPUPowerLimits[0])
	}
	if cfg.GPUPowerLimits[1] != 300 {
		t.Errorf("expected GPU 1 power limit 300, got %d", cfg.GPUPowerLimits[1])
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/config -run TestConfig_GPUPowerLimits`
Expected: FAIL - field not recognized

**Step 3: Add GPUPowerLimits field to Config struct**

In `internal/config/config.go`, add to `Config` struct (after line 36):

```go
type Config struct {
	Server            ServerConfig           `yaml:"server"`
	VLLM              VLLMConfig             `yaml:"vllm"`
	Behavior          BehaviorConfig         `yaml:"behavior"`
	Scheduler         *SchedulerConfig       `yaml:"scheduler"`
	Models            map[string]ModelConfig `yaml:"models"`
	GPUPowerLimits    map[int]int            `yaml:"gpu_power_limits"`
	DefaultPowerLimit *int                   `yaml:"default_power_limit"`
	PowerLimitRequired bool                  `yaml:"power_limit_required"`
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/config -run TestConfig_GPUPowerLimits`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): add gpu_power_limits, default_power_limit, power_limit_required fields"
```

---

## Task 2: Add Per-Model Power Limit Config Fields

**Files:**
- Modify: `internal/config/config.go:79-93` (ModelConfig struct)
- Test: `internal/config/config_test.go`

**Step 1: Write failing test for model power_limit**

```go
func TestConfig_ModelPowerLimit(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limit: 300
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Models["test"].PowerLimit == nil {
		t.Fatal("expected PowerLimit to be set")
	}
	if *cfg.Models["test"].PowerLimit != 300 {
		t.Errorf("expected 300, got %d", *cfg.Models["test"].PowerLimit)
	}
}

func TestConfig_ModelPowerLimitsMap(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limits:
      0: 250
      1: 300
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Models["test"].PowerLimits == nil {
		t.Fatal("expected PowerLimits to be set")
	}
	if cfg.Models["test"].PowerLimits[0] != 250 {
		t.Errorf("expected GPU 0 = 250, got %d", cfg.Models["test"].PowerLimits[0])
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/config -run TestConfig_ModelPower`
Expected: FAIL - fields not recognized

**Step 3: Add PowerLimit and PowerLimits fields to ModelConfig**

In `internal/config/config.go`, add to `ModelConfig` struct:

```go
type ModelConfig struct {
	Path                 string            `yaml:"path"`
	Alias                string            `yaml:"alias"`
	GPUs                 []int             `yaml:"gpus"`
	MinFreeMemMBPerGPU   *int              `yaml:"min_free_mem_mb_per_gpu"`
	Pinned               *bool             `yaml:"pinned"`
	TensorParallelSize   *int              `yaml:"tensor_parallel_size"`
	PipelineParallelSize *int              `yaml:"pipeline_parallel_size"`
	MaxModelLen          *int              `yaml:"max_model_len"`
	GPUMemoryUtilization *float64          `yaml:"gpu_memory_utilization"`
	DType                string            `yaml:"dtype"`
	Quantization         string            `yaml:"quantization"`
	ExtraArgs            []string          `yaml:"extra_args"`
	Env                  map[string]string `yaml:"env"`
	PowerLimit           *int              `yaml:"power_limit"`
	PowerLimits          map[int]int       `yaml:"power_limits"`
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/config -run TestConfig_ModelPower`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): add power_limit and power_limits fields to ModelConfig"
```

---

## Task 3: Add Validation for Mutual Exclusivity

**Files:**
- Modify: `internal/config/config.go:164-201` (Validate function)
- Test: `internal/config/config_test.go`

**Step 1: Write failing tests for mutual exclusivity**

```go
func TestConfig_GPUPowerLimitsAndDefaultMutuallyExclusive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
gpu_power_limits:
  0: 250
default_power_limit: 200
models:
  test:
    path: "/models/test"
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for mutually exclusive fields")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected 'mutually exclusive' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitAndPowerLimitsMutuallyExclusive(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
models:
  test:
    path: "/models/test"
    power_limit: 300
    power_limits:
      0: 250
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for mutually exclusive fields")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected 'mutually exclusive' error, got: %v", err)
	}
}

func TestConfig_ModelPowerLimitsGPUsMustBeSubset(t *testing.T) {
	yaml := `
server:
  port: 8080
vllm:
  port: 8000
scheduler:
  port_range_start: 8100
  port_range_end: 8199
models:
  test:
    path: "/models/test"
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 1000
    power_limits:
      0: 250
      2: 300
`
	_, err := config.Load([]byte(yaml))
	if err == nil {
		t.Fatal("expected error for GPU not in gpus list")
	}
	if !strings.Contains(err.Error(), "not in gpus list") {
		t.Errorf("expected 'not in gpus list' error, got: %v", err)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test -v ./internal/config -run "TestConfig_.*Exclusive|TestConfig_ModelPowerLimitsGPUs"`
Expected: FAIL - no validation yet

**Step 3: Add validation in Validate function**

In `internal/config/config.go`, add to `Validate()` method after line 177:

```go
func (c *Config) Validate() error {
	// ... existing validations ...

	// Validate power limit mutual exclusivity at config level
	if c.GPUPowerLimits != nil && c.DefaultPowerLimit != nil {
		return fmt.Errorf("gpu_power_limits and default_power_limit are mutually exclusive")
	}

	// Validate power limits are positive
	for gpuID, watts := range c.GPUPowerLimits {
		if watts <= 0 {
			return fmt.Errorf("gpu_power_limits[%d] must be positive, got %d", gpuID, watts)
		}
	}
	if c.DefaultPowerLimit != nil && *c.DefaultPowerLimit <= 0 {
		return fmt.Errorf("default_power_limit must be positive, got %d", *c.DefaultPowerLimit)
	}

	for name, model := range c.Models {
		// ... existing model validations ...

		// Validate model power limit mutual exclusivity
		if model.PowerLimit != nil && model.PowerLimits != nil {
			return fmt.Errorf("model %q: power_limit and power_limits are mutually exclusive", name)
		}

		// Validate model power_limits GPU indices are in gpus list
		if model.PowerLimits != nil && len(model.GPUs) > 0 {
			gpuSet := make(map[int]bool)
			for _, g := range model.GPUs {
				gpuSet[g] = true
			}
			for gpuID := range model.PowerLimits {
				if !gpuSet[gpuID] {
					return fmt.Errorf("model %q: power_limits GPU %d not in gpus list", name, gpuID)
				}
			}
		}

		// Validate power limits are positive
		if model.PowerLimit != nil && *model.PowerLimit <= 0 {
			return fmt.Errorf("model %q: power_limit must be positive", name)
		}
		for gpuID, watts := range model.PowerLimits {
			if watts <= 0 {
				return fmt.Errorf("model %q: power_limits[%d] must be positive", name, gpuID)
			}
		}
	}

	return nil
}
```

**Step 4: Run tests to verify they pass**

Run: `go test -v ./internal/config -run "TestConfig_.*Exclusive|TestConfig_ModelPowerLimitsGPUs"`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): add validation for power limit mutual exclusivity"
```

---

## Task 4: Create PowerManager Struct and Interface

**Files:**
- Create: `internal/gpu/power.go`
- Test: `internal/gpu/power_test.go`

**Step 1: Write failing test for PowerManager creation**

```go
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
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/gpu -run TestNewPowerManager`
Expected: FAIL - NewPowerManager not defined

**Step 3: Create PowerManager struct**

Create `internal/gpu/power.go`:

```go
package gpu

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"sync"
)

type PowerManager struct {
	binary   string
	defaults map[int]int // GPU index -> power limit in watts
	required bool        // fail on error vs warn

	mu       sync.Mutex
	current  map[int]int // GPU index -> current power limit (for revert tracking)
}

func NewPowerManager(binary string, defaults map[int]int, required bool) *PowerManager {
	if binary == "" {
		binary = "nvidia-smi"
	}
	copied := make(map[int]int, len(defaults))
	for k, v := range defaults {
		copied[k] = v
	}
	return &PowerManager{
		binary:   binary,
		defaults: copied,
		required: required,
		current:  make(map[int]int),
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/gpu -run TestNewPowerManager`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/gpu/power.go internal/gpu/power_test.go
git commit -m "feat(gpu): add PowerManager struct"
```

---

## Task 5: Implement SetLimit Method

**Files:**
- Modify: `internal/gpu/power.go`
- Test: `internal/gpu/power_test.go`

**Step 1: Write failing test for SetLimit**

```go
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
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/gpu -run TestPowerManager_SetLimit_CommandFormat`
Expected: FAIL - SetLimit not defined

**Step 3: Implement SetLimit method**

Add to `internal/gpu/power.go`:

```go
func (p *PowerManager) SetLimit(ctx context.Context, gpuIndex, watts int) error {
	cmd := exec.CommandContext(ctx, p.binary,
		"-i", strconv.Itoa(gpuIndex),
		"-pl", strconv.Itoa(watts),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("nvidia-smi -i %d -pl %d failed: %w (output: %s)", gpuIndex, watts, err, string(output))
	}

	p.mu.Lock()
	p.current[gpuIndex] = watts
	p.mu.Unlock()

	return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/gpu -run TestPowerManager_SetLimit_CommandFormat`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/gpu/power.go internal/gpu/power_test.go
git commit -m "feat(gpu): implement PowerManager.SetLimit"
```

---

## Task 6: Implement RevertToDefault Method

**Files:**
- Modify: `internal/gpu/power.go`
- Test: `internal/gpu/power_test.go`

**Step 1: Write failing test for RevertToDefault**

```go
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
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/gpu -run TestPowerManager_RevertToDefault`
Expected: FAIL - RevertToDefault not defined

**Step 3: Implement RevertToDefault method**

Add to `internal/gpu/power.go`:

```go
func (p *PowerManager) RevertToDefault(ctx context.Context, gpuIndex int) error {
	defaultWatts, ok := p.defaults[gpuIndex]
	if !ok {
		// No default configured for this GPU, nothing to revert to
		return nil
	}

	p.mu.Lock()
	currentWatts, hasOverride := p.current[gpuIndex]
	p.mu.Unlock()

	// Only revert if we actually applied an override
	if !hasOverride || currentWatts == defaultWatts {
		return nil
	}

	return p.SetLimit(ctx, gpuIndex, defaultWatts)
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/gpu -run TestPowerManager_RevertToDefault`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/gpu/power.go internal/gpu/power_test.go
git commit -m "feat(gpu): implement PowerManager.RevertToDefault"
```

---

## Task 7: Implement ApplyStartupLimits Method

**Files:**
- Modify: `internal/gpu/power.go`
- Test: `internal/gpu/power_test.go`

**Step 1: Write failing test for ApplyStartupLimits**

```go
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
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/gpu -run TestPowerManager_ApplyStartupLimits`
Expected: FAIL - ApplyStartupLimits not defined

**Step 3: Implement ApplyStartupLimits method**

Add to `internal/gpu/power.go`:

```go
import (
	"log/slog"
	"sort"
)

func (p *PowerManager) ApplyStartupLimits(ctx context.Context) error {
	if len(p.defaults) == 0 {
		return nil
	}

	// Sort GPU indices for deterministic order
	gpuIndices := make([]int, 0, len(p.defaults))
	for idx := range p.defaults {
		gpuIndices = append(gpuIndices, idx)
	}
	sort.Ints(gpuIndices)

	var firstErr error
	for _, gpuIndex := range gpuIndices {
		watts := p.defaults[gpuIndex]
		if err := p.SetLimit(ctx, gpuIndex, watts); err != nil {
			slog.Warn("failed to set GPU power limit",
				"gpu", gpuIndex,
				"watts", watts,
				"err", err,
			)
			if p.required && firstErr == nil {
				firstErr = err
			}
		} else {
			slog.Info("set GPU power limit",
				"gpu", gpuIndex,
				"watts", watts,
			)
		}
	}

	return firstErr
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/gpu -run TestPowerManager_ApplyStartupLimits`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/gpu/power.go internal/gpu/power_test.go
git commit -m "feat(gpu): implement PowerManager.ApplyStartupLimits"
```

---

## Task 8: Add ApplyModelLimits and RevertModelLimits Methods

**Files:**
- Modify: `internal/gpu/power.go`
- Test: `internal/gpu/power_test.go`

**Step 1: Write failing tests**

```go
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
```

**Step 2: Run test to verify it fails**

Run: `go test -v ./internal/gpu -run "TestPowerManager_ApplyModelLimits|TestPowerManager_RevertModelLimits"`
Expected: FAIL - methods not defined

**Step 3: Implement ApplyModelLimits and RevertModelLimits**

Add to `internal/gpu/power.go`:

```go
// ApplyModelLimits sets power limits for the given GPUs based on model config.
// If powerLimit is non-nil, it's applied to all GPUs.
// If powerLimits is non-nil, per-GPU limits are used.
// Returns list of GPUs that were modified (for tracking reverts).
func (p *PowerManager) ApplyModelLimits(ctx context.Context, gpus []int, powerLimit *int, powerLimits map[int]int) error {
	if powerLimit == nil && powerLimits == nil {
		return nil
	}

	var firstErr error
	for _, gpuIndex := range gpus {
		var watts int
		if powerLimits != nil {
			if w, ok := powerLimits[gpuIndex]; ok {
				watts = w
			} else {
				continue // No override for this GPU
			}
		} else if powerLimit != nil {
			watts = *powerLimit
		} else {
			continue
		}

		if err := p.SetLimit(ctx, gpuIndex, watts); err != nil {
			slog.Warn("failed to set model power limit",
				"gpu", gpuIndex,
				"watts", watts,
				"err", err,
			)
			if p.required && firstErr == nil {
				firstErr = err
			}
		} else {
			slog.Info("set model power limit",
				"gpu", gpuIndex,
				"watts", watts,
			)
		}
	}

	return firstErr
}

// RevertModelLimits reverts power limits for the given GPUs to their defaults.
func (p *PowerManager) RevertModelLimits(ctx context.Context, gpus []int) error {
	var firstErr error
	for _, gpuIndex := range gpus {
		if err := p.RevertToDefault(ctx, gpuIndex); err != nil {
			slog.Warn("failed to revert GPU power limit",
				"gpu", gpuIndex,
				"err", err,
			)
			// Always best-effort for reverts, don't fail
		}
	}
	return firstErr
}
```

**Step 4: Run test to verify it passes**

Run: `go test -v ./internal/gpu -run "TestPowerManager_ApplyModelLimits|TestPowerManager_RevertModelLimits"`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/gpu/power.go internal/gpu/power_test.go
git commit -m "feat(gpu): add ApplyModelLimits and RevertModelLimits methods"
```

---

## Task 9: Create PowerManager in main.go

**Files:**
- Modify: `cmd/jukebox/main.go`

**Step 1: Add PowerManager creation and startup limits**

In `cmd/jukebox/main.go`, after config loading (around line 57), add:

```go
// Create PowerManager if power limits are configured
var powerMgr *gpu.PowerManager
if cfg.GPUPowerLimits != nil || cfg.DefaultPowerLimit != nil {
	defaults := cfg.GPUPowerLimits
	if defaults == nil && cfg.DefaultPowerLimit != nil {
		// Build defaults from inventory if using default_power_limit
		// For now, we'll apply to all GPUs we find
		defaults = make(map[int]int)

		nvidiaBinary := "nvidia-smi"
		if cfg.Scheduler != nil && cfg.Scheduler.NvidiaSMIBinary != "" {
			nvidiaBinary = cfg.Scheduler.NvidiaSMIBinary
		}
		inv := gpu.NvidiaSMIInventory{Binary: nvidiaBinary}
		gpus, err := inv.List(context.Background())
		if err != nil {
			slog.Warn("failed to list GPUs for default power limit", "err", err)
		} else {
			for _, g := range gpus {
				defaults[g.Index] = *cfg.DefaultPowerLimit
			}
		}
	}

	nvidiaBinary := "nvidia-smi"
	if cfg.Scheduler != nil && cfg.Scheduler.NvidiaSMIBinary != "" {
		nvidiaBinary = cfg.Scheduler.NvidiaSMIBinary
	}
	powerMgr = gpu.NewPowerManager(nvidiaBinary, defaults, cfg.PowerLimitRequired)

	if err := powerMgr.ApplyStartupLimits(ctx); err != nil {
		slog.Error("failed to apply startup power limits", "err", err)
		os.Exit(1)
	}
}
```

**Step 2: Run build to verify no compile errors**

Run: `go build -o /dev/null ./cmd/jukebox`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add cmd/jukebox/main.go
git commit -m "feat(main): create PowerManager and apply startup power limits"
```

---

## Task 10: Integrate PowerManager with Scheduler

**Files:**
- Modify: `internal/jukebox/scheduler.go`

**Step 1: Add PowerManager field to Scheduler**

In `internal/jukebox/scheduler.go`, add to Scheduler struct and constructor:

```go
type Scheduler struct {
	cfg      *config.Config
	inv      gpu.Inventory
	ports    *ports.Pool
	powerMgr *gpu.PowerManager  // Add this field
	now      func() time.Time
	// ... rest of fields
}

func NewScheduler(cfg *config.Config, inv gpu.Inventory, portPool *ports.Pool, now func() time.Time) *Scheduler {
	return NewSchedulerWithFactory(cfg, inv, portPool, now, nil, nil)
}

func NewSchedulerWithFactory(cfg *config.Config, inv gpu.Inventory, portPool *ports.Pool, now func() time.Time, factory InstanceFactory, powerMgr *gpu.PowerManager) *Scheduler {
	// ... existing code ...
	s := &Scheduler{
		cfg:      cfg,
		inv:      inv,
		ports:    portPool,
		powerMgr: powerMgr,
		// ...
	}
	// ...
}
```

**Step 2: Apply model power limits before starting vLLM**

In `AcquireRoute`, before `mgr.Start()` (around line 249):

```go
// Apply model power limits if configured
if s.powerMgr != nil {
	if err := s.powerMgr.ApplyModelLimits(ctx, modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
		s.ports.Release(port)
		return Route{}, err
	}
}

startCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
_, startErr := mgr.Start(startCtx, resolvedName)
cancel()
```

**Step 3: Revert power limits on instance stop**

In `drainAndStopInstance`, after stopping the manager (around line 551):

```go
_ = inst.mgr.Stop(stopCtx)
cancel()

// Revert power limits to defaults
if s.powerMgr != nil {
	_ = s.powerMgr.RevertModelLimits(context.Background(), inst.gpus)
}
```

**Step 4: Run build to verify no compile errors**

Run: `go build -o /dev/null ./cmd/jukebox`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add internal/jukebox/scheduler.go
git commit -m "feat(scheduler): integrate PowerManager for model load/unload"
```

---

## Task 11: Update main.go to Pass PowerManager to Scheduler

**Files:**
- Modify: `cmd/jukebox/main.go`

**Step 1: Pass PowerManager to Scheduler constructor**

Update the scheduler creation in main.go (around line 115):

```go
sched := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, powerMgr)
```

**Step 2: Run build and tests**

Run: `go build -o /dev/null ./cmd/jukebox && go test ./...`
Expected: Build and tests pass

**Step 3: Commit**

```bash
git add cmd/jukebox/main.go
git commit -m "feat(main): pass PowerManager to Scheduler"
```

---

## Task 12: Integrate PowerManager with Coordinator (Swap Mode)

**Files:**
- Modify: `internal/jukebox/coordinator.go`

**Step 1: Add PowerManager field to Coordinator**

```go
type Coordinator struct {
	cfg      *config.Config
	mgr      Manager
	tr       *inflight.Tracker
	powerMgr *gpu.PowerManager  // Add this field
	now      func() time.Time
	// ... rest of fields
}

func NewCoordinator(cfg *config.Config, mgr Manager, tr *inflight.Tracker, now func() time.Time) *Coordinator {
	return NewCoordinatorWithPower(cfg, mgr, tr, now, nil)
}

func NewCoordinatorWithPower(cfg *config.Config, mgr Manager, tr *inflight.Tracker, now func() time.Time, powerMgr *gpu.PowerManager) *Coordinator {
	if now == nil {
		now = time.Now
	}
	return &Coordinator{
		cfg:      cfg,
		mgr:      mgr,
		tr:       tr,
		powerMgr: powerMgr,
		now:      now,
		requests: make(chan ensureReq),
		state:    StateIdle,
	}
}
```

**Step 2: Apply/revert power limits in doSwap**

In `doSwap`, before starting vLLM and after stopping:

```go
func (c *Coordinator) doSwap(model, requestID string) error {
	_, modelCfg, err := c.cfg.ResolveModel(model)
	if err != nil {
		return err
	}

	// If we already have a running model, stop it after draining in-flight.
	if st := c.Status(); st.State == StateStopping || st.State == StateReady {
		// ... existing drain and stop code ...

		// Revert power limits after stop
		if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
	}

	c.mu.Lock()
	c.state = StateStarting
	c.mu.Unlock()

	// Apply power limits before start
	if c.powerMgr != nil {
		if err := c.powerMgr.ApplyModelLimits(context.Background(), modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			return err
		}
	}

	startCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	// ... rest of start code ...
}
```

**Step 3: Run build to verify no compile errors**

Run: `go build -o /dev/null ./cmd/jukebox`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add internal/jukebox/coordinator.go
git commit -m "feat(coordinator): integrate PowerManager for swap mode"
```

---

## Task 13: Update main.go for Coordinator PowerManager

**Files:**
- Modify: `cmd/jukebox/main.go`

**Step 1: Pass PowerManager to Coordinator constructor**

Update coordinator creation in main.go (around line 76):

```go
coord := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, time.Now, powerMgr)
```

**Step 2: Run full test suite**

Run: `go test -v ./...`
Expected: All tests pass

**Step 3: Commit**

```bash
git add cmd/jukebox/main.go
git commit -m "feat(main): pass PowerManager to Coordinator"
```

---

## Task 14: Add Integration Test

**Files:**
- Create: `internal/gpu/power_integration_test.go`

**Step 1: Write integration test with fake nvidia-smi**

```go
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
```

**Step 2: Run integration test**

Run: `go test -v ./internal/gpu -run TestPowerManager_FullLifecycle`
Expected: PASS

**Step 3: Commit**

```bash
git add internal/gpu/power_integration_test.go
git commit -m "test(gpu): add PowerManager integration test"
```

---

## Task 15: Update Example Configs

**Files:**
- Modify: `configs/example.yaml`
- Modify: `configs/scheduler_example.yaml`

**Step 1: Add power limit examples to configs**

Add to `configs/example.yaml`:

```yaml
# GPU Power Management (optional)
# Option A: Per-GPU power limits at startup
# gpu_power_limits:
#   0: 250
#   1: 250

# Option B: Single default for all GPUs (mutually exclusive with gpu_power_limits)
# default_power_limit: 250

# If true, fail startup when power limit cannot be set (default: false = warn only)
# power_limit_required: false
```

Add to `configs/scheduler_example.yaml`:

```yaml
# GPU Power Management (optional)
# gpu_power_limits:
#   0: 250
#   1: 250
#   2: 250
#   3: 250
#   4: 150

# power_limit_required: false

models:
  small:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    gpus: [0]
    min_free_mem_mb_per_gpu: 4000
    # Per-model power limit override (optional)
    # power_limit: 100

  big:
    path: "/models/Meta-Llama-3-70B-Instruct"
    gpus: [0,1,2,3]
    min_free_mem_mb_per_gpu: 40000
    # Per-GPU power limits for this model (optional)
    # power_limits:
    #   0: 350
    #   1: 300
    #   2: 300
    #   3: 300
```

**Step 2: Commit**

```bash
git add configs/example.yaml configs/scheduler_example.yaml
git commit -m "docs(config): add power limit examples to config files"
```

---

## Task 16: Update CLAUDE.md and Run Final Tests

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add power limits section to CLAUDE.md**

Add under Configuration section:

```markdown
## GPU Power Limits

Power limits can be configured at startup and overridden per-model:

```yaml
# Option A: Per-GPU limits at startup
gpu_power_limits:
  0: 250
  1: 300

# Option B: Single default (mutually exclusive with above)
default_power_limit: 250

# Fail if power limit cannot be set (default: false)
power_limit_required: false

models:
  mymodel:
    power_limit: 300          # Single value for all model GPUs
    # OR
    power_limits:             # Per-GPU within model
      0: 350
      1: 300
```

Power limits are applied via `nvidia-smi -i <gpu> -pl <watts>` and reverted to defaults when models unload.
```

**Step 2: Run full test suite**

Run: `make check`
Expected: All checks pass

**Step 3: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: add GPU power limits documentation to CLAUDE.md"
```

---

Plan complete and saved to `docs/plans/2025-12-24-gpu-power-limits-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
