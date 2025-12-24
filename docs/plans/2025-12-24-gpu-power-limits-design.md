# GPU Power Limits Design

## Overview

Add support for limiting GPU power consumption via configuration. Power limits are applied at startup with optional per-model overrides that revert when the model unloads.

## Configuration Schema

```yaml
# Option A: Per-GPU power limits at startup
gpu_power_limits:
  0: 250
  1: 250
  2: 200
  3: 200
  4: 150

# Option B: Single default (mutually exclusive with gpu_power_limits)
# default_power_limit: 250

# Strictness control (default: false = warn and continue)
power_limit_required: false

models:
  devstral:
    path: "cyankiwi/Devstral-..."
    gpus: [0,1,2,3]
    # Simple: one value for all GPUs this model uses
    power_limit: 300

  qwen-small:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    gpus: [4]
    # Granular: per-GPU within model
    power_limits:
      4: 100
```

### Validation Rules

- `gpu_power_limits` and `default_power_limit` are mutually exclusive - config validation fails if both are set
- `power_limit` (int) and `power_limits` (map) on a model are mutually exclusive
- Power limits are in watts (positive integers)
- GPU indices in model `power_limits` must be subset of the model's `gpus` list

## Implementation

### Approach

Use `nvidia-smi -i <gpu_index> -pl <watts>` to set power limits. This is simple, well-tested, and uses tools already in the codebase pattern.

### New Package: `internal/gpu/power.go`

```go
type PowerManager struct {
    binary   string            // "nvidia-smi"
    defaults map[int]int       // GPU index -> startup power limit (watts)
    required bool              // fail on error vs warn
}

func (p *PowerManager) ApplyStartupLimits(ctx context.Context) error
func (p *PowerManager) SetLimit(ctx context.Context, gpuIndex, watts int) error
func (p *PowerManager) RevertToDefault(ctx context.Context, gpuIndex int) error
```

### Integration Points

1. **Startup** (`cmd/jukebox/main.go`): Create `PowerManager`, call `ApplyStartupLimits()` before starting the HTTP server

2. **Model load** (`internal/vllm/runtime.go` or `internal/jukebox/scheduler.go`): Before starting vLLM process, call `SetLimit()` for each GPU if model has power overrides

3. **Model unload** (`internal/jukebox/scheduler.go` for eviction, `internal/jukebox/coordinator.go` for swap): After stopping vLLM, call `RevertToDefault()` for affected GPUs

## Lifecycle

### Startup Sequence

1. Parse config, validate mutual exclusivity rules
2. Build defaults map: either from `gpu_power_limits` (per-GPU) or `default_power_limit` (all GPUs get same value)
3. Call `ApplyStartupLimits()` - iterates defaults map, runs `nvidia-smi -i X -pl Y` for each
4. If any fail and `power_limit_required: true`, exit with error
5. If any fail and `power_limit_required: false`, log warning, continue

### Model Load Sequence

1. Scheduler/Coordinator decides to load model
2. Check if model has `power_limit` or `power_limits`
3. If yes, for each GPU in model's `gpus` list:
   - Determine target watts (from `power_limit` or `power_limits[gpu]`)
   - Call `SetLimit(ctx, gpuIndex, watts)`
   - On failure: respect `power_limit_required` setting
4. Start vLLM process

### Model Unload Sequence

1. Stop vLLM process, wait for exit
2. For each GPU that had a power override applied:
   - Call `RevertToDefault(ctx, gpuIndex)`
   - On failure: always warn (best-effort revert)

### State Tracking

`PowerManager` tracks which GPUs currently have overrides applied, so it knows what to revert.

## Testing Strategy

### Unit Tests (`internal/gpu/power_test.go`)

- Mock `nvidia-smi` execution (similar to existing inventory tests)
- Test `ApplyStartupLimits` with various configs
- Test `SetLimit` / `RevertToDefault` state tracking
- Test error handling with `power_limit_required: true/false`

### Config Validation Tests (`internal/config/config_test.go`)

- Mutual exclusivity: `gpu_power_limits` + `default_power_limit` fails
- Per-model `power_limit` + `power_limits` fails
- GPU indices in `power_limits` must be in model's `gpus` list

### Integration Tests (Smoke Tests)

- Extend `scripts/smoke.sh` with fake nvidia-smi that logs calls
- Verify power limit commands are called in correct order

No real GPU required - all tests use mocked nvidia-smi, matching the existing test patterns.

## Files to Modify/Create

- `internal/config/config.go` - Add new config fields
- `internal/config/config_test.go` - Validation tests
- `internal/gpu/power.go` - New PowerManager implementation
- `internal/gpu/power_test.go` - Unit tests
- `internal/jukebox/scheduler.go` - Integration for scheduler mode
- `internal/jukebox/coordinator.go` - Integration for swap mode
- `cmd/jukebox/main.go` - Startup initialization
