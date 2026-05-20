# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/claude-code) when working with this codebase.

## Project Overview

vLLM Jukebox is an OpenAI-compatible HTTP server written in Go that orchestrates LLM inference backends. The default backend is vLLM; `llama-server` (llama.cpp) is supported in swap mode as an alternative per-model runtime. It operates in two modes:

- **Swap mode**: Single vLLM instance that swaps models on-demand based on incoming requests
- **Scheduler mode**: Multiple concurrent vLLM instances (one per GPU set) with LRU eviction for non-pinned models

## Build and Test Commands

```bash
# Build
make build                    # Build binary to bin/jukebox
go build -o bin/jukebox ./cmd/jukebox  # Direct build

# Test
make test                     # Run all tests with verbose output
make test-short               # Run tests without verbose
make test-cover               # Generate coverage report

# Code quality
make fmt                      # Format code
make vet                      # Run go vet
make lint                     # Run golangci-lint
make check                    # Run fmt-check, vet, and tests

# Smoke tests
make smoke                    # Basic smoke test (no real vLLM needed)
make smoke-scheduler          # Scheduler mode smoke test
make smoke-vllm               # Real vLLM integration test (requires GPU)

# Run
make run                      # Run with configs/example.yaml
make run-scheduler            # Run with configs/scheduler_example.yaml
```

## Project Structure

```
cmd/jukebox/main.go          # Entry point
internal/
  config/                    # YAML configuration parsing and validation
  gpu/                       # GPU inventory management (nvidia-smi parsing)
  httpserver/                # Fiber-based HTTP server, handlers, middleware
  inflight/                  # In-flight request tracking for graceful drains
  jukebox/                   # Core business logic
    coordinator.go           # Swap coordination (single goroutine, serializes swaps)
    scheduler.go             # Multi-instance scheduler (LRU eviction, GPU allocation)
    router.go                # Request routing interface
    legacy_router.go         # Swap mode routing implementation
  metrics/                   # Prometheus metrics
  ports/                     # Port pool management for scheduler mode
  proxy/                     # HTTP proxy to vLLM with SSE streaming support
  vllm/                      # vLLM process lifecycle management
configs/                     # Example YAML configurations
scripts/                     # Smoke test scripts
```

## Key Architectural Concepts

1. **Coordinator** (`internal/jukebox/coordinator.go`): Serializes model swap operations via a single goroutine. Enforces cooldown between swaps and handles graceful drains.

2. **Scheduler** (`internal/jukebox/scheduler.go`): Manages multiple vLLM instances. Tracks GPU memory, performs LRU eviction of non-pinned instances, allocates ports from a pool.

3. **Router Interface** (`internal/jukebox/router.go`): Common interface for both modes - `EnsureModelReady(model) -> backend URL`.

4. **Proxy** (`internal/proxy/proxy.go`): Forwards requests to vLLM backends. Handles SSE streaming and optional model name rewriting in responses.

5. **In-flight Tracker** (`internal/inflight/tracker.go`): Tracks active requests per model for graceful shutdown/swap draining.

## Dependencies

- `github.com/gofiber/fiber/v2` - HTTP framework
- `github.com/prometheus/client_golang` - Metrics
- `gopkg.in/yaml.v3` - Configuration parsing
- `github.com/google/uuid` - Request IDs

## Coding Conventions

- Standard Go project layout with `cmd/` and `internal/`
- All internal packages are under `internal/` (not importable externally)
- Tests are colocated with source files (`*_test.go`)
- Configuration is YAML-based, parsed into strongly-typed Go structs
- Errors are wrapped with context using `fmt.Errorf("context: %w", err)`
- Logging uses Go's standard `log` package with prefixes

## API Endpoints

Model-bearing (proxy to vLLM):
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/tokenize`
- `POST /v1/detokenize`
- `POST /v1/messages`

Jukebox-handled:
- `GET /v1/models` - List configured models
- `GET /health` - Health check
- `GET /status` - Detailed status (protect in production)
- `GET /metrics` - Prometheus metrics

## Configuration

Two example configs are provided:
- `configs/example.yaml` - Swap mode (single instance)
- `configs/scheduler_example.yaml` - Scheduler mode (multi-instance)

Key config sections: `server`, `vllm`, `scheduler` (optional), `behavior`, `models`

## Runtimes

Each model picks an inference runtime via the `runtime` field (default: `vllm`).
Two runtimes are supported today:

- `vllm` — the default. Uses the `vllm:` block's `binary` and the existing
  vLLM CLI shape (`serve <path> --host ... --tensor-parallel-size ...`).
- `llama_cpp` — runs llama.cpp's `llama-server`. Requires the top-level
  `llama_cpp:` block.

```yaml
llama_cpp:
  binary: "llama-server"   # required if any model uses runtime: llama_cpp
  default_args: []         # optional; prepended to every llama_cpp launch

models:
  my-gguf:
    runtime: llama_cpp
    path: "org/repo-GGUF:Q4_K_M"   # treated as --hf-repo
    # or: path: "/models/model.gguf" # treated as -m
    max_model_len: 32768            # mapped to --ctx-size
    extra_args:
      - "--n-gpu-layers"
      - "all"
      - "--jinja"
```

Field mapping for `runtime: llama_cpp`:

| YAML field            | llama-server flag                                        |
|-----------------------|----------------------------------------------------------|
| `path`                | `--hf-repo <p>` if not a local path, else `-m <p>`       |
| `max_model_len`       | `--ctx-size <n>`                                         |
| `extra_args`          | passthrough (after `llama_cpp.default_args`)             |
| `tensor_parallel_size`, `pipeline_parallel_size`, `gpu_memory_utilization`, `dtype`, `quantization` | ignored (with a startup warning) — translate to the equivalent llama.cpp flags via `extra_args` |

The `--hf-repo` vs `-m` choice mirrors the heuristic in
`scripts/llama-server-wrapper.sh`: a value is treated as a local file when it
exists on disk, is an absolute or `./`/`../` path, or ends in `.gguf`.

Scheduler mode does not yet support `runtime: llama_cpp`; the config
validator rejects that combination.

### Multi-Token Prediction (MTP) for Qwen 3.5 / 3.6

llama.cpp implements MTP as a speculative-decoding mode that runs against
the model's own NextN layers — no separate draft model is needed. Enable it
by adding to a model's `extra_args`:

```yaml
extra_args:
  - "--spec-type"
  - "draft-mtp"
```

Two gotchas to know before you turn it on:

- The GGUF must include the NextN/MTP layers. Many community quants
  (including the default `unsloth/Qwen3.6-35B-A3B-GGUF`) strip them to save
  space and llama-server will exit at load with
  `context type MTP requested but model doesn't contain MTP layers`. Use
  an `-MTP-GGUF` variant such as `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (the
  shipped `configs/qwen36-35b-a3b-llamacpp.yaml` example points at this).
- MTP being functional doesn't guarantee a throughput win. On Qwen 3.6
  35B-A3B (MoE, ~3B active) at default `--spec-draft-n-max 3` we measured
  ~59% draft acceptance but ~0% net tok/s gain (129 → 128 tok/s) because
  the dense MTP draft pass costs about as much as the MoE main pass. Try
  tuning `--spec-draft-n-max` (try lower for cheaper drafts, higher for
  more aggressive speculation) if you want to push throughput further.

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

## vLLM Log Files

vLLM process output can be redirected to rotating log files:

```yaml
vllm:
  log_dir: "/var/log/vllm"    # Directory for log files
  log_max_size_mb: 50         # Max size before rotation (default: 50)
  log_max_files: 5            # Rotated files to keep (default: 5)

models:
  mymodel:
    path: "..."
    log_file: "/custom/path.log"  # Per-model override (optional)
```

- If `log_dir` is set, each model logs to `<log_dir>/<model-name>.log`
- Per-model `log_file` overrides the auto-generated path
- Logs are appended with size-based rotation
- Tail buffer still available via `/status` endpoint

## Testing Notes

- Smoke tests use fake vLLM servers and fake nvidia-smi for isolation
- Real vLLM tests are opt-in via `RUN_VLLM_SMOKE=1`
- Tests don't require GPU access (mocked where needed)
