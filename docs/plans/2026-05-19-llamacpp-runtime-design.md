# llama.cpp runtime support (v1) — design

**Date:** 2026-05-19
**Status:** Design accepted; ready for implementation plan.
**Scope:** First-class llama.cpp (`llama-server`) support in vllm-jukebox, swap mode only. MTP, scheduler-mode coverage, and building llama.cpp from source are out of scope for this change.

## Motivation

The existing approach (`scripts/llama-server-wrapper.sh` consumed via the `vllm.binary` field) translates vLLM-style flags into llama-server flags from bash. It works but:

- Silently drops fields like `--gpu-memory-utilization`, `--dtype`, `--tensor-parallel-size`.
- Forces a vLLM-shaped CLI layout on a non-vLLM binary.
- Leaks vLLM assumptions into the readiness path (`vllm.VerifyModelLoaded` compares `m.ID` to the configured `path`, which happens to line up for llama-server but is coincidence rather than design).

A first-class runtime makes llama.cpp a peer of vLLM and gives us the abstraction we need to add MTP, scheduler-mode, and other backends later without further bash plumbing.

## Configuration changes

### New per-model field

```yaml
models:
  some-model:
    runtime: llama_cpp        # one of: vllm (default), llama_cpp
```

Default is `vllm`, so every existing config keeps working unchanged.

### New top-level block

Required if any model picks `runtime: llama_cpp`:

```yaml
llama_cpp:
  binary: "llama-server"      # path or basename resolved via PATH
  default_args: []            # optional, prepended to every llama_cpp launch
```

The existing `vllm:` block is untouched. Process-management settings on it (`startup_timeout`, `shutdown_timeout`, `drain_timeout`, `swap_cooldown`, `log_dir`, `log_max_size_mb`, `log_max_files`) are runtime-agnostic in practice and continue to apply to whichever backend runs. They are not renamed in v1 to avoid churn.

### Model-field semantics under `runtime: llama_cpp`

| YAML field         | Effect under llama_cpp                                                              |
|--------------------|--------------------------------------------------------------------------------------|
| `path`             | `--hf-repo <path>` if it looks like a HF repo, else `-m <path>` (see heuristic)     |
| `max_model_len`    | `--ctx-size <n>`                                                                     |
| `extra_args`       | Passthrough after `default_args`                                                    |
| `tensor_parallel_size`, `pipeline_parallel_size`, `gpu_memory_utilization`, `dtype`, `quantization` | Ignored; emit a one-time startup warning so misconfiguration is visible |

**HF-repo vs local heuristic** (matches the existing wrapper script): treat the value as a local path when it satisfies any of — file exists on disk, starts with `/`, starts with `./` or `../`, or has a `.gguf` suffix. Anything else is passed as `--hf-repo`. The repo form may include a quant tag (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M`); we pass it through unchanged.

### Validation additions

- If any model sets `runtime: llama_cpp`, the top-level `llama_cpp.binary` field is required (non-empty).
- Unknown `runtime` values are rejected at config-load time.
- In scheduler mode (out of scope for v1 functionally, but the validator runs), reject `runtime: llama_cpp` with a clear error directing the user to swap mode.

## Code structure

### New package: `internal/runtime/`

```go
// internal/runtime/runtime.go
type Runtime interface {
    Name() string
    Binary(cfg *config.Config) string
    BuildArgs(cfg *config.Config, model config.ModelConfig,
              resolvedName string, port int) ([]string, error)
    VerifyModelLoaded(ctx context.Context, baseURL,
                      expectedID, expectedPath string) error
}

func For(name string) (Runtime, error)   // factory; "" → vllm
```

### Implementations

- **`internal/runtime/vllm.go`** — wraps the existing logic in `internal/vllm/manager.go`'s `BuildServeArgsForPort` and `internal/vllm/health.go`'s `VerifyModelLoaded`. The wrapper is thin; the underlying functions stay so existing call sites and tests keep working.
- **`internal/runtime/llamacpp.go`** — new. Produces `--host 127.0.0.1 --port <p> --ctx-size <n>` plus the path-form selection and the default/extra args, then a verifier that hits `/v1/models` and matches against both the configured ID and the resolved `path` (llama-server reports either the alias, the repo, or the GGUF basename as `id`).

### Touch points in existing code

1. `internal/config/config.go`
   - Add `Runtime string` to `ModelConfig`.
   - Add `LlamaCppConfig` type and `LlamaCpp *LlamaCppConfig` to `Config`.
   - Wire defaults + validation.
2. `internal/vllm/runtime.go` (`Manager.Start`)
   - Resolve the per-model runtime, look up `runtime.For(...)`.
   - Use `rt.Binary(cfg)` instead of `m.cfg.VLLM.Binary` directly.
   - Use `rt.BuildArgs(...)` instead of `BuildServeArgsForPort`.
   - The existing `wrapBinaryArgs` `uvx` special case stays put; it only triggers when binary basename is `uvx`, which is a vLLM thing.
3. `internal/vllm/runtime.go` (`Manager.VerifyReady`)
   - Replace direct `VerifyModelLoaded` call with `rt.VerifyModelLoaded(...)`.
4. `internal/vllm/manager.go` — leave `BuildServeArgs` / `BuildServeArgsForPort` in place. The `runtime/vllm.go` adapter calls them.

The `internal/vllm` package keeps its name for historical reasons; renaming to `backend` or `runtime` is deferred.

## Tests

- **`internal/runtime/llamacpp_test.go`** — unit tests:
  - HF-repo path → `--hf-repo`
  - Local `.gguf` path → `-m`
  - `max_model_len: 65536` → `--ctx-size 65536`
  - vLLM-only fields set under `runtime: llama_cpp` produce no CLI flags
  - `default_args` precede `extra_args`
- **`internal/runtime/vllm_test.go`** — regression: the adapter's args match the existing `BuildServeArgsForPort` output byte-for-byte for a representative config.
- **`internal/config/config_test.go`** — additions:
  - `runtime: llama_cpp` without `llama_cpp.binary` → error
  - Unknown `runtime` value → error
  - Default `runtime` is `vllm`
  - Existing test fixtures still parse identically.

No integration tests against a real `llama-server` in this change. Smoke tests stay as they are.

## Example config

New file: `configs/qwen36-35b-a3b-llamacpp.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8080

vllm:
  port: 8000
  startup_timeout: 1800s
  shutdown_timeout: 30s
  drain_timeout: 60s
  swap_cooldown: 30s
  log_dir: "/tmp/vllm-logs"
  log_max_size_mb: 100
  log_max_files: 3

llama_cpp:
  binary: "llama-server"

behavior:
  rewrite_model_name: true

models:
  qwen36-35b-a3b:
    runtime: llama_cpp
    path: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
    max_model_len: 65536
    extra_args:
      - "--jinja"
      - "--flash-attn"
      - "on"
      - "--n-gpu-layers"
      - "all"
      - "--split-mode"
      - "layer"
      - "--cache-type-k"
      - "q8_0"
      - "--cache-type-v"
      - "q8_0"
      - "--parallel"
      - "4"
      - "--batch-size"
      - "2048"
      - "--ubatch-size"
      - "512"
```

The exact GGUF repo/quant tag is illustrative; the user will pick the one that fits their hardware.

## Docs

- Update `CLAUDE.md`:
  - Mention llama.cpp under "Project Overview" as a supported runtime.
  - Add a short "Runtimes" subsection describing the `runtime` field, the `llama_cpp` block, and the field-semantics table above.
- No README changes for v1.

## Rollout

1. Land the runtime abstraction (no behavior change for existing vLLM configs).
2. Land the llama_cpp runtime + config plumbing + example + tests.
3. `make build && make test` must pass; manual smoke against the new example config is on the user once `llama-server` is installed.
4. Keep `scripts/llama-server-wrapper.sh` and `configs/qwopus-glm-18b-q8-llama-server.yaml` in tree as a fallback until the native path is validated; delete them in a follow-up.

## Out of scope

- MTP (Multi-Token Prediction) flag wiring — deferred until the user has a llama.cpp build that includes the feature.
- Scheduler-mode support for llama.cpp.
- Building llama.cpp from source / managing the binary install.
- Renaming the `internal/vllm` package or the YAML `vllm:` block.
