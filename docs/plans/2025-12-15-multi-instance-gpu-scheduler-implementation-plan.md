# Multi-Instance GPU Scheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GPU-aware multi-instance scheduler to vLLM Jukebox so it can run multiple vLLM processes concurrently (one per model/GPU-set/port), evict non-pinned instances by LRU to make room for larger models, and route all model-bearing OpenAI endpoints to the correct backend.

**Architecture:** Introduce a scheduler that owns instance state and serializes scheduling operations (start/evict), while allowing already-ready instances to keep serving concurrently. GPU inventory is sourced from `nvidia-smi` output. Each instance has its own port from a configured port range and its own in-flight tracker for draining.

**Tech Stack:** Go, Fiber, `os/exec` for `nvidia-smi`, existing `inflight.Tracker`, existing proxy + model rewrite, Prometheus client.

---

## Task 1: Add scheduler + model scheduling config (TDD)

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`
- Modify: `docs/spec.md`
- Modify: `README.md`

**Step 1: Add config structs**

Add:
- `type SchedulerConfig struct { NvidiaSMIBinary string; PortRangeStart int; PortRangeEnd int; MaxInstances *int; MinInstanceUptime Duration }`
- New fields on `ModelConfig`: `GPUs []int`, `MinFreeMemMBPerGPU *int`, `Pinned *bool`
- New top-level `Config` field: `Scheduler *SchedulerConfig`

**Step 2: Write failing config tests**

Add tests (table-driven where possible):
- `TestLoad_Scheduler_AllowsNull`
- `TestValidate_Scheduler_PortRangeRequiredWhenSchedulerEnabled`
- `TestValidate_Scheduler_PortRangeInvalid`
- `TestValidate_Scheduler_MaxInstancesMustFitPortRange`
- `TestValidate_Scheduler_ModelsRequireGPUsAndMinFreeWhenSchedulerEnabled`
- `TestValidate_Scheduler_RejectsCUDAVisibleDevicesInModelEnv`
- `TestValidate_Scheduler_RejectsCUDAVisibleDevicesInDefaultEnv`
- `TestValidate_Scheduler_ModelGPUsMustBeUniqueAndNonNegative`

**Step 3: Implement validation + defaults**

Rules:
- If `scheduler` is nil: preserve current behavior; ignore scheduler-only fields.
- If `scheduler` is set:
  - `port_range_start` and `port_range_end` must be valid ports, start <= end.
  - `max_instances` (if set) must be >0 and <= number of ports in range.
  - For each non-alias model: `gpus` must be non-empty; `min_free_mem_mb_per_gpu` must be set and >0.
  - Reject any `CUDA_VISIBLE_DEVICES` in `vllm.default_env` and in `models.<name>.env` (owned by scheduler).
  - `gpus` must not contain duplicates and must be >=0.
  - Default `nvidia_smi_binary` to `"nvidia-smi"` if empty.
  - Default `min_instance_uptime` to `30s` if unset/zero.

**Step 4: Update docs**

Document scheduler mode in:
- `README.md` (add a section with example config)
- `docs/spec.md` (schema additions + notes about CUDA_VISIBLE_DEVICES ownership)

Verification:
- Run: `go test ./...`
Expected: all tests pass.

---

## Task 2: Implement GPU inventory via `nvidia-smi` (TDD)

**Files:**
- Create: `internal/gpu/inventory.go`
- Create: `internal/gpu/inventory_test.go`

**Step 1: Write failing parser tests**

Add tests for parsing:
- normal lines: `"0, 81920, 61234"`
- extra whitespace
- empty output
- malformed rows (too few columns / non-int)

**Step 2: Implement parser + command runner**

Implement:
- `type GPU struct { Index int; TotalMB int; FreeMB int }`
- `func ParseInventoryCSV(out string) ([]GPU, error)`
- `type Inventory interface { List(ctx context.Context) ([]GPU, error) }`
- `type NvidiaSMIInventory struct { Binary string }` with `List(ctx)` executing:
  - `nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits`

Verification:
- Run: `go test ./...`
Expected: all tests pass.

---

## Task 3: Implement port pool allocator (TDD)

**Files:**
- Create: `internal/ports/pool.go`
- Create: `internal/ports/pool_test.go`

**Step 1: Tests**

- allocate all ports in range then fail
- release and re-allocate reuse works

**Step 2: Implementation**

Expose:
- `type Pool struct{ ... }`
- `func New(start,end int) *Pool`
- `func (p *Pool) Acquire() (int, bool)`
- `func (p *Pool) Release(port int)`

Verification:
- Run: `go test ./...`

---

## Task 4: Refactor vLLM manager into per-instance process manager (TDD)

**Files:**
- Modify: `internal/vllm/manager.go`
- Modify: `internal/vllm/runtime.go`
- Modify: `internal/vllm/manager_test.go`
- Modify: `internal/vllm/launcher_test.go`

**Step 1: Split “build args” to accept port**

Change:
- `BuildServeArgs(cfg, requestedModel)` → `BuildServeArgs(cfg, requestedModel, port int)`

Update tests accordingly.

**Step 2: Add an instance-scoped manager**

Make `vllm.Manager` parameterized by port and per-instance env additions:
- `type InstanceManager struct { cfg *config.Config; port int; cudaVisibleDevices string; ... }`
- It should use `BuildServeArgs(..., port)` and `BuildEnv(os.Environ(), defaultEnv, modelEnv+CUDA_VISIBLE_DEVICES)`
- `BaseURL()` returns `127.0.0.1:<port>`

**Step 3: Keep legacy single-manager wrapper**

Keep existing `NewManager(cfg)` for non-scheduler mode (uses `cfg.VLLM.Port`).

Verification:
- Run: `go test ./...`

---

## Task 5: Implement scheduler (instances, eviction, waiters) (TDD)

**Files:**
- Create: `internal/jukebox/scheduler.go`
- Modify: `internal/jukebox/coordinator.go` (or replace usage) / `internal/jukebox/*_test.go`
- Modify: `internal/httpserver/server.go` (interfaces)
- Modify: `internal/httpserver/handlers_proxy.go`

**Step 1: Define scheduler interface + route**

Add types:
- `type Route struct { BaseURL string; UpstreamModel string; Done func(); InstanceID string }`
- `type Router interface { Status() Status; AcquireRoute(ctx, requestedModel, requestID string) (Route, error) }`

**Step 2: Write failing unit tests for scheduler decisions**

Use fake inventory + fake vLLM launcher:
- ready instance routes while busy scheduling another model
- pinned conflict rejects
- LRU eviction order for overlapping GPU sets
- min-uptime prevents eviction
- one waiter per model (second waiter gets 503)

**Step 3: Implement scheduler**

Key implementation details:
- Maintain `instances map[string]*instance` keyed by resolved model name (one instance per model in v1).
- `busy` gate: use a channel semaphore (size 1) for scheduling operations.
- Fast path reads under RWMutex and returns ready instance if not draining.
- If scheduling required and busy:
  - register waiter if none exists for model, then block waiting to acquire scheduling permit within `swap_wait_timeout`.
  - if waiter exists, return `RejectSwapInProgress` (`503`).
- Scheduling operation:
  - mark conflicting instances draining; wait per-instance drain; stop; release port.
  - re-check `nvidia-smi` free mem on required GPUs.
  - start instance; verify health/model; mark ready.

**Step 4: Update HTTP server wiring**

Change httpserver to depend on the new `Router` interface.

Verification:
- Run: `go test ./...`

---

## Task 6: Route all model-bearing endpoints via scheduler

**Files:**
- Modify: `internal/httpserver/server.go`
- Modify: `internal/httpserver/handlers_proxy.go`

Changes:
- Treat `/v1/embeddings`, `/v1/tokenize`, `/v1/detokenize` as switching endpoints (extract model and acquire route).
- Remove “pass-through when vLLM ready” assumption (ambiguous with multi-instance).

Verification:
- Run: `go test ./...`

---

## Task 7: Update status + metrics for multi-instance

**Files:**
- Modify: `internal/httpserver/handlers_status.go`
- Modify: `internal/httpserver/handlers_health.go`
- Modify: `internal/metrics/metrics.go`
- Modify: `internal/httpserver/handlers_metrics_test.go` (if needed)

Changes:
- Status:
  - include instances list (model, port, gpu IDs, pid, state, pinned, draining, inflight, startedAt, lastUsedAt)
  - include busy flag + waiters count
- Health:
  - `accepting_requests` true if at least one instance ready OR scheduler is able to schedule (not hard-error).
- Metrics:
  - keep `jukebox_in_flight_requests` as total (sum across instances)
  - add optional per-instance gauges (running instances, inflight by instance)

Verification:
- Run: `go test ./...`

---

## Task 8: Wire scheduler mode in main

**Files:**
- Modify: `cmd/jukebox/main.go`

Changes:
- If `cfg.Scheduler == nil`: keep legacy `Coordinator` behavior.
- If `cfg.Scheduler != nil`:
  - create `gpu.Inventory` (nvidia-smi)
  - create port pool from config
  - create scheduler + run any background goroutines if needed
  - preload behavior:
    - honor `behavior.default_model` by scheduling it once at startup

Verification:
- Run: `go test ./...`

---

## Task 9: Add a minimal “scheduler mode” smoke script (optional)

**Files:**
- Create: `scripts/smoke_scheduler.sh`
- Modify: `README.md`

Approach:
- Use a dummy script binary in place of vLLM (like existing tests do) that binds ports and serves `/health` + `/v1/models`.
- Exercise routing to two models, eviction, and waiting behavior.

Verification:
- Run: `./scripts/smoke.sh` and `./scripts/smoke_scheduler.sh`

