# Multi-Instance GPU Scheduler Design (vLLM Jukebox)

**Date:** 2025-12-15  
**Status:** Draft (validated via brainstorming)  
**Goal:** Extend vLLM Jukebox from “single vLLM process with model swapping” to “multiple concurrent vLLM instances” with a GPU-aware scheduler that can evict non-pinned models (LRU) to make room for larger models.

---

## Summary

We add a **scheduler** that:

- Discovers GPUs and free VRAM via `nvidia-smi`.
- Runs **multiple vLLM processes concurrently**, each bound to:
  - a configured GPU set (`CUDA_VISIBLE_DEVICES`),
  - an allocated port from a configured range,
  - one loaded model.
- Routes **all OpenAI endpoints that accept `model`** to the correct vLLM instance.
- When a requested model cannot be started due to conflicts or insufficient free VRAM, it can **evict multiple non-pinned instances** (LRU) that overlap the requested model’s GPU set, respecting `min_instance_uptime` to reduce thrash.
- While a scheduling operation is in progress, **requests for already-ready models continue serving**; only requests that *require scheduling* are rejected with `503` + `Retry-After`.

---

## Current State (Baseline)

Today, Jukebox:

- Fronts **a single vLLM process** and does “swap on demand”.
- Uses a single coordinator (`internal/jukebox/coordinator.go`) that serializes swaps and blocks all “model switching” endpoints during transitions.
- Proxies to a single upstream `BaseURL` (`http://127.0.0.1:<vllm.port>`) from `internal/httpserver/handlers_proxy.go`.
- Tracks in-flight requests globally to drain before swap.

This design cannot serve multiple models simultaneously.

---

## Goals / Non-Goals

### Goals

- Serve multiple models concurrently (multiple vLLM processes).
- Support **fixed GPU sets per model** (exact GPU IDs).
- Use **hybrid fit checks**:
  - GPU set must be available (no overlapping assigned GPUs).
  - Each GPU in the set must have `free_mb >= min_free_mem_mb_per_gpu` based on `nvidia-smi`.
- Eviction:
  - `pinned: true` models are **never evicted** if running.
  - Among non-pinned, evict by **LRU**, but only if instance uptime ≥ `min_instance_uptime`.
- Keep existing semantics:
  - Triggering request can wait up to `swap_wait_timeout`.
  - Non-triggering requests that require scheduling get `503` + `Retry-After`.
  - Requests for already-ready models continue working during scheduling.

### Non-Goals (initially)

- Precise per-model VRAM estimation beyond `min_free_mem_mb_per_gpu`.
- Request queueing beyond “one waiter per model”.
- Load balancing multiple instances for the same model.
- Automatic GPU set selection (“any GPUs”)—GPU IDs are explicit per model.
- Multi-node scheduling.

---

## Proposed Architecture

### Core Concepts

- **GPU Inventory**: a snapshot from `nvidia-smi` of GPU IDs and memory totals/free.
- **Instance**: one vLLM process with:
  - `modelName` (resolved, non-alias),
  - `gpuIDs []int`,
  - `port int`,
  - `pid int`,
  - `startedAt time.Time`,
  - `lastUsedAt time.Time` (LRU),
  - `state`: `starting|ready|draining|stopping|error`,
  - per-instance `inflight.Tracker`.
- **Scheduler**: owns the instance registry and serializes scheduling operations.

### Data Flow (Model-Bearing Requests)

1. HTTP handler extracts `model` from body.
2. Handler asks scheduler: `AcquireRoute(model, requestID)`.
3. Scheduler returns:
   - `BaseURL` for the chosen instance,
   - `UpstreamModel` (path/ID for JSON rewriting),
   - an `inflight` token (or a `done()` function) to track per-instance in-flight.
4. Handler proxies to that `BaseURL`.

### Data Flow (Scheduling in Progress)

- If a request targets a model that already has a **ready** instance: proceed (even if scheduler is busy).
- If a request targets a model that is missing, starting, draining, or otherwise not ready:
  - If scheduler is busy, either:
    - register as the single waiter for that model and wait up to `swap_wait_timeout`, or
    - immediately return `503` + `Retry-After` if a waiter already exists.

---

## Configuration Changes (Proposed)

### `scheduler`

```yaml
scheduler:
  nvidia_smi_binary: "nvidia-smi"   # optional
  port_range_start: 8100            # required
  port_range_end: 8199              # required
  max_instances: 8                  # optional cap
  min_instance_uptime: 30s          # optional; prevents thrash
```

### `models.<name>` additions

```yaml
models:
  llama-3-70b:
    path: "..."
    gpus: [0,1,2,3]                 # required for schedulable models
    min_free_mem_mb_per_gpu: 40000  # required for hybrid fit checks
    pinned: true                    # optional
```

Notes:
- Aliases remain supported; scheduler resolves request to a single canonical model config.
- Model `env` may continue to exist, but **`CUDA_VISIBLE_DEVICES` is owned by scheduler** (config should reject it to avoid footguns).

---

## GPU Discovery (via `nvidia-smi`)

Implement `internal/gpu`:

- `ListGPUs(ctx) ([]GPU, error)`
- Executes:
  - `nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits`
- Parsing rules:
  - Trim whitespace, split CSV, `Atoi` fields.
  - Return stable by `index` ordering.

Usage:
- Startup: validate model GPU IDs exist; validate `port_range`.
- Placement: check current `free_mb` against per-model `min_free_mem_mb_per_gpu`.
- Post-eviction: re-check to ensure we didn’t free “in our own registry only” while external processes still consume VRAM.

---

## Port Allocation

Scheduler manages a port pool:

- Range: `[port_range_start, port_range_end]` inclusive.
- Track ports in use by running instances.
- On instance creation: allocate a free port.
- On stop: release the port.
- Validate `max_instances <= len(range)` when both are set.

---

## Scheduling + Eviction Algorithm

Terminology:
- `target` model has required GPU set `G_target`.
- A running instance `i` has GPU set `G_i`.
- `conflict(i) := intersects(G_i, G_target)`.

### AcquireRoute(model)

Fast path (no scheduling):
- If an instance for that model is `ready` and not `draining`, return it and update `lastUsedAt`.

Scheduling path (serialized):
1. Resolve aliases to canonical `modelName`.
2. If `busy`:
   - if no existing waiter for `modelName`, register waiter and wait (up to `swap_wait_timeout`).
   - else return `503` + `Retry-After`.
3. Set `busy=true` for the duration of scheduling.
4. Compute `conflictingInstances := {i | conflict(i)}`.
5. If any `i` is `pinned`: reject (resource conflict).
6. Evict non-pinned conflicts in LRU order, subject to `min_instance_uptime`:
   - For each instance `i` (oldest `lastUsedAt` first):
     - If `uptime(i) < min_instance_uptime`, skip it (cannot evict yet).
     - Mark `i.draining=true` so new requests won’t be routed to it.
     - Wait for `i.inflight` drain (bounded by drain timeout).
     - Stop vLLM process and release GPUs/port.
   - If conflicts remain after evictable instances exhausted: reject (cannot free required GPUs).
7. Re-check `nvidia-smi` inventory free mem for all GPUs in `G_target`:
   - If any GPU free mem < threshold: reject (external consumer / fragmentation risk).
8. Start a new vLLM instance on `G_target` + allocated port.
9. Verify ready + verify model loaded.
10. Set instance `ready`, update `lastUsedAt`, notify waiters, `busy=false`.

### Min-Uptime Thrash Protection

`min_instance_uptime` applies only to eviction.

If a placement requires evicting a too-young instance, placement can fail even if it’s “logically possible”, which reduces rapid flip-flopping.

---

## Per-Instance Drain Semantics

We move from a single global in-flight tracker to **per-instance** trackers:

- Routing to a `ready` instance increments that instance’s in-flight count.
- When instance enters `draining`, no new requests route to it; remaining in-flight requests are allowed to finish.
- Eviction waits on that instance’s drain (bounded), then stops it.

This enables “keep serving other models while scheduling”.

---

## Endpoint Routing

We route **all OpenAI endpoints that accept `model`** through the scheduler (examples):

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/tokenize`, `POST /v1/detokenize` (if they include `model` in body for the upstream)

Endpoints without `model` stay global:

- `GET /health`, `GET /status`, `GET /metrics`, `GET /v1/models`

---

## Error Handling & Response Semantics

- Unknown model name: `400` “model_not_found” (same as today).
- Scheduler busy and request requires scheduling: `503` + `Retry-After` (same shape as current swap).
- Pinned conflict: `503` (message should clarify pinned overlap / insufficient resources).
- VRAM insufficient per `nvidia-smi`: `503` (resource unavailable).
- vLLM start/verify failure: `500` or `503` depending on whether it’s transient; preserve existing backoff behavior conceptually (can be adapted per-instance).

---

## Observability

### Status

`GET /status` should include:
- GPU inventory snapshot (optional/redacted)
- Instances list:
  - model, port, gpuIDs, pid, state, pinned, draining, inflight, startedAt, lastUsedAt
- Scheduler busy flag + active scheduling op summary
- Waiters (count and which models)

### Metrics

Add/adjust Prometheus:
- `jukebox_running_instances{model,port}=1`
- `jukebox_instance_in_flight{model,port}`
- `jukebox_evictions_total{model}` and `...{reason}`
- `jukebox_schedule_rejections_total{reason}`
- Keep existing request counters/histograms; add labels if needed.

---

## Implementation Outline (What Changes Where)

### New packages

- `internal/gpu`: `nvidia-smi` inventory + parsing
- `internal/scheduler` or extend `internal/jukebox`: instance registry + scheduling/eviction logic
- `internal/ports`: port pool allocator (or keep inside scheduler)

### Refactors

- `internal/vllm/runtime.go`: split into reusable per-instance manager (port/env/gpuIDs are parameters).
- `internal/httpserver/handlers_proxy.go`: model-based routing returns per-request upstream base URL.
- `internal/inflight`: support multiple trackers (existing tracker can be re-used per instance).

### Config

- `internal/config/config.go`: add scheduler and per-model fields + validation.

---

## Testing Strategy

- Unit tests for:
  - `nvidia-smi` output parsing (table-driven).
  - Port allocator behavior (range exhaustion, reuse).
  - Scheduler decisions:
    - pinned conflict rejects,
    - LRU eviction order,
    - min-uptime prevents eviction,
    - waiter behavior (one waiter per model),
    - “ready models keep serving while busy”.
- Integration test (optional / opt-in like existing vLLM smoke):
  - start multiple dummy “vLLM-like” servers on ports,
  - scheduler routes correctly and enforces draining/eviction.

