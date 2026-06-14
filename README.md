# vLLM Jukebox

vLLM Jukebox is an OpenAI-compatible HTTP server that can run in either:

- **Swap mode**: fronts a **single** vLLM instance and automatically **swaps the loaded model** based on each incoming request’s `model`.
- **Scheduler mode**: runs **multiple concurrent** vLLM instances (one per configured GPU set + port), routes requests by `model`, and can evict non-pinned instances (LRU) to make room for larger models.

This is useful when:
- You want one stable OpenAI-compatible endpoint, but multiple models (with swap-on-demand).
- You’re OK with only one model being loaded at a time (no multi-instance/zero-downtime swaps).
- You have multiple GPUs and want multiple models served concurrently (scheduler mode).

## Features

- OpenAI-ish endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`
- Anthropic protocol support: `POST /v1/messages`
- Graceful swaps: drains in-flight requests before restarting vLLM
- Aliases: multiple client-facing model names can point to one underlying model config
- Prometheus metrics at `GET /metrics`
- Operational endpoints: `GET /health`, `GET /status`

## Requirements

- Go (to build `jukebox`)
- A way to run vLLM:
  - Recommended: `uvx` (default `vllm.binary: "uvx"`), which runs vLLM as `uvx vllm serve ...`
  - Alternative: set `vllm.binary: "vllm"` if you have `vllm` installed directly
- A model available to vLLM:
  - A local filesystem path (e.g. `/models/Qwen2.5-0.5B-Instruct`), or
  - A HuggingFace model ID (e.g. `Qwen/Qwen2.5-0.5B-Instruct`)

Notes:
- Some HuggingFace models are gated (e.g. `meta-llama/*`) and require access + `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`).
- vLLM needs sufficient free GPU memory; tune `gpu_memory_utilization` / `max_model_len` if you hit OOMs.

## Installation

### Download pre-built binary

Download the latest release from the [Releases page](https://github.com/erans/vllm-jukebox/releases):

```bash
# Download and extract
curl -sL https://github.com/erans/vllm-jukebox/releases/latest/download/jukebox-linux-amd64.tar.gz | tar xz
chmod +x jukebox
```

### Build from source

```bash
go build -o bin/jukebox ./cmd/jukebox
```

## Run

```bash
./bin/jukebox -config configs/example.yaml
```

## Configuration

Jukebox is configured with a single YAML file.

### Minimal config

```yaml
server:
  host: "127.0.0.1"
  port: 8080

vllm:
  # Use "uvx" to run `uvx vllm serve ...`
  # You can also set this to an absolute path, e.g. "/usr/local/bin/uvx".
  binary: "uvx"
  port: 8000

models:
  qwen:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
```

### Multiple models + aliases

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  log_requests: true

vllm:
  binary: "uvx"
  port: 8000
  startup_timeout: 300s
  shutdown_timeout: 30s
  drain_timeout: 60s
  swap_cooldown: 30s
  swap_wait_timeout: 60s
  defaults:
    gpu_memory_utilization: 0.70
    dtype: "float16"
    max_model_len: 2048

behavior:
  # Optional: preload one model at startup.
  default_model: "qwen"
  # If true, rewrites response JSON/SSE `model` fields to match the request.
  rewrite_model_name: true

models:
  qwen:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    tensor_parallel_size: 1

  # Alias example (client asks for "gpt-3.5-turbo", but we serve qwen)
  gpt-3.5-turbo:
    alias: qwen
```

### Scheduler mode (multi-instance, multi-GPU)

In scheduler mode, each non-alias model declares an **exact GPU set** and a **minimum free VRAM requirement per GPU**. Jukebox allocates a unique port per running instance from the configured port range and sets `CUDA_VISIBLE_DEVICES` automatically (do not set it in `env`).

```yaml
server:
  host: "0.0.0.0"
  port: 8080

vllm:
  binary: "uvx"
  startup_timeout: 300s
  shutdown_timeout: 30s
  drain_timeout: 60s
  swap_wait_timeout: 60s

scheduler:
  port_range_start: 8100
  port_range_end: 8199
  max_instances: 8
  min_instance_uptime: 30s
  nvidia_smi_binary: "nvidia-smi"

models:
  small:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    gpus: [0]
    min_free_mem_mb_per_gpu: 4000

  big:
    path: "/models/Meta-Llama-3-70B-Instruct"
    gpus: [0,1,2,3]
    min_free_mem_mb_per_gpu: 40000

  pinned-hot:
    path: "/models/Some-Always-On-Model"
    gpus: [4]
    min_free_mem_mb_per_gpu: 16000
    pinned: true
```

### Sleep mode (vLLM 0.22+)

For models marked `sleep_mode: true`, jukebox uses vLLM's sleep/wake API in place of full process stop/start when the instance is evicted or auto-suspended. Wake from L1 sleep is ~5-15s vs ~30-100s for a cold start — and in practice on a small embedding model with weights resident, **as fast as ~200ms** (CPU↔GPU transfer, no model reload).

**What L1 sleep does** (verified on a 4× RTX 3090 host running vLLM with PR #32947 + PR #44778 applied):

| L1 sleep | Behavior |
|---|---|
| Weights → CPU RAM | ✓ |
| KV cache discarded | ✓ |
| Wake is fast (no disk reload) | ✓ |
| **GPU VRAM actually freed** | ✓ — measured ~18-19 GiB freed per GPU on a 27B AWQ model with TP=2 PP=2 |

The earlier worry that "L1 keeps cumem reservation" applied to vLLM versions BEFORE PR #32947 (merged Feb 2026) where a `pool_ctx and model_ctx` typo prevented the CuMemAllocator from capturing weight allocations. With #32947 applied, the allocator releases on sleep and reclaims on wake. Verify your vLLM image has it before relying on real VRAM reclaim — see `scripts/check-vllm-sleep.sh`.

`sleep_level: 2` is the heavier option (discards weights entirely; wake reloads from disk). Use it only when CPU RAM is constrained.

Requirements:
- vLLM 0.22+ (the sleep API)
- `runtime: vllm` (llama.cpp does not have an equivalent)
- For `lifecycle: managed` models, jukebox automatically launches vLLM with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`. For `lifecycle: external` models, the operator must include both in their own command line / env.

```yaml
scheduler:
  port_range_start: 8100
  port_range_end: 8199

models:
  big:
    path: "/models/Big-Model"
    gpus: [0, 1, 2, 3]
    min_free_mem_mb_per_gpu: 40000
    pinned: true                  # never auto-sleep
    sleep_mode: true              # use sleep instead of stop on eviction (n/a when pinned)

  bursty-embeddings:
    path: "BAAI/bge-m3"
    runner: pooling               # passes --task embed to vLLM
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
    sleep_mode: true
    sleep_level: 1                # 1 = L1 (weights → CPU RAM, fastest wake)
    wake_timeout: 30s             # max wait for /health=200 after /wake_up
    idle_timeout: 600s            # auto-sleep after 10 min idle
```

### Lifecycle: external (jukebox doesn't own the process)

For setups where vLLM runs as a separate container (or systemd service, or whatever — anything jukebox didn't start), declare `lifecycle: external` and point jukebox at its hostname + port. Jukebox proxies requests and, with `sleep_mode: true`, drives sleep/wake — but never starts or stops the process.

```yaml
scheduler:
  port_range_start: 8100
  port_range_end: 8199

models:
  vllm-main:
    lifecycle: external
    host: "vllm-main"             # docker-compose service name, DNS hostname, or IP
    port: 8000
    gpus: [0, 1, 2, 3]            # informational only — jukebox does not allocate
    min_free_mem_mb_per_gpu: 40000
    pinned: true

  vllm-embeddings:
    lifecycle: external
    host: "vllm-embeddings"
    port: 8000
    gpus: [1]
    min_free_mem_mb_per_gpu: 4000
    sleep_mode: true
    idle_timeout: 600s
```

The external vLLM service must include `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1` in its launch arguments / environment for sleep-mode operations to function.

### NVLink topology hints (optional)

If you declare your host's NVLink-bonded GPU pairs, jukebox warns at startup when a multi-GPU model's `gpus` layout would force tensor-parallel collective ops across a non-NVLink hop:

```yaml
scheduler:
  nvlink_pairs:
    - [0, 3]                      # GPU0 ↔ GPU3 NVLink-bonded
    - [1, 2]                      # GPU1 ↔ GPU2 NVLink-bonded
```

Models with 2 GPUs whose pair isn't in `nvlink_pairs`, or 4-GPU models that don't span exactly two configured pairs, get a structured warning. Single-GPU models and unusual sizes are ignored. Validation is purely opt-in — leave `nvlink_pairs` unset to disable.

### Model config reference

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes* | Model path or HuggingFace ID. Optional when `lifecycle: external`. |
| `runtime` | enum | No | `vllm` (default) or `llama_cpp`. |
| `runner` | enum | No | `generate` (default) or `pooling` (vLLM `--task embed`). vLLM-only. |
| `alias` | string | No | Point this name at another model. Incompatible with `lifecycle: external`. |
| `gpus` | []int | Yes** | Exact GPU IDs (informational for external lifecycle). |
| `min_free_mem_mb_per_gpu` | int | Yes** | Placement guardrail (scheduler mode). |
| `tensor_parallel_size` / `pipeline_parallel_size` | int | No | Standard vLLM TP/PP. |
| `max_model_len`, `gpu_memory_utilization`, `dtype`, `quantization` | mixed | No | Passed to vLLM. |
| `extra_args` | []string | No | Appended to the vLLM `serve` command. |
| `env` | map | No | Per-model environment variables. |
| `pinned` | bool | No | Never auto-evict or auto-sleep. |
| `sleep_mode` | bool | No | Use sleep/wake instead of stop/start on eviction. vLLM-only. |
| `sleep_level` | int | No | 1 (L1, weights→RAM, fast wake — default) or 2 (L2, discard all, slower). |
| `wake_timeout` | duration | No | Max wait for `/health=200` after `/wake_up`. Default 120s. |
| `idle_timeout` | duration | No | Auto-sleep after this much idle. 0 (default) = never. |
| `lifecycle` | enum | No | `managed` (default — jukebox spawns) or `external` (operator-managed process). |
| `host` | string | No*** | External service hostname. Defaults to `127.0.0.1`. |
| `port` | int | Yes*** | External service port. |
| `power_limit` / `power_limits` | int / map | No | Per-model `nvidia-smi -pl` overrides. |
| `log_file` | string | No | Override the per-model vLLM log path. |

\* Required unless `alias` or `lifecycle: external`. \*\* Required for non-alias models when `scheduler` is enabled. \*\*\* Required when `lifecycle: external`.

Model naming rules:
- Client-facing model names are the YAML keys under `models:`.
- Aliases (`alias: other_name`) let you support multiple names for the same underlying model config.
- Jukebox forwards the *configured* `path` to vLLM (so vLLM sees a real model ID/path even if the client requested an alias).

## Endpoints

Model-bearing endpoints (extract `model` from body, ensure a backend instance is ready, then proxy to vLLM):
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/tokenize`
- `POST /v1/detokenize`

Anthropic endpoints (proxy to vLLM):
- `POST /v1/messages`

Notes:
- These endpoints require a `model` field in the JSON body (matching OpenAI semantics).
- In scheduler mode, each request is routed to the vLLM instance for that model (potentially triggering eviction/start).

Jukebox endpoints:
- `GET /v1/models` (returns configured models, not vLLM’s)
- `GET /health`
- `GET /status` (bind to localhost / protect in production)
- `GET /metrics` (Prometheus)

## Example request

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}]
  }'
```

## Smoke tests

Lightweight (no real vLLM required):
```bash
./scripts/smoke.sh
```

Scheduler mode (no real vLLM / no GPU required; uses fake `nvidia-smi` + fake vLLM server):
```bash
./scripts/smoke_scheduler.sh
```

Real vLLM integration (opt-in; requires `uvx` and a model):
```bash
RUN_VLLM_SMOKE=1 VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct ./scripts/smoke_vllm.sh
```

Useful knobs for the real smoke test:
- `VLLM_GPU_MEMORY_UTILIZATION`
- `VLLM_MAX_MODEL_LEN`
- `VLLM_DTYPE`
- `VLLM_STARTUP_TIMEOUT_SECS`
- `JBOX_REQUEST_TIMEOUT_SECS`

## Troubleshooting

- vLLM exits immediately with a gated-model error: export `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`) and ensure you have access.
- vLLM fails with GPU memory errors: stop other GPU-heavy processes, or lower `gpu_memory_utilization` / `max_model_len`.
- Model load/scheduling takes time: the triggering request can block (up to `swap_wait_timeout`); other requests that require scheduling get `503` with `Retry-After`. Requests for already-ready models continue serving.
- Scheduler mode: do not set `CUDA_VISIBLE_DEVICES` in `vllm.default_env` or `models.<name>.env` (the scheduler owns it).
