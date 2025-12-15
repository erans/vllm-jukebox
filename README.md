# vLLM Jukebox

vLLM Jukebox is an OpenAI-compatible HTTP server that fronts a **single** vLLM instance and automatically **swaps the loaded model** based on each incoming request’s `model`.

This is useful when:
- You want one stable OpenAI-compatible endpoint, but multiple models (with swap-on-demand).
- You’re OK with only one model being loaded at a time (no multi-instance/zero-downtime swaps).

## Features

- OpenAI-ish endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`
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

## Build

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
  # You can also set this to an absolute path, e.g. "/opt/homebrew/bin/uvx".
  # Windows-style "uvx.exe" is also supported.
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

Model naming rules:
- Client-facing model names are the YAML keys under `models:`.
- Aliases (`alias: other_name`) let you support multiple names for the same underlying model config.
- Jukebox forwards the *configured* `path` to vLLM (so vLLM sees a real model ID/path even if the client requested an alias).

## Endpoints

Model-switching endpoints (extract `model` from body, ensure model is loaded, then proxy to vLLM):
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Pass-through endpoints (proxied only when vLLM is ready):
- `POST /v1/embeddings`
- `POST /v1/tokenize`
- `POST /v1/detokenize`

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
- Model swap takes time: the first request for a new model blocks (up to `swap_wait_timeout`); other requests get `503` with `Retry-After`.
