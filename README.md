# vLLM Jukebox

OpenAI-compatible API server that fronts a single vLLM instance and swaps the loaded model based on each request’s `model`.

## Quickstart

Build:
```bash
go build -o bin/jukebox ./cmd/jukebox
```

Run:
```bash
./bin/jukebox -config configs/example.yaml
```

Health:
```bash
curl -s http://127.0.0.1:8080/health
```

Metrics:
```bash
curl -s http://127.0.0.1:8080/metrics
```

List configured models:
```bash
curl -s http://127.0.0.1:8080/v1/models
```

## OpenAI endpoints

Model-switching endpoints:
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Pass-through (only when vLLM is ready):
- `POST /v1/embeddings`
- `POST /v1/tokenize`
- `POST /v1/detokenize`

## vLLM via `uvx`

Set `vllm.binary: "uvx"` and Jukebox will run vLLM as `uvx vllm serve ...`.

## Production note

`/status` can reveal operational details; bind it to localhost or put it behind auth in production.

## Smoke tests

Lightweight (no real vLLM required):
```bash
./scripts/smoke.sh
```

Real vLLM integration (opt-in; requires `uvx` and a model):
```bash
RUN_VLLM_SMOKE=1 VLLM_MODEL=/path/to/model ./scripts/smoke_vllm.sh
```
