#!/usr/bin/env bash
set -euo pipefail

# Real vLLM integration smoke test.
#
# This is intentionally opt-in because it may require:
# - GPU access (depending on your vLLM build/runtime)
# - a model that is already available locally or downloadable
#
# Usage:
#   RUN_VLLM_SMOKE=1 VLLM_MODEL=/path/to/model ./scripts/smoke_vllm.sh
#   RUN_VLLM_SMOKE=1 VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct ./scripts/smoke_vllm.sh
#
# Note:
#   Some HuggingFace model IDs (e.g. meta-llama/*) are gated and require access + a token
#   (export `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`) or vLLM will fail fast during startup.
#
# Optional:
#   VLLM_STARTUP_TIMEOUT_SECS=600
#   JBOX_REQUEST_TIMEOUT_SECS=120
#   VLLM_GPU_MEMORY_UTILIZATION=0.70
#   VLLM_MAX_MODEL_LEN=2048
#   VLLM_DTYPE=float16

if [[ "${RUN_VLLM_SMOKE:-}" != "1" ]]; then
  echo "skipping: set RUN_VLLM_SMOKE=1 to run real vLLM smoke"
  exit 0
fi

if [[ -z "${VLLM_MODEL:-}" ]]; then
  echo "error: set VLLM_MODEL to a local model path or HuggingFace ID" >&2
  exit 2
fi

if [[ "${VLLM_MODEL}" == meta-llama/* ]]; then
  if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    echo "note: VLLM_MODEL=${VLLM_MODEL} is usually gated; you likely need access + HUGGING_FACE_HUB_TOKEN (or HF_TOKEN)" >&2
  fi
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! command -v uvx >/dev/null 2>&1; then
  echo "error: uvx not found in PATH" >&2
  exit 2
fi

mkdir -p bin
go build -o bin/jukebox ./cmd/jukebox

vllm_port="$(
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

jbox_port="$(
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

startup_timeout="${VLLM_STARTUP_TIMEOUT_SECS:-600}"
req_timeout="${JBOX_REQUEST_TIMEOUT_SECS:-120}"
gpu_mem="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
max_len="${VLLM_MAX_MODEL_LEN:-2048}"
dtype="${VLLM_DTYPE:-float16}"

tmp_cfg="$(mktemp)"
tmp_out="$(mktemp)"

cleanup() {
  rm -f "$tmp_cfg" "$tmp_out" 2>/dev/null || true
  if [[ -n "${pid:-}" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat >"$tmp_cfg" <<YAML
server:
  host: "127.0.0.1"
  port: ${jbox_port}

vllm:
  binary: "uvx"
  port: ${vllm_port}
  startup_timeout: ${startup_timeout}s
  shutdown_timeout: 30s
  drain_timeout: 60s
  swap_cooldown: 0s
  swap_wait_timeout: ${startup_timeout}s

behavior:
  rewrite_model_name: true

models:
  smoke:
    path: "${VLLM_MODEL}"
    tensor_parallel_size: 1
    gpu_memory_utilization: ${gpu_mem}
    max_model_len: ${max_len}
    dtype: "${dtype}"
YAML

bin/jukebox -config "$tmp_cfg" &
pid=$!

for _ in $(seq 1 80); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "jukebox exited early" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${jbox_port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

# Trigger model load + validate SSE streaming proxying.
payload='{"model":"smoke","stream":true,"messages":[{"role":"user","content":"Say hello in one short sentence."}]}'

deadline=$(( $(date +%s) + req_timeout ))
while true; do
  now=$(date +%s)
  if (( now > deadline )); then
    echo "timed out waiting for chat completion to succeed" >&2
    exit 1
  fi

  # We expect either:
  # - 200 with SSE output, or
  # - 503 during swap/startup, with Retry-After.
  status="$(curl -sS -o "$tmp_out" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST "http://127.0.0.1:${jbox_port}/v1/chat/completions" \
    --data "$payload" || true)"

  if [[ "$status" == "200" ]]; then
    break
  fi
  if [[ "$status" == "503" ]]; then
    sleep 1
    continue
  fi

  echo "unexpected status: $status" >&2
  cat "$tmp_out" >&2 || true
  exit 1
done

if ! rg -q "^data: " "$tmp_out"; then
  echo "expected SSE output (data: lines), got:" >&2
  head -n 50 "$tmp_out" >&2 || true
  exit 1
fi
if ! rg -q "data: \\[DONE\\]" "$tmp_out"; then
  echo "expected SSE DONE marker, got:" >&2
  tail -n 50 "$tmp_out" >&2 || true
  exit 1
fi

curl -fsS "http://127.0.0.1:${jbox_port}/status" >/dev/null
curl -fsS "http://127.0.0.1:${jbox_port}/metrics" >/dev/null

echo "vllm smoke ok"
