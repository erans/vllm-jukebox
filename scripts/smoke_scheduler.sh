#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mkdir -p bin
go build -o bin/jukebox ./cmd/jukebox

jbox_port="$(
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

# Find a range of consecutive free ports for vLLM instances.
range="$(
  python3 - <<'PY'
import random, socket

def range_free(start, count):
    socks = []
    try:
        for p in range(start, start + count):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", p))
            socks.append(s)
        return True
    except OSError:
        return False
    finally:
        for s in socks:
            s.close()

COUNT = 8
for _ in range(200):
    start = random.randint(20000, 40000)
    if range_free(start, COUNT):
        print(f"{start} {start+COUNT-1}")
        raise SystemExit(0)
raise SystemExit("failed to find free port range")
PY
)"

range_start="$(awk '{print $1}' <<<"$range")"
range_end="$(awk '{print $2}' <<<"$range")"

tmp_dir="$(mktemp -d)"
tmp_cfg="${tmp_dir}/cfg.yaml"
fake_smi="${tmp_dir}/fake_nvidia_smi.sh"
fake_vllm="${tmp_dir}/fake_vllm.py"
fake_vllm_launcher="${tmp_dir}/fake_vllm_launcher.sh"

cleanup() {
  rm -rf "$tmp_dir" 2>/dev/null || true
  if [[ -n "${pid:-}" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat >"$fake_smi" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
# Emulate: nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits
echo "0, 100000, 100000"
echo "1, 100000, 100000"
SH
chmod +x "$fake_smi"

cat >"$fake_vllm" <<'PY'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

def parse_args(argv):
    # vllm-jukebox calls: serve <model_path> --host 127.0.0.1 --port <port> ...
    if len(argv) < 3 or argv[1] != "serve":
        raise SystemExit(f"expected: serve <model> ...; got {argv!r}")
    model = argv[2]
    port = None
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    if port is None:
        raise SystemExit("missing --port")
    return model, port

MODEL, PORT = parse_args(sys.argv)

class Handler(BaseHTTPRequestHandler):
    def _write_json(self, code, obj, content_type="application/json"):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._write_json(200, {"object":"list","data":[{"id": MODEL, "object":"model"}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._write_json(200, {"id":"x","object":"chat.completion","model": MODEL})
            return
        if self.path == "/v1/embeddings":
            self._write_json(200, {"object":"list","data":[],"model": MODEL})
            return
        if self.path in ("/v1/tokenize", "/v1/detokenize"):
            self._write_json(200, {"ok": True, "model": MODEL})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args, **_kwargs):
        # quiet
        return

httpd = HTTPServer(("127.0.0.1", PORT), Handler)
httpd.serve_forever()
PY

cat >"$fake_vllm_launcher" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exec python3 "${FAKE_VLLM_PY}" "$@"
SH
chmod +x "$fake_vllm_launcher"

cat >"$tmp_cfg" <<YAML
server:
  host: "127.0.0.1"
  port: ${jbox_port}

vllm:
  binary: "${fake_vllm_launcher}"
  startup_timeout: 5s
  shutdown_timeout: 2s
  drain_timeout: 2s
  swap_wait_timeout: 5s

scheduler:
  port_range_start: ${range_start}
  port_range_end: ${range_end}
  max_instances: 3
  min_instance_uptime: 0s
  nvidia_smi_binary: "${fake_smi}"

behavior:
  rewrite_model_name: true

models:
  small0:
    path: "/models/small0"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
  small1:
    path: "/models/small1"
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
  big:
    path: "/models/big"
    gpus: [0,1]
    min_free_mem_mb_per_gpu: 10
YAML

FAKE_VLLM_PY="$fake_vllm" bin/jukebox -config "$tmp_cfg" &
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

curl -fsS "http://127.0.0.1:${jbox_port}/health" >/dev/null

# Start two small models.
curl -fsS "http://127.0.0.1:${jbox_port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"small0","messages":[{"role":"user","content":"hi"}]}' >/dev/null

curl -fsS "http://127.0.0.1:${jbox_port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"small1","messages":[{"role":"user","content":"hi"}]}' >/dev/null

# Request big model; should evict the small models and start big.
curl -fsS "http://127.0.0.1:${jbox_port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big","messages":[{"role":"user","content":"hi"}]}' >/dev/null

curl -fsS "http://127.0.0.1:${jbox_port}/status" | python3 -c '
import json, sys
raw = sys.stdin.read()
if not raw.strip():
  print("empty /status response", file=sys.stderr)
  sys.exit(1)
st = json.loads(raw)
instances = st.get("instances") or []
models = sorted([i.get("model") for i in instances if isinstance(i, dict)])
if "big" not in models:
  print("expected big instance in status; got models=", models, file=sys.stderr)
  sys.exit(1)
if "small0" in models or "small1" in models:
  print("expected small models evicted; got models=", models, file=sys.stderr)
  sys.exit(1)
'

curl -fsS "http://127.0.0.1:${jbox_port}/metrics" >/dev/null

echo "scheduler smoke ok"
