#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mkdir -p bin
go build -o bin/jukebox ./cmd/jukebox

port="$(
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

tmp_cfg="$(mktemp)"

sed -e 's/^  host: "0\.0\.0\.0"$/  host: "127.0.0.1"/' -e "s/^  port: 8080$/  port: ${port}/" configs/example.yaml >"$tmp_cfg"

bin/jukebox -config "$tmp_cfg" &
pid=$!

cleanup() {
  rm -f "$tmp_cfg" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "jukebox exited early" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
curl -fsS "http://127.0.0.1:${port}/status" >/dev/null
curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null

echo "smoke ok"
