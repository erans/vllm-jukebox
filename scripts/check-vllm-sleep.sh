#!/usr/bin/env bash
# check-vllm-sleep.sh — preflight check for jukebox's sleep-mode integration.
#
# Probes a vLLM instance and reports whether the sleep API is properly
# exposed (i.e. the service was launched with both --enable-sleep-mode
# and VLLM_SERVER_DEV_MODE=1). Useful when bringing up a new vllm-*
# container before pointing jukebox at it.
#
# Usage:
#   scripts/check-vllm-sleep.sh <base_url>
#   scripts/check-vllm-sleep.sh http://vllm-main:8000
#
# Exits 0 if sleep mode is reachable and the endpoints behave correctly.
# Exits non-zero with a diagnostic if anything is off.

set -euo pipefail

BASE_URL="${1:-}"
if [ -z "$BASE_URL" ]; then
  echo "usage: $0 <vllm_base_url>" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

# 1. /health must return 2xx.
echo "═══ /health ═══"
if ! curl -fsS -o /dev/null -w '  %{http_code}\n' "$BASE_URL/health"; then
  red "  ✗ /health did not return 2xx — is vLLM running?"
  exit 1
fi
green "  ✓ /health 2xx"

# 2. /v1/models must return at least one model.
echo "═══ /v1/models ═══"
MODELS_JSON="$(curl -fsS "$BASE_URL/v1/models")"
MODEL_COUNT="$(printf '%s' "$MODELS_JSON" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("data",[])))')"
if [ "$MODEL_COUNT" -lt 1 ]; then
  red "  ✗ /v1/models returned 0 models — vLLM didn't load anything"
  exit 1
fi
MODEL_NAMES="$(printf '%s' "$MODELS_JSON" | python3 -c 'import sys,json;print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
green "  ✓ /v1/models served: $MODEL_NAMES"

# 3. /is_sleeping must exist (404 here = sleep mode NOT enabled).
echo "═══ /is_sleeping (sleep-API gate) ═══"
IS_SLEEPING_CODE="$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/is_sleeping")"
case "$IS_SLEEPING_CODE" in
  200)
    green "  ✓ /is_sleeping returned 200 — sleep API is exposed"
    IS_SLEEPING="$(curl -fsS "$BASE_URL/is_sleeping")"
    echo "    state: $IS_SLEEPING"
    ;;
  404)
    red "  ✗ /is_sleeping returned 404 — sleep API NOT exposed"
    yellow "    Fix: relaunch vLLM with both:"
    yellow "      --enable-sleep-mode   (CLI flag)"
    yellow "      VLLM_SERVER_DEV_MODE=1   (environment variable)"
    exit 1
    ;;
  *)
    red "  ✗ /is_sleeping returned unexpected status: $IS_SLEEPING_CODE"
    exit 1
    ;;
esac

# 4. Round-trip: sleep → /is_sleeping=true → wake → /is_sleeping=false.
# Skip if the model is already sleeping (we don't want to surprise an
# operator by fully cycling a hot model).
ALREADY_SLEEPING="$(printf '%s' "$IS_SLEEPING" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("is_sleeping",False))')"
if [ "$ALREADY_SLEEPING" = "True" ]; then
  yellow "  - skipping round-trip (model already sleeping)"
  exit 0
fi

echo "═══ round-trip: /sleep → /is_sleeping → /wake_up → /is_sleeping ═══"
echo "  POST /sleep?level=1"
SLEEP_CODE="$(curl -sS -X POST -o /dev/null -w '%{http_code}' "$BASE_URL/sleep?level=1")"
if [ "$SLEEP_CODE" != "200" ]; then
  red "    ✗ /sleep returned $SLEEP_CODE"
  exit 1
fi
green "    ✓ slept"

# Tiny delay for vLLM to settle.
sleep 1
SLEEPING_NOW="$(curl -fsS "$BASE_URL/is_sleeping" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("is_sleeping"))')"
if [ "$SLEEPING_NOW" != "True" ]; then
  red "    ✗ /is_sleeping says $SLEEPING_NOW after /sleep — expected True"
  red "      attempting recovery via /wake_up..."
  curl -sS -X POST "$BASE_URL/wake_up" -o /dev/null
  exit 1
fi
green "    ✓ /is_sleeping = True"

echo "  POST /wake_up"
WAKE_CODE="$(curl -sS -X POST -o /dev/null -w '%{http_code}' "$BASE_URL/wake_up")"
if [ "$WAKE_CODE" != "200" ]; then
  red "    ✗ /wake_up returned $WAKE_CODE — model may now be stuck asleep!"
  exit 1
fi
green "    ✓ woke"

# Poll /health until ready or timeout.
echo "  poll /health until 200 (timeout 30s)"
DEADLINE=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if curl -fsS -o /dev/null "$BASE_URL/health"; then
    green "    ✓ /health 200"
    break
  fi
  sleep 0.5
done
if ! curl -fsS -o /dev/null "$BASE_URL/health"; then
  red "    ✗ /health never returned 200 within 30s after /wake_up"
  exit 1
fi

SLEEPING_AFTER_WAKE="$(curl -fsS "$BASE_URL/is_sleeping" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("is_sleeping"))')"
if [ "$SLEEPING_AFTER_WAKE" != "False" ]; then
  red "    ✗ /is_sleeping says $SLEEPING_AFTER_WAKE after /wake_up — expected False"
  exit 1
fi
green "    ✓ /is_sleeping = False"

echo ""
green "✓ All checks passed — vLLM at $BASE_URL is sleep-mode-ready for jukebox"
