package vllmcli

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ErrWakeVerifyConfigError is returned by VerifyWakeWithProbe when the
// failure is structural — most commonly the served model name passed to
// the probe doesn't match what vLLM is actually serving, so vLLM returns
// a 404 / 400 NotFoundError. Recreating the container won't change the
// served-model-name (it's baked into the launch flags), so the caller
// MUST distinguish this from a transient phantom-wake: redeploying loops
// forever, while a config-error wants a loud log + operator action.
//
// Match via errors.Is(err, ErrWakeVerifyConfigError) at the call site.
var ErrWakeVerifyConfigError = errors.New("wake-verify config error (model name mismatch?)")

// ErrWakeVerifyPhantom is returned by VerifyWakeWithProbe when the
// probe confirmed the engine is wedged — HTTP timeout/hang, 5xx, empty
// choices, malformed JSON. These are the genuine phantom-wake signatures
// and the caller SHOULD trigger an async RedeployMember.
//
// Match via errors.Is(err, ErrWakeVerifyPhantom) at the call site.
var ErrWakeVerifyPhantom = errors.New("wake-verify phantom (engine wedged)")

// VerifyWakeWithProbe issues a tiny single-token /v1/completions request
// against baseURL to confirm that an apparently-successful /wake_up has
// actually produced a working engine (and is not a "phantom wake" where
// /wake_up returns 200 but the worker processes are wedged).
//
// servedModelName: the value vLLM is actually serving (i.e. what its
// --served-model-name flag was set to, or the path basename if unset).
// If this differs from the jukebox config key, vLLM returns 404 /
// NotFoundError and the probe interprets that as a CONFIG error rather
// than a phantom wake — see ErrWakeVerifyConfigError above.
//
// Background: vllm-project/vllm cumem_tag wake_up has been observed
// returning 200 from the API server while one or more PP/TP workers
// have hit "CUDA Error: invalid argument at cumem_allocator.cpp:169"
// during create_and_map. /health continues to return 200 (the FastAPI
// process is alive) but every subsequent inference request hangs forever
// because the shm_broadcast block is gone. The PR upstream that adds
// /health/decode (#45097) catches one slice of this; this probe catches
// the rest by exercising the actual decode path end-to-end with the
// smallest possible payload.
//
// Design choices:
//   - max_tokens: 1, temperature: 0 — minimizes compute; the goal is
//     "did decode return at all", not "is generation good".
//   - Unique prompt with crypto/rand suffix — defeats the prefix-cache
//     fast path so the probe really exercises the engine, not a cached
//     trie hit. A wedged engine fails on the cache miss path the same
//     way a real consumer request would.
//   - Timeout bound by ctx + the timeout parameter AND a hard
//     http.Client.Timeout (defense-in-depth above context cancellation).
//     A wedged vLLM that sends HTTP 200 headers then hangs on body write
//     does NOT reliably interrupt via context across all HTTP/1.1
//     server impls — the Client.Timeout is the last-line guarantee
//     that we don't hold the cold-load mutex past budget.
//   - Returns nil only on HTTP 2xx with at least one choice in the
//     response. Anything else is wrapped with the appropriate sentinel:
//     400/404 model-not-found → ErrWakeVerifyConfigError; everything
//     else (timeout, 5xx, malformed JSON, empty choices) →
//     ErrWakeVerifyPhantom.
//
// Note: this function does NOT call /health/decode — that's a complementary
// upstream signal already polled separately during the standard wake
// health-wait loop. This probe specifically catches the "lying /health"
// failure mode where /health is fine but decode is wedged.
func VerifyWakeWithProbe(ctx context.Context, baseURL, servedModelName string, timeout time.Duration) error {
	if timeout <= 0 {
		return fmt.Errorf("VerifyWakeWithProbe: timeout must be > 0")
	}
	if strings.TrimSpace(baseURL) == "" {
		return fmt.Errorf("VerifyWakeWithProbe: baseURL is empty")
	}
	if strings.TrimSpace(servedModelName) == "" {
		return fmt.Errorf("VerifyWakeWithProbe: servedModelName is empty")
	}

	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// 8 bytes of randomness = 16 hex chars = enough entropy to defeat
	// prefix-cache collisions even under heavy probe load.
	var nonce [8]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		// crypto/rand failure is essentially unheard of on Linux;
		// surface it as a probe failure rather than silently degrading
		// to a fixed prompt that would hit the prefix cache.
		return fmt.Errorf("VerifyWakeWithProbe: generate nonce: %w", err)
	}
	prompt := "wake-verify-" + hex.EncodeToString(nonce[:])

	body, err := json.Marshal(struct {
		Model       string  `json:"model"`
		Prompt      string  `json:"prompt"`
		MaxTokens   int     `json:"max_tokens"`
		Temperature float64 `json:"temperature"`
		Stream      bool    `json:"stream"`
	}{
		Model:       servedModelName,
		Prompt:      prompt,
		MaxTokens:   1,
		Temperature: 0,
		Stream:      false,
	})
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: marshal body: %w", err)
	}

	u := strings.TrimRight(baseURL, "/") + "/v1/completions"
	req, err := http.NewRequestWithContext(probeCtx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Per-call client with a hard Timeout that bounds the ENTIRE request
	// lifecycle (dial + TLS + write + body read). http.DefaultClient has
	// Timeout=0 — a wedged server that flushes headers then hangs on body
	// can outlive context cancellation on some HTTP/1.1 paths and leave
	// us holding coldLoadMu past budget. The +500ms slack is so context
	// cancellation fires first (yielding the clean "context deadline
	// exceeded" error) before the hard timeout's "Client.Timeout exceeded
	// while awaiting headers" kicks in.
	client := &http.Client{Timeout: timeout + 500*time.Millisecond}

	resp, err := client.Do(req)
	if err != nil {
		// Network-layer failures (timeout, connection refused, body-read
		// hang) all surface as transport errors. These are phantom-wake
		// signatures: the API server is alive enough to accept the TCP
		// connection or is fully unreachable. Either way, redeploy is
		// the correct response.
		return fmt.Errorf("%w: POST /v1/completions: %v", ErrWakeVerifyPhantom, err)
	}
	defer resp.Body.Close()

	// Bound the body read with the same client timeout — io.ReadAll on a
	// hanging body would block past the Client.Timeout in pathological
	// cases (chunked encoding with empty trailing chunks). The probeCtx
	// deadline still applies via the underlying request context.
	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("%w: read /v1/completions body: %v", ErrWakeVerifyPhantom, err)
	}

	// 400/404 with a model-not-found-shaped error is a config mismatch:
	// jukebox is sending model="vllm-main" but vLLM was launched with
	// --served-model-name="some-other-name" (or defaults to the path
	// basename). Redeploying the container won't change this — the
	// served-model-name is baked into launch flags. Surface as a
	// distinct sentinel so the caller logs loud and skips redeploy.
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusBadRequest {
		if looksLikeModelNotFound(rawBody) {
			return fmt.Errorf("%w: vLLM returned %s for model %q (body=%s)",
				ErrWakeVerifyConfigError, resp.Status, servedModelName, truncate(rawBody, 200))
		}
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%w: POST /v1/completions returned %s (body=%s)",
			ErrWakeVerifyPhantom, resp.Status, truncate(rawBody, 200))
	}

	var decoded struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Text         string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rawBody, &decoded); err != nil {
		return fmt.Errorf("%w: decode response: %v", ErrWakeVerifyPhantom, err)
	}
	if len(decoded.Choices) == 0 {
		return fmt.Errorf("%w: response had zero choices", ErrWakeVerifyPhantom)
	}
	// Don't require non-empty text — max_tokens:1 with temperature:0 on
	// some models legitimately emits an empty string (e.g. EOS as the
	// first sampled token). The fact that decode completed AT ALL is
	// what proves the engine isn't wedged.
	return nil
}

// looksLikeModelNotFound inspects a vLLM error body to decide whether a
// 400/404 is a model-name mismatch (vs some other transient 400 like a
// malformed request). vLLM's /v1/completions returns 400 with a JSON
// body containing "model" / "not found" / "NotFoundError" when the
// requested model name isn't in its registry. We match conservatively
// — if the body shape is ambiguous, fall through and treat it as a
// phantom-wake signal (the safer default, since redeploy at worst costs
// a cold-load while ignoring a real config-error loops forever).
func looksLikeModelNotFound(body []byte) bool {
	s := strings.ToLower(string(body))
	// Need "model" reference + a not-found-ish phrase. Either substring
	// alone is too noisy (a phantom-wake body can mention "model").
	hasModelRef := strings.Contains(s, "model")
	hasNotFound := strings.Contains(s, "not found") ||
		strings.Contains(s, "notfounderror") ||
		strings.Contains(s, "does not exist") ||
		strings.Contains(s, "is not available")
	return hasModelRef && hasNotFound
}

// truncate caps a byte slice for log readability. Returns the original
// string when under cap; otherwise prefix + "...(N bytes truncated)".
func truncate(b []byte, max int) string {
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + fmt.Sprintf("...(%d bytes truncated)", len(b)-max)
}
