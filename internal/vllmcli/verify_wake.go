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

	"vllm-jukebox/internal/config"
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

// ErrWakeVerifyEndpointUnsupported is returned by VerifyWakeWithProbe when
// the probe endpoint legitimately does not exist on the woken model — a
// route-level 404 ("Not Found" from the web framework, NOT a model-name
// 404 from the OpenAI-compat layer). This is the load-bearing defense for
// the RAG path: embedding models (bge-m3) and reranker models
// (mxbai-rerank-base-v2) are launched with `--task embed` and DO NOT
// register `/v1/completions`. Probing /v1/completions against them returns
// a bare framework 404 (`{"detail":"Not Found"}`) which the pre-fix code
// misread as a phantom-wedge → 503 + schedule-recreate on EVERY request
// (and the recreate then failed because evict_action=sleep). A wedged
// engine times out, 5xxs, or hangs — it never cleanly returns a route-404.
// So a route-404 is definitively NOT a wedge: the engine is alive enough to
// route, the requested verb just isn't served by this model type. The
// caller treats this as "wake confirmed healthy" (the engine answered) and
// MUST NOT redeploy.
//
// Match via errors.Is(err, ErrWakeVerifyEndpointUnsupported) at the call site.
var ErrWakeVerifyEndpointUnsupported = errors.New("wake-verify probe endpoint not supported by this model type")

// VerifyWakeWithProbe issues a tiny single-token request against baseURL
// to confirm that an apparently-successful /wake_up has actually produced
// a working engine (and is not a "phantom wake" where /wake_up returns 200
// but the worker processes are wedged).
//
// runner: the jukebox model Runner ("generate" or "pooling", per
// config.RunnerGenerate / config.RunnerPooling). It selects the probe
// endpoint so the probe exercises a verb the model actually serves:
//   - generate (or empty/unknown) → POST /v1/completions with a 1-token
//     decode request. This is the genuine cumem decode-path exercise.
//   - pooling → POST /v1/embeddings with a tiny input. Embedding and
//     reranker models are launched with `--task embed` and do NOT register
//     /v1/completions; probing the generative endpoint against them returns
//     a route-404 that the pre-fix code misclassified as a phantom-wedge,
//     black-holing the whole RAG path. The embeddings endpoint touches the
//     same cumem-managed forward pass, so a healthy pooling engine answers
//     2xx and a wedged one still times out / 5xxs.
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
// the rest by exercising the actual forward path end-to-end with the
// smallest possible payload.
//
// Design choices:
//   - max_tokens: 1, temperature: 0 (generate path) — minimizes compute;
//     the goal is "did decode return at all", not "is generation good".
//   - Unique input with crypto/rand suffix — defeats the prefix-cache
//     fast path so the probe really exercises the engine, not a cached
//     trie hit. A wedged engine fails on the cache miss path the same
//     way a real consumer request would.
//   - Timeout bound by ctx + the timeout parameter AND a hard
//     http.Client.Timeout (defense-in-depth above context cancellation).
//     A wedged vLLM that sends HTTP 200 headers then hangs on body write
//     does NOT reliably interrupt via context across all HTTP/1.1
//     server impls — the Client.Timeout is the last-line guarantee
//     that we don't hold the cold-load mutex past budget.
//   - Returns nil on HTTP 2xx with a valid-shaped response. A route-level
//     404 (the web framework's "Not Found", body like {"detail":"Not
//     Found"}) → ErrWakeVerifyEndpointUnsupported (NOT a wedge — see that
//     sentinel). A model-name 404/400 (OpenAI-compat NotFoundError) →
//     ErrWakeVerifyConfigError. Everything else (timeout, 5xx, malformed
//     JSON, empty result) → ErrWakeVerifyPhantom.
//
// Note: this function does NOT call /health/decode — that's a complementary
// upstream signal already polled separately during the standard wake
// health-wait loop. This probe specifically catches the "lying /health"
// failure mode where /health is fine but the engine is wedged.
func VerifyWakeWithProbe(ctx context.Context, baseURL, servedModelName, runner string, timeout time.Duration) error {
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
	probeInput := "wake-verify-" + hex.EncodeToString(nonce[:])

	// Pick the probe endpoint + payload by model type. Pooling models
	// (embeddings / rerankers, launched --task embed) do not serve
	// /v1/completions; sending one returns a route-404 that the phantom
	// detector would misread as a wedge. The embeddings endpoint touches
	// the same cumem forward pass, so it's an equivalent liveness probe.
	pooling := runner == config.RunnerPooling
	var endpoint string
	var body []byte
	var err error
	if pooling {
		endpoint = "/v1/embeddings"
		body, err = json.Marshal(struct {
			Model string `json:"model"`
			Input string `json:"input"`
		}{
			Model: servedModelName,
			Input: probeInput,
		})
	} else {
		endpoint = "/v1/completions"
		body, err = json.Marshal(struct {
			Model       string  `json:"model"`
			Prompt      string  `json:"prompt"`
			MaxTokens   int     `json:"max_tokens"`
			Temperature float64 `json:"temperature"`
			Stream      bool    `json:"stream"`
		}{
			Model:       servedModelName,
			Prompt:      probeInput,
			MaxTokens:   1,
			Temperature: 0,
			Stream:      false,
		})
	}
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: marshal body: %w", err)
	}

	u := strings.TrimRight(baseURL, "/") + endpoint
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
		return fmt.Errorf("%w: POST %s: %v", ErrWakeVerifyPhantom, endpoint, err)
	}
	defer resp.Body.Close()

	// Bound the body read with the same client timeout — io.ReadAll on a
	// hanging body would block past the Client.Timeout in pathological
	// cases (chunked encoding with empty trailing chunks). The probeCtx
	// deadline still applies via the underlying request context.
	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("%w: read %s body: %v", ErrWakeVerifyPhantom, endpoint, err)
	}

	// Route-level 404: the web framework (Starlette/FastAPI) returns a
	// bare {"detail":"Not Found"} when the requested PATH isn't a
	// registered route. This is exactly what an embedding/reranker model
	// returns for /v1/completions. It is NOT a wedge (a wedged engine
	// times out / 5xxs / hangs — it never cleanly route-404s) and it is
	// NOT a model-name mismatch (no "model" reference, no NotFoundError
	// type). The engine is alive enough to route; the verb just isn't
	// served by this model type. Treat as a non-fatal "endpoint
	// unsupported" so the caller routes the model normally instead of
	// black-holing it with a redeploy that would fail (sleep-evict
	// members aren't redeploy-eligible) and 503 every request. Checked
	// BEFORE the model-name 404 branch because a route-404 body never
	// satisfies looksLikeModelNotFound anyway, but ordering it first
	// makes the intent explicit.
	if resp.StatusCode == http.StatusNotFound && looksLikeRouteNotFound(rawBody) {
		return fmt.Errorf("%w: POST %s returned %s (body=%s) — model type %q does not serve this endpoint",
			ErrWakeVerifyEndpointUnsupported, endpoint, resp.Status, truncate(rawBody, 200), runner)
	}

	// 405 Method Not Allowed is another route-level signal that the verb
	// isn't served here (the path exists for a different method). Same
	// reasoning as the route-404: not a wedge, not a model mismatch.
	if resp.StatusCode == http.StatusMethodNotAllowed {
		return fmt.Errorf("%w: POST %s returned %s (body=%s)",
			ErrWakeVerifyEndpointUnsupported, endpoint, resp.Status, truncate(rawBody, 200))
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
		return fmt.Errorf("%w: POST %s returned %s (body=%s)",
			ErrWakeVerifyPhantom, endpoint, resp.Status, truncate(rawBody, 200))
	}

	// Validate the response envelope shape per endpoint. The fact that a
	// well-formed result came back AT ALL is what proves the engine isn't
	// wedged; we don't inspect the actual values.
	if pooling {
		var decoded struct {
			Data []json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(rawBody, &decoded); err != nil {
			return fmt.Errorf("%w: decode embeddings response: %v", ErrWakeVerifyPhantom, err)
		}
		if len(decoded.Data) == 0 {
			return fmt.Errorf("%w: embeddings response had zero data entries", ErrWakeVerifyPhantom)
		}
		return nil
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

// looksLikeRouteNotFound reports whether a 404 body is a web-framework
// route-not-found (the PATH isn't registered) rather than an OpenAI-compat
// model-not-found. vLLM is served by FastAPI/Starlette, whose default 404
// for an unregistered route is exactly `{"detail":"Not Found"}`. The
// distinguishing feature vs a model-name 404 is the ABSENCE of any model
// reference / NotFoundError type and the PRESENCE of the framework's
// "detail" envelope. We match conservatively: a body that mentions the
// model or carries the OpenAI error shape is NOT treated as a route-404
// (it falls through to looksLikeModelNotFound). An empty body on a 404 is
// also treated as a route-404 (some proxies strip the body) — still not a
// wedge signature.
func looksLikeRouteNotFound(body []byte) bool {
	s := strings.ToLower(strings.TrimSpace(string(body)))
	// A model-name mismatch body always references the model and/or the
	// OpenAI NotFoundError type. If those markers are present, this is NOT
	// a route-404 — let the model-not-found branch own it.
	if strings.Contains(s, "model") || strings.Contains(s, "notfounderror") {
		return false
	}
	if s == "" {
		return true
	}
	// Starlette/FastAPI default: {"detail":"not found"}. Match the
	// "detail" + "not found" shape, or a plain "not found" string.
	if strings.Contains(s, "not found") {
		return true
	}
	return false
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
