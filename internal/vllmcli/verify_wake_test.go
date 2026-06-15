package vllmcli_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/vllmcli"
)

// TestVerifyWakeWithProbe_SuccessOnHealthyDecode proves the happy path:
// a 200 OK response with at least one choice means the engine is alive.
func TestVerifyWakeWithProbe_SuccessOnHealthyDecode(t *testing.T) {
	var seenPath, seenMethod string
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		seenMethod = r.Method
		seenBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"text":"x","finish_reason":"length"}]}`)
	}))
	defer srv.Close()

	if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "test-model", "", 2*time.Second); err != nil {
		t.Fatalf("VerifyWakeWithProbe: %v", err)
	}
	if seenPath != "/v1/completions" {
		t.Errorf("expected POST to /v1/completions; got %s", seenPath)
	}
	if seenMethod != http.MethodPost {
		t.Errorf("expected POST method; got %s", seenMethod)
	}
	// Verify the request body shape: must carry max_tokens:1 + a
	// unique-per-call prompt + the model id we supplied.
	var decoded struct {
		Model     string `json:"model"`
		Prompt    string `json:"prompt"`
		MaxTokens int    `json:"max_tokens"`
	}
	if err := json.Unmarshal(seenBody, &decoded); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if decoded.Model != "test-model" {
		t.Errorf("expected model=test-model; got %q", decoded.Model)
	}
	if decoded.MaxTokens != 1 {
		t.Errorf("expected max_tokens=1; got %d", decoded.MaxTokens)
	}
	if !strings.HasPrefix(decoded.Prompt, "wake-verify-") {
		t.Errorf("expected prompt prefixed with 'wake-verify-' for cache-defeat; got %q", decoded.Prompt)
	}
	if len(decoded.Prompt) <= len("wake-verify-") {
		t.Errorf("expected nonce-suffixed prompt; got bare prefix %q", decoded.Prompt)
	}
}

// TestVerifyWakeWithProbe_PromptUniquePerCall proves the prefix-cache-
// defeating nonce is actually random across consecutive invocations. A
// hardcoded prompt would let prefix-cache mask a wedged engine on
// retries (cached trie hit doesn't exercise decode).
func TestVerifyWakeWithProbe_PromptUniquePerCall(t *testing.T) {
	prompts := make(map[string]struct{}, 8)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var decoded struct {
			Prompt string `json:"prompt"`
		}
		_ = json.Unmarshal(body, &decoded)
		prompts[decoded.Prompt] = struct{}{}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"text":"x"}]}`)
	}))
	defer srv.Close()

	for i := 0; i < 8; i++ {
		if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", "", time.Second); err != nil {
			t.Fatalf("iter %d: %v", i, err)
		}
	}
	if len(prompts) != 8 {
		t.Errorf("expected 8 unique prompts across 8 invocations; got %d unique (prefix-cache risk if duplicates)", len(prompts))
	}
}

// TestVerifyWakeWithProbe_PhantomOnTimeout proves that a hanging /v1/completions
// (the canonical phantom-wake signature — endpoint accepts the request
// but workers are wedged so it never decodes) is detected as a failed
// verification within the timeout budget.
//
// We use a server that holds the response open until a test-controlled
// release channel fires. The client's context deadline expires first
// and trips the verify timeout; the test then releases the handler so
// httptest.Server.Close() doesn't block teardown waiting on the
// goroutine.
func TestVerifyWakeWithProbe_PhantomOnTimeout(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	start := time.Now()
	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", "", 250*time.Millisecond)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("expected timeout error on wedged decode; got nil")
	}
	if elapsed < 200*time.Millisecond {
		t.Errorf("returned too fast (%s) — did the timeout enforce?", elapsed)
	}
	if elapsed > 2*time.Second {
		t.Errorf("returned too slow (%s) — timeout budget exceeded by >> 1.75s", elapsed)
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom sentinel on timeout; got %v", err)
	}
}

// TestVerifyWakeWithProbe_PhantomOnHeadersThenHang covers HIGH-2 from
// the adversarial review: a wedged vLLM that sends HTTP 200 status +
// headers, then never sends a body. http.DefaultClient has Timeout=0;
// context cancellation does NOT reliably interrupt body reads across
// all HTTP/1.1 paths (the server can flush headers and hold the
// connection open well past ctx.Done()). The fix uses a per-call
// http.Client with Timeout = budget + 500ms slack, which DOES bound
// the entire request lifecycle including the body read.
//
// Without the fix this test takes ~60s and would fail the elapsed
// budget. With the fix, the probe must return within budget + slack
// (~1.5s for a 1s budget).
func TestVerifyWakeWithProbe_PhantomOnHeadersThenHang(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Flush 200 OK + headers immediately, then hang on body write
		// for 60s. Mimics a wedged vLLM that accepts the request,
		// returns control to the FastAPI handler shell which writes
		// the response start, then deadlocks before any body bytes
		// can be produced by the wedged decode worker.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		select {
		case <-r.Context().Done():
		case <-time.After(60 * time.Second):
		}
	}))
	defer srv.Close()

	start := time.Now()
	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", "", 1*time.Second)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("expected error on headers-then-hang wedge; got nil")
	}
	// MUST return within budget + http.Client.Timeout slack (1s + 500ms)
	// plus a little scheduling jitter. If this exceeds 2s, the
	// http.DefaultClient regression is back.
	if elapsed > 2*time.Second {
		t.Errorf("probe took %s for headers-then-hang — http.Client.Timeout not enforcing (HIGH-2 regression?)", elapsed)
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom sentinel on headers-then-hang; got %v", err)
	}
}

// TestVerifyWakeWithProbe_PhantomOn5xx proves a 5xx from /v1/completions
// (the failure mode where the server-side decode loop has crashed but
// the API server is still serving) is detected as a failed verification.
func TestVerifyWakeWithProbe_PhantomOn5xx(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error":"engine wedged"}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", "", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on 500; got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to mention HTTP status; got %v", err)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("expected exactly one POST; got %d", got)
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom sentinel on 500; got %v", err)
	}
}

// TestVerifyWakeWithProbe_PhantomOnEmptyChoices proves that an HTTP 200
// with a malformed payload (zero choices) is detected as a phantom. This
// catches the rare case where the engine half-recovers and returns a
// well-formed JSON envelope with no actual decode output.
func TestVerifyWakeWithProbe_PhantomOnEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[]}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", "", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on empty choices; got nil")
	}
	if !strings.Contains(err.Error(), "zero choices") {
		t.Errorf("expected 'zero choices' in error message; got %v", err)
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom sentinel on empty choices; got %v", err)
	}
}

// TestVerifyWakeWithProbe_ConfigErrorOn404ModelNotFound covers HIGH-3
// from the adversarial review: if the probe's `model` field doesn't
// match what vLLM is serving (--served-model-name mismatch), vLLM
// returns 404 with a model-not-found error body. The pre-fix behavior
// classified this as a phantom and triggered RedeployMember, which
// recreates the container with the SAME --served-model-name flag,
// hits the same 404, redeploys again — infinite loop.
//
// Post-fix: this MUST return ErrWakeVerifyConfigError so the caller
// can log loud + suppress redeploy.
func TestVerifyWakeWithProbe_ConfigErrorOn404ModelNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":{"message":"The model 'vllm-main' does not exist.","type":"NotFoundError","code":404}}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "vllm-main", "", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on 404 model-not-found; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyConfigError) {
		t.Errorf("expected ErrWakeVerifyConfigError sentinel on 404 model-not-found; got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("config error MUST NOT also be classified as phantom (would trigger redeploy loop); got %v", err)
	}
}

// TestVerifyWakeWithProbe_ConfigErrorOn400ModelNotFound covers the
// vLLM-actually-returns-400-not-404 path. The OpenAI-compat layer in
// vLLM returns 400 with NotFoundError in the body for unknown models
// (this is observed behavior, despite 404 being more semantically
// correct).
func TestVerifyWakeWithProbe_ConfigErrorOn400ModelNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"object":"error","message":"The model `+"`"+`vllm-main`+"`"+` does not exist.","type":"NotFoundError","code":404}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "vllm-main", "", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on 400 NotFoundError; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyConfigError) {
		t.Errorf("expected ErrWakeVerifyConfigError sentinel on 400 NotFoundError; got %v", err)
	}
}

// TestVerifyWakeWithProbe_4xxWithoutModelHintIsPhantom proves the
// conservative fall-through: a 400 that doesn't look like a model-name
// mismatch is treated as a phantom (the safer default — redeploy at
// worst costs a cold-load; ignoring a real wedge keeps requests hanging).
func TestVerifyWakeWithProbe_4xxWithoutModelHintIsPhantom(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"malformed request: expected field foo"}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "vllm-main", "", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on generic 400; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom (conservative fall-through for 4xx without model hint); got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyConfigError) {
		t.Errorf("generic 400 must NOT be classified as config-error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_ServedNameUsedInRequest proves the
// servedModelName parameter is actually what's sent on the wire (not
// some stale internal name). HIGH-3 fix: jukebox config key may differ
// from --served-model-name; caller passes the served name through.
func TestVerifyWakeWithProbe_ServedNameUsedInRequest(t *testing.T) {
	var seenModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var decoded struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &decoded)
		seenModel = decoded.Model
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"text":"x"}]}`)
	}))
	defer srv.Close()

	const served = "Qwen/Qwen3-32B-AWQ"
	if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, served, "", time.Second); err != nil {
		t.Fatalf("probe: %v", err)
	}
	if seenModel != served {
		t.Errorf("expected vLLM to receive served-model-name %q in body; got %q", served, seenModel)
	}
}

// TestVerifyWakeWithProbe_EmptyBaseURLRejected proves the input
// validation: an empty baseURL fails fast rather than firing an HTTP
// request against the empty string. Defends the production wake-path's
// "skip when baseURL is empty" gate from accidentally triggering this
// code path on a regression.
func TestVerifyWakeWithProbe_EmptyBaseURLRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "", "m", "", time.Second)
	if err == nil || !strings.Contains(err.Error(), "baseURL") {
		t.Fatalf("expected baseURL validation error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_EmptyModelRejected proves model name is required.
func TestVerifyWakeWithProbe_EmptyModelRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "http://example.test", "", "", time.Second)
	if err == nil || !strings.Contains(err.Error(), "servedModelName") {
		t.Fatalf("expected servedModelName validation error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_ZeroTimeoutRejected proves the timeout floor:
// a zero/negative budget is rejected to defend against silent-no-op
// regressions (a zero-deadline ctx returns immediately, which would
// make every probe call false-pass).
func TestVerifyWakeWithProbe_ZeroTimeoutRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "http://example.test", "m", "", 0)
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("expected timeout validation error; got %v", err)
	}
}

// --- model-type-aware probe (RAG-path black-hole fix) ---

// TestVerifyWakeWithProbe_PoolingProbesEmbeddings proves the load-bearing
// fix: a pooling model (runner: "pooling" — an embedding model like
// bge-m3) is probed against /v1/embeddings, NOT /v1/completions. Pre-fix
// the probe always hit /v1/completions, which pooling models don't serve
// (they're launched --task embed) → route-404 → misclassified phantom →
// 503 + schedule-recreate on every request, black-holing the RAG path.
func TestVerifyWakeWithProbe_PoolingProbesEmbeddings(t *testing.T) {
	var seenPath string
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		seenBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`)
	}))
	defer srv.Close()

	if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "bge-m3", "pooling", 2*time.Second); err != nil {
		t.Fatalf("pooling probe against healthy embeddings endpoint should succeed; got %v", err)
	}
	if seenPath != "/v1/embeddings" {
		t.Errorf("expected pooling model to be probed at /v1/embeddings; got %s", seenPath)
	}
	var decoded struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}
	if err := json.Unmarshal(seenBody, &decoded); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if decoded.Model != "bge-m3" {
		t.Errorf("expected model=bge-m3 in embeddings body; got %q", decoded.Model)
	}
	if !strings.HasPrefix(decoded.Input, "wake-verify-") {
		t.Errorf("expected cache-defeating input prefixed 'wake-verify-'; got %q", decoded.Input)
	}
}

// TestVerifyWakeWithProbe_PoolingEmptyDataIsPhantom proves a pooling model
// that returns HTTP 200 with zero data entries (half-recovered engine) is
// still detected as a phantom — the type-aware path doesn't weaken the
// wedge detection for pooling models.
func TestVerifyWakeWithProbe_PoolingEmptyDataIsPhantom(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[]}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "bge-m3", "pooling", 2*time.Second)
	if err == nil {
		t.Fatalf("expected phantom on empty embeddings data; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("expected ErrWakeVerifyPhantom on zero data entries; got %v", err)
	}
}

// TestVerifyWakeWithProbe_RouteNotFoundIsEndpointUnsupported is the
// core defense-in-depth assertion. This reproduces the EXACT live failure:
// a reranker/embedding model whose verify probe hits an endpoint it
// doesn't serve returns the FastAPI/Starlette route-404 body
// {"detail":"Not Found"}. Pre-fix this was classified as a phantom-wedge
// → 503 + schedule-recreate (which then failed for sleep-evict members).
// Post-fix it MUST be ErrWakeVerifyEndpointUnsupported (NOT phantom, NOT
// config-error) so the caller routes the model normally.
func TestVerifyWakeWithProbe_RouteNotFoundIsEndpointUnsupported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"detail":"Not Found"}`)
	}))
	defer srv.Close()

	// Probe a pooling model whose /v1/embeddings somehow route-404s (or a
	// reranker that serves neither verb we probe) — the backstop must fire.
	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "mxbai-rerank-base-v2", "pooling", 2*time.Second)
	if err == nil {
		t.Fatalf("expected endpoint-unsupported error on route-404; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyEndpointUnsupported) {
		t.Errorf("expected ErrWakeVerifyEndpointUnsupported on {\"detail\":\"Not Found\"}; got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("route-404 MUST NOT be classified as phantom (would 503 + schedule-recreate the RAG path); got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyConfigError) {
		t.Errorf("route-404 MUST NOT be classified as config-error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_GenerativeRouteNotFoundIsEndpointUnsupported
// proves the same backstop on the generative path: even if a generative
// model's /v1/completions somehow route-404s, that bare {"detail":"Not
// Found"} is not a wedge signature.
func TestVerifyWakeWithProbe_GenerativeRouteNotFoundIsEndpointUnsupported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"detail":"Not Found"}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "qwen3", "generate", 2*time.Second)
	if err == nil {
		t.Fatalf("expected endpoint-unsupported error on generative route-404; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyEndpointUnsupported) {
		t.Errorf("expected ErrWakeVerifyEndpointUnsupported; got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("route-404 MUST NOT be phantom; got %v", err)
	}
}

// TestVerifyWakeWithProbe_GenerativeTimeoutStillPhantom is the negative
// control paired with the route-404 test: a GENERATIVE model whose probe
// hangs (the genuine cumem-wedge signature) MUST still be flagged phantom.
// This guards against the fix accidentally swallowing real wedges.
func TestVerifyWakeWithProbe_GenerativeTimeoutStillPhantom(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "qwen3", "generate", 250*time.Millisecond)
	if err == nil {
		t.Fatalf("expected phantom on wedged generative decode; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyPhantom) {
		t.Errorf("generative wedge MUST still be ErrWakeVerifyPhantom; got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyEndpointUnsupported) {
		t.Errorf("a timeout is a wedge, not endpoint-unsupported; got %v", err)
	}
}

// TestVerifyWakeWithProbe_ModelNotFoundStillConfigError guards that the
// new route-404 branch doesn't steal the model-name-mismatch 404 (which
// references the model + NotFoundError and MUST stay ErrWakeVerifyConfigError).
func TestVerifyWakeWithProbe_ModelNotFoundStillConfigError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":{"message":"The model 'qwen3' does not exist.","type":"NotFoundError","code":404}}`)
	}))
	defer srv.Close()

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "qwen3", "generate", 2*time.Second)
	if err == nil {
		t.Fatalf("expected config error on model-not-found 404; got nil")
	}
	if !errors.Is(err, vllmcli.ErrWakeVerifyConfigError) {
		t.Errorf("model-name 404 must stay ErrWakeVerifyConfigError; got %v", err)
	}
	if errors.Is(err, vllmcli.ErrWakeVerifyEndpointUnsupported) {
		t.Errorf("model-name 404 must NOT be reclassified as endpoint-unsupported; got %v", err)
	}
}
