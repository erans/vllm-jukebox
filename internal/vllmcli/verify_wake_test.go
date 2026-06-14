package vllmcli_test

import (
	"context"
	"encoding/json"
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

	if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "test-model", 2*time.Second); err != nil {
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
		if err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", time.Second); err != nil {
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
	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", 250*time.Millisecond)
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

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on 500; got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to mention HTTP status; got %v", err)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("expected exactly one POST; got %d", got)
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

	err := vllmcli.VerifyWakeWithProbe(context.Background(), srv.URL, "m", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error on empty choices; got nil")
	}
	if !strings.Contains(err.Error(), "zero choices") {
		t.Errorf("expected 'zero choices' in error message; got %v", err)
	}
}

// TestVerifyWakeWithProbe_EmptyBaseURLRejected proves the input
// validation: an empty baseURL fails fast rather than firing an HTTP
// request against the empty string. Defends the production wake-path's
// "skip when baseURL is empty" gate from accidentally triggering this
// code path on a regression.
func TestVerifyWakeWithProbe_EmptyBaseURLRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "", "m", time.Second)
	if err == nil || !strings.Contains(err.Error(), "baseURL") {
		t.Fatalf("expected baseURL validation error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_EmptyModelRejected proves model name is required.
func TestVerifyWakeWithProbe_EmptyModelRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "http://example.test", "", time.Second)
	if err == nil || !strings.Contains(err.Error(), "model") {
		t.Fatalf("expected model validation error; got %v", err)
	}
}

// TestVerifyWakeWithProbe_ZeroTimeoutRejected proves the timeout floor:
// a zero/negative budget is rejected to defend against silent-no-op
// regressions (a zero-deadline ctx returns immediately, which would
// make every probe call false-pass).
func TestVerifyWakeWithProbe_ZeroTimeoutRejected(t *testing.T) {
	err := vllmcli.VerifyWakeWithProbe(context.Background(), "http://example.test", "m", 0)
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("expected timeout validation error; got %v", err)
	}
}
