package httpserver_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
)

// TestSwitchingProxy_ColdLoadingModelReturns503AsyncKick is the canonical
// async-503 contract test for wake-from-Stopped (per Kagi PATCH 16).
//
// When the requested model is admissionStopped, the proxy handler MUST:
//  1. Return 503 with Retry-After: 60 immediately (no blocking). 60s
//     fits within OpenAI/Anthropic SDK retry budgets so clients re-poll
//     instead of surfacing user-visible errors. See handlers_proxy.go
//     for rationale (commit reverting 300→60).
//  2. Call Router.KickColdLoad to spawn the background cold-load.
//  3. NOT call Router.AcquireRoute (which would synchronously wake and
//     block the request for ~5 min — past client timeouts).
//
// The OpenAI error envelope must carry code="warming_up" so consumers
// can distinguish cold-load 503s from transient swap 503s.
func TestSwitchingProxy_ColdLoadingModelReturns503AsyncKick(t *testing.T) {
	router := &stubCoord{
		coldLoadModels: map[string]bool{"big-moe": true},
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"big-moe": {Path: "/models/big-moe"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"big-moe","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	// Status — 503.
	if resp.StatusCode != http.StatusServiceUnavailable {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 503 for cold-loading model, got %d (body: %s)", resp.StatusCode, body)
	}

	// Retry-After header — 60s. Fits inside OpenAI/Anthropic SDK retry
	// budgets so clients re-poll instead of surfacing user-visible errors.
	if got := resp.Header.Get("Retry-After"); got != "60" {
		t.Errorf("expected Retry-After: 60, got %q", got)
	}

	// Body — OpenAI error envelope with code=warming_up.
	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)
	if !strings.Contains(bodyStr, `"code":"warming_up"`) {
		t.Errorf("expected code=warming_up in body, got: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, `"type":"service_unavailable"`) {
		t.Errorf("expected type=service_unavailable in body, got: %s", bodyStr)
	}
	// The message should mention cold-starting / retry so consumers know
	// what to do (the specific wording is operational tuning, but the
	// keyword "cold" should be there).
	if !strings.Contains(strings.ToLower(bodyStr), "cold") {
		t.Errorf("expected body message to mention cold-start, got: %s", bodyStr)
	}

	// KickColdLoad must have been invoked exactly once with the right name.
	if router.kickCalls != 1 {
		t.Errorf("expected exactly 1 KickColdLoad call, got %d", router.kickCalls)
	}
	if router.lastKickedModel != "big-moe" {
		t.Errorf("expected KickColdLoad(%q), got %q", "big-moe", router.lastKickedModel)
	}

	// AcquireRoute must NOT have been called — the whole point of async-503
	// is to avoid the synchronous wake path.
	if router.acquireCalls != 0 {
		t.Errorf("expected AcquireRoute NOT called when model is cold-loading, got %d calls", router.acquireCalls)
	}
}

// TestSwitchingProxy_NonColdLoadingModelFallsThroughToAcquireRoute
// asserts the negative: admissionSleeping (not Stopped) models keep
// the existing fast sync wake path — DO NOT get 503'd. Mirrors the
// task spec's "if false, fall through to the existing synchronous path".
func TestSwitchingProxy_NonColdLoadingModelFallsThroughToAcquireRoute(t *testing.T) {
	// Backend that responds 200 to anything.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	router := &stubCoord{
		// coldLoadModels intentionally empty / nil → IsModelColdLoading
		// returns false for every name.
		baseURL:       backend.URL,
		upstreamModel: "/models/small-model",
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"small-model": {Path: "/models/small-model"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"small-model","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 fast-path, got %d (body: %s)", resp.StatusCode, body)
	}
	if router.kickCalls != 0 {
		t.Errorf("expected KickColdLoad NOT called for non-Stopped model, got %d", router.kickCalls)
	}
	if router.acquireCalls != 1 {
		t.Errorf("expected AcquireRoute called exactly once for fast-path, got %d", router.acquireCalls)
	}
}

// TestSwitchingProxy_ReprobeClears503AndFallsThroughToAcquireRoute is the
// Bug #3 fall-through contract: when admission still thinks a model is
// cold-loading (admissionStopped) but the on-request health re-probe
// (ReprobeStoppedExternal) discovers the external container is in fact
// healthy — recreated out-of-band and now answering /health=200 — the
// handler must NOT fast-fail with a stale "warming_up" 503. Instead the
// re-probe reconciles admission to Sleeping, returns true, and the
// handler falls THROUGH to AcquireRoute → backend → 200.
//
// This pins the precise behavior at handlers_proxy.go ~line 148-167:
//
//	reprobed := cl.IsModelColdLoading(modelName) && cl.ReprobeStoppedExternal(...)
//	if !reprobed { ... 503 warming_up ... }
//	// reprobed==true → fall through to AcquireRoute (normal wake).
//
// Critically, when the re-probe CLEARS the warming_up state we must NOT
// KickColdLoad — the container is already up; a kick would spawn a
// redundant (and potentially OOM-storming) cold-load of a model that no
// longer needs one. So: status 200, reprobeCalls==1, acquireCalls==1,
// kickCalls==0.
//
// Companion to TestSwitchingProxy_ColdLoadingModelReturns503AsyncKick
// (reprobeClears empty → re-probe returns false → 503 + kick), which
// exercises the OTHER side of the same branch.
func TestSwitchingProxy_ReprobeClears503AndFallsThroughToAcquireRoute(t *testing.T) {
	// Backend that responds 200 to anything — the fall-through path must
	// reach it via AcquireRoute.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","model":"m"}`))
	}))
	defer backend.Close()

	router := &stubCoord{
		// admission still reports m as cold-loading (Stopped)...
		coldLoadModels: map[string]bool{"m": true},
		// ...but the on-request re-probe finds the container healthy and
		// clears warming_up → handler must fall through to AcquireRoute.
		reprobeClears: map[string]bool{"m": true},
		baseURL:       backend.URL,
		upstreamModel: "/models/m",
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"m": {Path: "/models/m"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	// Status — 200, NOT 503. The re-probe cleared the stale warming_up
	// state so the request is served normally.
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 (reprobe cleared warming_up → fall-through to AcquireRoute), got %d (body: %s)", resp.StatusCode, body)
	}
	// No Retry-After — that header is only set on the 503 fast-fail path.
	if got := resp.Header.Get("Retry-After"); got != "" {
		t.Errorf("expected no Retry-After (no 503), got %q", got)
	}

	// The re-probe must have been consulted exactly once.
	if router.reprobeCalls != 1 {
		t.Errorf("expected exactly 1 ReprobeStoppedExternal call, got %d", router.reprobeCalls)
	}
	// Fall-through reached AcquireRoute exactly once.
	if router.acquireCalls != 1 {
		t.Errorf("expected exactly 1 AcquireRoute call (fall-through path), got %d", router.acquireCalls)
	}
	if router.lastModel != "m" {
		t.Errorf("expected AcquireRoute(m), got %q", router.lastModel)
	}
	// KickColdLoad must NOT fire — the container is already healthy; kicking
	// a fresh cold-load here would be a redundant OOM-storm risk.
	if router.kickCalls != 0 {
		t.Errorf("expected KickColdLoad NOT called when reprobe cleared warming_up, got %d", router.kickCalls)
	}
}

// TestAnthropicProxy_ColdLoadingModelReturns529 mirrors the OpenAI test
// for the Anthropic endpoint. The error envelope differs (Anthropic
// shape) AND the status code differs: Anthropic upstream returns 529
// for overloaded_error, not 503. The Retry-After + KickColdLoad
// contract is identical to the OpenAI side.
func TestAnthropicProxy_ColdLoadingModelReturns529(t *testing.T) {
	router := &stubCoord{
		coldLoadModels: map[string]bool{"big-moe": true},
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"big-moe": {Path: "/models/big-moe"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"big-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":10}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	// 529 = Anthropic's overloaded_error code (not a stdlib http.StatusXxx).
	if resp.StatusCode != 529 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 529 (Anthropic overloaded_error), got %d (body: %s)", resp.StatusCode, body)
	}
	if got := resp.Header.Get("Retry-After"); got != "60" {
		t.Errorf("expected Retry-After: 60, got %q", got)
	}
	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)
	// Anthropic envelope: top-level type=error, error.type=overloaded_error
	if !strings.Contains(bodyStr, `"type":"overloaded_error"`) {
		t.Errorf("expected overloaded_error envelope, got: %s", bodyStr)
	}

	if router.kickCalls != 1 {
		t.Errorf("expected 1 KickColdLoad call, got %d", router.kickCalls)
	}
	if router.acquireCalls != 0 {
		t.Errorf("expected AcquireRoute NOT called, got %d", router.acquireCalls)
	}
}

// minimalRouter implements ONLY the required Router interface — NOT
// the optional ColdLoadAware extension. Used to exercise the
// source-compatibility fall-through: a router without
// IsModelColdLoading/KickColdLoad must NOT crash or wedge the proxy
// handler; instead the handler skips the async-503 branch entirely
// and falls through to AcquireRoute as before.
type minimalRouter struct {
	st            jukebox.Status
	acquireCalls  int
	lastModel     string
	baseURL       string
	upstreamModel string
}

func (m *minimalRouter) Status() jukebox.Status { return m.st }

func (m *minimalRouter) AcquireRoute(_ context.Context, requestedModel, _ string) (jukebox.Route, error) {
	m.acquireCalls++
	m.lastModel = requestedModel
	return jukebox.Route{
		BaseURL:       m.baseURL,
		UpstreamModel: m.upstreamModel,
		Done:          func() {},
	}, nil
}

// Compile-time assertion: minimalRouter satisfies Router. The negative
// assertion (NOT ColdLoadAware) is enforced at runtime in each test
// below — Go can't express "does not implement X" as a compile-time
// constraint.
var _ jukebox.Router = (*minimalRouter)(nil)

// TestSwitchingProxy_RouterWithoutColdLoadAware_SkipsAsyncPath asserts
// the source-compatibility contract: a Router that does NOT implement
// the optional ColdLoadAware interface must not break the proxy.
//
// The handler's type-assert (cl, ok := opts.Router.(jukebox.ColdLoadAware))
// must short-circuit when the router doesn't satisfy the optional
// interface, falling through to AcquireRoute exactly as a pre-Phase-4
// router would. No 503-warming-up response, no Retry-After header, no
// KickColdLoad call (it doesn't exist on this router). The request
// proceeds to the fast path and gets back whatever AcquireRoute + the
// upstream backend produce.
func TestSwitchingProxy_RouterWithoutColdLoadAware_SkipsAsyncPath(t *testing.T) {
	// Guard the invariant at runtime — if a future refactor accidentally
	// adds IsModelColdLoading/KickColdLoad to minimalRouter, this test
	// loses its teeth.
	if _, isColdLoadAware := any(&minimalRouter{}).(jukebox.ColdLoadAware); isColdLoadAware {
		t.Fatalf("test setup bug: minimalRouter should NOT implement ColdLoadAware")
	}

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	router := &minimalRouter{
		baseURL:       backend.URL,
		upstreamModel: "/models/m",
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"m": {Path: "/models/m"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	// Must NOT have been 503'd from the cold-load branch — the type
	// assertion failed so the branch is skipped entirely. Fall-through
	// to AcquireRoute → backend → 200.
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 (cold-load branch skipped), got %d (body: %s)", resp.StatusCode, body)
	}
	// Retry-After header must be empty — the cold-load branch was the
	// only thing that would have set it on a happy-path request.
	if got := resp.Header.Get("Retry-After"); got != "" {
		t.Errorf("expected no Retry-After (no cold-load 503), got %q", got)
	}
	// AcquireRoute must have been the path taken.
	if router.acquireCalls != 1 {
		t.Errorf("expected exactly 1 AcquireRoute call (fall-through path), got %d", router.acquireCalls)
	}
	if router.lastModel != "m" {
		t.Errorf("expected AcquireRoute(m), got %q", router.lastModel)
	}
}

// TestAnthropicProxy_RouterWithoutColdLoadAware_SkipsAsyncPath mirrors
// the OpenAI test for the Anthropic endpoint — same source-compatibility
// contract, just with the Anthropic envelope. A minimalRouter (no
// ColdLoadAware) must fall through to AcquireRoute without 529'ing or
// setting Retry-After.
func TestAnthropicProxy_RouterWithoutColdLoadAware_SkipsAsyncPath(t *testing.T) {
	if _, isColdLoadAware := any(&minimalRouter{}).(jukebox.ColdLoadAware); isColdLoadAware {
		t.Fatalf("test setup bug: minimalRouter should NOT implement ColdLoadAware")
	}

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"model":"m"}`))
	}))
	defer backend.Close()

	router := &minimalRouter{
		baseURL:       backend.URL,
		upstreamModel: "/models/m",
	}
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"m": {Path: "/models/m"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Config: cfg, Router: router})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":10}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 (cold-load branch skipped), got %d (body: %s)", resp.StatusCode, body)
	}
	if got := resp.Header.Get("Retry-After"); got != "" {
		t.Errorf("expected no Retry-After (no cold-load 529), got %q", got)
	}
	if router.acquireCalls != 1 {
		t.Errorf("expected exactly 1 AcquireRoute call (fall-through path), got %d", router.acquireCalls)
	}
}
