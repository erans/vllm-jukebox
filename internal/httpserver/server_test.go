package httpserver_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
)

type stubCoord struct {
	st jukebox.Status

	ensureErr     error
	ensureCalls   int
	lastModel     string
	lastRequestID string
}

func (s stubCoord) Status() jukebox.Status { return s.st }

func (s *stubCoord) EnsureModel(ctx context.Context, requestedModel, requestID string) error {
	s.ensureCalls++
	s.lastModel = requestedModel
	s.lastRequestID = requestID
	return s.ensureErr
}

func TestHealth_ReadyReturns200AndAcceptingRequests(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady, CurrentModel: "m", PID: 123}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if resp.Header.Get("X-Request-ID") == "" {
		t.Fatalf("expected X-Request-ID header to be set")
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["accepting_requests"] != true {
		t.Fatalf("expected accepting_requests=true, got %v", body["accepting_requests"])
	}
}

func TestHealth_SchemaMatchesSpec(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady, CurrentModel: "m", PID: 123, UptimeSeconds: 7}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	type healthResp struct {
		Status            string `json:"status"`
		AcceptingRequests bool   `json:"accepting_requests"`
		VLLM              struct {
			State         string `json:"state"`
			Model         string `json:"model"`
			PID           int    `json:"pid"`
			UptimeSeconds int64  `json:"uptime_seconds"`
		} `json:"vllm"`
	}

	dec := json.NewDecoder(resp.Body)
	dec.DisallowUnknownFields()
	var decoded healthResp
	if err := dec.Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
}

func TestStatus_SchemaMatchesSpec(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady, CurrentModel: "m", InFlight: 2, UptimeSeconds: 3}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/status", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	type statusResp struct {
		State                        string   `json:"state"`
		AcceptingRequests            bool     `json:"accepting_requests"`
		CurrentModel                 string   `json:"current_model"`
		InFlightRequests             int64    `json:"in_flight_requests"`
		UptimeSeconds                int64    `json:"uptime_seconds"`
		SwapCooldownRemainingSeconds int64    `json:"swap_cooldown_remaining_seconds"`
		LastSwap                     any      `json:"last_swap"`
		FailureCount                 int      `json:"failure_count"`
		AvailableModels              []string `json:"available_models"`
	}

	dec := json.NewDecoder(resp.Body)
	dec.DisallowUnknownFields()
	var decoded statusResp
	if err := dec.Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
}

func TestHealth_StartingReturns503(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateStarting}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}
}

func TestHealth_ErrorReturns500(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateError}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}
}

// Sanity: we should return a usable Fiber app.
func TestNewApp_ReturnsFiberApp(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateIdle}},
	})
	if app == nil {
		t.Fatalf("expected non-nil")
	}
	if _, ok := any(app).(*fiber.App); !ok {
		t.Fatalf("expected *fiber.App")
	}
}

func TestServer_LogRequestsCanBeDisabled(t *testing.T) {
	cfg, err := config.Load([]byte(`
server:
  log_requests: false
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
}

func TestServer_RecoveryMiddlewareReturns500OnPanic(t *testing.T) {
	cfg, err := config.Load([]byte(`
server:
  log_requests: false
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})
	app.Get("/_panic", func(c *fiber.Ctx) error {
		panic("boom")
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/_panic", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}
}

func TestModels_ReturnsConfiguredModels(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  base:
    path: "/models/base"
  alias:
    alias: base
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(decoded.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(decoded.Data))
	}
}

func TestSwitchingProxy_EnsuresModelAndRewritesModelNameInJSON(t *testing.T) {
	var backendReqBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		_ = r.Body.Close()
		backendReqBody = b
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","model":"base"}`))
	}))
	t.Cleanup(backend.Close)

	u := strings.TrimPrefix(backend.URL, "http://")
	_, portStr, _ := strings.Cut(u, ":")

	cfg, err := config.Load([]byte(`
behavior:
  rewrite_model_name: true
vllm:
  port: ` + portStr + `
models:
  base:
    path: "/models/base"
  alias:
    alias: base
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	coord := &stubCoord{st: jukebox.Status{State: jukebox.StateReady}}
	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: coord,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"alias","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "req_123")

	resp, err := app.Test(req, 3000)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if coord.ensureCalls != 1 || coord.lastModel != "alias" || coord.lastRequestID != "req_123" {
		t.Fatalf("expected EnsureModel(alias, req_123) once, got calls=%d model=%q reqid=%q", coord.ensureCalls, coord.lastModel, coord.lastRequestID)
	}

	bodyBytes, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(bodyBytes), `"model":"alias"`) {
		t.Fatalf("expected rewritten model in body, got: %s", string(bodyBytes))
	}
	if !strings.Contains(string(backendReqBody), `"model":"/models/base"`) {
		t.Fatalf("expected request model to be rewritten for vLLM, got: %s", string(backendReqBody))
	}
}

func TestSwitchingProxy_SwapInProgressReturns503WithRetryAfter(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	coord := &stubCoord{
		st:        jukebox.Status{State: jukebox.StateStarting},
		ensureErr: &jukebox.RejectError{Reason: jukebox.RejectSwapInProgress, RetryAfter: 5 * time.Second},
	}
	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: coord,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatalf("expected Retry-After header")
	}
}

func TestSwitchingProxy_EnsureFailureReturns500(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	coord := &stubCoord{
		st:        jukebox.Status{State: jukebox.StateError},
		ensureErr: errors.New("boom"),
	}
	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: coord,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}
}

func TestSwitchingProxy_UnknownModelReturns400WithSpecErrorShape(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"nope"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}

	var decoded struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	dec := json.NewDecoder(resp.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Error.Code != "model_not_found" || decoded.Error.Type != "invalid_request_error" {
		t.Fatalf("unexpected error: %+v", decoded.Error)
	}
}

func TestSwitchingProxy_InvalidJSONReturns400(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{not json`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestSwitchingProxy_MissingModelReturns400(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[]}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestSwitchingProxy_RewritesModelNameInSSE(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"model\":\"base\",\"x\":1}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(backend.Close)

	u := strings.TrimPrefix(backend.URL, "http://")
	_, portStr, _ := strings.Cut(u, ":")

	cfg, err := config.Load([]byte(`
behavior:
  rewrite_model_name: true
vllm:
  port: ` + portStr + `
models:
  base:
    path: "/models/base"
  alias:
    alias: base
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"alias","stream":true}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, 3000)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(bodyBytes), `"model":"alias"`) {
		t.Fatalf("expected rewritten model in SSE, got: %s", string(bodyBytes))
	}
}

func TestPassthroughProxy_DoesNotCallEnsureModel(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(backend.Close)

	u := strings.TrimPrefix(backend.URL, "http://")
	_, portStr, _ := strings.Cut(u, ":")

	cfg, err := config.Load([]byte(`
vllm:
  port: ` + portStr + `
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	coord := &stubCoord{st: jukebox.Status{State: jukebox.StateReady}}
	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: coord,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if coord.ensureCalls != 0 {
		t.Fatalf("expected EnsureModel not called, got %d", coord.ensureCalls)
	}
}

func TestUnknownV1Endpoint_Returns501WithOpenAIErrorShape(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	app := httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateReady}},
	})

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/v1/unknown", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", resp.StatusCode)
	}

	var decoded struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	dec := json.NewDecoder(resp.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Error.Code != "not_implemented" {
		t.Fatalf("unexpected code: %q", decoded.Error.Code)
	}
}
