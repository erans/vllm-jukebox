package httpserver

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

type mockAnthropicRouter struct {
	route jukebox.Route
	err   error
}

func (m *mockAnthropicRouter) AcquireRoute(ctx context.Context, model, requestID string) (jukebox.Route, error) {
	return m.route, m.err
}

func (m *mockAnthropicRouter) Status() jukebox.Status {
	return jukebox.Status{State: jukebox.StateReady}
}

// mockAnthropicRouter intentionally does NOT implement ColdLoadAware.
// The handler must type-assert and skip the async-503 path when the
// assertion fails — this stub exercises that fall-through.

func TestAnthropicProxy_ExtractsModelAndRoutes(t *testing.T) {
	// Create a fake backend that returns Anthropic-style response
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"msg_123","type":"message","role":"assistant","content":[{"type":"text","text":"Hello"}],"model":"backend-model"}`))
	}))
	defer backend.Close()

	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
		Behavior: config.BehaviorConfig{RewriteModelName: true},
	}

	router := &mockAnthropicRouter{
		route: jukebox.Route{
			BaseURL:       backend.URL,
			UpstreamModel: "claude",
		},
	}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	body := []byte(`{"model":"claude","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}`)
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, -1)
	require.NoError(t, err)

	assert.Equal(t, 200, resp.StatusCode)

	respBody, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(respBody), `"model":"claude"`) // rewritten
}

func TestAnthropicProxy_InvalidJSONReturnsAnthropicError(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{invalid`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 400, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), `"type":"invalid_request_error"`)
}

func TestAnthropicProxy_MissingModelReturns400(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{"messages":[]}`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 400, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), "missing required field")
}

func TestAnthropicProxy_UnknownModelReturns400(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{"model":"unknown"}`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 400, resp.StatusCode)
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), "not found")
}

func TestAnthropicProxy_SwapInProgressReturns503WithRetryAfter(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{
		err: &jukebox.RejectError{Reason: jukebox.RejectSwapInProgress, RetryAfter: 5 * time.Second},
	}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{"model":"claude"}`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 503, resp.StatusCode)
	assert.NotEmpty(t, resp.Header.Get("Retry-After"))
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), `"type":"overloaded_error"`)
}

// TestAnthropicProxy_AdminInterventionReturnsApiError verifies that
// RejectAdminIntervention surfaces as error.type="api_error" (NOT
// "overloaded_error") so retry-aware Anthropic SDKs treat the response
// as terminal. The Retry-After header is also absent (the RejectError
// carries no RetryAfter for this reason). See the mapAnthropicError
// non-retryable switch for the full rationale.
func TestAnthropicProxy_AdminInterventionReturnsApiError(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{
		err: &jukebox.RejectError{Reason: jukebox.RejectAdminIntervention},
	}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{"model":"claude"}`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 503, resp.StatusCode)
	assert.Empty(t, resp.Header.Get("Retry-After"))
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), `"type":"api_error"`)
	assert.NotContains(t, string(body), `"type":"overloaded_error"`)
	assert.Contains(t, string(body), "operator intervention required")
}

// TestAnthropicProxy_InsufficientReturnsApiError verifies that
// RejectInsufficient (structural infeasibility) surfaces as
// error.type="api_error" so retry-aware Anthropic SDKs treat it as
// terminal rather than backing-off-and-retrying against an
// unsolvable budget condition.
func TestAnthropicProxy_InsufficientReturnsApiError(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{
		err: &jukebox.RejectError{Reason: jukebox.RejectInsufficient},
	}

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(`{"model":"claude"}`)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 503, resp.StatusCode)
	assert.Empty(t, resp.Header.Get("Retry-After"))
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), `"type":"api_error"`)
	assert.NotContains(t, string(body), `"type":"overloaded_error"`)
	assert.Contains(t, string(body), "exceeds available GPU budget")
}
