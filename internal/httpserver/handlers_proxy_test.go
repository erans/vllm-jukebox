package httpserver

import (
	"bytes"
	"context"
	"net"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/gofiber/fiber/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

// countingRouter returns a Route whose Done decrements an in-flight counter so
// tests can assert the in-flight slot is released on every exit path.
type countingRouter struct {
	route      jukebox.Route
	doneCalled *int64
}

func (m *countingRouter) AcquireRoute(ctx context.Context, model, requestID string) (jukebox.Route, error) {
	// Track one in-flight slot; Done releases it.
	atomic.AddInt64(m.doneCalled, 1) // simulate Acquire incrementing in-flight
	m.route.Done = func() {
		// release: decrement in-flight
		atomic.AddInt64(m.doneCalled, -1)
	}
	return m.route, nil
}

func (m *countingRouter) Status() jukebox.Status {
	return jukebox.Status{State: jukebox.StateReady}
}

// closedPortURL reserves a TCP port and immediately closes it so the proxy's
// client.Do returns a connection error (the error path we want to exercise).
func closedPortURL(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())
	return "http://" + addr
}

// newCountingRouter builds a router whose Route points at a closed upstream,
// forcing ForwardFiber to return a client.Do error.
func newCountingRouter(t *testing.T, doneCalled *int64) *countingRouter {
	return &countingRouter{
		route: jukebox.Route{
			BaseURL: closedPortURL(t),
		},
		doneCalled: doneCalled,
	}
}

// TestSwitchingProxy_ReleasesInflightOnForwardError asserts that when
// ForwardFiber returns an error (upstream unreachable), the in-flight slot
// acquired via AcquireRoute is released exactly once via route.Done. Without
// the error-path release, the slot would leak and WaitForDrain would hang.
func TestSwitchingProxy_ReleasesInflightOnForwardError(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	require.NoError(t, err)

	var doneCalled int64
	router := newCountingRouter(t, &doneCalled)

	app := fiber.New()
	app.Post("/v1/chat/completions", switchingProxyHandler(Options{Config: cfg, Router: router}))

	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, -1)
	require.NoError(t, err)
	defer resp.Body.Close()

	// ForwardFiber hit a connection error; Fiber surfaces a 5xx (default 500) error.
	assert.GreaterOrEqual(t, resp.StatusCode, 500)

	// The in-flight slot must have been released: counter returns to 0.
	assert.Equal(t, int64(0), atomic.LoadInt64(&doneCalled),
		"in-flight slot leaked: route.Done was not called on ForwardFiber error path (switchingProxyHandler)")
}

// TestAnthropicProxy_ReleasesInflightOnForwardError is the same guarantee for
// the Anthropic /v1/messages handler.
func TestAnthropicProxy_ReleasesInflightOnForwardError(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	require.NoError(t, err)

	var doneCalled int64
	router := newCountingRouter(t, &doneCalled)

	app := fiber.New()
	app.Post("/v1/messages", anthropicProxyHandler(Options{Config: cfg, Router: router}))

	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":100}`)
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, -1)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.GreaterOrEqual(t, resp.StatusCode, 500)
	assert.Equal(t, int64(0), atomic.LoadInt64(&doneCalled),
		"in-flight slot leaked: route.Done was not called on ForwardFiber error path (anthropicProxyHandler)")
}
