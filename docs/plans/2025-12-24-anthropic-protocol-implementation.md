# Anthropic Protocol Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native Anthropic API protocol support (`/v1/messages`) alongside existing OpenAI endpoints.

**Architecture:** Add Anthropic endpoint handler that reuses existing model extraction and routing logic, with protocol-specific error formatting and SSE model rewriting.

**Tech Stack:** Go, Fiber HTTP framework, existing proxy package

---

### Task 1: Extract error helpers to separate file

**Files:**
- Create: `internal/httpserver/errors.go`
- Modify: `internal/httpserver/handlers_proxy.go`

**Step 1: Write the test to verify OpenAI error format**

Create `internal/httpserver/errors_test.go`:

```go
package httpserver

import (
	"io"
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriteOpenAIError_Format(t *testing.T) {
	app := fiber.New()
	app.Get("/test", func(c *fiber.Ctx) error {
		return writeOpenAIError(c, 400, "test message", "invalid_request_error", "test_code")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 400, resp.StatusCode)
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))

	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"message":"test message"`)
	assert.Contains(t, string(body), `"type":"invalid_request_error"`)
	assert.Contains(t, string(body), `"code":"test_code"`)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/httpserver -run TestWriteOpenAIError_Format -v`
Expected: PASS (function already exists in handlers_proxy.go)

**Step 3: Create errors.go and move writeOpenAIError**

Create `internal/httpserver/errors.go`:

```go
package httpserver

import (
	"github.com/gofiber/fiber/v2"
)

func writeOpenAIError(c *fiber.Ctx, status int, message, errType, code string) error {
	c.Set("Content-Type", "application/json")
	return c.Status(status).JSON(fiber.Map{
		"error": fiber.Map{
			"message": message,
			"type":    errType,
			"code":    code,
		},
	})
}
```

**Step 4: Remove writeOpenAIError from handlers_proxy.go**

Remove the `writeOpenAIError` function from `internal/httpserver/handlers_proxy.go` (it was at the bottom of the file).

**Step 5: Run tests to verify refactor is correct**

Run: `go test ./internal/httpserver -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add internal/httpserver/errors.go internal/httpserver/errors_test.go internal/httpserver/handlers_proxy.go
git commit -m "refactor(httpserver): extract error helpers to errors.go"
```

---

### Task 2: Add Anthropic error helper

**Files:**
- Modify: `internal/httpserver/errors.go`
- Modify: `internal/httpserver/errors_test.go`

**Step 1: Write the failing test**

Add to `internal/httpserver/errors_test.go`:

```go
func TestWriteAnthropicError_Format(t *testing.T) {
	app := fiber.New()
	app.Get("/test", func(c *fiber.Ctx) error {
		return writeAnthropicError(c, 400, "invalid_request_error", "test message")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	require.NoError(t, err)

	assert.Equal(t, 400, resp.StatusCode)
	assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))

	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), `"type":"error"`)
	assert.Contains(t, string(body), `"error":{`)
	assert.Contains(t, string(body), `"type":"invalid_request_error"`)
	assert.Contains(t, string(body), `"message":"test message"`)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/httpserver -run TestWriteAnthropicError_Format -v`
Expected: FAIL with "undefined: writeAnthropicError"

**Step 3: Implement writeAnthropicError**

Add to `internal/httpserver/errors.go`:

```go
func writeAnthropicError(c *fiber.Ctx, status int, errType, message string) error {
	c.Set("Content-Type", "application/json")
	return c.Status(status).JSON(fiber.Map{
		"type": "error",
		"error": fiber.Map{
			"type":    errType,
			"message": message,
		},
	})
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/httpserver -run TestWriteAnthropicError_Format -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/httpserver/errors.go internal/httpserver/errors_test.go
git commit -m "feat(httpserver): add Anthropic error response helper"
```

---

### Task 3: Create Anthropic proxy handler

**Files:**
- Create: `internal/httpserver/handlers_anthropic.go`
- Create: `internal/httpserver/handlers_anthropic_test.go`

**Step 1: Write the failing test for happy path**

Create `internal/httpserver/handlers_anthropic_test.go`:

```go
package httpserver

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

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
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/httpserver -run TestAnthropicProxy_ExtractsModelAndRoutes -v`
Expected: FAIL with "undefined: anthropicProxyHandler"

**Step 3: Implement the handler**

Create `internal/httpserver/handlers_anthropic.go`:

```go
package httpserver

import (
	"fmt"
	"net/http"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/proxy"
)

func anthropicProxyHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Router == nil {
			return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "server not configured")
		}

		modelName, err := extractModel(c.Body())
		if err != nil {
			return writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		}
		if modelName == "" {
			return writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", "missing required field 'model'")
		}

		c.Locals(requestedModelLocal, modelName)

		_, _, err = opts.Config.ResolveModel(modelName)
		if err != nil {
			return writeAnthropicError(
				c,
				http.StatusBadRequest,
				"invalid_request_error",
				fmt.Sprintf("model %q not found in configuration", modelName),
			)
		}

		requestID, _ := c.Locals(requestIDHeader).(string)
		if requestID == "" {
			requestID = c.Get(requestIDHeader)
		}

		route, err := opts.Router.AcquireRoute(c.UserContext(), modelName, requestID)
		if err != nil {
			return mapAnthropicError(c, err)
		}
		if route.Done != nil {
			defer route.Done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
		})
	}
}

func mapAnthropicError(c *fiber.Ctx, err error) error {
	// For now, map all errors to overloaded or api_error
	// Can be refined based on error types like in OpenAI handler
	return writeAnthropicError(c, http.StatusServiceUnavailable, "overloaded_error", "model loading in progress, please retry")
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/httpserver -run TestAnthropicProxy_ExtractsModelAndRoutes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/httpserver/handlers_anthropic.go internal/httpserver/handlers_anthropic_test.go
git commit -m "feat(httpserver): add Anthropic proxy handler"
```

---

### Task 4: Add Anthropic error mapping tests

**Files:**
- Modify: `internal/httpserver/handlers_anthropic_test.go`

**Step 1: Write tests for error cases**

Add to `internal/httpserver/handlers_anthropic_test.go`:

```go
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
```

**Step 2: Run tests to verify they pass**

Run: `go test ./internal/httpserver -run TestAnthropicProxy -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add internal/httpserver/handlers_anthropic_test.go
git commit -m "test(httpserver): add Anthropic error handling tests"
```

---

### Task 5: Add detailed error mapping for Anthropic

**Files:**
- Modify: `internal/httpserver/handlers_anthropic.go`
- Modify: `internal/httpserver/handlers_anthropic_test.go`

**Step 1: Write test for swap-in-progress error**

Add to `internal/httpserver/handlers_anthropic_test.go`:

```go
func TestAnthropicProxy_SwapInProgressReturns503WithRetryAfter(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"claude": {Path: "/models/claude"},
		},
	}
	router := &mockAnthropicRouter{
		err: &jukebox.RejectError{Reason: jukebox.RejectSwapping, RetryAfter: 5 * time.Second},
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
```

Add import at top:
```go
import (
	"time"
	// ... other imports
)
```

**Step 2: Run test to verify it fails**

Run: `go test ./internal/httpserver -run TestAnthropicProxy_SwapInProgressReturns503 -v`
Expected: FAIL (Retry-After header not set)

**Step 3: Implement detailed error mapping**

Update `mapAnthropicError` in `internal/httpserver/handlers_anthropic.go`:

```go
import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/proxy"
)

func mapAnthropicError(c *fiber.Ctx, err error) error {
	var rej *jukebox.RejectError
	if errors.As(err, &rej) {
		if rej.RetryAfter > 0 {
			retryAfterSeconds := int64(rej.RetryAfter.Round(time.Second).Seconds())
			if retryAfterSeconds <= 0 {
				retryAfterSeconds = 1
			}
			c.Set("Retry-After", fmt.Sprintf("%d", retryAfterSeconds))
		}

		if rej.Reason == jukebox.RejectBackoff {
			return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "model in error state, please retry")
		}

		msg := "model loading in progress, please retry"
		switch rej.Reason {
		case jukebox.RejectPinnedConflict:
			msg = "insufficient resources: requested model conflicts with a pinned model"
		case jukebox.RejectNoCapacity:
			msg = "insufficient capacity to start requested model, please retry"
		case jukebox.RejectInsufficient:
			msg = "insufficient GPU resources to start requested model, please retry"
		case jukebox.RejectMinUptime:
			msg = "insufficient GPU resources (min uptime), please retry"
		}
		return writeAnthropicError(c, http.StatusServiceUnavailable, "overloaded_error", msg)
	}

	return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "backend unavailable")
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./internal/httpserver -run TestAnthropicProxy -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add internal/httpserver/handlers_anthropic.go internal/httpserver/handlers_anthropic_test.go
git commit -m "feat(httpserver): add detailed Anthropic error mapping"
```

---

### Task 6: Register Anthropic endpoint in server

**Files:**
- Modify: `internal/httpserver/server.go`

**Step 1: Add the route**

In `internal/httpserver/server.go`, add after the OpenAI endpoints (around line 41):

```go
	// Anthropic endpoints
	app.Post("/v1/messages", anthropicProxyHandler(opts))
```

**Step 2: Run all tests**

Run: `go test ./internal/httpserver -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add internal/httpserver/server.go
git commit -m "feat(httpserver): register /v1/messages Anthropic endpoint"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md**

Add to the Endpoints section in `README.md`:

```markdown
Anthropic endpoints (proxy to vLLM):
- `POST /v1/messages`
```

And add a note about Anthropic support in the Features section.

**Step 2: Update CLAUDE.md**

Add `/v1/messages` to the API Endpoints section.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add Anthropic protocol support documentation"
```

---

### Task 8: Run full test suite and verify

**Step 1: Run all tests**

Run: `make check`
Expected: All tests PASS, no vet errors

**Step 2: Build binary**

Run: `make build`
Expected: Binary builds successfully

**Step 3: Manual smoke test (optional)**

Start jukebox and test Anthropic endpoint:
```bash
./bin/jukebox -config configs/mine/scheduler.yaml &
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-small", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}'
```

---
