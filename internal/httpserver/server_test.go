package httpserver_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
)

type stubCoord struct {
	st jukebox.Status
}

func (s stubCoord) Status() jukebox.Status { return s.st }

func TestHealth_ReadyReturns200AndAcceptingRequests(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: stubCoord{st: jukebox.Status{State: jukebox.StateReady, CurrentModel: "m", PID: 123}},
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

func TestHealth_StartingReturns503(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: stubCoord{st: jukebox.Status{State: jukebox.StateStarting}},
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
		Coordinator: stubCoord{st: jukebox.Status{State: jukebox.StateError}},
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
		Coordinator: stubCoord{st: jukebox.Status{State: jukebox.StateIdle}},
	})
	if app == nil {
		t.Fatalf("expected non-nil")
	}
	if _, ok := any(app).(*fiber.App); !ok {
		t.Fatalf("expected *fiber.App")
	}
}

