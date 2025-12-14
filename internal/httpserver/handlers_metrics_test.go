package httpserver_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
)

func TestMetrics_ExposesPrometheusText(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateIdle}},
	})

	// Touch at least one metric instance before scraping.
	_, _ = app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))

	resp, err := app.Test(httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	bodyBytes, _ := io.ReadAll(resp.Body)
	body := string(bodyBytes)
	if !strings.Contains(body, "jukebox_requests_total") {
		t.Fatalf("expected jukebox_requests_total in metrics output")
	}
}
