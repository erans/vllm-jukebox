package httpserver_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/metrics"
)

func TestMetrics_ExposesPrometheusText(t *testing.T) {
	app := httpserver.NewApp(httpserver.Options{
		Coordinator: &stubCoord{st: jukebox.Status{State: jukebox.StateIdle}},
	})

	// Touch at least one metric instance before scraping.
	_, _ = app.Test(httptest.NewRequest(http.MethodGet, "/health", nil))
	metrics.SwapsTotal.WithLabelValues("a", "b").Add(0)
	metrics.SwapRejectionsTotal.WithLabelValues("cooldown").Add(0)
	metrics.SwapDurationSeconds.Observe(0)
	metrics.RequestDurationSeconds.WithLabelValues("unknown").Observe(0)
	metrics.State.WithLabelValues("ready").Set(0)
	metrics.InFlightRequests.Set(0)
	metrics.CurrentModel.WithLabelValues("m").Set(0)
	metrics.ConsecutiveFailures.Set(0)

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
	for _, name := range []string{
		"jukebox_requests_total",
		"jukebox_swaps_total",
		"jukebox_swap_rejections_total",
		"jukebox_swap_duration_seconds",
		"jukebox_request_duration_seconds",
		"jukebox_state",
		"jukebox_in_flight_requests",
		"jukebox_current_model",
		"jukebox_consecutive_failures",
	} {
		if !strings.Contains(body, name) {
			t.Fatalf("expected %s in metrics output", name)
		}
	}
}
