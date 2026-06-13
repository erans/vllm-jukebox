package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v2"
)

// TestForwardFiber_Upstream429RewrittenTo503 asserts the proxy's
// 429→503+Retry-After rewrite. vLLM emits 429 when its in-engine
// admission queue is saturated (commonly during wake / cold-start);
// 429 tells SDKs "client is rate-limited" which is the wrong signal —
// the engine is briefly busy, not the caller misbehaving. The proxy
// rewrites to 503 + Retry-After so SDKs retry within their normal
// budget.
func TestForwardFiber_Upstream429RewrittenTo503(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"queue full"}`))
	}))
	defer upstream.Close()

	app := fiber.New()
	app.Post("/v1/embeddings", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "bge-m3",
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"model":"bge-m3","input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("upstream 429 should be rewritten to 503; got status=%d", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != upstream429RetryAfterSeconds {
		t.Errorf("expected Retry-After=%q on rewritten 503, got %q", upstream429RetryAfterSeconds, got)
	}
	// Body must pass through untouched — clients can still inspect the
	// upstream error message even though we rewrote the status.
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "queue full") {
		t.Errorf("expected upstream body to pass through; got %q", string(body))
	}
}

// TestForwardFiber_Upstream429StripsStaleRetryAfter asserts that any
// Retry-After header the upstream provided alongside the 429 is
// REPLACED by the proxy's value (not appended), so clients don't see
// two values or a stale upstream value.
func TestForwardFiber_Upstream429StripsStaleRetryAfter(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "999")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer upstream.Close()

	app := fiber.New()
	app.Post("/x", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "m",
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(""))
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}
	// fasthttp/fiber stores header values as a single string for
	// single-value headers; check we did NOT end up with the upstream's
	// 999 leaking through.
	got := resp.Header.Get("Retry-After")
	if got != upstream429RetryAfterSeconds {
		t.Errorf("upstream Retry-After should be replaced; got %q want %q", got, upstream429RetryAfterSeconds)
	}
	if strings.Contains(got, "999") {
		t.Errorf("stale upstream Retry-After 999 leaked through: %q", got)
	}
}

// TestForwardFiber_NonRewrittenStatusesPassThrough sanity-checks that
// the 429→503 rewrite is narrowly scoped — every other status code
// reaches the client untouched, including 503 (which the proxy must
// NOT double-rewrite).
func TestForwardFiber_NonRewrittenStatusesPassThrough(t *testing.T) {
	for _, status := range []int{
		http.StatusOK,
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusNotFound,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
	} {
		status := status
		t.Run(http.StatusText(status), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(status)
			}))
			defer upstream.Close()

			app := fiber.New()
			app.Post("/x", func(c *fiber.Ctx) error {
				return ForwardFiber(c, ForwardOptions{
					BaseURL:        upstream.URL,
					RequestedModel: "m",
				})
			})

			req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(""))
			resp, err := app.Test(req, -1)
			if err != nil {
				t.Fatalf("app.Test: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != status {
				t.Errorf("status %d should pass through unchanged; got %d", status, resp.StatusCode)
			}
			if got := resp.Header.Get("Retry-After"); got != "" {
				t.Errorf("only 429 should add Retry-After; got %q for upstream %d", got, status)
			}
		})
	}
}
