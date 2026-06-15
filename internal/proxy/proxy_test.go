package proxy

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v2"
)

// TestMain clears any ambient HTTP(S)_PROXY before the first HTTP request
// in this package. Go's net/http caches the proxy config exactly once per
// process (httpproxy.FromEnvironment via a sync.Once), so a per-test
// t.Setenv can't undo it after another test has already triggered a
// request. Some sandboxed CI / dev environments inject an HTTP_PROXY; with
// it set, the default http.Client (which honors http.ProxyFromEnvironment)
// routes a request for an unresolvable backend to the *proxy* (which may
// answer 4xx) instead of producing the real DNS/transport error these
// tests exercise. Clearing it before any request is issued makes the
// transport-error tests deterministic and environment-independent.
func TestMain(m *testing.M) {
	for _, k := range []string{
		"HTTP_PROXY", "http_proxy",
		"HTTPS_PROXY", "https_proxy",
		"ALL_PROXY", "all_proxy",
	} {
		_ = os.Unsetenv(k)
	}
	os.Exit(m.Run())
}

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

// TestForwardFiber_UnresolvableHostReturns503 is the regression test for
// the silent-200-on-dead-backend bug. When a backend host is unresolvable
// (NXDOMAIN — `dial tcp: lookup vllm-embeddings: no such host`) the Go
// http.Client.Do call fails before any byte reaches the wire. The proxy
// MUST map that transport failure to a 503 (model_unavailable) with a
// clean JSON body — NOT let the raw transport error escape as a 200 with
// the error text (and the internal hostname) in the body. Pre-fix this
// returned 200 with the dial error leaked into the body; consumers
// (openwebui RAG) treated the garbage as a successful completion.
func TestForwardFiber_UnresolvableHostReturns503(t *testing.T) {
	// RFC 6761 reserves .invalid as guaranteed-NXDOMAIN. Using a name
	// that contains an internal-looking hostname lets us assert it does
	// NOT leak into the client-facing body.
	const internalHost = "vllm-embeddings-internal.invalid"

	app := fiber.New()
	app.Post("/v1/embeddings", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        "http://" + internalHost + ":8000",
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
		t.Fatalf("unresolvable backend must return 503, got status=%d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)

	// Body must be a clean JSON error envelope, not the raw transport error.
	var env struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("503 body must be valid JSON error envelope; got %q (unmarshal err: %v)", bodyStr, err)
	}
	if env.Error.Code != "model_unavailable" {
		t.Errorf("expected error.code=model_unavailable, got %q (body=%q)", env.Error.Code, bodyStr)
	}

	// No internal hostname / DNS internals may leak to the consumer.
	for _, leak := range []string{internalHost, "no such host", "dial tcp", "lookup", ":8000"} {
		if strings.Contains(bodyStr, leak) {
			t.Errorf("internal detail %q leaked into client-facing body: %q", leak, bodyStr)
		}
	}
}

// TestForwardFiber_ConnectionRefusedReturns503 covers the other half of
// the dead-backend family: the host resolves but nothing is listening
// (connection refused / *net.OpError dial error). Like NXDOMAIN this must
// surface as a sanitized 503, never a silent 200.
func TestForwardFiber_ConnectionRefusedReturns503(t *testing.T) {
	// Bind a listener to grab a free port, then close it so the port is
	// guaranteed refused.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := l.Addr().String()
	_ = l.Close()

	app := fiber.New()
	app.Post("/v1/embeddings", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        "http://" + addr,
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
		t.Fatalf("connection-refused backend must return 503, got status=%d", resp.StatusCode)
	}

	// The rewritten 503 must carry a Retry-After so SDKs ride through a
	// recreate/reload window inside their retry budget (this is the
	// raw-500-on-recreate contract folded into the shared classifier).
	if got := resp.Header.Get("Retry-After"); got != upstream429RetryAfterSeconds {
		t.Errorf("expected Retry-After=%q on transport-error 503, got %q", upstream429RetryAfterSeconds, got)
	}

	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)
	if !strings.Contains(bodyStr, "model_unavailable") {
		t.Errorf("expected model_unavailable error code, got %q", bodyStr)
	}
	for _, leak := range []string{addr, "connection refused", "dial tcp"} {
		if strings.Contains(bodyStr, leak) {
			t.Errorf("internal detail %q leaked into client-facing body: %q", leak, bodyStr)
		}
	}
}

// TestForwardFiber_UpstreamTransportErrorFiresStatusZero asserts the
// circuit-breaker observability contract is preserved: a transport
// failure still fires OnComplete(0) so the breaker treats it as a 5xx
// signal, even though the client sees a sanitized 503.
func TestForwardFiber_UpstreamTransportErrorFiresStatusZero(t *testing.T) {
	var fired []int
	app := fiber.New()
	app.Post("/x", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        "http://nonexistent-backend.invalid:8000",
			RequestedModel: "m",
			OnComplete:     func(status int) { fired = append(fired, status) },
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(""))
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if len(fired) != 1 || fired[0] != 0 {
		t.Errorf("transport error must fire OnComplete exactly once with status=0; got %v", fired)
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

// TestForwardFiber_BufferedMidBodyReadErrorReturns503 is the regression test
// for the buffered-response silent-truncation bug. The upstream sends a 200
// with a Content-Length larger than the bytes it actually writes, then hangs
// up the connection. io.ReadAll on the proxy side fails mid-body with
// io.ErrUnexpectedEOF. Pre-fix the proxy returned that raw Go error to
// Fiber's default error handler: the status stayed at the upstream's 200
// (already Set before the body read) and the raw error string escaped into
// the body — a partial/garbage response that openwebui RAG treats as a
// complete 200. The fix routes the mid-body read error through the same
// sanitizer the pre-response transport path uses: a clean 503 +
// model_unavailable envelope + Retry-After, with no raw error / internal
// detail leaked.
func TestForwardFiber_BufferedMidBodyReadErrorReturns503(t *testing.T) {
	// httptest's handler can't easily close mid-body with a declared
	// Content-Length, so hijack the conn and write a raw HTTP/1.1 response
	// whose Content-Length lies, then close the socket before satisfying it.
	const internalMarker = "vllm-main-internal-secret"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Errorf("ResponseWriter does not support hijacking")
			return
		}
		conn, bufrw, err := hj.Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		// Declare 4096 bytes, then write far fewer and slam the connection.
		// The marker stands in for any internal detail the read error might
		// otherwise carry — it must not leak to the client.
		_, _ = bufrw.WriteString("HTTP/1.1 200 OK\r\n")
		_, _ = bufrw.WriteString("Content-Type: application/json\r\n")
		_, _ = bufrw.WriteString("Content-Length: 4096\r\n")
		_, _ = bufrw.WriteString("\r\n")
		_, _ = bufrw.WriteString(`{"partial":"` + internalMarker + `"`)
		_ = bufrw.Flush()
		_ = conn.Close() // hang up before the declared body is complete
	}))
	defer upstream.Close()

	app := fiber.New()
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "vllm-main",
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"vllm-main"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, 5000)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("buffered mid-body read error must surface as 503, got status=%d", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != upstream429RetryAfterSeconds {
		t.Errorf("expected Retry-After=%q on sanitized 503, got %q", upstream429RetryAfterSeconds, got)
	}

	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)

	// Must be the clean JSON error envelope, not the partial upstream body
	// or a raw Go error.
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("503 body must be valid JSON error envelope; got %q (unmarshal err: %v)", bodyStr, err)
	}
	if env.Error.Code != "model_unavailable" {
		t.Errorf("expected error.code=model_unavailable, got %q (body=%q)", env.Error.Code, bodyStr)
	}
	// Neither the partial upstream payload nor any raw read-error text may leak.
	for _, leak := range []string{internalMarker, "partial", "unexpected EOF", "io."} {
		if strings.Contains(bodyStr, leak) {
			t.Errorf("internal/partial detail %q leaked into client-facing body: %q", leak, bodyStr)
		}
	}
}
