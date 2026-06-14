package proxy

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/metrics"
)

// upstream429RetryAfterSeconds is the Retry-After value sent to the client
// when an upstream backend returns 429. The value targets the engine's
// warm-up curve: vLLM typically goes from cold-engine queue-full to
// admitting at steady-state within a few seconds once the first batch
// schedules. 5s is short enough that OpenAI / Anthropic SDK default
// retry budgets absorb it without surfacing to the application, and long
// enough that a tight retry loop doesn't immediately re-hit the same 429.
const upstream429RetryAfterSeconds = "5"

type ForwardOptions struct {
	BaseURL           string
	RewriteModelName  bool
	RequestedModel    string
	UpstreamModel     string
	RequestID         string
	AdditionalHeaders map[string]string
	Timeout           time.Duration
	// OnComplete, if non-nil, is invoked exactly once with the
	// upstream HTTP status code after the request finishes (success
	// or failure). status == 0 means the proxy never reached upstream
	// (network error / timeout / ctx cancel mid-flight). Used by the
	// circuit breaker to observe per-model 5xx streams; safe to leave
	// nil for callers that don't need post-flight notification.
	OnComplete func(status int)
}

// fire invokes opts.OnComplete with the given status if set. Safe to
// call with status=0 to mean "proxy never reached upstream".
func (opts ForwardOptions) fire(status int) {
	if opts.OnComplete != nil {
		opts.OnComplete(status)
	}
}

func ForwardFiber(c *fiber.Ctx, opts ForwardOptions) error {
	targetURL := strings.TrimRight(opts.BaseURL, "/") + c.OriginalURL()

	var body io.Reader
	if b := c.Body(); len(b) > 0 {
		if opts.UpstreamModel != "" && strings.Contains(strings.ToLower(c.Get("Content-Type")), "application/json") {
			if rewritten, ok, err := RewriteJSONModel(b, opts.UpstreamModel); err == nil && ok {
				b = rewritten
			}
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(requestContext(c), c.Method(), targetURL, body)
	if err != nil {
		opts.fire(0)
		return writeTransportError(c, err, opts)
	}

	for k, values := range c.GetReqHeaders() {
		if shouldSkipRequestHeader(k) {
			continue
		}
		for _, v := range values {
			req.Header.Add(k, v)
		}
	}
	if opts.RequestID != "" {
		req.Header.Set("X-Request-ID", opts.RequestID)
	}
	for k, v := range opts.AdditionalHeaders {
		req.Header.Set(k, v)
	}

	client := &http.Client{}
	if opts.Timeout > 0 {
		client.Timeout = opts.Timeout
	}

	resp, err := client.Do(req)
	if err != nil {
		opts.fire(0)
		return writeTransportError(c, err, opts)
	}

	for k, values := range resp.Header {
		if shouldSkipResponseHeader(k) {
			continue
		}
		for _, v := range values {
			c.Append(k, v)
		}
	}

	// Rewrite upstream 429 (Too Many Requests) to 503 (Service Unavailable)
	// with a short Retry-After. Rationale:
	//
	//   * 429 is semantically "the CLIENT exceeded a rate limit" — it
	//     tells well-behaved SDKs to back off aggressively (often with
	//     long exponential delays) or to surface the error to the
	//     application as terminal. vLLM emits 429 when its in-engine
	//     admission queue is saturated, which during engine warm-up
	//     happens at request volumes the steady-state engine handles
	//     fine. From the consumer's POV the right semantic is "service
	//     is briefly busy, retry shortly" — that's 503 + Retry-After.
	//
	//   * Observed under chaos-style burst load (concurrency=32 against
	//     a freshly-woken embedding backend): ~half of requests hit 429
	//     during the wake window, yet the same load admits cleanly once
	//     the engine is warm. Returning 429 caused SDKs to give up
	//     instead of riding through; rewriting to 503 + Retry-After:5
	//     keeps the request in the SDK's retry budget.
	//
	// The rewrite is unconditional — we don't gate on wake state because
	// (a) the proxy package intentionally has no jukebox-state coupling,
	// and (b) a 429 from a warm vLLM is also better surfaced as 503 to
	// the caller (jukebox is the gateway; client-side rate-limiting is
	// not the gateway's concern). The Retry-After header tells clients
	// the engine should be ready momentarily.
	//
	// We strip any upstream-supplied Retry-After before setting ours,
	// since vLLM does not currently emit one for 429s and a stale value
	// (if upstream ever adds one) would defeat the purpose.
	//
	// Breaker integration: opts.fire() receives the UPSTREAM status code
	// (the un-rewritten one) so the breaker sees what the engine
	// actually returned. The client-facing status is what we Set on the
	// fiber context. Status is locked in; the upstream call effectively
	// succeeded from the breaker's POV (a 5xx body counts as a 5xx —
	// the callback gets the real code). Subsequent body-copy errors are
	// downstream-client problems (client hung up, etc.), not engine
	// crashes, so we don't re-fire OnComplete on those.
	upstreamStatus := resp.StatusCode
	if upstreamStatus == http.StatusTooManyRequests {
		model := opts.RequestedModel
		if model == "" {
			model = "unknown"
		}
		metrics.Upstream429RewrittenTotal.WithLabelValues(model).Inc()
		c.Response().Header.Del("Retry-After")
		c.Set("Retry-After", upstream429RetryAfterSeconds)
		c.Status(http.StatusServiceUnavailable)
	} else {
		c.Status(upstreamStatus)
	}
	opts.fire(upstreamStatus)

	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(strings.ToLower(contentType), "text/event-stream") {
		c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
			defer resp.Body.Close()
			if opts.RewriteModelName && opts.RequestedModel != "" {
				_ = RewriteSSEModel(resp.Body, w, opts.RequestedModel)
			} else {
				_, _ = copyWithFlush(resp.Body, w)
			}
			_ = w.Flush()
		})
		return nil
	}

	data, readErr := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if readErr != nil {
		return readErr
	}

	if opts.RewriteModelName && opts.RequestedModel != "" && strings.Contains(strings.ToLower(contentType), "application/json") {
		if rewritten, ok, err := RewriteJSONModel(data, opts.RequestedModel); err == nil && ok {
			data = rewritten
		}
	}

	return c.Send(data)
}

// writeTransportError maps a Go http.Client transport failure (the err
// returned by client.Do *before* any HTTP response is received) onto a
// clean 503 (Service Unavailable) JSON error envelope.
//
// This is the single coherent classifier for the whole dead/unreachable
// backend family. It folds together two originally-separate fixes that
// both lived on this exact transport-error path:
//
//   - the silent-200-on-dead-backend fix (NXDOMAIN / unresolvable host):
//     previously the raw transport error was returned to Fiber, which
//     never overwrote the default 200 status the response carried at the
//     time the logging middleware read it, so the request was logged as a
//     SUCCESS, and the raw error string — including the internal backend
//     hostname (e.g. "dial tcp: lookup vllm-embeddings: no such host") —
//     leaked into the body. Consumers (notably openwebui RAG) treated that
//     garbage as a valid 200 completion. The generalization of "/health
//     lies 200".
//
//   - the raw-500-on-recreate fix (connection-refused to a KNOWN host):
//     during a vLLM-main `compose up --force-recreate` window the upstream
//     socket is briefly gone; the raw "connect: connection refused" error
//     (with the internal mesh IP) otherwise bubbled to Fiber's default
//     handler as a hard 500 — a terminal signal SDKs do not retry — while
//     the engine is merely reloading.
//
// Both arrive as the same shape (client.Do returns a *url.Error wrapping a
// *net.OpError or *net.DNSError), so they share ONE rewrite here rather
// than two competing implementations. Failure taxonomy (all map to 503,
// never 200/500):
//   - *net.DNSError (NXDOMAIN / "no such host") — backend hostname is
//     unresolvable (container down, mesh-DNS race, typo in active.yaml).
//   - *net.OpError dial errors (connection refused, no route to host) —
//     host resolves but nothing is listening (recreate rm-gap / reload).
//   - context deadline / timeout and any other transport error — the
//     backend never produced an HTTP response, so from the consumer's POV
//     the model is unavailable.
//
// The client-facing body is sanitized: it carries a generic
// "model_unavailable" code and a fixed message, never the underlying
// hostname/port/dial detail. A Retry-After is set so well-behaved SDKs
// ride through a reload window inside their retry budget. The real error
// is logged server-side (with the request id) for operators, and a metric
// (labelled by model + coarse reason) makes the transport-failure rate
// observable in Prometheus. The circuit breaker has already been notified
// via opts.fire(0) at the call site (status==0 = "never reached upstream",
// treated as a 5xx signal).
func writeTransportError(c *fiber.Ctx, err error, opts ForwardOptions) error {
	reason := "unreachable"
	var dnsErr *net.DNSError
	var opErr *net.OpError
	switch {
	case errors.As(err, &dnsErr):
		reason = "dns"
	case errors.As(err, &opErr):
		// Dial-time OpError (connection refused, no route, etc) — this is
		// the recreate/reload-window case as well as a permanently-down
		// known host.
		reason = "dial"
	case errors.Is(err, context.DeadlineExceeded):
		reason = "timeout"
	}

	model := opts.RequestedModel
	if model == "" {
		model = "unknown"
	}
	metrics.UpstreamTransportErrorsTotal.WithLabelValues(model, reason).Inc()

	// Log the *real* error server-side — operators need the hostname/port.
	slog.Warn("upstream_transport_error",
		"request_id", opts.RequestID,
		"model", model,
		"reason", reason,
		"error", err.Error(),
	)

	// Sanitized client-facing body — no internal hostname/port/dial detail.
	// Replace any upstream-supplied Retry-After (defensive; none exists on
	// this path today) before setting ours.
	c.Response().Header.Del("Retry-After")
	c.Set("Content-Type", "application/json")
	c.Set("Retry-After", upstream429RetryAfterSeconds)
	return c.Status(http.StatusServiceUnavailable).JSON(fiber.Map{
		"error": fiber.Map{
			"message": "Upstream model server is currently unavailable; please retry.",
			"type":    "service_unavailable",
			"code":    "model_unavailable",
		},
	})
}

func copyWithFlush(r io.Reader, w *bufio.Writer) (int64, error) {
	buf := make([]byte, 32*1024)
	var total int64
	for {
		n, err := r.Read(buf)
		if n > 0 {
			written, werr := w.Write(buf[:n])
			total += int64(written)
			_ = w.Flush()
			if werr != nil {
				return total, werr
			}
		}
		if err != nil {
			if err == io.EOF {
				return total, nil
			}
			return total, err
		}
	}
}

func requestContext(c *fiber.Ctx) context.Context {
	if c == nil {
		return context.Background()
	}
	if ctx := c.UserContext(); ctx != nil {
		return ctx
	}
	return context.Background()
}

func shouldSkipRequestHeader(key string) bool {
	k := strings.ToLower(key)
	switch k {
	case "host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func shouldSkipResponseHeader(key string) bool {
	k := strings.ToLower(key)
	switch k {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}
