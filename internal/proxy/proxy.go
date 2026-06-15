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
	"sync"
	"sync/atomic"
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

	// Release, if non-nil, is invoked EXACTLY ONCE when the response
	// has been fully delivered to the downstream client — i.e. when the
	// upstream resource backing this request is no longer in use. For
	// buffered responses that is just before ForwardFiber returns. For
	// STREAMING (text/event-stream) responses, Fiber runs the body copy
	// in a SetBodyStreamWriter callback that executes AFTER the handler
	// returns; Release therefore fires from INSIDE that callback once the
	// stream EOFs (or the copy errors / the client hangs up), NOT when
	// ForwardFiber returns.
	//
	// This is the load-bearing hook for in-flight accounting: the
	// scheduler's per-model inflight counter (which gates the drain
	// barrier before sleep/eviction AND the per-model concurrency cap)
	// MUST stay incremented for the entire lifetime of a streaming
	// response. A handler that instead `defer`s its release would
	// decrement the counter the instant the handler returns — before a
	// single SSE token has been copied — making an active stream
	// invisible to the drain barrier (the scheduler observes inflight==0
	// and sleeps/evicts the instance mid-stream, killing the client's
	// stream and freeing VRAM the stream still needs). That mid-stream
	// sleep is exactly the cumem drain-barrier the fix is defending
	// (vllm#45520): inflight==0 lets /sleep fire during active decode and
	// corrupts the cumem allocator (cudaErrorIllegalAddress).
	//
	// Pass the route's Done/release closure here INSTEAD of deferring it
	// in the handler. Idempotent at this layer via a sync.Once; the
	// handler keeps a sync.Once-guarded defer as a belt-and-suspenders
	// safety net for any error path that never reaches the proxy.
	Release func()
}

// fire invokes opts.OnComplete with the given status if set. Safe to
// call with status=0 to mean "proxy never reached upstream".
func (opts ForwardOptions) fire(status int) {
	if opts.OnComplete != nil {
		opts.OnComplete(status)
	}
}

// sharedTransport is the single, process-wide HTTP transport used for ALL
// upstream forwarding. It MUST be a singleton: a *new* http.Client per
// request (the original code) is fine for connection pooling (a nil
// Transport falls back to http.DefaultTransport, which IS shared) — but
// http.DefaultTransport caps idle keep-alive connections per host at
// DefaultMaxIdleConnsPerHost = 2. Jukebox forwards every request for a
// resident model to a SINGLE upstream host (http://127.0.0.1:<port> or the
// mesh URL). Under the real workload — Claude Code fanning tens of
// concurrent requests at the pinned main model from 2+ people — only 2
// connections are kept warm; every request beyond those closes its TCP
// connection on completion instead of returning it to the pool, so the
// next request pays a fresh dial (and churns sockets into TIME_WAIT).
//
// Raising MaxIdleConnsPerHost to match expected concurrency lets the pool
// actually retain connections, eliminating per-request dial latency and
// socket churn under burst load. Cloned from DefaultTransport so all other
// dialer/timeout defaults are preserved.
var sharedTransport http.RoundTripper = func() http.RoundTripper {
	t := http.DefaultTransport.(*http.Transport).Clone()
	// Generous per-host idle pool: the gateway is a fan-in point onto a
	// small number of upstream hosts, so a high per-host cap is exactly the
	// right shape (unlike a general-purpose client hitting many hosts).
	t.MaxIdleConnsPerHost = 64
	t.MaxIdleConns = 256
	return t
}()

func ForwardFiber(c *fiber.Ctx, opts ForwardOptions) error {
	// release is the exactly-once wrapper around opts.Release. Every exit
	// path MUST call it (early transport errors, buffered success,
	// streaming success). For streaming, the SetBodyStreamWriter callback
	// owns the release so the inflight token survives until the stream
	// EOFs — see ForwardOptions.Release for the full rationale.
	var releaseOnce sync.Once
	release := func() {
		releaseOnce.Do(func() {
			if opts.Release != nil {
				opts.Release()
			}
		})
	}

	targetURL := strings.TrimRight(opts.BaseURL, "/") + c.OriginalURL()

	// reqCtx is a cancelable child of the inbound request context. It exists
	// so the streaming path can install an IDLE watchdog: fasthttp/Fiber in
	// this version provides NO per-request client-disconnect signal
	// (RequestCtx.Done() closes only on server shutdown), and the proxy sets
	// no client.Timeout (a hard total timeout would kill legitimately long
	// but progressing reasoning streams). The result, pre-fix, is that a
	// WEDGED or dead upstream stream — one that produces no further bytes —
	// blocks resp.Body.Read forever, pinning the load-bearing inflight token
	// (and the vLLM decode slot it represents) indefinitely. Under
	// concurrency that's a slow resource leak: every client that hangs up
	// while its upstream stream is stalled leaves a phantom inflight that the
	// drain barrier and the per-model concurrency cap both still count.
	//
	// The watchdog (streaming path only) cancels reqCtx if NO read progress
	// happens within streamIdleTimeout, unblocking the read and releasing the
	// token. Each successful read bumps the activity clock, so a slow-but-
	// progressing stream is never killed. Buffered (non-stream) responses
	// rely on the transport's own dialer/response-header timeouts plus
	// io.ReadAll; we still cancel reqCtx on every return to free the context.
	reqCtx, cancel := context.WithCancel(requestContext(c))

	var body io.Reader
	if b := c.Body(); len(b) > 0 {
		if opts.UpstreamModel != "" && strings.Contains(strings.ToLower(c.Get("Content-Type")), "application/json") {
			if rewritten, ok, err := RewriteJSONModel(b, opts.UpstreamModel); err == nil && ok {
				b = rewritten
			}
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(reqCtx, c.Method(), targetURL, body)
	if err != nil {
		cancel()
		opts.fire(0)
		release()
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

	client := &http.Client{Transport: sharedTransport}
	if opts.Timeout > 0 {
		client.Timeout = opts.Timeout
	}

	resp, err := client.Do(req)
	if err != nil {
		cancel()
		opts.fire(0)
		release()
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
		// Idle watchdog: reads bump `lastActivity`; the watchdog cancels the
		// upstream request (unblocking a stalled Read) if no progress occurs
		// within streamIdleTimeout. Cancel is also called when the stream
		// ends so the watchdog goroutine always exits. The activityReader
		// wraps resp.Body so both the rewrite and the plain-copy paths share
		// one progress source.
		ar := newActivityReader(resp.Body)
		stopWatchdog := startIdleWatchdog(ar, streamIdleTimeout, cancel)
		c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
			// CRITICAL: release fires from INSIDE this callback, which Fiber
			// runs AFTER the handler returns. Holding the inflight token
			// until the stream EOFs (or the copy errors / client hangs up)
			// keeps an active SSE stream visible to the scheduler's drain
			// barrier + concurrency cap. Deferring release in the handler
			// instead would decrement inflight before the first token copies,
			// letting the cumem drain barrier fire /sleep mid-decode
			// (vllm#45520).
			defer release()
			defer cancel()       // stop the watchdog + free reqCtx
			defer stopWatchdog() // ensure the watchdog goroutine exits
			defer resp.Body.Close()
			var copyErr error
			if opts.RewriteModelName && opts.RequestedModel != "" {
				copyErr = RewriteSSEModel(ar, w, opts.RequestedModel)
			} else {
				_, copyErr = copyWithFlush(ar, w)
			}
			// Watchdog-attributable abort: the idle watchdog cancels reqCtx
			// when an upstream SSE stream makes ZERO read progress for longer
			// than streamIdleTimeout (the #45094 wedge signature). That cancel
			// surfaces here as a read error on resp.Body. The HTTP status +
			// headers were already flushed to the client at stream start, so
			// we CANNOT change the status. A bare Flush of a partial body
			// looks IDENTICAL to a complete response on the wire (no terminal
			// data: [DONE], no error) — silent truncation: the client (and
			// openwebui RAG) treats a truncated reasoning/answer stream as a
			// finished one. An in-band SSE error frame is the only honest
			// terminal signal available once headers are sent. We attribute
			// via reqCtx.Err(): during the stream the watchdog goroutine is
			// the ONLY caller of cancel() before these defers run, so a
			// non-nil reqCtx error here means the watchdog fired (not a
			// normal EOF, which leaves reqCtx live until the deferred cancel).
			if copyErr != nil && reqCtx.Err() != nil {
				writeStreamStallErrorFrame(w)
			}
			_ = w.Flush()
		})
		return nil
	}

	data, readErr := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	cancel()
	if readErr != nil {
		release()
		// Mid-body read failure on a BUFFERED response: the upstream sent
		// response headers (so c.Status was already Set to the real upstream
		// code) but then the body read failed — connection reset, upstream
		// crash mid-flight, declared Content-Length never satisfied, etc.
		// Returning the raw Go error to Fiber's default error handler is the
		// silent-truncation bug for buffered responses: the status stays at
		// whatever upstream sent (commonly 200), and the raw error string —
		// which can carry the internal upstream host/port — escapes into the
		// body. A consumer (openwebui RAG) treats that partial/garbage as a
		// complete 200. We never wrote any body bytes yet (io.ReadAll buffers
		// before c.Send), so it is safe to discard the upstream status and
		// emit the same sanitized 503 + Retry-After the pre-response transport
		// path uses, and bump the transport-error metric. We do NOT re-fire
		// OnComplete: it already fired above with the REAL upstream status, and
		// a body-copy failure is a transport problem, not a second engine verdict.
		return writeTransportError(c, readErr, opts)
	}

	if opts.RewriteModelName && opts.RequestedModel != "" && strings.Contains(strings.ToLower(contentType), "application/json") {
		if rewritten, ok, err := RewriteJSONModel(data, opts.RequestedModel); err == nil && ok {
			data = rewritten
		}
	}

	release()
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

// streamIdleTimeout bounds how long a streaming upstream response may make
// ZERO read progress before the proxy gives up on it and cancels the
// upstream request. It is an IDLE timeout, not a total one: every chunk of
// bytes received resets the clock, so a slow-but-progressing reasoning
// stream (which can legitimately run for minutes) is never killed. The
// value must comfortably exceed vLLM's worst-case inter-token gap under
// heavy concurrent batching while still being short enough that a truly
// wedged engine (#45094: running>0, 0 decode progress, /health lies 200)
// releases its pinned inflight token + decode slot in bounded time instead
// of forever. 120s is well past any honest inter-token gap yet bounds the
// phantom-inflight leak. Overridable for tests via the package var.
var streamIdleTimeout = 120 * time.Second

// activityReader wraps an io.Reader and records the time of the last read
// that returned bytes. The idle watchdog reads lastActivity to decide
// whether the stream has stalled. Safe for concurrent access (the watchdog
// goroutine reads while the copy goroutine writes) via atomic int64
// holding a UnixNano timestamp.
type activityReader struct {
	r            io.Reader
	lastActivity atomic.Int64 // UnixNano of last byte-producing read
}

func newActivityReader(r io.Reader) *activityReader {
	ar := &activityReader{r: r}
	ar.lastActivity.Store(time.Now().UnixNano())
	return ar
}

func (a *activityReader) Read(p []byte) (int, error) {
	n, err := a.r.Read(p)
	if n > 0 {
		a.lastActivity.Store(time.Now().UnixNano())
	}
	return n, err
}

// startIdleWatchdog launches a goroutine that calls cancel() if the reader
// makes no progress for `timeout`. Returns a stop func that terminates the
// watchdog; stop is idempotent and must be called when the stream ends so
// the goroutine never leaks. A non-positive timeout disables the watchdog
// (returns a no-op stop) — used to turn the feature off entirely.
func startIdleWatchdog(ar *activityReader, timeout time.Duration, cancel context.CancelFunc) (stop func()) {
	if timeout <= 0 {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		// Tick at a fraction of the timeout so detection latency is bounded
		// without busy-spinning. quarter-timeout gives <=1.25x worst-case.
		tick := timeout / 4
		if tick <= 0 {
			tick = timeout
		}
		t := time.NewTicker(tick)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				last := time.Unix(0, ar.lastActivity.Load())
				if time.Since(last) >= timeout {
					metrics.StreamIdleTimeoutsTotal.Inc()
					cancel()
					return
				}
			}
		}
	}()
	var once sync.Once
	return func() { once.Do(func() { close(done) }) }
}

// streamStallErrorFrame is the verbatim in-band SSE payload the proxy writes
// to a streaming response that the idle watchdog aborted mid-flight. Because
// the HTTP status + headers were already sent when the stream opened, this
// terminal frame is the ONLY way to tell the client the stream did NOT
// complete normally — without it, a truncated stream is byte-indistinguishable
// from a finished one (silent truncation on an engine wedge). The shape mirrors
// the OpenAI streaming error convention (a `data:` line carrying an `error`
// object) so SDK stream parsers surface it as an error rather than swallowing
// it as content. It is intentionally NOT followed by a `data: [DONE]` line:
// [DONE] signals successful completion, which is exactly the false signal we
// are correcting. The code "upstream_stalled" is the in-band analogue of the
// transport path's "model_unavailable".
const streamStallErrorFrame = "data: {\"error\":{\"message\":\"Upstream model server stopped producing output (stream stalled); the response is incomplete.\",\"type\":\"upstream_error\",\"code\":\"upstream_stalled\"}}\n\n"

// writeStreamStallErrorFrame writes the terminal stall error frame to the SSE
// body writer. A write error is ignored: if the client already hung up there
// is nobody to inform, and the caller flushes immediately after.
func writeStreamStallErrorFrame(w *bufio.Writer) {
	_, _ = w.WriteString(streamStallErrorFrame)
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
