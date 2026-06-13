package proxy

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
)

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
		return err
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
		return err
	}

	for k, values := range resp.Header {
		if shouldSkipResponseHeader(k) {
			continue
		}
		for _, v := range values {
			c.Append(k, v)
		}
	}

	c.Status(resp.StatusCode)
	// Status is locked in; the upstream call effectively succeeded
	// from the breaker's POV (a 5xx body counts as a 5xx — the
	// callback gets the real code). Subsequent body-copy errors are
	// downstream-client problems (client hung up, etc.), not engine
	// crashes, so we don't re-fire OnComplete on those.
	opts.fire(resp.StatusCode)

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
