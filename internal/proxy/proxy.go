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
	RequestID         string
	AdditionalHeaders map[string]string
	Timeout           time.Duration
}

func ForwardFiber(c *fiber.Ctx, opts ForwardOptions) error {
	targetURL := strings.TrimRight(opts.BaseURL, "/") + c.OriginalURL()

	var body io.Reader
	if b := c.Body(); len(b) > 0 {
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(requestContext(c), c.Method(), targetURL, body)
	if err != nil {
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
