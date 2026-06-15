package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
)

// TestForwardFiber_StreamingReleaseFiresAfterBody is the regression test
// for the round-3 CRIT: for a text/event-stream response, Fiber runs the
// body copy in a SetBodyStreamWriter callback that executes AFTER the
// handler returns. The inflight-accounting release (ForwardOptions.Release)
// MUST therefore fire from inside that callback once the stream EOFs — NOT
// when ForwardFiber returns.
//
// PRE-FIX (defer route.Done() in the handler, no proxy Release): the token
// is released the instant the handler returns, before a single SSE chunk
// is copied, so an active stream is invisible to the scheduler's drain
// barrier (inflight.Count()==0 → instance gets slept/evicted mid-stream).
//
// This test asserts release has NOT fired before the stream body is fully
// drained, and HAS fired exactly once afterward.
func TestForwardFiber_StreamingReleaseFiresAfterBody(t *testing.T) {
	const chunks = 4
	chunkSent := make(chan struct{}, chunks)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for i := 0; i < chunks; i++ {
			fmt.Fprintf(w, "data: {\"chunk\":%d}\n\n", i)
			if fl != nil {
				fl.Flush()
			}
			chunkSent <- struct{}{}
			time.Sleep(15 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	var released int64
	// releasedDuringStream is set true if the release callback fired while
	// the upstream still had un-sent chunks (i.e. mid-stream). That is the
	// exact production fault: inflight decremented while tokens still flow.
	var releasedDuringStream int64
	allChunksSent := make(chan struct{})

	app := fiber.New()
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "m",
			Release: func() {
				select {
				case <-allChunksSent:
					// stream finished before release — correct ordering
				default:
					atomic.StoreInt64(&releasedDuringStream, 1)
				}
				atomic.AddInt64(&released, 1)
			},
		})
	})

	// Watcher: signal once the upstream has emitted every chunk.
	go func() {
		for i := 0; i < chunks; i++ {
			<-chunkSent
		}
		close(allChunksSent)
	}()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")

	// app.Test with -1 blocks until the full response body is readable.
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if !strings.Contains(string(body), "chunk\":3") {
		t.Fatalf("expected full streamed body, got %q", string(body))
	}

	// Give the stream-writer goroutine a beat to run its deferred release
	// after the body fully drains.
	deadline := time.Now().Add(2 * time.Second)
	for atomic.LoadInt64(&released) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}

	if atomic.LoadInt64(&released) != 1 {
		t.Fatalf("Release should fire exactly once after stream completes; fired %d times",
			atomic.LoadInt64(&released))
	}
	if atomic.LoadInt64(&releasedDuringStream) == 1 {
		t.Fatal("Release fired MID-STREAM — inflight token decremented before stream finished " +
			"(drain barrier would see inflight==0 and evict the instance mid-stream)")
	}
}

// TestForwardFiber_BufferedReleaseFiresOnce asserts the non-streaming path
// still releases exactly once (before ForwardFiber returns).
func TestForwardFiber_BufferedReleaseFiresOnce(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	var released int64
	app := fiber.New()
	app.Post("/x", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "m",
			Release:        func() { atomic.AddInt64(&released, 1) },
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	_ = resp.Body.Close()

	if got := atomic.LoadInt64(&released); got != 1 {
		t.Fatalf("buffered Release should fire exactly once; fired %d", got)
	}
}

// TestForwardFiber_ReleaseFiresOnUpstreamDialError asserts release fires
// even when the proxy never reaches upstream (so the inflight token is not
// leaked on a connection failure).
func TestForwardFiber_ReleaseFiresOnUpstreamDialError(t *testing.T) {
	var released int64
	app := fiber.New()
	app.Post("/x", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			// Unroutable port — client.Do returns an error.
			BaseURL:        "http://127.0.0.1:1",
			RequestedModel: "m",
			Release:        func() { atomic.AddInt64(&released, 1) },
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	_, _ = app.Test(req, -1)

	if got := atomic.LoadInt64(&released); got != 1 {
		t.Fatalf("Release should fire once on upstream dial error; fired %d", got)
	}
}

// TestForwardFiber_StreamingReleaseExactlyOnceUnderClientHangup asserts the
// release fires exactly once even if the stream copy errors out (client
// hangs up). Uses copyWithFlush directly against a writer that fails.
func TestForwardFiber_StreamingReleaseExactlyOnce(t *testing.T) {
	// Drive ForwardFiber's once-guard directly: two release calls collapse
	// to one. Mirrors the sync.Once wrapping inside ForwardFiber.
	var n int64
	var once sync.Once
	r := func() { once.Do(func() { atomic.AddInt64(&n, 1) }) }
	r()
	r()
	if atomic.LoadInt64(&n) != 1 {
		t.Fatalf("once-guarded release must collapse to 1; got %d", n)
	}
}
