package proxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
)

// TestForwardFiber_StreamIdleWatchdogReleasesOnStall is the regression test
// for the phantom-inflight leak: an upstream SSE stream that emits a few
// chunks then STALLS forever (the #45094 wedge signature — running>0, 0
// decode progress, /health lies 200) must not pin its load-bearing inflight
// token indefinitely. With no per-request client-disconnect signal and no
// client.Timeout, resp.Body.Read blocks forever pre-fix, so Release never
// fires and the drain barrier + concurrency cap count a phantom request.
//
// The idle watchdog cancels the upstream request after streamIdleTimeout of
// zero read progress, unblocking the read so the stream-writer's deferred
// Release fires. This test sends 2 chunks, then hangs the upstream handler;
// with a short idle timeout the proxy must release within a bounded window.
//
// PRE-FIX (no watchdog): Release never fires; the test times out asserting
// release==1.
func TestForwardFiber_StreamIdleWatchdogReleasesOnStall(t *testing.T) {
	prev := streamIdleTimeout
	streamIdleTimeout = 200 * time.Millisecond
	defer func() { streamIdleTimeout = prev }()

	hang := make(chan struct{})
	defer close(hang)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for i := 0; i < 2; i++ {
			fmt.Fprintf(w, "data: {\"chunk\":%d}\n\n", i)
			if fl != nil {
				fl.Flush()
			}
		}
		// Stall forever (until the server tears down) — no more bytes.
		<-r.Context().Done()
	}))
	defer upstream.Close()

	var released int64
	app := fiber.New()
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{
			BaseURL:        upstream.URL,
			RequestedModel: "m",
			Release:        func() { atomic.AddInt64(&released, 1) },
		})
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")

	// app.Test drains the body; the watchdog should fire ~streamIdleTimeout
	// after the last chunk, ending the body copy and firing Release.
	resp, err := app.Test(req, 5000)
	if err == nil && resp != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	deadline := time.Now().Add(3 * time.Second)
	for atomic.LoadInt64(&released) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if atomic.LoadInt64(&released) != 1 {
		t.Fatalf("idle watchdog must release the inflight token on a stalled "+
			"stream; release fired %d times (phantom inflight leak)",
			atomic.LoadInt64(&released))
	}
}

// TestForwardFiber_StreamIdleWatchdogDoesNotKillProgressingStream guards the
// other half of the contract: the watchdog is an IDLE timeout, not a total
// one. A slow stream that keeps emitting chunks at an interval SHORTER than
// the idle timeout must run to completion uninterrupted, even if its total
// duration exceeds the idle timeout. This protects legitimately long
// reasoning streams from being killed.
func TestForwardFiber_StreamIdleWatchdogDoesNotKillProgressingStream(t *testing.T) {
	prev := streamIdleTimeout
	streamIdleTimeout = 150 * time.Millisecond
	defer func() { streamIdleTimeout = prev }()

	const chunks = 8
	const gap = 50 * time.Millisecond // < idle timeout; total 400ms > timeout
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for i := 0; i < chunks; i++ {
			fmt.Fprintf(w, "data: {\"chunk\":%d}\n\n", i)
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(gap)
		}
	}))
	defer upstream.Close()

	app := fiber.New()
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return ForwardFiber(c, ForwardOptions{BaseURL: upstream.URL, RequestedModel: "m"})
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req, -1)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	// All chunks must be present — the progressing stream was NOT killed.
	if !strings.Contains(string(body), fmt.Sprintf("chunk\":%d", chunks-1)) {
		t.Fatalf("progressing stream was truncated by the idle watchdog; got %q", string(body))
	}
}

// TestStartIdleWatchdog_StopPreventsLeakAndCancel is a focused unit test on
// the watchdog primitive: stop() must terminate the goroutine and a stopped
// watchdog must never call cancel(). Run with -race.
func TestStartIdleWatchdog_StopPreventsCancel(t *testing.T) {
	ar := newActivityReader(strings.NewReader(""))
	var canceled int64
	stop := startIdleWatchdog(ar, 50*time.Millisecond, func() { atomic.AddInt64(&canceled, 1) })
	stop()
	// Even well past the timeout, a stopped watchdog must not cancel.
	time.Sleep(150 * time.Millisecond)
	if atomic.LoadInt64(&canceled) != 0 {
		t.Fatalf("stopped watchdog called cancel %d times; must be 0", atomic.LoadInt64(&canceled))
	}
	// stop is idempotent.
	stop()
}

// TestStartIdleWatchdog_FiresOnIdle asserts the watchdog DOES cancel when the
// reader makes no progress for the timeout.
func TestStartIdleWatchdog_FiresOnIdle(t *testing.T) {
	ar := newActivityReader(strings.NewReader(""))
	var canceled int64
	ctx, cancel := context.WithCancel(context.Background())
	_ = startIdleWatchdog(ar, 60*time.Millisecond, func() {
		atomic.AddInt64(&canceled, 1)
		cancel()
	})
	select {
	case <-ctx.Done():
		if atomic.LoadInt64(&canceled) != 1 {
			t.Fatalf("expected exactly one cancel, got %d", atomic.LoadInt64(&canceled))
		}
	case <-time.After(1 * time.Second):
		t.Fatal("watchdog did not fire on a fully idle reader")
	}
}
