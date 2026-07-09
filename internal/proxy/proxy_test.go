package proxy_test

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"vllm-jukebox/internal/proxy"
)

// Test that OnDone is NOT called while an SSE stream is still open, and IS
// called once the stream writer completes. This is the regression test for the
// critical bug where the handler's defer route.Done() fired on handler return.
func TestForwardFiber_SSE_OnDoneHeldUntilStreamCompletes(t *testing.T) {
	streamStarted := make(chan struct{})
	streamRelease := make(chan struct{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: chunk1\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		close(streamStarted)
		<-streamRelease // hold the stream open
		fmt.Fprintf(w, "data: chunk2\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	var doneCount int32
	onDone := func() { atomic.AddInt32(&doneCount, 1) }

	app := fiber.New()
	app.Use(recover.New())
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL: upstream.URL,
			OnDone:  onDone,
		})
	})

	// Drive the request in a goroutine; fasthttp's stream writer runs after the
	// handler returns, but we need the request to progress far enough to start
	// the stream. Use a real http.Client against the fiber app.
	listenAddr := "127.0.0.1:0"
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go app.Listener(ln)
	defer app.Shutdown()
	base := "http://" + ln.Addr().String()

	client := &http.Client{}
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Accept", "text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()

	select {
	case <-streamStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("stream never started")
	}

	// While the stream is open, OnDone must NOT have fired.
	if got := atomic.LoadInt32(&doneCount); got != 0 {
		t.Fatalf("OnDone fired while stream open: %d", got)
	}

	close(streamRelease)
	io.Copy(io.Discard, resp.Body) // drain to let the writer finish

	// After the stream completes, OnDone must fire exactly once.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt32(&doneCount) == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := atomic.LoadInt32(&doneCount); got != 1 {
		t.Fatalf("OnDone not fired exactly once after stream: %d", got)
	}
}

// Buffered path: OnDone fires after c.Send, before ForwardFiber returns.
func TestForwardFiber_Buffered_OnDoneAfterSend(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	var doneCount int32
	onDone := func() { atomic.AddInt32(&doneCount, 1) }

	app := fiber.New()
	app.Use(recover.New())
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		if err := proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL: upstream.URL,
			OnDone:  onDone,
		}); err != nil {
			return err
		}
		// OnDone must have fired by the time ForwardFiber returns for buffered.
		if got := atomic.LoadInt32(&doneCount); got != 1 {
			t.Fatalf("OnDone not fired after buffered send: %d", got)
		}
		return nil
	})

	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	go app.Listener(ln)
	defer app.Shutdown()
	base := "http://" + ln.Addr().String()

	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}
