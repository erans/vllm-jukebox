package jukebox_test

// Regression tests for the T4 502 RCA: ComfyUI readiness must be
// probed on every AcquireRoute, not gated by ComfyUIManager's in-memory
// sleeping flag.
//
// Background: ComfyUIManager.IsSleeping reports a process-local flag.
// If ComfyUI is started/restarted/recreated outside jukebox's view
// (cron restart, operator `docker start`, container recreation), the
// flag reports sleeping=false even though the upstream port is not
// yet bound. The downstream image-gen handler then opens a TCP
// connection against an unbound port and the client sees `connect:
// refused` surfaced as 502.
//
// Fix: AcquireRoute always invokes /system_stats (via VerifyReady)
// for LifecycleComfyUI peers with a bounded budget. On budget
// exhaustion we surface 503 RejectColdLoading — the honest mapping
// for "model still loading" — instead of letting the request hit an
// unbound port and 502.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

// fakeComfyServer is a minimal httptest-backed ComfyUI stand-in. The
// systemStatsOK flag lets tests flip between "ready" (200 on
// /system_stats) and "warming up" (500 / connection-refused-via-close).
type fakeComfyServer struct {
	srv             *httptest.Server
	systemStatsHits atomic.Int32
	// healthy controls the response: when false, /system_stats returns
	// 500 so the probe loop keeps ticking until its budget exhausts.
	healthy atomic.Bool
}

func newFakeComfyServer() *fakeComfyServer {
	f := &fakeComfyServer{}
	f.healthy.Store(true)
	mux := http.NewServeMux()
	mux.HandleFunc("/system_stats", func(w http.ResponseWriter, _ *http.Request) {
		f.systemStatsHits.Add(1)
		if !f.healthy.Load() {
			http.Error(w, "warming up", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"system":{}}`))
	})
	f.srv = httptest.NewServer(mux)
	return f
}

func (f *fakeComfyServer) Close() {
	f.srv.Close()
}

func (f *fakeComfyServer) Host() string {
	host, _, _ := net.SplitHostPort(strings.TrimPrefix(f.srv.URL, "http://"))
	return host
}

func (f *fakeComfyServer) Port() int {
	_, port, _ := net.SplitHostPort(strings.TrimPrefix(f.srv.URL, "http://"))
	var p int
	for _, c := range port {
		p = p*10 + int(c-'0')
	}
	return p
}

// TestAcquireRoute_ComfyUI_UnreachablePortReturns503WithinBudget proves
// the architectural fix: when a LifecycleComfyUI peer's upstream port
// is not bound, AcquireRoute MUST surface a 503-class RejectError
// within the readiness budget. Before the fix this path 502'd because
// the in-memory sleeping=false flag short-circuited the readiness
// probe and the request flowed through to a `connect: refused`.
func TestAcquireRoute_ComfyUI_UnreachablePortReturns503WithinBudget(t *testing.T) {
	// Allocate a TCP port and immediately close it so it is unbound.
	// AcquireRoute will see ECONNREFUSED inside ComfyUIManager.VerifyReady.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	hostport := l.Addr().String()
	host, portStr, _ := net.SplitHostPort(hostport)
	var port int
	for _, c := range portStr {
		port = port*10 + int(c-'0')
	}
	if err := l.Close(); err != nil {
		t.Fatalf("l.Close: %v", err)
	}

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  imagegen:
    lifecycle: comfyui
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
`, host, port)

	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	// Shrink the readiness budget so this test takes ms, not 60s. The
	// production budget exists to cover ComfyUI plugin warm-up; in this
	// test the warm-up never completes (port stays unbound) so we just
	// need enough headroom for one or two probe ticks.
	restore := jukebox.SetComfyUIReadyProbeBudgetForTest(150 * time.Millisecond)
	defer restore()

	// RegisterExternalInstances also seeds ComfyUI-lifecycle peers
	// (registerComfyUIInstances runs at the tail). We don't care about
	// the startup probe's outcome — it logs a warning when unreachable
	// and continues.
	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	start := time.Now()
	_, acqErr := s.AcquireRoute(context.Background(), "imagegen", "req-test")
	dur := time.Since(start)

	if acqErr == nil {
		t.Fatalf("expected AcquireRoute to fail when ComfyUI port is unbound, got nil error")
	}

	// Must be a RejectError so the HTTP layer maps it to 503 (not 502).
	var rej *jukebox.RejectError
	if !errors.As(acqErr, &rej) {
		t.Fatalf("expected *jukebox.RejectError, got %T: %v", acqErr, acqErr)
	}
	if rej.Reason != jukebox.RejectColdLoading {
		t.Fatalf("expected Reason=%q, got %q", jukebox.RejectColdLoading, rej.Reason)
	}
	if rej.RetryAfter <= 0 {
		t.Fatalf("expected non-zero Retry-After hint, got %v", rej.RetryAfter)
	}

	// Budget bound: must surface within ~budget plus a small margin
	// for goroutine wake-up. The 60s production budget is reduced to
	// 150ms above; allow up to 1s for CI noise.
	if dur > time.Second {
		t.Fatalf("AcquireRoute took %v — must surface within readiness budget (~150ms + slack)", dur)
	}
}

// TestAcquireRoute_ComfyUI_HealthyPeerStillRoutes proves the fix does
// not regress the happy path: a reachable ComfyUI peer must still
// produce a route after the readiness probe succeeds.
func TestAcquireRoute_ComfyUI_HealthyPeerStillRoutes(t *testing.T) {
	fake := newFakeComfyServer()
	defer fake.Close()

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  imagegen:
    lifecycle: comfyui
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
`, fake.Host(), fake.Port())

	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	restore := jukebox.SetComfyUIReadyProbeBudgetForTest(2 * time.Second)
	defer restore()

	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	startHits := fake.systemStatsHits.Load()
	route, err := s.AcquireRoute(context.Background(), "imagegen", "req-happy")
	if err != nil {
		t.Fatalf("AcquireRoute(imagegen) on healthy peer: %v", err)
	}
	if route.BaseURL != fake.srv.URL {
		t.Fatalf("BaseURL mismatch: got %q want %q", route.BaseURL, fake.srv.URL)
	}
	if route.Done != nil {
		route.Done()
	}

	// Confirm AcquireRoute actually hit /system_stats — proves the
	// probe ran (decoupled from the in-memory sleeping flag, which was
	// false before the call).
	if endHits := fake.systemStatsHits.Load(); endHits <= startHits {
		t.Fatalf("expected /system_stats hits to grow on AcquireRoute (start=%d end=%d)", startHits, endHits)
	}
}
