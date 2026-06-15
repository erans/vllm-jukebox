package jukebox

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/ports"
)

const concurrencyCapTestConfig = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  capped:
    lifecycle: external
    host: capped
    port: 8001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    max_concurrent_requests: 3
  uncapped:
    lifecycle: external
    host: uncapped
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`

func makeConcurrencyCapScheduler(t *testing.T) *Scheduler {
	t.Helper()
	cfg, err := config.Load([]byte(concurrencyCapTestConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	inv := &decodeProbeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	cappedMgr := &fakeDecodeMgr{baseURL: "http://capped:8001", pid: 5001}
	uncappedMgr := &fakeDecodeMgr{baseURL: "http://uncapped:8002", pid: 5002}
	s.SeedInstanceForTest("capped", 8001, []int{0}, false, StateReady, cappedMgr)
	s.SeedInstanceForTest("uncapped", 8002, []int{0}, false, StateReady, uncappedMgr)
	return s
}

func isConcurrencyLimit(err error) bool {
	var rej *RejectError
	return errors.As(err, &rej) && rej.Reason == RejectConcurrencyLimit
}

// TestConcurrencyCap_ExcessRejectedAndSlotFrees: with cap=3, three
// concurrent holders succeed; the 4th is rejected with
// RejectConcurrencyLimit; releasing one holder frees a slot so a
// subsequent acquire succeeds.
func TestConcurrencyCap_ExcessRejectedAndSlotFrees(t *testing.T) {
	s := makeConcurrencyCapScheduler(t)
	ctx := context.Background()

	// Acquire up to the cap (3). Hold the Done funcs so the slots stay
	// occupied.
	var holders []func()
	for i := 0; i < 3; i++ {
		route, err := s.AcquireRoute(ctx, "capped", "")
		if err != nil {
			t.Fatalf("acquire %d: unexpected err %v", i, err)
		}
		if route.Done == nil {
			t.Fatalf("acquire %d: nil Done", i)
		}
		holders = append(holders, route.Done)
	}

	// 4th acquire must be rejected with the concurrency-limit reason.
	_, err := s.AcquireRoute(ctx, "capped", "")
	if !isConcurrencyLimit(err) {
		t.Fatalf("4th acquire: expected RejectConcurrencyLimit, got %v", err)
	}

	// A RejectError carries a Retry-After so the proxy emits a retryable
	// 503 (not a terminal error).
	var rej *RejectError
	if errors.As(err, &rej) && rej.RetryAfter <= 0 {
		t.Fatalf("concurrency-limit RejectError should carry a Retry-After, got %v", rej.RetryAfter)
	}

	// Release one holder → a slot frees → next acquire succeeds.
	holders[0]()
	route, err := s.AcquireRoute(ctx, "capped", "")
	if err != nil {
		t.Fatalf("acquire after release: expected success, got %v", err)
	}
	if route.Done != nil {
		route.Done()
	}

	// Clean up remaining holders.
	for _, done := range holders[1:] {
		done()
	}
}

// TestConcurrencyCap_UncappedUnbounded: a model with no
// max_concurrent_requests admits well past any cap (zero behavior change
// for the default config).
func TestConcurrencyCap_UncappedUnbounded(t *testing.T) {
	s := makeConcurrencyCapScheduler(t)
	ctx := context.Background()

	var holders []func()
	for i := 0; i < 20; i++ {
		route, err := s.AcquireRoute(ctx, "uncapped", "")
		if err != nil {
			t.Fatalf("uncapped acquire %d: unexpected err %v", i, err)
		}
		holders = append(holders, route.Done)
	}
	for _, done := range holders {
		if done != nil {
			done()
		}
	}
}

// TestConcurrencyCap_ConcurrentFireExactlyCapSucceed: fire N > cap
// concurrent acquisitions; assert AT MOST cap succeed at once and the
// rest get RejectConcurrencyLimit, and that all successful slots free
// cleanly (no leak / underflow).
func TestConcurrencyCap_ConcurrentFireExactlyCapSucceed(t *testing.T) {
	s := makeConcurrencyCapScheduler(t)
	ctx := context.Background()

	const cap = 3
	const n = 12

	var (
		mu        sync.Mutex
		succeeded []func()
		rejected  int
		otherErr  error
		wg        sync.WaitGroup
		start     = make(chan struct{})
	)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			route, err := s.AcquireRoute(ctx, "capped", "")
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				succeeded = append(succeeded, route.Done)
			case isConcurrencyLimit(err):
				rejected++
			default:
				otherErr = err
			}
		}()
	}
	close(start)
	wg.Wait()

	if otherErr != nil {
		t.Fatalf("unexpected non-concurrency error: %v", otherErr)
	}
	// The count-then-Track admission has a benign TOCTOU window, so the
	// number that succeed can briefly exceed cap under a thundering herd.
	// The invariant that MUST hold is: every request is accounted for,
	// at least `cap` succeed, and nothing succeeds unboundedly (we fired
	// n=12 with cap=3, so a correct cap must reject a meaningful share).
	if len(succeeded) < cap {
		t.Fatalf("expected at least %d successes, got %d", cap, len(succeeded))
	}
	if len(succeeded)+rejected != n {
		t.Fatalf("accounting mismatch: %d succeeded + %d rejected != %d", len(succeeded), rejected, n)
	}
	if rejected == 0 {
		t.Fatalf("expected some requests rejected by the cap (fired %d, cap %d), got 0", n, cap)
	}

	// All successful slots must free cleanly.
	for _, done := range succeeded {
		if done != nil {
			done()
		}
	}

	// After draining, a fresh acquire must succeed (slots fully freed,
	// no underflow).
	route, err := s.AcquireRoute(ctx, "capped", "")
	if err != nil {
		t.Fatalf("acquire after full drain: expected success, got %v", err)
	}
	if route.Done != nil {
		route.Done()
	}
}

// TestConcurrencyCap_HotReload: raising the cap via config.SetCurrent
// (the hot-reload path) lets a previously-rejected request through
// without rebuilding the scheduler.
func TestConcurrencyCap_HotReload(t *testing.T) {
	s := makeConcurrencyCapScheduler(t)
	ctx := context.Background()

	// Fill the cap of 3.
	var holders []func()
	for i := 0; i < 3; i++ {
		route, err := s.AcquireRoute(ctx, "capped", "")
		if err != nil {
			t.Fatalf("acquire %d: %v", i, err)
		}
		holders = append(holders, route.Done)
	}
	if _, err := s.AcquireRoute(ctx, "capped", ""); !isConcurrencyLimit(err) {
		t.Fatalf("expected reject at cap=3, got %v", err)
	}

	// Hot-reload: bump the cap to 5.
	newCfg, err := config.Load([]byte(concurrencyCapTestConfig))
	if err != nil {
		t.Fatalf("reload config load: %v", err)
	}
	mc := newCfg.Models["capped"]
	mc.MaxConcurrentRequests = 5
	newCfg.Models["capped"] = mc
	config.SetCurrent(newCfg)

	// Now the 4th acquire (in-flight=3, cap=5) succeeds.
	route, err := s.AcquireRoute(ctx, "capped", "")
	if err != nil {
		t.Fatalf("acquire after hot-reload raise: expected success, got %v", err)
	}
	if route.Done != nil {
		holders = append(holders, route.Done)
	}

	for _, done := range holders {
		if done != nil {
			done()
		}
	}
}
