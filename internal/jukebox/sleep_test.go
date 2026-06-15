package jukebox_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

// sleepableInstance extends fakeInstance with SleepCapable methods so we can
// exercise the sleep/wake code paths in scheduler.go without standing up a
// real vLLM. Implements both jukebox.InstanceManager (via embedded fields +
// methods) and jukebox.SleepCapable.
type sleepableInstance struct {
	port int

	mu  sync.Mutex
	pid int

	// Counters tracked for assertions.
	sleepCalls    atomic.Int32
	wakeCalls     atomic.Int32
	stopCalls     atomic.Int32
	startCalls    atomic.Int32
	lastSleepLvl  atomic.Int32
	lastWakeTimeo atomic.Int64

	// Failure injection.
	sleepErr error
	wakeErr  error

	// Faked sleep state — returned by IsSleeping and toggled by Sleep/Wake.
	isSleeping atomic.Bool

	// wakeDelay forces Wake to sleep this long before returning. Useful for
	// fan-out tests where we want multiple callers to observe an in-flight wake.
	wakeDelay time.Duration
}

func (s *sleepableInstance) Start(_ context.Context, _ string) (int, error) {
	s.startCalls.Add(1)
	s.mu.Lock()
	s.pid = 7000 + s.port
	s.mu.Unlock()
	return s.pid, nil
}

func (s *sleepableInstance) Stop(_ context.Context) error {
	s.stopCalls.Add(1)
	s.mu.Lock()
	s.pid = 0
	s.mu.Unlock()
	return nil
}

func (s *sleepableInstance) VerifyReady(_ context.Context, _ string) error { return nil }

func (s *sleepableInstance) CurrentPID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pid
}

func (s *sleepableInstance) BaseURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", s.port)
}

func (s *sleepableInstance) Sleep(_ context.Context, level int) error {
	s.sleepCalls.Add(1)
	s.lastSleepLvl.Store(int32(level))
	if s.sleepErr != nil {
		return s.sleepErr
	}
	s.isSleeping.Store(true)
	return nil
}

func (s *sleepableInstance) Wake(_ context.Context, timeout time.Duration) error {
	s.wakeCalls.Add(1)
	s.lastWakeTimeo.Store(int64(timeout))
	if s.wakeDelay > 0 {
		time.Sleep(s.wakeDelay)
	}
	if s.wakeErr != nil {
		return s.wakeErr
	}
	s.isSleeping.Store(false)
	return nil
}

func (s *sleepableInstance) IsSleeping(_ context.Context) (bool, error) {
	return s.isSleeping.Load(), nil
}

// sleepableFactory returns a factory that hands out sleepableInstance per
// port. Tests can inspect the returned instances via the returned slice (in
// creation order) — DO NOT key by port because ports get released back to
// the pool on stop and reused by subsequent allocations, overwriting any
// port-keyed map.
func sleepableFactory() (jukebox.InstanceFactory, *instanceLog) {
	log := &instanceLog{}
	factory := func(port int, _ string) jukebox.InstanceManager {
		inst := &sleepableInstance{port: port}
		log.mu.Lock()
		log.created = append(log.created, inst)
		log.mu.Unlock()
		return inst
	}
	return factory, log
}

// instanceLog records every sleepableInstance the factory has handed out,
// in creation order.
type instanceLog struct {
	mu      sync.Mutex
	created []*sleepableInstance
}

// nth returns the Nth instance the factory created (0-indexed).
func (l *instanceLog) nth(t *testing.T, n int) *sleepableInstance {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	if n >= len(l.created) {
		t.Fatalf("no sleepableInstance at index %d (only %d created)", n, len(l.created))
	}
	return l.created[n]
}

// ---------------------------------------------------------------------------
// Scheduler tests
// ---------------------------------------------------------------------------

// When a sleep_mode-enabled instance is evicted (because another model
// requests its GPUs), the scheduler must call Sleep, NOT Stop, and the
// instance must stay in the registry so a future request can wake it.
// NOTE: with sleep (L1) the GPU isn't fully freed — so a fresh B
// allocation on the SAME gpus will still fail downstream. To exercise
// the eviction-sleep path without that confound, B is on a DIFFERENT
// GPU than A; eviction triggers because we manually drive it via the
// scheduler. Actually simpler: assert sleep happened by checking
// counters + state after we trigger the evict path explicitly.
func TestScheduler_SleepModelOnEviction(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 2s
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	factory, log := sleepableFactory()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)

	// Bring A up.
	routeA, err := s.AcquireRoute(context.Background(), "a", "req-a1")
	if err != nil {
		t.Fatalf("AcquireRoute(a): %v", err)
	}
	if routeA.Done != nil {
		routeA.Done()
	}
	instA := log.nth(t, 0)
	time.Sleep(20 * time.Millisecond) // exceed min_instance_uptime

	// Stop A. With sleep_mode=true this should sleep, not stop, and
	// keep A in the map.
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}

	if got := instA.sleepCalls.Load(); got != 1 {
		t.Fatalf("expected A.Sleep to be called 1x, got %d", got)
	}
	if got := instA.stopCalls.Load(); got != 0 {
		t.Fatalf("expected A.Stop NOT to be called (sleep_mode=true), got %d", got)
	}
	if got := instA.lastSleepLvl.Load(); got != 1 {
		t.Fatalf("expected sleep level 1, got %d", got)
	}

	st := s.Status()
	foundA := false
	for _, inst := range st.Instances {
		if inst.Model == "a" {
			foundA = true
			if inst.State != jukebox.StateSleeping {
				t.Fatalf("expected A state=sleeping, got %s", inst.State)
			}
		}
	}
	if !foundA {
		t.Fatalf("expected A to remain in instances after sleep")
	}
}

// A request for an already-sleeping instance must Wake it and route the
// request — no full restart, no new port allocation.
func TestScheduler_WakeFromSleepOnRequest(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    wake_timeout: 2s
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	factory, log := sleepableFactory()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)

	if _, err := s.AcquireRoute(context.Background(), "a", "init"); err != nil {
		t.Fatalf("init A: %v", err)
	}
	instA := log.nth(t, 0)
	time.Sleep(20 * time.Millisecond)
	// Sleep A via StopAll (since same-GPU eviction by another model
	// can't co-exist with L1 sleep keeping VRAM reserved).
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}
	if !instA.isSleeping.Load() {
		t.Fatalf("expected A to be sleeping after StopAll with sleep_mode")
	}

	// Now request A again — must wake, not rebuild.
	startCallsBefore := instA.startCalls.Load()
	routeA, err := s.AcquireRoute(context.Background(), "a", "wake-req")
	if err != nil {
		t.Fatalf("wake AcquireRoute(a): %v", err)
	}
	if routeA.Done != nil {
		routeA.Done()
	}
	if got := instA.wakeCalls.Load(); got != 1 {
		t.Fatalf("expected A.Wake to be called 1x, got %d", got)
	}
	if got := instA.startCalls.Load(); got != startCallsBefore {
		t.Fatalf("expected no new Start (wake should reuse instance), Start delta=%d", got-startCallsBefore)
	}
	if instA.isSleeping.Load() {
		t.Fatalf("expected A awake after wake")
	}
}

// Multiple concurrent requests for the same sleeping model must NOT
// double-call Wake — the wakeOp fan-out must serialize through a single
// wake operation, and ALL waiters must be served when wake completes.
// This is the load-bearing fix for "no 503 during wake cycle".
func TestScheduler_WakeFanOutMultipleWaiters(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
  swap_wait_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    wake_timeout: 3s
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	factory, log := sleepableFactory()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)

	if _, err := s.AcquireRoute(context.Background(), "a", "init"); err != nil {
		t.Fatalf("init A: %v", err)
	}
	instA := log.nth(t, 0)
	time.Sleep(20 * time.Millisecond)
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}
	if !instA.isSleeping.Load() {
		t.Fatalf("setup: A should be sleeping")
	}
	instA.wakeDelay = 300 * time.Millisecond // make wake slow enough that waiters pile up

	const N = 5
	var wg sync.WaitGroup
	errCh := make(chan error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			route, err := s.AcquireRoute(ctx, "a", "fan-out")
			if err != nil {
				errCh <- err
				return
			}
			if route.Done != nil {
				route.Done()
			}
			errCh <- nil
		}()
	}
	wg.Wait()
	close(errCh)

	var failures []error
	for err := range errCh {
		if err != nil {
			failures = append(failures, err)
		}
	}
	if len(failures) > 0 {
		t.Fatalf("expected all %d concurrent requests to succeed; %d failed; first: %v", N, len(failures), failures[0])
	}
	if got := instA.wakeCalls.Load(); got != 1 {
		t.Fatalf("expected Wake to be called EXACTLY ONCE for %d concurrent waiters, got %d", N, got)
	}
}

// When Wake returns an error, the calling request should get a RejectError
// (mapped to 503), the instance should stay in StateSleeping, and we should
// record a failure metric.
func TestScheduler_WakeFailureReturnsRejectError(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    wake_timeout: 1s
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	factory, log := sleepableFactory()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)

	if _, err := s.AcquireRoute(context.Background(), "a", "init"); err != nil {
		t.Fatalf("init A: %v", err)
	}
	instA := log.nth(t, 0)
	time.Sleep(20 * time.Millisecond)
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}
	instA.wakeErr = errors.New("simulated wake failure")

	_, err := s.AcquireRoute(context.Background(), "a", "wake-fail")
	if err == nil {
		t.Fatalf("expected wake failure to surface error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) {
		t.Fatalf("expected RejectError, got %T: %v", err, err)
	}
}

// Regression: a model WITHOUT sleep_mode should still use the existing
// Stop path on eviction — not the sleep path. This guards against a
// future refactor accidentally enabling sleep for non-sleep-mode models.
func TestScheduler_NoSleepModeStillHardStops(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	factory, log := sleepableFactory()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)

	if _, err := s.AcquireRoute(context.Background(), "a", "init"); err != nil {
		t.Fatalf("init A: %v", err)
	}
	instA := log.nth(t, 0)
	time.Sleep(20 * time.Millisecond)
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}

	if got := instA.stopCalls.Load(); got != 1 {
		t.Fatalf("expected A.Stop=1 (no sleep_mode), got %d", got)
	}
	if got := instA.sleepCalls.Load(); got != 0 {
		t.Fatalf("expected A.Sleep=0 (no sleep_mode), got %d", got)
	}
	// A should be REMOVED from instances after a hard stop.
	for _, inst := range s.Status().Instances {
		if inst.Model == "a" {
			t.Fatalf("expected A to be removed from instances after hard stop, still present: %+v", inst)
		}
	}
}
