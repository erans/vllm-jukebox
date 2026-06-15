package jukebox

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/ports"
)

// drainRaceInventory is a minimal gpu.Inventory for the barrier test.
type drainRaceInventory struct{ gpus []gpu.GPU }

func (d *drainRaceInventory) List(_ context.Context) ([]gpu.GPU, error) {
	out := make([]gpu.GPU, len(d.gpus))
	copy(out, d.gpus)
	return out, nil
}

// barrierInstance is an InstanceManager that detects the black-hole bug: any
// forward (BaseURL read) that happens after Stop() has been called is a
// violation, because it means a request was routed onto an engine that the
// evictor had already decided to tear down.
type barrierInstance struct {
	port      int
	pid       int64 // atomic; 0 once stopped
	stopped   int32 // atomic flag set when Stop() is invoked
	violation *int32
}

func (b *barrierInstance) Start(_ context.Context, _ string) (int, error) {
	atomic.StoreInt64(&b.pid, 123)
	return 123, nil
}

func (b *barrierInstance) Stop(_ context.Context) error {
	atomic.StoreInt32(&b.stopped, 1)
	atomic.StoreInt64(&b.pid, 0)
	return nil
}

func (b *barrierInstance) VerifyReady(_ context.Context, _ string) error { return nil }

func (b *barrierInstance) CurrentPID() int { return int(atomic.LoadInt64(&b.pid)) }

func (b *barrierInstance) BaseURL() string {
	// A route reads BaseURL when it forwards. If Stop() already ran, this is a
	// request being forwarded into a dying/stopped engine — the exact
	// black-hole AUDIT-E describes.
	if atomic.LoadInt32(&b.stopped) == 1 {
		atomic.AddInt32(b.violation, 1)
	}
	return "http://127.0.0.1:0"
}

func newBarrierScheduler(t *testing.T, violation *int32) (*Scheduler, func(port int, cuda string) InstanceManager) {
	t.Helper()
	cfg, err := config.Load([]byte(`
scheduler:
  port_range_start: 8200
  port_range_end: 8299
  min_instance_uptime: 0s
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 5s
  shutdown_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	inv := &drainRaceInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8200, 8299)
	factory := func(port int, _ string) InstanceManager {
		return &barrierInstance{port: port, violation: violation}
	}
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, nil)
	return s, factory
}

// TestDrainBarrier_NoRouteOntoDrainingInstance is the regression test for the
// AUDIT-E drain-barrier TOCTOU. It races many AcquireRoute("a") calls against
// concurrent eviction of the "a" instance. The invariant: NO request may
// forward (read BaseURL) onto an instance after the evictor has Stop()'d it.
//
// Pre-fix (readiness check + inflight.Track() in separate critical sections,
// and drainAndStopInstance reading the count outside the lock that set
// draining) this reliably records violations under -race. Post-fix (check +
// Track sealed under s.mu, count re-read under the same lock that sets
// draining) it records zero.
func TestDrainBarrier_NoRouteOntoDrainingInstance(t *testing.T) {
	const rounds = 200

	for round := 0; round < rounds; round++ {
		var violation int32
		s, _ := newBarrierScheduler(t, &violation)

		// Stand up the "a" instance.
		route, err := s.AcquireRoute(context.Background(), "a", "warmup")
		if err != nil {
			t.Fatalf("round %d: warmup AcquireRoute: %v", round, err)
		}
		if route.Done != nil {
			route.Done()
		}

		// Grab the live *schedInstance so the evictor can target it directly,
		// mirroring what evictConflicts/StopAll do internally.
		s.mu.RLock()
		inst := s.instances["a"]
		s.mu.RUnlock()
		if inst == nil {
			t.Fatalf("round %d: expected instance 'a' to exist", round)
		}

		var wg sync.WaitGroup

		// Router goroutines hammer the ready path while eviction happens.
		const routers = 8
		for i := 0; i < routers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < 4; j++ {
					r, ok := s.tryRouteReady(context.Background(), "a", "/models/a")
					if ok && r.Done != nil {
						r.Done()
					}
				}
			}()
		}

		// Evictor goroutine drains+stops the instance concurrently.
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = s.drainAndStopInstance(context.Background(), inst, true)
		}()

		wg.Wait()

		if v := atomic.LoadInt32(&violation); v != 0 {
			t.Fatalf("round %d: %d request(s) forwarded onto a stopped instance "+
				"(drain-barrier TOCTOU: route Track()'d after evictor Stop()'d)", round, v)
		}
	}
}
