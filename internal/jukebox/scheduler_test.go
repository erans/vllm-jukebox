package jukebox_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

type fakeInventory struct {
	mu   sync.Mutex
	gpus []gpu.GPU
	err  error
}

func (f *fakeInventory) List(_ context.Context) ([]gpu.GPU, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	out := make([]gpu.GPU, len(f.gpus))
	copy(out, f.gpus)
	return out, nil
}

type fakeInstanceController struct {
	mu          sync.Mutex
	startBlocks map[string]<-chan struct{}
	stopOrder   []string
}

type fakeInstance struct {
	ctrl *fakeInstanceController
	port int

	mu  sync.Mutex
	pid int

	startedModel string
}

func (f *fakeInstance) Start(ctx context.Context, modelName string) (int, error) {
	f.ctrl.mu.Lock()
	ch := f.ctrl.startBlocks[modelName]
	f.ctrl.mu.Unlock()

	if ch != nil {
		select {
		case <-ch:
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}
	f.mu.Lock()
	f.pid = 123
	f.startedModel = modelName
	f.mu.Unlock()
	return 123, nil
}

func (f *fakeInstance) Stop(_ context.Context) error {
	f.mu.Lock()
	model := f.startedModel
	f.mu.Unlock()
	f.ctrl.mu.Lock()
	f.ctrl.stopOrder = append(f.ctrl.stopOrder, fmt.Sprintf("%s@%d", model, f.port))
	f.ctrl.mu.Unlock()
	f.mu.Lock()
	f.pid = 0
	f.mu.Unlock()
	return nil
}

func (f *fakeInstance) VerifyReady(_ context.Context, _ string) error { return nil }

func (f *fakeInstance) CurrentPID() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pid
}

func (f *fakeInstance) BaseURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", f.port)
}

func mustLoadSchedulerCfg(t *testing.T, yaml string) *config.Config {
	t.Helper()
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	return cfg
}

func TestScheduler_ReadyModelRoutesWhileAnotherModelIsScheduling(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
  b:
    path: "/models/b"
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
`)

	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
		{Index: 1, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8100, 8109)

	blockA := make(chan struct{})
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{"a": blockA}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }

	now := time.Now()
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, func() time.Time { return now }, factory)

	// Start model b (non-blocking).
	routeB, err := s.AcquireRoute(context.Background(), "b", "req-b")
	if err != nil {
		t.Fatalf("AcquireRoute(b): %v", err)
	}
	if routeB.Done != nil {
		routeB.Done()
	}

	// Start model a in background (blocked).
	errCh := make(chan error, 1)
	go func() {
		_, err := s.AcquireRoute(context.Background(), "a", "req-a")
		errCh <- err
	}()

	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		if s.Status().SchedulerBusy {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected scheduler to become busy")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// While a is blocked in scheduling, b should still route immediately.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := s.AcquireRoute(ctx, "b", "req-b2"); err != nil {
		t.Fatalf("AcquireRoute(b) while scheduling: %v", err)
	}

	close(blockA)
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("AcquireRoute(a) returned error: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for model a to finish scheduling")
	}
}

func TestScheduler_PinnedConflictRejected(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
  startup_timeout: 5s
models:
  small:
    path: "/models/small"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    pinned: true
  big:
    path: "/models/big"
    gpus: [0,1]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
		{Index: 1, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8100, 8109)

	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory)

	routeSmall, err := s.AcquireRoute(context.Background(), "small", "req-1")
	if err != nil {
		t.Fatalf("AcquireRoute(small): %v", err)
	}
	if routeSmall.Done != nil {
		routeSmall.Done()
	}
	_, err = s.AcquireRoute(context.Background(), "big", "req-2")
	if err == nil {
		t.Fatalf("expected error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) || rej.Reason != jukebox.RejectPinnedConflict {
		t.Fatalf("expected RejectPinnedConflict, got %T %v", err, err)
	}
}

func TestScheduler_LRUEvictsMultipleNonPinnedConflicts(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 0s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  s0:
    path: "/models/s0"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
  s1:
    path: "/models/s1"
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
  big:
    path: "/models/big"
    gpus: [0,1]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
		{Index: 1, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8100, 8109)

	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }

	var nowMu sync.Mutex
	now := time.Unix(1000, 0)
	nowFn := func() time.Time {
		nowMu.Lock()
		defer nowMu.Unlock()
		return now
	}
	advance := func(d time.Duration) {
		nowMu.Lock()
		now = now.Add(d)
		nowMu.Unlock()
	}

	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, nowFn, factory)

	// Start s0 then s1 (s0 is older / LRU).
	routeS0, err := s.AcquireRoute(context.Background(), "s0", "req-s0")
	if err != nil {
		t.Fatalf("AcquireRoute(s0): %v", err)
	}
	if routeS0.Done != nil {
		routeS0.Done()
	}
	advance(1 * time.Second)
	routeS1, err := s.AcquireRoute(context.Background(), "s1", "req-s1")
	if err != nil {
		t.Fatalf("AcquireRoute(s1): %v", err)
	}
	if routeS1.Done != nil {
		routeS1.Done()
	}

	// Touch s0 to make s1 the LRU.
	advance(1 * time.Second)
	routeS0b, err := s.AcquireRoute(context.Background(), "s0", "req-s0b")
	if err != nil {
		t.Fatalf("AcquireRoute(s0) touch: %v", err)
	}
	if routeS0b.Done != nil {
		routeS0b.Done()
	}

	if _, err := s.AcquireRoute(context.Background(), "big", "req-big"); err != nil {
		t.Fatalf("AcquireRoute(big): %v", err)
	}

	ctrl.mu.Lock()
	defer ctrl.mu.Unlock()
	if len(ctrl.stopOrder) != 2 {
		t.Fatalf("expected 2 evictions, got %v", ctrl.stopOrder)
	}
}

func TestScheduler_RejectsWhenGPUMissing(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
  startup_timeout: 5s
models:
  m:
    path: "/models/m"
    gpus: [9]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory)

	_, err := s.AcquireRoute(context.Background(), "m", "req-1")
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestScheduler_RejectsWhenInsufficientVRAM(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
  startup_timeout: 5s
models:
  m:
    path: "/models/m"
    gpus: [0]
    min_free_mem_mb_per_gpu: 999
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory)

	_, err := s.AcquireRoute(context.Background(), "m", "req-1")
	if err == nil {
		t.Fatalf("expected error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) || rej.Reason != jukebox.RejectInsufficient {
		t.Fatalf("expected RejectInsufficient, got %T %v", err, err)
	}
}

func TestScheduler_MaxInstancesEnforced(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  max_instances: 1
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
  b:
    path: "/models/b"
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
		{Index: 1, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8100, 8109)
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory)

	routeA, err := s.AcquireRoute(context.Background(), "a", "req-a")
	if err != nil {
		t.Fatalf("AcquireRoute(a): %v", err)
	}
	if routeA.Done != nil {
		routeA.Done()
	}

	_, err = s.AcquireRoute(context.Background(), "b", "req-b")
	if err == nil {
		t.Fatalf("expected error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) || rej.Reason != jukebox.RejectNoCapacity {
		t.Fatalf("expected RejectNoCapacity, got %T %v", err, err)
	}
}

func TestScheduler_MinInstanceUptimePreventsEviction(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1h
vllm:
  port: 8000
  startup_timeout: 5s
models:
  small:
    path: "/models/small"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
  big:
    path: "/models/big"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }

	var nowMu sync.Mutex
	now := time.Unix(1000, 0)
	nowFn := func() time.Time {
		nowMu.Lock()
		defer nowMu.Unlock()
		return now
	}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, nowFn, factory)

	routeSmall, err := s.AcquireRoute(context.Background(), "small", "req-1")
	if err != nil {
		t.Fatalf("AcquireRoute(small): %v", err)
	}
	if routeSmall.Done != nil {
		routeSmall.Done()
	}

	_, err = s.AcquireRoute(context.Background(), "big", "req-2")
	if err == nil {
		t.Fatalf("expected error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) || rej.Reason != jukebox.RejectMinUptime {
		t.Fatalf("expected RejectMinUptime, got %T %v", err, err)
	}
	if rej.RetryAfter <= 0 {
		t.Fatalf("expected RetryAfter > 0, got %v", rej.RetryAfter)
	}
}

func TestScheduler_OnlyOneWaiterPerModel(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
  startup_timeout: 5s
  swap_wait_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 100, FreeMB: 100}}}
	pool := ports.New(8100, 8109)

	blockA := make(chan struct{})
	ctrl := &fakeInstanceController{startBlocks: map[string]<-chan struct{}{"a": blockA}}
	factory := func(port int, _ string) jukebox.InstanceManager { return &fakeInstance{ctrl: ctrl, port: port} }
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory)

	// First request acquires scheduling permit and blocks in Start.
	firstDone := make(chan struct{})
	go func() {
		_, _ = s.AcquireRoute(context.Background(), "a", "req-1")
		close(firstDone)
	}()

	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		if s.Status().SchedulerBusy {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected scheduler to become busy")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Second request becomes the single waiter (blocks).
	secondErr := make(chan error, 1)
	go func() {
		_, err := s.AcquireRoute(context.Background(), "a", "req-2")
		secondErr <- err
	}()

	deadline = time.Now().Add(500 * time.Millisecond)
	for {
		st := s.Status()
		found := false
		for _, w := range st.Waiters {
			if w == "a" {
				found = true
				break
			}
		}
		if found {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected waiter registration")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Third request should be rejected immediately.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, err := s.AcquireRoute(ctx, "a", "req-3")
	if err == nil {
		t.Fatalf("expected error")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) || rej.Reason != jukebox.RejectSwapInProgress {
		t.Fatalf("expected RejectSwapInProgress, got %T %v", err, err)
	}

	close(blockA)
	select {
	case <-firstDone:
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for first request")
	}
	select {
	case err := <-secondErr:
		if err != nil {
			t.Fatalf("expected second waiter to succeed, got %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for second request")
	}
}
