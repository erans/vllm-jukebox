package jukebox_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
)

type fakeManager struct {
	startBlock <-chan struct{}
	verifyErr  error
	startErr   error
	stopErr    error

	startedModel string
	startCalls   int
	stopCalls    int
	pid          int
}

func (m *fakeManager) Start(ctx context.Context, modelName string) (int, error) {
	m.startCalls++
	m.startedModel = modelName
	if m.startBlock != nil {
		select {
		case <-m.startBlock:
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}
	if m.startErr != nil {
		return 0, m.startErr
	}
	m.pid = 12345
	return m.pid, nil
}

func (m *fakeManager) Stop(ctx context.Context) error {
	m.stopCalls++
	if m.stopErr != nil {
		return m.stopErr
	}
	return nil
}

func (m *fakeManager) VerifyReady(ctx context.Context, expectedModel string) error {
	if m.verifyErr != nil {
		return m.verifyErr
	}
	return nil
}

func (m *fakeManager) CurrentPID() int {
	return m.pid
}

func TestCoordinator_StartsFromIdleAndBecomesReady(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{}

	now := time.Date(2025, 12, 14, 0, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }

	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "m", "req_1"); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}

	st := c.Status()
	if st.State != jukebox.StateReady {
		t.Fatalf("expected state ready, got %s", st.State)
	}
	if st.CurrentModel != "m" {
		t.Fatalf("expected currentModel=m, got %q", st.CurrentModel)
	}
	if mgr.startCalls != 1 {
		t.Fatalf("expected 1 start call, got %d", mgr.startCalls)
	}
}

func TestCoordinator_RejectsOtherRequestsDuringSwap(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  swap_wait_timeout: 2s
models:
  a:
    path: "/models/a"
  b:
    path: "/models/b"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	block := make(chan struct{})
	mgr := &fakeManager{startBlock: block}

	clock := func() time.Time { return time.Unix(0, 0) }
	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	errs := make(chan error, 1)
	go func() {
		errs <- c.EnsureModel(context.Background(), "a", "req_swap")
	}()

	// Wait until swap has started.
	deadline := time.Now().Add(250 * time.Millisecond)
	for {
		if c.Status().State == jukebox.StateStarting || c.Status().State == jukebox.StateStopping {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for swap to begin")
		}
		time.Sleep(5 * time.Millisecond)
	}

	err2 := c.EnsureModel(context.Background(), "b", "req_other")
	var rej *jukebox.RejectError
	if !errors.As(err2, &rej) || rej.Reason != jukebox.RejectSwapInProgress {
		t.Fatalf("expected swap-in-progress rejection, got %v", err2)
	}

	close(block)

	select {
	case err1 := <-errs:
		if err1 != nil {
			t.Fatalf("expected swap success, got %v", err1)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("timed out waiting for triggering request to complete")
	}
}

func TestCoordinator_RestartsIfProcessCrashedWhileReady(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{}
	clock := func() time.Time { return time.Unix(0, 0) }

	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "m", "req_1"); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	if mgr.startCalls != 1 {
		t.Fatalf("expected 1 start call, got %d", mgr.startCalls)
	}

	// Simulate crash: pid is now unknown/zero.
	mgr.pid = 0

	if err := c.EnsureModel(context.Background(), "m", "req_2"); err != nil {
		t.Fatalf("EnsureModel after crash: %v", err)
	}
	if mgr.startCalls != 2 {
		t.Fatalf("expected restart (2 start calls), got %d", mgr.startCalls)
	}
}

func TestCoordinator_StatusReportsErrorIfReadyButNoPID(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{}
	clock := func() time.Time { return time.Unix(0, 0) }

	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "m", "req_1"); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	mgr.pid = 0

	st := c.Status()
	if st.State != jukebox.StateError {
		t.Fatalf("expected status state error, got %s", st.State)
	}
}

func TestCoordinator_StopsStartedProcessIfVerifyFails(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  shutdown_timeout: 1s
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{verifyErr: errors.New("verify failed")}
	clock := func() time.Time { return time.Unix(0, 0) }

	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	err = c.EnsureModel(context.Background(), "m", "req_1")
	if err == nil {
		t.Fatalf("expected EnsureModel to fail")
	}
	if mgr.startCalls != 1 {
		t.Fatalf("expected 1 start call, got %d", mgr.startCalls)
	}
	if mgr.stopCalls != 1 {
		t.Fatalf("expected 1 stop call after verify failure, got %d", mgr.stopCalls)
	}
}

// TestCoordinator_BoundedRetryBackoffAndCircuitOpen proves HIGH #2: a verify
// failure does NOT lead to an immediate retry (backoff > 0 even at the first
// failure), and after enough consecutive failures the circuit opens and stops
// auto-retrying entirely instead of looping on 3-5 min cold-loads.
func TestCoordinator_BoundedRetryBackoffAndCircuitOpen(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  shutdown_timeout: 1s
  swap_cooldown: 0s
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{verifyErr: errors.New("forward-pass verification failed (engine not serving): poisoned")}

	// Mutable clock so we can advance past each backoff window to drive the
	// failure count up deterministically.
	now := time.Unix(0, 0)
	clock := func() time.Time { return now }

	c := jukebox.NewCoordinator(cfg, mgr, &tr, clock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	// First attempt actually runs the swap and fails verification.
	if err := c.EnsureModel(context.Background(), "m", "req_1"); err == nil {
		t.Fatalf("expected first EnsureModel to fail verification")
	}
	if mgr.startCalls != 1 {
		t.Fatalf("expected exactly 1 start call after first failure, got %d", mgr.startCalls)
	}

	// HIGH #2 core: the immediate follow-up must be rejected with a NON-ZERO
	// backoff (the old backoffDelay(1)==0 let it cold-reload instantly → loop).
	err2 := c.EnsureModel(context.Background(), "m", "req_2")
	var rej *jukebox.RejectError
	if !errors.As(err2, &rej) || rej.Reason != jukebox.RejectBackoff {
		t.Fatalf("expected backoff rejection at failureCount==1, got %v", err2)
	}
	if rej.RetryAfter <= 0 {
		t.Fatalf("backoff RetryAfter must be > 0 at first failure, got %v", rej.RetryAfter)
	}
	// No new swap should have been attempted while backing off.
	if mgr.startCalls != 1 {
		t.Fatalf("backoff must NOT trigger another cold-load, start calls=%d", mgr.startCalls)
	}

	// Drive consecutive failures up by advancing past each backoff window.
	// Each successful re-entry runs the swap (which fails again).
	for i := 0; i < 10; i++ {
		now = now.Add(10 * time.Minute) // past any backoff window
		err := c.EnsureModel(context.Background(), "m", "req_loop")
		var r *jukebox.RejectError
		if errors.As(err, &r) && r.Reason == jukebox.RejectCircuitOpen {
			// Circuit opened: from here on, NO further cold-loads happen.
			startsAtOpen := mgr.startCalls
			now = now.Add(10 * time.Minute)
			err3 := c.EnsureModel(context.Background(), "m", "req_after_open")
			var r3 *jukebox.RejectError
			if !errors.As(err3, &r3) || r3.Reason != jukebox.RejectCircuitOpen {
				t.Fatalf("expected circuit to stay open, got %v", err3)
			}
			if mgr.startCalls != startsAtOpen {
				t.Fatalf("circuit-open must stop cold-loads: starts went %d -> %d", startsAtOpen, mgr.startCalls)
			}
			return // success: bounded
		}
	}
	t.Fatalf("circuit never opened after repeated failures (start calls=%d) — retry loop is unbounded", mgr.startCalls)
}
