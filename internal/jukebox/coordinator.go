package jukebox

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/inflight"
)

type State string

const (
	StateIdle     State = "idle"
	StateStarting State = "starting"
	StateReady    State = "ready"
	StateStopping State = "stopping"
	StateError    State = "error"
)

type RejectReason string

const (
	RejectSwapInProgress RejectReason = "in_progress"
	RejectCooldown       RejectReason = "cooldown"
	RejectBackoff        RejectReason = "backoff"
	RejectWaitTimeout    RejectReason = "wait_timeout"
)

type RejectError struct {
	Reason     RejectReason
	RetryAfter time.Duration
	Message    string
}

func (e *RejectError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	if e.RetryAfter > 0 {
		return fmt.Sprintf("request rejected (%s), retry after %s", e.Reason, e.RetryAfter)
	}
	return fmt.Sprintf("request rejected (%s)", e.Reason)
}

type Manager interface {
	Start(ctx context.Context, modelName string) (pid int, err error)
	Stop(ctx context.Context) error
	VerifyReady(ctx context.Context, expectedModel string) error
	CurrentPID() int
}

type Status struct {
	State        State
	CurrentModel string
	InFlight     int64
	PID          int
	FailureCount int
}

type Coordinator struct {
	cfg *config.Config
	mgr Manager
	tr  *inflight.Tracker
	now func() time.Time

	requests chan ensureReq
	events   chan swapDone

	mu             sync.RWMutex
	state          State
	currentModel   string
	lastSwapTime   time.Time
	failureCount   int
	lastFailure    time.Time
	swapInProgress bool
}

type ensureReq struct {
	requestedModel string
	requestID      string
	resp           chan ensureReply
}

type ensureReply struct {
	wait <-chan error
	err  error
}

type swapDone struct {
	model string
	err   error
}

func NewCoordinator(cfg *config.Config, mgr Manager, tr *inflight.Tracker, now func() time.Time) *Coordinator {
	if now == nil {
		now = time.Now
	}
	return &Coordinator{
		cfg:      cfg,
		mgr:      mgr,
		tr:       tr,
		now:      now,
		requests: make(chan ensureReq),
		events:   make(chan swapDone, 1),
		state:    StateIdle,
	}
}

func (c *Coordinator) Run(ctx context.Context) {
	for {
		select {
		case req := <-c.requests:
			c.handleEnsure(req)
		case ev := <-c.events:
			c.handleSwapDone(ev)
		case <-ctx.Done():
			return
		}
	}
}

func (c *Coordinator) EnsureModel(ctx context.Context, requestedModel, requestID string) error {
	replyCh := make(chan ensureReply, 1)
	select {
	case c.requests <- ensureReq{requestedModel: requestedModel, requestID: requestID, resp: replyCh}:
	case <-ctx.Done():
		return ctx.Err()
	}

	var reply ensureReply
	select {
	case reply = <-replyCh:
	case <-ctx.Done():
		return ctx.Err()
	}

	if reply.err != nil {
		return reply.err
	}
	if reply.wait == nil {
		return nil
	}

	waitTimeout := c.cfg.VLLM.SwapWaitTimeout.Duration
	if waitTimeout <= 0 {
		waitTimeout = 60 * time.Second
	}
	timer := time.NewTimer(waitTimeout)
	defer timer.Stop()

	select {
	case err := <-reply.wait:
		return err
	case <-timer.C:
		return &RejectError{
			Reason:     RejectWaitTimeout,
			RetryAfter: 0,
			Message:    "model swap in progress, please retry",
		}
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *Coordinator) Status() Status {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var inFlight int64
	if c.tr != nil {
		inFlight = c.tr.Count()
	}

	pid := 0
	if c.mgr != nil {
		pid = c.mgr.CurrentPID()
	}

	return Status{
		State:        c.state,
		CurrentModel: c.currentModel,
		InFlight:     inFlight,
		PID:          pid,
		FailureCount: c.failureCount,
	}
}

func (c *Coordinator) handleEnsure(req ensureReq) {
	resolvedName, _, err := c.cfg.ResolveModel(req.requestedModel)
	if err != nil {
		req.resp <- ensureReply{err: err}
		return
	}

	c.mu.RLock()
	state := c.state
	current := c.currentModel
	lastSwap := c.lastSwapTime
	failures := c.failureCount
	lastFailure := c.lastFailure
	inProgress := c.swapInProgress
	c.mu.RUnlock()

	if state == StateReady && current == resolvedName {
		req.resp <- ensureReply{}
		return
	}

	if inProgress || state == StateStarting || state == StateStopping {
		req.resp <- ensureReply{err: &RejectError{Reason: RejectSwapInProgress, RetryAfter: 5 * time.Second}}
		return
	}

	if state == StateError && failures > 0 {
		delay := backoffDelay(failures)
		if delay > 0 {
			until := lastFailure.Add(delay)
			if remaining := time.Until(until); remaining > 0 {
				req.resp <- ensureReply{err: &RejectError{Reason: RejectBackoff, RetryAfter: remaining}}
				return
			}
		}
	}

	if cooldown := c.cfg.VLLM.SwapCooldown.Duration; cooldown > 0 && !lastSwap.IsZero() {
		nextAllowed := lastSwap.Add(cooldown)
		if now := c.now(); now.Before(nextAllowed) {
			req.resp <- ensureReply{err: &RejectError{Reason: RejectCooldown, RetryAfter: nextAllowed.Sub(now)}}
			return
		}
	}

	done := make(chan error, 1)
	c.mu.Lock()
	c.swapInProgress = true
	if state == StateReady {
		c.state = StateStopping
	} else {
		c.state = StateStarting
	}
	c.mu.Unlock()

	go c.performSwap(resolvedName, req.requestID, done)
	req.resp <- ensureReply{wait: done}
}

func (c *Coordinator) handleSwapDone(ev swapDone) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.swapInProgress = false

	if ev.err == nil {
		c.state = StateReady
		c.currentModel = ev.model
		c.lastSwapTime = c.now()
		c.failureCount = 0
		c.lastFailure = time.Time{}
		return
	}

	c.state = StateError
	c.failureCount++
	c.lastFailure = c.now()
}

func (c *Coordinator) performSwap(model, requestID string, done chan<- error) {
	err := c.doSwap(model, requestID)

	select {
	case done <- err:
	default:
	}
	close(done)

	select {
	case c.events <- swapDone{model: model, err: err}:
	default:
		// If the coordinator is shutting down, dropping this event is fine; state won't be updated.
	}
}

func (c *Coordinator) doSwap(model, requestID string) error {
	// If we already have a running model, stop it after draining in-flight.
	if st := c.Status(); st.State == StateStopping || st.State == StateReady {
		if c.tr != nil {
			drainCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.DrainTimeout.Duration)
			err := c.tr.WaitForDrain(drainCtx)
			cancel()
			if err != nil && !errors.Is(err, context.DeadlineExceeded) {
				return err
			}
		}

		stopCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		stopErr := c.mgr.Stop(stopCtx)
		cancel()
		if stopErr != nil {
			return stopErr
		}
	}

	c.mu.Lock()
	c.state = StateStarting
	c.mu.Unlock()

	startCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	_, err := c.mgr.Start(startCtx, model)
	cancel()
	if err != nil {
		return err
	}

	verifyCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	err = c.mgr.VerifyReady(verifyCtx, model)
	cancel()
	if err != nil {
		return err
	}

	return nil
}

func backoffDelay(failureCount int) time.Duration {
	switch {
	case failureCount <= 1:
		return 0
	case failureCount == 2:
		return 5 * time.Second
	case failureCount == 3:
		return 15 * time.Second
	case failureCount == 4:
		return 30 * time.Second
	default:
		return 60 * time.Second
	}
}

