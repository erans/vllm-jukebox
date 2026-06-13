package jukebox

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/metrics"
)

type State string

const (
	StateIdle     State = "idle"
	StateStarting State = "starting"
	StateReady    State = "ready"
	StateStopping State = "stopping"
	StateSleeping State = "sleeping"
	// StateStopped — container is `docker compose stop`'d (used by
	// stop-on-evict for evict_action: stop members). Distinct from
	// StateSleeping in that the container's process tree is gone and
	// /wake_up does NOT apply — restart goes through `docker compose
	// up` + health-wait via the redeploy-member CLI verb.
	StateStopped State = "stopped"
	StateError   State = "error"
)

type RejectReason string

const (
	RejectSwapInProgress RejectReason = "in_progress"
	RejectCooldown       RejectReason = "cooldown"
	RejectBackoff        RejectReason = "backoff"
	RejectWaitTimeout    RejectReason = "wait_timeout"
	RejectPinnedConflict RejectReason = "pinned_conflict"
	RejectNoCapacity     RejectReason = "no_capacity"
	RejectInsufficient   RejectReason = "insufficient_resources"
	RejectMinUptime      RejectReason = "min_uptime"
	// RejectColdLoading — the requested model is currently admissionStopped
	// (its container is `docker compose stop`'d) and a cold-load has been
	// kicked off in the background. The client should retry after the
	// RetryAfter duration. Distinct from RejectSwapInProgress because
	// the wall-clock is much longer (~5 min for moe/longctx) and the
	// proxy-layer async-503 path uses this reason to plumb the right
	// human-readable message + Retry-After through to the consumer.
	RejectColdLoading RejectReason = "cold_loading"
	// RejectAdminIntervention — admission's books may have drifted from
	// physical container state (cleanup `docker stop` failed mid-cold-load
	// or peer-stop failed mid-redeploy). Auto-retry could stomp on a
	// recovering process; operator must reconcile manually. Maps to 503
	// with NO Retry-After so SDKs surface the error rather than retry-loop.
	RejectAdminIntervention RejectReason = "admin_intervention"
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
	State         State
	CurrentModel  string
	InFlight      int64
	PID           int
	FailureCount  int
	UptimeSeconds int64
	LastReadyAt   time.Time
	LastSwap      *LastSwap
	LastSwapAt    time.Time

	// Scheduler mode (multi-instance). These fields are best-effort and are
	// currently unused by the legacy single-instance coordinator.
	SchedulerBusy bool
	Waiters       []string
	Instances     []InstanceStatus
}

type InstanceStatus struct {
	Model      string
	Port       int
	GPUs       []int
	PID        int
	State      State
	Pinned     bool
	Draining   bool
	InFlight   int64
	StartedAt  time.Time
	LastUsedAt time.Time
}

type Coordinator struct {
	cfg      *config.Config
	mgr      Manager
	tr       *inflight.Tracker
	powerMgr *gpu.PowerManager
	now      func() time.Time

	requests chan ensureReq

	mu             sync.RWMutex
	state          State
	currentModel   string
	lastSwapTime   time.Time
	failureCount   int
	lastFailure    time.Time
	swapInProgress bool

	lastReadyAt time.Time
	lastSwap    LastSwap
	hasLastSwap bool
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
	from      string
	model     string
	requestID string
	duration  time.Duration
	err       error
}

type LastSwap struct {
	From               string
	To                 string
	At                 time.Time
	Duration           time.Duration
	TriggeredByRequest string
}

func NewCoordinator(cfg *config.Config, mgr Manager, tr *inflight.Tracker, now func() time.Time) *Coordinator {
	return NewCoordinatorWithPower(cfg, mgr, tr, now, nil)
}

func NewCoordinatorWithPower(cfg *config.Config, mgr Manager, tr *inflight.Tracker, now func() time.Time, powerMgr *gpu.PowerManager) *Coordinator {
	if now == nil {
		now = time.Now
	}
	return &Coordinator{
		cfg:      cfg,
		mgr:      mgr,
		tr:       tr,
		powerMgr: powerMgr,
		now:      now,
		requests: make(chan ensureReq),
		state:    StateIdle,
	}
}

func (c *Coordinator) Run(ctx context.Context) {
	for {
		select {
		case req := <-c.requests:
			c.handleEnsure(req)
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
		metrics.SwapRejectionsTotal.WithLabelValues(string(RejectWaitTimeout)).Inc()
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
	now := c.now()
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

	state := c.state
	// If we think we're ready but there's no running process, surface this as error immediately
	// (covers unexpected vLLM exit without needing a new request to arrive).
	if state == StateReady && pid == 0 {
		state = StateError
	}

	var uptimeSeconds int64
	if state == StateReady && !c.lastReadyAt.IsZero() {
		uptimeSeconds = int64(now.Sub(c.lastReadyAt).Seconds())
		if uptimeSeconds < 0 {
			uptimeSeconds = 0
		}
	}

	var lastSwap *LastSwap
	var lastSwapAt time.Time
	if c.hasLastSwap {
		copy := c.lastSwap
		lastSwap = &copy
		lastSwapAt = copy.At
	}

	return Status{
		State:         state,
		CurrentModel:  c.currentModel,
		InFlight:      inFlight,
		PID:           pid,
		FailureCount:  c.failureCount,
		UptimeSeconds: uptimeSeconds,
		LastReadyAt:   c.lastReadyAt,
		LastSwap:      lastSwap,
		LastSwapAt:    lastSwapAt,
	}
}

func (c *Coordinator) handleEnsure(req ensureReq) {
	resolvedName, _, err := liveResolveModel(c.cfg, req.requestedModel)
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
		if c.mgr == nil || c.mgr.CurrentPID() == 0 {
			// Treat "ready but no pid" as error and fall through to perform a restart.
		} else {
			req.resp <- ensureReply{}
			return
		}
	}

	// Sleep-mode wake path: if we're currently sleeping the requested
	// model, wake it (re-uses CPU-RAM weights, no full restart).
	if state == StateSleeping && current == resolvedName {
		c.mu.Lock()
		c.swapInProgress = true
		c.state = StateStarting
		c.mu.Unlock()
		metrics.SetState(string(StateStarting))

		done := make(chan error, 1)
		go c.performWake(context.Background(), req.requestID, done)
		req.resp <- ensureReply{wait: done}
		return
	}

	isCrashed := state == StateReady && current == resolvedName && c.mgr != nil && c.mgr.CurrentPID() == 0

	if inProgress || state == StateStarting || state == StateStopping {
		metrics.SwapRejectionsTotal.WithLabelValues(string(RejectSwapInProgress)).Inc()
		req.resp <- ensureReply{err: &RejectError{Reason: RejectSwapInProgress, RetryAfter: 5 * time.Second}}
		return
	}

	if state == StateError && failures > 0 {
		delay := backoffDelay(failures)
		if delay > 0 {
			until := lastFailure.Add(delay)
			if remaining := time.Until(until); remaining > 0 {
				metrics.SwapRejectionsTotal.WithLabelValues(string(RejectBackoff)).Inc()
				req.resp <- ensureReply{err: &RejectError{Reason: RejectBackoff, RetryAfter: remaining}}
				return
			}
		}
	}

	if !isCrashed {
		if cooldown := c.cfg.VLLM.SwapCooldown.Duration; cooldown > 0 && !lastSwap.IsZero() {
			nextAllowed := lastSwap.Add(cooldown)
			if now := c.now(); now.Before(nextAllowed) {
				metrics.SwapRejectionsTotal.WithLabelValues(string(RejectCooldown)).Inc()
				req.resp <- ensureReply{err: &RejectError{Reason: RejectCooldown, RetryAfter: nextAllowed.Sub(now)}}
				return
			}
		}
	}

	done := make(chan error, 1)
	c.mu.RLock()
	fromModel := c.currentModel
	c.mu.RUnlock()
	c.mu.Lock()
	c.swapInProgress = true
	if state == StateReady {
		c.state = StateStopping
	} else {
		c.state = StateStarting
	}
	c.mu.Unlock()
	metrics.SetState(string(c.Status().State))

	go c.performSwap(fromModel, resolvedName, req.requestID, done)
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
		c.lastReadyAt = c.now()
		c.failureCount = 0
		c.lastFailure = time.Time{}

		c.lastSwap = LastSwap{
			From:               ev.from,
			To:                 ev.model,
			At:                 c.lastSwapTime,
			Duration:           ev.duration,
			TriggeredByRequest: ev.requestID,
		}
		c.hasLastSwap = true
		metrics.SetState(string(c.state))
		if ev.from != "" || ev.model != "" {
			metrics.SwapsTotal.WithLabelValues(ev.from, ev.model).Inc()
			metrics.SwapDurationSeconds.Observe(ev.duration.Seconds())
		}
		metrics.ConsecutiveFailures.Set(0)
		if ev.from != "" && ev.from != ev.model {
			metrics.CurrentModel.WithLabelValues(ev.from).Set(0)
		}
		if ev.model != "" {
			metrics.CurrentModel.WithLabelValues(ev.model).Set(1)
		}
		slog.Info(
			"model_swap_completed",
			"request_id", ev.requestID,
			"from_model", ev.from,
			"to_model", ev.model,
			"duration_ms", ev.duration.Milliseconds(),
		)
		return
	}

	c.state = StateError
	c.failureCount++
	c.lastFailure = c.now()
	metrics.SetState(string(c.state))
	metrics.ConsecutiveFailures.Set(float64(c.failureCount))

	slog.Error(
		"model_swap_failed",
		"request_id", ev.requestID,
		"from_model", ev.from,
		"to_model", ev.model,
		"duration_ms", ev.duration.Milliseconds(),
		"failure_count", c.failureCount,
		"err", ev.err,
	)
}

func (c *Coordinator) performSwap(fromModel, model, requestID string, done chan<- error) {
	start := c.now()
	slog.Info(
		"model_swap_started",
		"request_id", requestID,
		"from_model", fromModel,
		"to_model", model,
	)
	err := c.doSwap(model, requestID)
	duration := c.now().Sub(start)

	c.handleSwapDone(swapDone{from: fromModel, model: model, requestID: requestID, duration: duration, err: err})

	select {
	case done <- err:
	default:
	}
	close(done)
}

func (c *Coordinator) doSwap(model, requestID string) error {
	_, modelCfg, err := liveResolveModel(c.cfg, model)
	if err != nil {
		return err
	}

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

		// Revert power limits after stop (if we have GPUs configured)
		if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
	}

	c.mu.Lock()
	c.state = StateStarting
	c.mu.Unlock()

	// Apply power limits before start (if configured)
	if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
		if err := c.powerMgr.ApplyModelLimits(context.Background(), modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			return err
		}
	}

	startCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	_, err = c.mgr.Start(startCtx, model)
	cancel()
	if err != nil {
		return err
	}

	verifyCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	err = c.mgr.VerifyReady(verifyCtx, model)
	cancel()
	if err != nil {
		// Ensure we don't leave a partially-started vLLM process around if verification fails.
		stopCtx, stopCancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		_ = c.mgr.Stop(stopCtx)
		stopCancel()
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
