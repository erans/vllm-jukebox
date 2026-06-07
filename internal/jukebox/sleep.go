package jukebox

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/vllm"
)

// SleepCapable is the optional interface that an InstanceManager (or the
// swap-mode coordinator's Manager) implements when it supports vLLM's
// sleep-mode API. Both vllm.Manager and vllm.ExternalManager satisfy it.
type SleepCapable interface {
	Sleep(ctx context.Context, level int) error
	Wake(ctx context.Context, timeout time.Duration) error
	IsSleeping(ctx context.Context) (bool, error)
}

// wakeOp tracks an in-flight wake for a specific model. Multiple
// concurrent requests for a sleeping model coordinate through a shared
// wakeOp so we never reject a request with 503 during a wake cycle (the
// reason the spec explicitly bumps "request queuing during model
// switches" from non-goal to v1 requirement for sleep-mode).
type wakeOp struct {
	done chan struct{}
	err  error
}

// settleAfterSleep is how long jukebox waits after vLLM reports
// is_sleeping=true before considering GPU memory fully released. CUDA
// managed-memory release is async; without this delay a back-to-back
// "sleep A → wake B" sequence can OOM.
const settleAfterSleep = 2 * time.Second

// sleepPollTimeout caps how long we'll poll /is_sleeping after POSTing
// /sleep before assuming the call was synchronous and proceeding.
const sleepPollTimeout = 5 * time.Second

// idleCheckInterval is how often the auto-suspend goroutine wakes to
// check for idle instances eligible for sleeping.
const idleCheckInterval = 30 * time.Second

// ---------------------------------------------------------------------------
// Scheduler-mode sleep / wake
// ---------------------------------------------------------------------------

// sleepInstance drains the instance, calls SleepCapable.Sleep, polls
// IsSleeping until true (or sleepPollTimeout), waits settleAfterSleep,
// and flips state to StateSleeping. The instance is KEPT in s.instances
// — sleep is the replacement for stop in eviction/auto-suspend paths,
// and a future request for this model will trigger wake instead of a
// fresh build.
//
// Reason is recorded on the SleepsTotal counter — "evict" when called
// from drainAndStopInstance, "idle" from auto-suspend, "manual" from
// admin paths.
func (s *Scheduler) sleepInstance(ctx context.Context, inst *schedInstance, level int, reason string) error {
	if inst == nil {
		return fmt.Errorf("sleepInstance: nil instance")
	}
	sc, ok := inst.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("instance %q: manager does not support sleep", inst.model)
	}

	start := s.now()

	s.mu.Lock()
	inst.draining = true
	prevState := inst.state
	inst.state = StateStopping
	s.mu.Unlock()

	// Drain in-flight requests so /sleep doesn't race with active
	// generation. vLLM's PR #16536 lands graceful error handling for
	// requests-during-sleep, but waiting is still the polite default.
	drainTimeout := s.cfg.VLLM.DrainTimeout.Duration
	if drainTimeout <= 0 {
		drainTimeout = 60 * time.Second
	}
	drainCtx, cancel := context.WithTimeout(context.Background(), drainTimeout)
	_ = inst.inflight.WaitForDrain(drainCtx)
	cancel()

	if err := sc.Sleep(ctx, level); err != nil {
		metrics.SleepFailuresTotal.WithLabelValues(inst.model, "sleep_api").Inc()
		// Restore prior state so callers can fall back to a hard stop.
		s.mu.Lock()
		inst.state = prevState
		inst.draining = false
		s.mu.Unlock()
		return fmt.Errorf("sleep API: %w", err)
	}

	// Poll /is_sleeping briefly. vLLM /sleep may be synchronous or async
	// depending on version; this loop is cheap insurance.
	pollCtx, pollCancel := context.WithTimeout(ctx, sleepPollTimeout)
	for {
		sleeping, err := sc.IsSleeping(pollCtx)
		if err == nil && sleeping {
			break
		}
		select {
		case <-time.After(200 * time.Millisecond):
			continue
		case <-pollCtx.Done():
			// Treat as best-effort confirmation.
		}
		break
	}
	pollCancel()

	// Settle: CUDA managed-memory release is async. Without this delay,
	// a "sleep A → wake B" sequence can OOM the GPU.
	time.Sleep(settleAfterSleep)

	s.mu.Lock()
	inst.state = StateSleeping
	inst.draining = false
	s.mu.Unlock()

	dur := s.now().Sub(start)
	metrics.SleepsTotal.WithLabelValues(inst.model, reason).Inc()
	metrics.SleepDurationSeconds.WithLabelValues(inst.model).Observe(dur.Seconds())
	slog.Info("instance_sleeping",
		"model", inst.model,
		"reason", reason,
		"level", level,
		"duration_ms", dur.Milliseconds(),
	)
	return nil
}

// tryRouteFromSleep handles the "instance exists but is sleeping" case.
// Returns (route, true, nil) on a successful wake + route, or
// (_, true, err) on wake failure. (_, false, nil) means "caller should
// proceed to the regular slow-path scheduling logic" (instance doesn't
// exist, or isn't sleeping, or a race left it in some other state).
//
// Multiple concurrent requests for the same sleeping model share a
// single wakeOp — the first creates it and performs the wake; the rest
// wait on op.done. None get 503-rejected during wake.
func (s *Scheduler) tryRouteFromSleep(ctx context.Context, resolvedName string, modelCfg config.ModelConfig) (Route, bool, error) {
	s.mu.Lock()
	inst := s.instances[resolvedName]
	if inst == nil || inst.state != StateSleeping {
		s.mu.Unlock()
		return Route{}, false, nil
	}
	op := s.wakeOps[resolvedName]
	if op != nil {
		// Wake already in flight — wait for it.
		s.mu.Unlock()
		timeout := modelCfg.EffectiveWakeTimeout()
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		select {
		case <-op.done:
			if op.err != nil {
				return Route{}, true, mapWakeError(op.err)
			}
			// Retry fast path now that wake is done.
			if route, ok := s.tryRouteReady(ctx, resolvedName, modelCfg.Path); ok {
				return route, true, nil
			}
			// Lost the race (e.g. auto-suspend slept it again immediately).
			return Route{}, false, nil
		case <-timer.C:
			metrics.SwapRejectionsTotal.WithLabelValues(string(RejectWaitTimeout)).Inc()
			return Route{}, true, &RejectError{Reason: RejectWaitTimeout, RetryAfter: 5 * time.Second, Message: "wake timed out, please retry"}
		case <-ctx.Done():
			return Route{}, true, ctx.Err()
		}
	}

	// We are the first — own the wake.
	op = &wakeOp{done: make(chan struct{})}
	s.wakeOps[resolvedName] = op
	s.mu.Unlock()

	op.err = s.performWake(ctx, inst, modelCfg, "request")

	s.mu.Lock()
	delete(s.wakeOps, resolvedName)
	s.mu.Unlock()
	close(op.done)

	if op.err != nil {
		return Route{}, true, mapWakeError(op.err)
	}
	if route, ok := s.tryRouteReady(ctx, resolvedName, modelCfg.Path); ok {
		return route, true, nil
	}
	return Route{}, false, nil
}

// performWake POSTs /wake_up, polls /health until ready, flips state,
// and records metrics. Does NOT take the scheduler semaphore — wake is
// a per-instance operation that doesn't touch GPU layouts, so it
// shouldn't block scheduling of other models.
func (s *Scheduler) performWake(ctx context.Context, inst *schedInstance, modelCfg config.ModelConfig, trigger string) error {
	sc, ok := inst.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("instance %q: manager does not support wake", inst.model)
	}
	start := s.now()
	timeout := modelCfg.EffectiveWakeTimeout()

	if err := sc.Wake(ctx, timeout); err != nil {
		metrics.SleepFailuresTotal.WithLabelValues(inst.model, "wake_api").Inc()
		slog.Error("instance_wake_failed", "model", inst.model, "err", err, "timeout", timeout)
		return err
	}

	s.mu.Lock()
	inst.state = StateReady
	inst.lastUsedAt = s.now()
	s.mu.Unlock()

	dur := s.now().Sub(start)
	metrics.WakesTotal.WithLabelValues(inst.model, trigger).Inc()
	metrics.WakeDurationSeconds.WithLabelValues(inst.model).Observe(dur.Seconds())
	slog.Info("instance_woken",
		"model", inst.model,
		"trigger", trigger,
		"duration_ms", dur.Milliseconds(),
	)
	return nil
}

// mapWakeError converts a raw wake error to the user-facing error shape.
func mapWakeError(err error) error {
	return &RejectError{
		Reason:     RejectSwapInProgress,
		RetryAfter: 10 * time.Second,
		Message:    fmt.Sprintf("wake failed: %v", err),
	}
}

// ---------------------------------------------------------------------------
// External-lifecycle bootstrap
// ---------------------------------------------------------------------------

// RegisterExternalInstances walks the config and registers a
// schedInstance for every model with lifecycle: external. These
// instances are seeded in StateReady (we assume the operator's
// external vLLM service is up); a startup health probe logs a warning
// if it isn't, but doesn't fail. The state machine reconciles on the
// first real request.
//
// Per-instance probes run in parallel: serial probing would stall
// startup by ~3s × N_externals before the HTTP server can come up.
func (s *Scheduler) RegisterExternalInstances(ctx context.Context) error {
	if s == nil || s.cfg == nil {
		return nil
	}

	type pending struct {
		name string
		mgr  *vllm.ExternalManager
	}
	var todo []pending

	for name, model := range s.cfg.Models {
		if model.EffectiveLifecycle() != config.LifecycleExternal {
			continue
		}
		// Defense: aliases-with-external are rejected at config load,
		// but skip-on-encounter is cheap and keeps this method robust to
		// future config changes.
		if model.Alias != "" {
			continue
		}

		mgr := vllm.NewExternalManager(model.Host, model.Port)
		inst := &schedInstance{
			model:      name,
			port:       model.Port,
			gpus:       append([]int(nil), model.GPUs...),
			pinned:     model.Pinned != nil && *model.Pinned,
			state:      StateReady,
			mgr:        mgr,
			startedAt:  s.now(),
			lastUsedAt: s.now(),
		}
		portLabel := strconv.Itoa(model.Port)
		inst.inflight.OnChange = func(count int64) {
			metrics.InstanceInFlightRequests.WithLabelValues(inst.model, portLabel).Set(float64(count))
		}

		// Register synchronously so a request arriving mid-probe still
		// has a route to take.
		s.mu.Lock()
		s.instances[name] = inst
		s.mu.Unlock()
		metrics.RunningInstances.WithLabelValues(name, portLabel).Set(1)
		slog.Info("registered external instance", "model", name, "url", mgr.BaseURL())
		todo = append(todo, pending{name: name, mgr: mgr})
	}

	// Probe in parallel. Wall-clock is bounded by the slowest probe
	// (~3s) regardless of N. Each probe is best-effort.
	var wg sync.WaitGroup
	for _, p := range todo {
		wg.Add(1)
		go func(p pending) {
			defer wg.Done()
			probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			if err := p.mgr.VerifyReady(probeCtx, p.name); err != nil {
				slog.Warn("external instance not reachable at startup; will reconcile on first request",
					"model", p.name,
					"url", p.mgr.BaseURL(),
					"err", err,
				)
			}
		}(p)
	}
	wg.Wait()
	return nil
}

// ---------------------------------------------------------------------------
// Auto-suspend (scheduler mode)
// ---------------------------------------------------------------------------

// IdleMonitor runs the auto-suspend loop. Tick every idleCheckInterval,
// scan instances, and sleep any non-pinned, sleep-mode-enabled instance
// whose lastUsedAt is older than its configured idle_timeout. Runs
// until ctx is cancelled. Start once from main.go as a goroutine.
func (s *Scheduler) IdleMonitor(ctx context.Context) {
	ticker := time.NewTicker(idleCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.checkIdle(ctx)
		}
	}
}

func (s *Scheduler) checkIdle(ctx context.Context) {
	now := s.now()

	// Snapshot under read lock.
	s.mu.RLock()
	candidates := make([]*schedInstance, 0, len(s.instances))
	for _, inst := range s.instances {
		if inst == nil {
			continue
		}
		if inst.pinned || inst.state != StateReady || inst.draining {
			continue
		}
		candidates = append(candidates, inst)
	}
	s.mu.RUnlock()

	for _, inst := range candidates {
		modelCfg, ok := s.cfg.Models[inst.model]
		if !ok || !modelCfg.SleepMode {
			continue
		}
		idleTimeout := modelCfg.IdleTimeout.Duration
		if idleTimeout <= 0 {
			continue
		}
		if inst.inflight.Count() > 0 {
			continue
		}
		idle := now.Sub(inst.lastUsedAt)
		if idle < idleTimeout {
			continue
		}
		slog.Info("auto_suspending_idle_instance",
			"model", inst.model,
			"idle_seconds", int64(idle.Seconds()),
			"timeout_seconds", int64(idleTimeout.Seconds()),
		)
		if err := s.sleepInstance(ctx, inst, modelCfg.EffectiveSleepLevel(), "idle"); err != nil {
			slog.Warn("auto_suspend_failed", "model", inst.model, "err", err)
		}
	}
}

// ---------------------------------------------------------------------------
// Swap-mode (coordinator) sleep / wake
// ---------------------------------------------------------------------------

// performSleep handles auto-suspend for the legacy single-instance
// coordinator. It drains, calls Sleep, polls IsSleeping, settles, and
// flips state to StateSleeping under the coordinator's lock.
func (c *Coordinator) performSleep(ctx context.Context, level int, reason string) error {
	if c.mgr == nil {
		return fmt.Errorf("coordinator has no manager")
	}
	sc, ok := c.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("coordinator manager does not support sleep")
	}

	c.mu.Lock()
	currentModel := c.currentModel
	prevState := c.state
	c.state = StateStopping
	c.swapInProgress = true
	c.mu.Unlock()
	metrics.SetState(string(StateStopping))

	start := c.now()

	if c.tr != nil {
		drainTimeout := c.cfg.VLLM.DrainTimeout.Duration
		if drainTimeout <= 0 {
			drainTimeout = 60 * time.Second
		}
		drainCtx, cancel := context.WithTimeout(context.Background(), drainTimeout)
		_ = c.tr.WaitForDrain(drainCtx)
		cancel()
	}

	if err := sc.Sleep(ctx, level); err != nil {
		metrics.SleepFailuresTotal.WithLabelValues(currentModel, "sleep_api").Inc()
		c.mu.Lock()
		c.state = prevState
		c.swapInProgress = false
		c.mu.Unlock()
		metrics.SetState(string(prevState))
		return err
	}

	pollCtx, pollCancel := context.WithTimeout(ctx, sleepPollTimeout)
	for {
		sleeping, err := sc.IsSleeping(pollCtx)
		if err == nil && sleeping {
			break
		}
		select {
		case <-time.After(200 * time.Millisecond):
			continue
		case <-pollCtx.Done():
		}
		break
	}
	pollCancel()
	time.Sleep(settleAfterSleep)

	c.mu.Lock()
	c.state = StateSleeping
	c.swapInProgress = false
	c.mu.Unlock()
	metrics.SetState(string(StateSleeping))

	dur := c.now().Sub(start)
	metrics.SleepsTotal.WithLabelValues(currentModel, reason).Inc()
	metrics.SleepDurationSeconds.WithLabelValues(currentModel).Observe(dur.Seconds())
	slog.Info("coordinator_sleeping",
		"model", currentModel,
		"reason", reason,
		"level", level,
		"duration_ms", dur.Milliseconds(),
	)
	return nil
}

// performWake handles waking the single coordinator-managed instance.
func (c *Coordinator) performWake(ctx context.Context, requestID string, done chan<- error) {
	defer func() { close(done) }()

	c.mu.Lock()
	current := c.currentModel
	c.mu.Unlock()

	_, modelCfg, err := c.cfg.ResolveModel(current)
	if err != nil {
		select {
		case done <- err:
		default:
		}
		return
	}
	sc, ok := c.mgr.(SleepCapable)
	if !ok {
		err := fmt.Errorf("coordinator manager does not support wake")
		select {
		case done <- err:
		default:
		}
		return
	}

	start := c.now()
	timeout := modelCfg.EffectiveWakeTimeout()
	wakeErr := sc.Wake(ctx, timeout)

	c.mu.Lock()
	if wakeErr == nil {
		c.state = StateReady
		c.lastReadyAt = c.now()
	} else {
		c.state = StateError
		c.failureCount++
		c.lastFailure = c.now()
	}
	c.swapInProgress = false
	c.mu.Unlock()
	metrics.SetState(string(c.Status().State))

	dur := c.now().Sub(start)
	if wakeErr == nil {
		metrics.WakesTotal.WithLabelValues(current, "request").Inc()
		metrics.WakeDurationSeconds.WithLabelValues(current).Observe(dur.Seconds())
		slog.Info("coordinator_woken",
			"model", current,
			"request_id", requestID,
			"duration_ms", dur.Milliseconds(),
		)
	} else {
		metrics.SleepFailuresTotal.WithLabelValues(current, "wake_api").Inc()
		slog.Error("coordinator_wake_failed",
			"model", current,
			"request_id", requestID,
			"err", wakeErr,
		)
	}

	select {
	case done <- wakeErr:
	default:
	}
}

// IdleMonitor for coordinator: auto-suspend the single instance after
// idle_timeout, if the current model has SleepMode and is not pinned.
func (c *Coordinator) IdleMonitor(ctx context.Context) {
	ticker := time.NewTicker(idleCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.checkIdle(ctx)
		}
	}
}

func (c *Coordinator) checkIdle(ctx context.Context) {
	c.mu.RLock()
	state := c.state
	current := c.currentModel
	lastReady := c.lastReadyAt
	swapInProgress := c.swapInProgress
	c.mu.RUnlock()

	if state != StateReady || swapInProgress || current == "" {
		return
	}
	if c.tr != nil && c.tr.Count() > 0 {
		return
	}
	modelCfg, ok := c.cfg.Models[current]
	if !ok || !modelCfg.SleepMode || modelCfg.IdleTimeout.Duration <= 0 {
		return
	}
	if modelCfg.Pinned != nil && *modelCfg.Pinned {
		return
	}
	since := c.now().Sub(lastReady)
	if since < modelCfg.IdleTimeout.Duration {
		return
	}
	slog.Info("coordinator_auto_suspending",
		"model", current,
		"idle_seconds", int64(since.Seconds()),
		"timeout_seconds", int64(modelCfg.IdleTimeout.Duration.Seconds()),
	)
	if err := c.performSleep(ctx, modelCfg.EffectiveSleepLevel(), "idle"); err != nil {
		slog.Warn("coordinator_auto_suspend_failed", "model", current, "err", err)
	}
}
