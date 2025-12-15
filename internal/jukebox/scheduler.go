package jukebox

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
	"vllm-jukebox/internal/vllm"
)

type Scheduler struct {
	cfg   *config.Config
	inv   gpu.Inventory
	ports *ports.Pool
	now   func() time.Time
	new   InstanceFactory

	// sched is a semaphore (size 1) that serializes scheduling operations.
	sched chan struct{}

	mu        sync.RWMutex
	instances map[string]*schedInstance // keyed by resolved model name
	waiters   map[string]bool           // at most one waiter per resolved model

	total inflight.Tracker
}

type schedInstance struct {
	model string
	port  int
	gpus  []int

	pinned     bool
	draining   bool
	startedAt  time.Time
	lastUsedAt time.Time

	state    State
	mgr      InstanceManager
	inflight inflight.Tracker
}

type InstanceManager interface {
	Start(ctx context.Context, modelName string) (pid int, err error)
	Stop(ctx context.Context) error
	VerifyReady(ctx context.Context, expectedModel string) error
	CurrentPID() int
	BaseURL() string
}

type InstanceFactory func(port int, cudaVisibleDevices string) InstanceManager

func NewScheduler(cfg *config.Config, inv gpu.Inventory, portPool *ports.Pool, now func() time.Time) *Scheduler {
	return NewSchedulerWithFactory(cfg, inv, portPool, now, nil)
}

func NewSchedulerWithFactory(cfg *config.Config, inv gpu.Inventory, portPool *ports.Pool, now func() time.Time, factory InstanceFactory) *Scheduler {
	if now == nil {
		now = time.Now
	}
	if factory == nil {
		factory = func(port int, cudaVisibleDevices string) InstanceManager {
			return vllm.NewInstanceManager(cfg, port, map[string]string{"CUDA_VISIBLE_DEVICES": cudaVisibleDevices})
		}
	}
	s := &Scheduler{
		cfg:       cfg,
		inv:       inv,
		ports:     portPool,
		now:       now,
		new:       factory,
		sched:     make(chan struct{}, 1),
		instances: map[string]*schedInstance{},
		waiters:   map[string]bool{},
	}
	s.total.OnChange = func(count int64) {
		metrics.InFlightRequests.Set(float64(count))
	}
	return s
}

func (s *Scheduler) Status() Status {
	now := s.now()
	s.mu.RLock()
	defer s.mu.RUnlock()

	instances := make([]InstanceStatus, 0, len(s.instances))
	for _, inst := range s.instances {
		pid := 0
		if inst.mgr != nil {
			pid = inst.mgr.CurrentPID()
		}
		instances = append(instances, InstanceStatus{
			Model:      inst.model,
			Port:       inst.port,
			GPUs:       append([]int(nil), inst.gpus...),
			PID:        pid,
			State:      inst.state,
			Pinned:     inst.pinned,
			Draining:   inst.draining,
			InFlight:   inst.inflight.Count(),
			StartedAt:  inst.startedAt,
			LastUsedAt: inst.lastUsedAt,
		})
	}
	sort.Slice(instances, func(i, j int) bool { return instances[i].Model < instances[j].Model })

	waiters := make([]string, 0, len(s.waiters))
	for name := range s.waiters {
		waiters = append(waiters, name)
	}
	sort.Strings(waiters)

	busy := len(s.sched) == cap(s.sched) // held if channel full

	accepting := false
	for _, inst := range s.instances {
		if inst.state == StateReady && !inst.draining && inst.mgr != nil && inst.mgr.CurrentPID() != 0 {
			accepting = true
			break
		}
	}

	state := StateIdle
	if accepting {
		state = StateReady
	} else if busy {
		state = StateStarting
	}

	// Best-effort uptime: pick the newest ready instance as representative.
	var lastReadyAt time.Time
	for _, inst := range s.instances {
		if inst.state == StateReady && !inst.startedAt.IsZero() && inst.startedAt.After(lastReadyAt) {
			lastReadyAt = inst.startedAt
		}
	}
	var uptimeSeconds int64
	if state == StateReady && !lastReadyAt.IsZero() {
		uptimeSeconds = int64(now.Sub(lastReadyAt).Seconds())
		if uptimeSeconds < 0 {
			uptimeSeconds = 0
		}
	}

	return Status{
		State:         state,
		InFlight:      s.total.Count(),
		UptimeSeconds: uptimeSeconds,
		LastReadyAt:   lastReadyAt,

		SchedulerBusy: busy,
		Waiters:       waiters,
		Instances:     instances,
	}
}

func (s *Scheduler) AcquireRoute(ctx context.Context, requestedModel, requestID string) (Route, error) {
	if s == nil || s.cfg == nil {
		return Route{}, fmt.Errorf("scheduler not configured")
	}

	resolvedName, modelCfg, err := s.cfg.ResolveModel(requestedModel)
	if err != nil {
		return Route{}, err
	}

	if route, ok := s.tryRouteReady(ctx, resolvedName, modelCfg.Path); ok {
		return route, nil
	}

	// Scheduling required.
	if err := s.acquireSchedulingPermitOrReject(ctx, resolvedName); err != nil {
		return Route{}, err
	}
	defer func() { <-s.sched }()

	// Double-check after acquiring the permit; another request may have started it.
	if route, ok := s.tryRouteReady(ctx, resolvedName, modelCfg.Path); ok {
		return route, nil
	}

	if s.cfg.Scheduler == nil {
		return Route{}, fmt.Errorf("scheduler config is required")
	}
	if modelCfg.Alias != "" {
		// ResolveModel already returns the aliased target config.
	}
	if len(modelCfg.GPUs) == 0 || modelCfg.MinFreeMemMBPerGPU == nil {
		return Route{}, fmt.Errorf("model %q is missing scheduler fields", resolvedName)
	}

	if s.cfg.Scheduler.MaxInstances != nil {
		s.mu.RLock()
		currentInstances := len(s.instances)
		s.mu.RUnlock()
		if currentInstances >= *s.cfg.Scheduler.MaxInstances {
			metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectNoCapacity)).Inc()
			return Route{}, &RejectError{Reason: RejectNoCapacity, RetryAfter: 5 * time.Second, Message: "scheduler at capacity"}
		}
	}

	if err := s.ensureGPUsExist(ctx, modelCfg.GPUs); err != nil {
		return Route{}, err
	}

	if err := s.evictConflicts(ctx, modelCfg.GPUs); err != nil {
		return Route{}, err
	}

	if err := s.ensureFreeVRAM(ctx, modelCfg.GPUs, *modelCfg.MinFreeMemMBPerGPU); err != nil {
		return Route{}, err
	}

	port, ok := s.ports.Acquire()
	if !ok {
		metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectNoCapacity)).Inc()
		return Route{}, &RejectError{Reason: RejectNoCapacity, RetryAfter: 5 * time.Second, Message: "no free ports available for new instance"}
	}

	cuda := joinInts(modelCfg.GPUs, ",")
	mgr := s.new(port, cuda)

	inst := &schedInstance{
		model:      resolvedName,
		port:       port,
		gpus:       append([]int(nil), modelCfg.GPUs...),
		pinned:     modelCfg.Pinned != nil && *modelCfg.Pinned,
		state:      StateStarting,
		mgr:        mgr,
		startedAt:  time.Time{},
		lastUsedAt: time.Time{},
	}
	portLabel := strconv.Itoa(port)
	inst.inflight.OnChange = func(count int64) {
		metrics.InstanceInFlightRequests.WithLabelValues(inst.model, portLabel).Set(float64(count))
	}

	startCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	_, startErr := mgr.Start(startCtx, resolvedName)
	cancel()
	if startErr != nil {
		s.ports.Release(port)
		return Route{}, startErr
	}

	verifyCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	verifyErr := mgr.VerifyReady(verifyCtx, resolvedName)
	cancel()
	if verifyErr != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), s.cfg.VLLM.ShutdownTimeout.Duration)
		_ = mgr.Stop(stopCtx)
		stopCancel()
		s.ports.Release(port)
		return Route{}, verifyErr
	}

	now := s.now()
	inst.state = StateReady
	inst.startedAt = now
	inst.lastUsedAt = now
	metrics.RunningInstances.WithLabelValues(inst.model, portLabel).Set(1)

	s.mu.Lock()
	// If a previous instance exists for this model (e.g., crashed), replace it.
	if old := s.instances[resolvedName]; old != nil {
		// Best-effort cleanup of the old instance without blocking.
		old.draining = true
	}
	s.instances[resolvedName] = inst
	s.mu.Unlock()

	return s.routeForInstance(ctx, inst, modelCfg.Path), nil
}

func (s *Scheduler) StopAll(ctx context.Context) error {
	if s == nil {
		return nil
	}
	// Best-effort: serialize with scheduling operations if possible, but don't block shutdown forever.
	select {
	case s.sched <- struct{}{}:
		defer func() { <-s.sched }()
	default:
	}

	s.mu.RLock()
	all := make([]*schedInstance, 0, len(s.instances))
	for _, inst := range s.instances {
		all = append(all, inst)
	}
	s.mu.RUnlock()

	for _, inst := range all {
		_ = s.drainAndStopInstance(ctx, inst, false)
	}
	return nil
}

func (s *Scheduler) tryRouteReady(ctx context.Context, resolvedModelName, upstreamModel string) (Route, bool) {
	s.mu.RLock()
	inst := s.instances[resolvedModelName]
	s.mu.RUnlock()
	if inst == nil {
		return Route{}, false
	}
	if inst.state != StateReady || inst.draining || inst.mgr == nil || inst.mgr.CurrentPID() == 0 {
		return Route{}, false
	}

	now := s.now()
	s.mu.Lock()
	// Re-check under lock.
	if inst2 := s.instances[resolvedModelName]; inst2 != nil && inst2 == inst && inst.state == StateReady && !inst.draining && inst.mgr != nil && inst.mgr.CurrentPID() != 0 {
		inst.lastUsedAt = now
	}
	s.mu.Unlock()

	return s.routeForInstance(ctx, inst, upstreamModel), true
}

func (s *Scheduler) routeForInstance(ctx context.Context, inst *schedInstance, upstreamModel string) Route {
	var instDone func()
	if inst != nil {
		instDone = inst.inflight.Track(ctx)
	}
	totalDone := s.total.Track(ctx)

	return Route{
		BaseURL:       inst.mgr.BaseURL(),
		UpstreamModel: upstreamModel,
		Done: func() {
			if instDone != nil {
				instDone()
			}
			if totalDone != nil {
				totalDone()
			}
		},
	}
}

func (s *Scheduler) acquireSchedulingPermitOrReject(ctx context.Context, model string) error {
	select {
	case s.sched <- struct{}{}:
		return nil
	default:
	}

	s.mu.Lock()
	if s.waiters[model] {
		s.mu.Unlock()
		metrics.SwapRejectionsTotal.WithLabelValues(string(RejectSwapInProgress)).Inc()
		metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectSwapInProgress)).Inc()
		return &RejectError{Reason: RejectSwapInProgress, RetryAfter: 5 * time.Second, Message: "scheduler busy, please retry"}
	}
	s.waiters[model] = true
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		delete(s.waiters, model)
		s.mu.Unlock()
	}()

	waitTimeout := s.cfg.VLLM.SwapWaitTimeout.Duration
	if waitTimeout <= 0 {
		waitTimeout = 60 * time.Second
	}
	timer := time.NewTimer(waitTimeout)
	defer timer.Stop()

	select {
	case s.sched <- struct{}{}:
		return nil
	case <-timer.C:
		metrics.SwapRejectionsTotal.WithLabelValues(string(RejectWaitTimeout)).Inc()
		metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectWaitTimeout)).Inc()
		return &RejectError{Reason: RejectWaitTimeout, Message: "scheduler busy, please retry"}
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Scheduler) ensureGPUsExist(ctx context.Context, gpuIDs []int) error {
	if s.inv == nil {
		return fmt.Errorf("gpu inventory not configured")
	}
	inv, err := s.inv.List(ctx)
	if err != nil {
		return err
	}
	exists := map[int]bool{}
	for _, g := range inv {
		exists[g.Index] = true
	}
	for _, id := range gpuIDs {
		if !exists[id] {
			return fmt.Errorf("gpu %d not found (from nvidia-smi)", id)
		}
	}
	return nil
}

func (s *Scheduler) ensureFreeVRAM(ctx context.Context, gpuIDs []int, minFreeMB int) error {
	if s.inv == nil {
		return fmt.Errorf("gpu inventory not configured")
	}
	inv, err := s.inv.List(ctx)
	if err != nil {
		return err
	}
	byID := map[int]gpu.GPU{}
	for _, g := range inv {
		byID[g.Index] = g
	}
	for _, id := range gpuIDs {
		g, ok := byID[id]
		if !ok {
			return fmt.Errorf("gpu %d not found (from nvidia-smi)", id)
		}
		if g.FreeMB < minFreeMB {
			metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectInsufficient)).Inc()
			return &RejectError{
				Reason:  RejectInsufficient,
				Message: fmt.Sprintf("insufficient free GPU memory on gpu %d: need %dMB, have %dMB", id, minFreeMB, g.FreeMB),
			}
		}
	}
	return nil
}

func (s *Scheduler) evictConflicts(ctx context.Context, targetGPUs []int) error {
	now := s.now()
	minUptime := 30 * time.Second
	if s.cfg.Scheduler != nil && s.cfg.Scheduler.MinInstanceUptime != nil {
		minUptime = s.cfg.Scheduler.MinInstanceUptime.Duration
	}

	s.mu.RLock()
	var conflicts []*schedInstance
	for _, inst := range s.instances {
		if inst == nil {
			continue
		}
		if overlaps(inst.gpus, targetGPUs) {
			conflicts = append(conflicts, inst)
		}
	}
	s.mu.RUnlock()

	for _, inst := range conflicts {
		if inst.pinned {
			metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectPinnedConflict)).Inc()
			return &RejectError{
				Reason:  RejectPinnedConflict,
				Message: fmt.Sprintf("requested model conflicts with pinned model %q", inst.model),
			}
		}
	}

	sort.Slice(conflicts, func(i, j int) bool {
		return conflicts[i].lastUsedAt.Before(conflicts[j].lastUsedAt)
	})

	for _, inst := range conflicts {
		if !overlaps(inst.gpus, targetGPUs) {
			continue
		}

		uptime := now.Sub(inst.startedAt)
		if !inst.startedAt.IsZero() && uptime < minUptime {
			remaining := minUptime - uptime
			if remaining < 0 {
				remaining = 0
			}
			metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectMinUptime)).Inc()
			return &RejectError{
				Reason:     RejectMinUptime,
				RetryAfter: remaining,
				Message:    "resources are busy (min uptime), please retry",
			}
		}

		if err := s.drainAndStopInstance(ctx, inst, true); err != nil {
			return err
		}

		// Recompute conflicts after each eviction.
		s.mu.RLock()
		still := false
		for _, other := range s.instances {
			if other != nil && overlaps(other.gpus, targetGPUs) {
				still = true
				break
			}
		}
		s.mu.RUnlock()
		if !still {
			return nil
		}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, other := range s.instances {
		if other != nil && overlaps(other.gpus, targetGPUs) {
			metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectInsufficient)).Inc()
			return &RejectError{Reason: RejectInsufficient, Message: "unable to free required GPUs for requested model"}
		}
	}
	return nil
}

func (s *Scheduler) drainAndStopInstance(ctx context.Context, inst *schedInstance, evicted bool) error {
	if inst == nil {
		return nil
	}

	s.mu.Lock()
	// Mark draining so new routes won't use it.
	inst.draining = true
	inst.state = StateStopping
	s.mu.Unlock()

	if inst.mgr != nil {
		if !inst.inflightIsDrained() {
			drainTimeout := s.cfg.VLLM.DrainTimeout.Duration
			if drainTimeout <= 0 {
				drainTimeout = 60 * time.Second
			}
			drainCtx, cancel := context.WithTimeout(context.Background(), drainTimeout)
			_ = inst.inflight.WaitForDrain(drainCtx)
			cancel()
		}

		stopTimeout := s.cfg.VLLM.ShutdownTimeout.Duration
		if stopTimeout <= 0 {
			stopTimeout = 30 * time.Second
		}
		stopCtx, cancel := context.WithTimeout(context.Background(), stopTimeout)
		_ = inst.mgr.Stop(stopCtx)
		cancel()
	}

	portLabel := strconv.Itoa(inst.port)
	metrics.RunningInstances.WithLabelValues(inst.model, portLabel).Set(0)
	metrics.InstanceInFlightRequests.WithLabelValues(inst.model, portLabel).Set(0)
	if evicted {
		metrics.EvictionsTotal.WithLabelValues(inst.model).Inc()
	}

	s.mu.Lock()
	delete(s.instances, inst.model)
	s.mu.Unlock()

	if s.ports != nil && inst.port != 0 {
		s.ports.Release(inst.port)
	}
	return nil
}

func (inst *schedInstance) inflightIsDrained() bool {
	return inst.inflight.Count() == 0
}

func overlaps(a, b []int) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	seen := map[int]bool{}
	for _, x := range a {
		seen[x] = true
	}
	for _, y := range b {
		if seen[y] {
			return true
		}
	}
	return false
}

func joinInts(nums []int, sep string) string {
	var b strings.Builder
	for i, n := range nums {
		if i > 0 {
			b.WriteString(sep)
		}
		b.WriteString(strconv.Itoa(n))
	}
	return b.String()
}
