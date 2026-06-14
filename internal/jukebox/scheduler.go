package jukebox

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
	"vllm-jukebox/internal/vllm"
)

type Scheduler struct {
	cfg      *config.Config
	inv      gpu.Inventory
	ports    *ports.Pool
	powerMgr *gpu.PowerManager
	now      func() time.Time
	new      InstanceFactory

	// admission is the optional VRAM-budget-aware wake coordinator.
	// nil = legacy behavior (jukebox calls /wake_up unmediated and
	// any OOM surfaces as a 503 to the client). Set via SetAdmission
	// after construction.
	admission *AdmissionController

	// sched is a semaphore (size 1) that serializes scheduling operations.
	sched chan struct{}

	mu        sync.RWMutex
	instances map[string]*schedInstance // keyed by resolved model name
	waiters   map[string]bool           // at most one waiter per resolved model
	wakeOps   map[string]*wakeOp        // in-flight wake per resolved model (fan-out)

	// coldLoadKicks tracks which models have an in-flight async
	// cold-load goroutine kicked by KickColdLoad. Used purely to
	// dedupe goroutine spawns (defense-in-depth — coldLoadStoppedMember
	// also TOCTOU-rechecks inside the global cold-load lock, so even
	// without this map only one cold-load would actually run). Cleared
	// by the goroutine on exit. Guarded by mu.
	coldLoadKicks map[string]bool

	// coldLoadFailures records the most recent cold-load failure per
	// model name, used by KickColdLoad to back off re-kicks within a
	// short window. Without this, a member-container that exits(1) on
	// startup (e.g. OOM at worker init) gets re-kicked instantly on
	// every inbound request — each kick re-acquires the GPU-set
	// cold-load lock and re-runs a doomed docker start. Live-observed
	// blast radius (2026-06-09): an infinite OOM/retry loop on moe
	// holding coldLoadMu so requests to the healthy pinned 27B
	// (Sparx) blocked until they timed out.
	//
	// Semantics:
	//   - Updated by the async cold-load goroutine on failure (after
	//     NotifyStartFailed / drift-risk paths run, i.e. immediately
	//     before returning the err).
	//   - Cleared by the same goroutine on success.
	//   - Force-cleared by RedeployMember on success — the operator's
	//     explicit redeploy is the canonical "I am resolving this"
	//     signal; leaving a stale failure entry would gate a future
	//     request-triggered KickColdLoad on an ancient failure that
	//     the operator just resolved.
	//   - Force-cleared by coldLoadStoppedMemberLocked when a cold-load
	//     failure rollback could NOT re-wake a pinned peer (admission
	//     is in a known-inconsistent state; the cooldown's "don't
	//     wedge the GPU-set lock" purpose doesn't apply, and the
	//     operator's escape via /admin/redeploy-member or a retry must
	//     not be blocked by the cooldown).
	//   - Consulted by KickColdLoad: if the most recent failure is
	//     within coldLoadFailureCooldown, the kick is suppressed
	//     (the model stays admissionStopped; a future request after
	//     the cooldown will re-attempt). This is a per-model cooldown,
	//     NOT a hard retry cap — once the cooldown expires the next
	//     kick fires unconditionally. Operators can force-recover via
	//     /admin/redeploy-member which both bypasses the gate AND
	//     clears any prior failure record for the redeployed model.
	//
	// Guarded by mu.
	coldLoadFailures map[string]coldLoadFailureRecord

	// coldLoadEvictionMu + coldLoadEviction track models that are
	// currently mid-eviction as part of an in-flight peer cold-load.
	// Populated synchronously at coldLoadStoppedMember's WithColdLoadLock
	// entry (target + every potential peer in the same swap group),
	// cleared on exit. The handler-side fast-fail gate consults this map
	// via IsInColdLoadEviction to close the gap between "cold-load
	// started" and "admission state flipped" — a gap observed live
	// 2026-06-10 as a 600s hang because admission state remained
	// admissionAwake throughout sleepInstance's StateReady → Stopping →
	// Sleeping walk, so the IsModelColdLoading gate (which checks for
	// admissionStopped) missed the request and it blocked downstream on
	// coldLoadMu inside performWake. See router.ColdLoadAware docstring.
	//
	// Mutex is separate from `mu` (the scheduler's main lock) because
	// the hot path (handler-side IsInColdLoadEviction) takes RLock once
	// per request and we don't want it to contend with scheduler
	// state writes. Read-mostly; writes only fire on cold-load entry +
	// exit (rare relative to request rate).
	coldLoadEvictionMu sync.RWMutex
	coldLoadEviction   map[string]struct{}

	// bootEpoch records the wall-clock time at which this Scheduler
	// instance was constructed. Used by the startup-adoption path in
	// external_start.go: if a managed peer container's docker.StartedAt
	// is BEFORE this timestamp, the container was already running when
	// JUKEBOX itself restarted — so the running peer is NOT an external
	// start (the operator didn't `docker start` it on a live jukebox; it
	// was simply alive across the jukebox bounce). We ADOPT it instead
	// of SIGKILLing.
	//
	// Without this field, every jukebox restart cascade-killed the
	// already-running vllm-* peers (admission reseeded their state from
	// the /is_sleeping probe to StateSleeping, then external-start
	// monitor's "Sleeping + docker-running = external start" rule
	// SIGKILLed them; observed 3x in 30 minutes 2026-06-14 during
	// deploy churn).
	//
	// Set in NewSchedulerWithFactory (= process boot time for the
	// scheduler). Never mutated after construction. Compared against
	// containerState.StartedAt parsed via time.Parse(time.RFC3339Nano).
	bootEpoch time.Time

	// bootAdoptedPeers is the set of model names already adopted by
	// adoptPreExistingPeer at this jukebox boot. Once a peer is in this
	// set, checkOneExternalStart skips the adopt path so the WARN log +
	// AdmissionStartupAdoptedTotal metric + lifecycle_transition audit
	// fire exactly once per peer per boot, not every 2s tick.
	//
	// Why this is needed: the StateSleeping → StateSleeping adopt branch
	// in adoptPreExistingPeer leaves inst.state unchanged, so the
	// candidate filter in checkExternalStarts re-selects the peer on
	// every subsequent tick and the adopt machinery re-fires. The
	// resulting log noise (every 2s, indefinitely, for every pre-existing
	// sleeping peer) makes real cold-load events undistinguishable from
	// boot-adopt re-runs in dashboards. Live-observed 2026-06-13 on llm
	// for vllm-main (Qwen3.6-27B). Guarded by mu.
	bootAdoptedPeers map[string]struct{}

	total inflight.Tracker

	// cb is the jukebox-native circuit breaker. Nil when the breaker is
	// disabled in config or hasn't been wired in yet. See circuit_breaker.go.
	cb *CircuitBreaker
}

// coldLoadFailureRecord captures the last cold-load failure for a model
// so KickColdLoad can suppress immediate re-kicks. Stored by value in
// the coldLoadFailures map.
type coldLoadFailureRecord struct {
	at  time.Time
	err string
}

// coldLoadFailureCooldown is the minimum gap between an async cold-load
// failure and the next eligible re-kick for the same model. Set to 30s
// to balance two requirements:
//   - Long enough to prevent a doomed cold-load (e.g. OOM at worker
//     init) from re-firing on every inbound request, which would wedge
//     the GPU-set lock and starve healthy peers.
//   - Short enough that a TRANSIENT failure (docker daemon hiccup,
//     transient network blip) recovers within an operator-acceptable
//     window without manual intervention.
//
// Test-overridable via SetColdLoadFailureCooldownForTest (see
// sleep_export_test.go).
var coldLoadFailureCooldown atomic.Int64

func init() {
	coldLoadFailureCooldown.Store(int64(30 * time.Second))
}

// coldLoadFailureCooldownDuration returns the current cooldown as a
// time.Duration, masking the atomic.Int64-of-nanoseconds storage shape.
func coldLoadFailureCooldownDuration() time.Duration {
	return time.Duration(coldLoadFailureCooldown.Load())
}

type schedInstance struct {
	model string
	port  int
	gpus  []int

	pinned     bool
	draining   bool
	startedAt  time.Time
	lastUsedAt time.Time
	// lastWakeAt records the wall-clock time of the most recent
	// successful /wake_up + /health-ready transition. Consumed by the
	// post-wake settle barrier in sleepInstance — defense against
	// vllm-project/vllm#45519 (/wake_up returns 200 before PP workers
	// settle; a /sleep landing inside the settle window wedges the
	// engine). Zero value means "never woken" — the gate is a no-op.
	lastWakeAt time.Time

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
	return NewSchedulerWithFactory(cfg, inv, portPool, now, nil, nil)
}

func NewSchedulerWithFactory(cfg *config.Config, inv gpu.Inventory, portPool *ports.Pool, now func() time.Time, factory InstanceFactory, powerMgr *gpu.PowerManager) *Scheduler {
	if now == nil {
		now = time.Now
	}
	if factory == nil {
		factory = func(port int, cudaVisibleDevices string) InstanceManager {
			return vllm.NewInstanceManager(cfg, port, map[string]string{"CUDA_VISIBLE_DEVICES": cudaVisibleDevices})
		}
	}
	s := &Scheduler{
		cfg:           cfg,
		inv:           inv,
		ports:         portPool,
		powerMgr:      powerMgr,
		now:           now,
		new:           factory,
		sched:         make(chan struct{}, 1),
		instances:     map[string]*schedInstance{},
		waiters:       map[string]bool{},
		wakeOps:       map[string]*wakeOp{},
		coldLoadKicks:    map[string]bool{},
		coldLoadFailures: map[string]coldLoadFailureRecord{},
		bootEpoch:        now(),
		bootAdoptedPeers: map[string]struct{}{},
	}
	s.total.OnChange = func(count int64) {
		metrics.InFlightRequests.Set(float64(count))
	}
	return s
}

// OnConfigReloaded is wired into the active.yaml hot-reload observer
// (main.go's config.Watch callback). It propagates the new config
// view to subsystems that hold per-construction caches the fsnotify
// atomic pointer swap alone cannot reach — today that is exclusively
// the AdmissionController's per-model state map.
//
// Most per-model fields (cold_load_timeout, evict_action, pinned,
// swap_group, ...) are read via liveModelCfg() at decision time and
// pick up changes automatically; only the AdmissionController carries
// a pre-built per-model record that must be reconciled when the
// operator adds/removes a model or flips its pinned/group membership.
//
// Safe to call concurrently with admission decisions; the controller
// re-acquires its own mutex.
func (s *Scheduler) OnConfigReloaded(cfg *config.Config) {
	if s == nil || cfg == nil {
		return
	}
	s.mu.RLock()
	a := s.admission
	s.mu.RUnlock()
	if a != nil {
		a.RefreshConfig(cfg)
	}
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
	anySleeping := false
	anyStopped := false
	for _, inst := range s.instances {
		if inst.state == StateReady && !inst.draining && inst.mgr != nil && inst.mgr.CurrentPID() != 0 {
			accepting = true
			break
		}
		if inst.state == StateSleeping {
			anySleeping = true
		}
		if inst.state == StateStopped {
			anyStopped = true
		}
	}

	state := StateIdle
	if accepting {
		state = StateReady
	} else if anySleeping {
		// Slept instances can be woken on demand, so jukebox is still
		// effectively accepting work — just at higher latency for the
		// first request. Report StateSleeping rather than StateIdle so
		// /health and metrics make this state visible.
		state = StateSleeping
	} else if anyStopped {
		// Stopped instances are warm-on-demand via the async-503 +
		// KickColdLoad cold-load path (~5min). Report StateStopped so
		// /health and /status promote accepting_requests=true (the LB
		// should keep routing traffic so the cold-load recovery loop
		// can complete on a real consumer request — draining traffic
		// here would cut off the recovery loop). Mirrors the StateSleeping
		// branch above for the warm-on-demand category.
		state = StateStopped
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

	resolvedName, modelCfg, err := liveResolveModel(s.cfg, requestedModel)
	if err != nil {
		return Route{}, err
	}

	// ComfyUI readiness gate (T4 502 RCA, fix/comfyui-always-probe-readiness).
	//
	// ComfyUIManager.IsSleeping reports an in-memory flag — there is no
	// /is_sleeping endpoint to authoritatively probe. If ComfyUI was
	// started/restarted/recreated outside jukebox's view (cron restart,
	// `docker start` by an operator, container recreation), the manager
	// believes sleeping=false even though :8188 may not be bound yet
	// (ComfyUI plugin warm-up runs ~63s post-container-start before
	// Fiber binds the port). The downstream OpenAI-shape image-gen
	// handler then opens a TCP connection against an unbound port and
	// the client sees `connect: refused` surfaced as 502.
	//
	// Fix: ALWAYS probe /system_stats for LifecycleComfyUI peers on
	// every AcquireRoute, decoupled from the in-memory sleep flag. The
	// probe is bounded by `comfyUIReadyProbeBudget` and ticks at the
	// internal 100ms cadence inside ComfyUIManager.VerifyReady. On
	// budget exhaustion we surface a 503 (RejectColdLoading) — honest
	// "model still loading", not 502 "upstream unreachable".
	if modelCfg.EffectiveLifecycle() == config.LifecycleComfyUI {
		s.mu.RLock()
		inst := s.instances[resolvedName]
		s.mu.RUnlock()
		if inst != nil && inst.mgr != nil {
			probeCtx, cancel := context.WithTimeout(ctx, comfyUIReadyProbeBudget)
			probeErr := inst.mgr.VerifyReady(probeCtx, resolvedName)
			cancel()
			if probeErr != nil {
				slog.Warn("comfyui_readiness_probe_failed",
					"model", resolvedName,
					"url", inst.mgr.BaseURL(),
					"budget_s", comfyUIReadyProbeBudget.Seconds(),
					"err", probeErr.Error(),
				)
				metrics.ScheduleRejectionsTotal.WithLabelValues(string(RejectColdLoading)).Inc()
				return Route{}, &RejectError{
					Reason:     RejectColdLoading,
					RetryAfter: 5 * time.Second,
					Message:    fmt.Sprintf("comfyui readiness probe timed out for %q: %v", resolvedName, probeErr),
				}
			}
		}
	}

	if route, ok := s.tryRouteReady(ctx, resolvedName, modelCfg.Path); ok {
		return route, nil
	}

	// Sleep-mode fast path: if there's already a slept instance for this
	// model, wake it rather than building a new one. Multiple concurrent
	// requests share a single wakeOp — none get 503'd during wake.
	if route, handled, err := s.tryRouteFromSleep(ctx, resolvedName, modelCfg); handled {
		if err != nil {
			return Route{}, err
		}
		if route.BaseURL != "" {
			return route, nil
		}
		// Wake "handled" the path but instance state shifted (e.g. auto-suspend
		// re-slept it). Fall through to full scheduling.
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

	// Apply model power limits if configured
	if s.powerMgr != nil {
		if err := s.powerMgr.ApplyModelLimits(ctx, modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			s.ports.Release(port)
			return Route{}, err
		}
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

	// 2026-06-10 (B-6-rev): drift-recovery probe.
	//
	// External sleep-capable instances can transition Ready -> Sleeping
	// out-of-band — manual /sleep, container restart, or a wake-then-
	// sleep cycle that completes between checkIdle ticks while the
	// scheduler is busy with main's heavy load. There is no periodic
	// /is_sleeping reconcile loop, so jukebox's bookkeeping can drift.
	//
	// If we route to a Ready-by-bookkeeping instance that is actually
	// sleeping, ForwardFiber POSTs to a sleeping vLLM. /v1/chat/completions
	// auto-wakes there, but /v1/rerank and /v1/embeddings DO NOT — the
	// server hangs ~60s, the client retries 3x, observed wall-clock 180s
	// (Phase B-3 live evidence 2026-06-10: rerank to L1-slept reranker
	// while main concurrent-busy hung 180s; manual /wake_up succeeded in
	// 226 ms; manual /v1/rerank to woken reranker succeeded in 57 ms).
	//
	// Probe budget:
	//   - lastUsedAt > 30s stale -> only fire when there's been a
	//     meaningful gap; fresh hot-path requests skip the probe entirely
	//     (zero added latency under request bursts).
	//   - 500ms timeout caps worst-case added latency when the probe
	//     itself is unreachable (we ignore the error and proceed with
	//     the Ready route — no worse than today's behaviour).
	//
	// On confirmed sleep we flip state to StateSleeping and return
	// (_, false) so the caller (AcquireRoute) falls through to
	// tryRouteFromSleep, which performs the wake.
	// Round-2 instrumentation: log probe entry/exit so the handler-side
	// trace can confirm the path actually fires under repro conditions.
	staleness := now.Sub(inst.lastUsedAt)
	sc, scOk := inst.mgr.(SleepCapable)
	if scOk && staleness > 30*time.Second {
		slog.Info("drift_probe_entry",
			"model", resolvedModelName,
			"staleness_ms", staleness.Milliseconds(),
		)
		probeCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
		sleeping, perr := sc.IsSleeping(probeCtx)
		cancel()
		if perr != nil {
			slog.Info("drift_probe_error",
				"model", resolvedModelName,
				"err", perr.Error(),
			)
		} else {
			slog.Info("drift_probe_result",
				"model", resolvedModelName,
				"sleeping", sleeping,
			)
		}
		if perr == nil && sleeping {
			s.mu.Lock()
			// Re-check under lock to avoid racing with a concurrent
			// state transition (NotifyStarted, NotifySleep, etc).
			if inst2 := s.instances[resolvedModelName]; inst2 != nil && inst2 == inst && inst.state == StateReady {
				inst.state = StateSleeping
				slog.Info("drift_probe_flipped_to_sleeping",
					"model", resolvedModelName,
				)
			}
			s.mu.Unlock()
			return Route{}, false
		}
		// Probe error or sleeping==false -> proceed with Ready route.
	} else if scOk {
		// Not stale enough — skip probe, go directly to Ready route.
	} else {
		// Manager doesn't implement SleepCapable — log once-per-instance
		// would be ideal but a per-request log is fine because this is
		// the unexpected path (every external/sleep-mode manager should
		// satisfy SleepCapable).
		slog.Debug("drift_probe_skipped_not_sleepcapable",
			"model", resolvedModelName,
		)
	}
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

	// Sleep-mode branch: if the model opts into sleep, try to sleep
	// instead of stopping. On success, KEEP the instance in the map
	// (port, power limits, GPU allocation all retained — wake will
	// restore the instance without re-scheduling). On failure, fall
	// through to the hard-stop path.
	modelCfg, modelOk := liveModelCfg(s.cfg, inst.model)
	if modelOk {
		// Pinned + non-evicted = graceful shutdown of a model the operator
		// declared as always-on. Leave it alone. Without this guard,
		// StopAll (called when jukebox SIGTERMs) would sleep every pinned
		// instance, leaving them asleep across an unrelated jukebox
		// restart — operationally surprising and adds a wake latency
		// hit to the next request that should not have happened.
		// Eviction is different: an explicit decision to free GPU memory.
		if inst.pinned && !evicted {
			slog.Info("skip sleep on shutdown for pinned instance",
				"model", inst.model,
				"hint", "pinned models stay awake across jukebox restarts; explicit eviction still slept",
			)
			return nil
		}
		// External-lifecycle instances are never hard-stopped by jukebox.
		if modelCfg.EffectiveLifecycle() == config.LifecycleExternal {
			if modelCfg.SleepMode {
				reason := "evict"
				if !evicted {
					reason = "manual"
				}
				return s.sleepInstance(ctx, inst, modelCfg.EffectiveSleepLevel(), reason)
			}
			slog.Warn("cannot drain external instance without sleep_mode (no-op)",
				"model", inst.model,
				"hint", "set sleep_mode: true to enable jukebox eviction of external instances",
			)
			return nil
		}
		// Managed-lifecycle with sleep_mode: try sleep first, fall back to stop.
		if modelCfg.SleepMode {
			reason := "evict"
			if !evicted {
				reason = "manual"
			}
			if err := s.sleepInstance(ctx, inst, modelCfg.EffectiveSleepLevel(), reason); err == nil {
				return nil
			}
			slog.Warn("sleep failed; falling back to hard stop", "model", inst.model)
		}
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

		// Revert power limits to defaults
		if s.powerMgr != nil {
			_ = s.powerMgr.RevertModelLimits(context.Background(), inst.gpus)
		}
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
