package jukebox

import (
	"context"
	"log/slog"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// ---------------------------------------------------------------------------
// Periodic state reconciler (Issue #8)
// ---------------------------------------------------------------------------
//
// Sibling of ExternalStartMonitor (external_start.go, Issue #5). Where
// ExternalStartMonitor handles the SPECIFIC fault "bare `docker start
// <peer>` bypasses cross-group eviction → OOM" by SIGKILLing the
// misordered start and re-routing through KickColdLoad, this reconciler
// catches the BROADER drift class:
//
//   - Admission says Ready / Sleeping, Docker says exited / dead /
//     created (peer was stopped out-of-band, e.g. operator ran
//     `docker stop vllm-foo`, or the container OOM-died after init).
//     Admission keeps "consuming" GPU budget for a peer that is in fact
//     dead — every subsequent admission decision is wrong until we
//     reconcile.
//
//   - Admission says Stopped, Docker says paused / created / restarting
//     (transitional states that are neither "running" nor "fully off").
//     Lower urgency than the running-while-Stopped case ExternalStartMonitor
//     handles, but still drift we want a record of.
//
// We do NOT subsume ExternalStartMonitor — that one issues `docker stop
// -t 0` and re-kicks, racing the in-flight OOM, which is a stronger
// action than this reconciler should take. The two run side-by-side:
//
//   - ExternalStartMonitor (2s tick): looks for the SPECIFIC
//     {admission ∈ {Stopped, Sleeping}, docker = running} fault and
//     SIGKILLs.
//   - StateReconciler (5s tick): looks for the BROADER
//     {admission ∈ {Ready, Sleeping}, docker NOT running} drift and
//     reconciles admission down (NotifyStopped + flip schedInstance.state
//     to StateStopped). Optionally kicks a cold-load if config marks the
//     peer as one that should always be live (pinned).
//
// Cost: 1 `docker inspect` per admission-tracked external model per
// 5s tick. Same shape as the external-start monitor (mostly
// indistinguishable in CPU). Worst case ~0.2% CPU on a healthy host.

// defaultStateReconcileInterval is the default poll cadence for the
// state reconciler. Tunable via SchedulerConfig
// .StateReconcileIntervalSeconds (0 = use default). Slower than
// ExternalStartMonitor's 2s because this reconciler's action is
// recording reality, not racing an OOM — there is no in-flight init we
// need to interrupt.
const defaultStateReconcileInterval = 5 * time.Second

// stateReconcilePollInterval is the live cadence read by the monitor
// (allows test overrides without test-only fields on the production
// struct). Backed by atomic.Int64 (nanoseconds) for race-detector
// cleanliness.
var stateReconcilePollInterval atomic.Int64

func stateReconcilePollIntervalDuration() time.Duration {
	if v := stateReconcilePollInterval.Load(); v > 0 {
		return time.Duration(v)
	}
	return defaultStateReconcileInterval
}

// StateReconciler runs the periodic state-reconciliation loop. Tick
// every stateReconcilePollIntervalDuration(), scan admission-tracked
// external instances, and reconcile admission state with the actual
// docker container state. Runs until ctx is cancelled. Start once from
// main.go as a goroutine — exactly like IdleMonitor or
// ExternalStartMonitor.
//
// First tick fires after one interval (not immediately) so boot-time
// RegisterExternalInstances has a chance to seed state before we probe.
func (s *Scheduler) StateReconciler(ctx context.Context) {
	interval := stateReconcilePollIntervalDuration()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	slog.Info("state_reconciler_starting",
		"interval_ms", interval.Milliseconds(),
	)
	for {
		select {
		case <-ctx.Done():
			slog.Info("state_reconciler_stopped")
			return
		case <-ticker.C:
			s.reconcileState(ctx)
		}
	}
}

// reconcileState performs one scan pass. For each admission-tracked
// external model, inspect its container and reconcile admission state
// to docker reality when they disagree.
//
// Selection: external lifecycle with admission enabled (same filter as
// checkExternalStarts). Internal lifecycle is jukebox-managed end-to-
// end and admission-disabled models are intentionally invisible to
// admission.
//
// Snapshot pattern: build candidate list under read lock, release,
// then iterate without holding s.mu — per-candidate work (inspect,
// admission updates, kick) takes its own locks downstream.
func (s *Scheduler) reconcileState(ctx context.Context) {
	if s == nil || s.cfg == nil || s.admission == nil {
		return
	}

	type candidate struct {
		name      string
		modelCfg  config.ModelConfig
		container string
		instState State
		pinned    bool
	}
	var candidates []candidate

	s.mu.RLock()
	for name, modelCfg := range s.cfg.Models {
		if modelCfg.EffectiveLifecycle() != config.LifecycleExternal {
			continue
		}
		if !modelCfg.AdmissionEnabled() {
			continue
		}
		if modelCfg.Host == "" {
			continue
		}
		inst := s.instances[name]
		if inst == nil {
			continue
		}
		// Reconcile only "settled" states. Starting / Stopping are
		// in-flight transitions owned by callers; intervening here would
		// race their own state machine and is the same anti-pattern
		// ExternalStartMonitor avoids via its Stopped/Sleeping filter.
		switch inst.state {
		case StateReady, StateSleeping, StateStopped:
			pinned := modelCfg.Pinned != nil && *modelCfg.Pinned
			candidates = append(candidates, candidate{
				name:      name,
				modelCfg:  modelCfg,
				container: modelCfg.Host,
				instState: inst.state,
				pinned:    pinned,
			})
		}
	}
	s.mu.RUnlock()

	// Deterministic ordering for log greppability.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].name < candidates[j].name
	})

	for _, c := range candidates {
		if err := ctx.Err(); err != nil {
			return
		}
		s.reconcileOne(ctx, c.name, c.modelCfg, c.container, c.instState, c.pinned)
	}
}

// reconcileOne inspects a single managed container and reconciles
// admission state to docker reality when they disagree.
//
// Split out from reconcileState so tests can exercise a single-
// container code path without driving the full ticker loop.
func (s *Scheduler) reconcileOne(ctx context.Context, name string, modelCfg config.ModelConfig, container string, observedState State, pinned bool) {
	// GUARD: skip peers currently mid-cold-load. The cold-load path
	// intentionally leaves inst.state at StateStopped + container running
	// for the duration of `docker start` + health-wait (~5min). Without
	// this guard, tick N+1 mis-classifies the in-flight cold-load as an
	// external-start, SIGKILLs the half-started vLLM, then KickColdLoad
	// refuses (coldLoadKicks dedupe still holds the goroutine slot from
	// tick N). Wedges the peer. Root cause diagnosed 2026-06-14.
	if s.IsInColdLoadEviction(name) {
		return
	}
	s.mu.RLock()
	kickInFlight := s.coldLoadKicks[name]
	s.mu.RUnlock()
	if kickInFlight {
		return
	}

	inspectCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	st, err := inspectContainerState(inspectCtx, container)
	cancel()
	if err != nil {
		// Best-effort: transient docker hiccup or a removed container.
		// DEBUG so /var/log doesn't fill on a misconfigured model.
		slog.Debug("state_reconcile_inspect_failed",
			"model", name,
			"container", container,
			"err", err,
		)
		return
	}

	dockerRunning := strings.EqualFold(st.Status, "running")
	// "exited" / "dead" / "created" / "removing" all mean "not using
	// GPU VRAM right now". "paused" is technically still holding VRAM
	// but is not serving — treat it as "not running" for admission
	// purposes so we surface the inconsistency.
	dockerOff := !dockerRunning && !strings.EqualFold(st.Status, "restarting")

	switch observedState {
	case StateReady, StateSleeping:
		if dockerOff {
			// CASE A: admission thinks the peer is up (Ready) or sleeping
			// in VRAM (Sleeping). Docker says the container is gone /
			// exited / paused. Admission is over-counting VRAM. Flip
			// admission to Stopped so subsequent decisions see the
			// freed budget.
			//
			// TOCTOU re-check before mutating: did our own lifecycle
			// just push this peer into a different state between snapshot
			// and now? If state moved off {Ready, Sleeping}, leave it
			// alone — the in-flight transition owns it.
			s.mu.RLock()
			inst := s.instances[name]
			var currentState State
			if inst != nil {
				currentState = inst.state
			}
			s.mu.RUnlock()
			if inst == nil {
				return
			}
			if currentState != StateReady && currentState != StateSleeping {
				return
			}

			slog.Warn("state_reconcile_drift_detected",
				"model", name,
				"container", container,
				"observed_admission_state", string(observedState),
				"current_admission_state", string(currentState),
				"docker_status", st.Status,
				"docker_exit_code", st.ExitCode,
				"docker_oom_killed", st.OOMKilled,
				"finished_at", st.FinishedAt,
				"pinned", pinned,
				"action", "reconcile_admission_to_stopped",
				"runbook", "periodic-state-reconciler-Issue-8",
			)
			metrics.AdmissionStateReconcileCorrectedTotal.WithLabelValues(name).Inc()
			LogLifecycleTransition(LifecycleEvent{
				Action: LifecycleEvict,
				Model:  name,
				Reason: "state-reconcile-drift-corrected",
				GPUs:   s.gpusForModel(name),
			})

			// Flip admission + schedInstance to Stopped.
			s.admission.NotifyStopped(name)
			s.mu.Lock()
			if inst := s.instances[name]; inst != nil {
				inst.state = StateStopped
			}
			s.mu.Unlock()

			// If the peer is pinned (config says "should always be
			// awake"), kick a cold-load so the proper admission-aware
			// path brings it back up with cross-group eviction.
			if pinned {
				if ok, reason := s.KickColdLoadWithReason(name); !ok {
					// Specific hint per refusal path (admission_not_stopped /
					// no_live_config / kick_in_flight / cooldown_active).
					// Split out 2026-06-14; see KickColdLoadWithReason.
					slog.Warn("state_reconcile_kick_cold_load_refused",
						"model", name,
						"container", container,
						"hint", string(reason),
					)
					metrics.AdmissionStateReconcileFailedTotal.WithLabelValues(name).Inc()
				}
			}
			return
		}
		// dockerRunning + admission ∈ {Ready, Sleeping} = happy path or
		// the case ExternalStartMonitor specifically handles when
		// admission is Sleeping. Either way, NOT this reconciler's job.
		return

	case StateStopped:
		// CASE B: admission thinks the peer is off. If docker says
		// running, ExternalStartMonitor's stronger remediation already
		// owns that case — do NOT duplicate the SIGKILL here (two racing
		// remediations would log spam and double-kick). Skip silently.
		//
		// If docker says created / restarting / paused, that's a
		// transitional state we just record at INFO; no action — the
		// next tick will resolve to running (→ ExternalStartMonitor) or
		// to exited (→ already matches admission).
		if dockerRunning {
			return
		}
		if !strings.EqualFold(st.Status, "exited") && !strings.EqualFold(st.Status, "dead") {
			slog.Info("state_reconcile_transitional",
				"model", name,
				"container", container,
				"admission_state", string(observedState),
				"docker_status", st.Status,
				"action", "no_op_will_resolve_next_tick",
			)
		}
		return
	}
}
