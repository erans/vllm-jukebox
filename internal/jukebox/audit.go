package jukebox

import (
	"log/slog"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/metrics"
)

// LifecycleAction is the verb describing a model lifecycle transition.
// The set is intentionally small: every Phase 4 stop-on-evict event
// must reuse one of these strings (or extend this enum) so that
// dashboards/alerts query a stable label cardinality.
type LifecycleAction string

const (
	// LifecycleEvict — admission slept a peer to free VRAM for an
	// incoming wake. Emitted per-victim. `Model` is the victim,
	// `Related` is the model that triggered the eviction.
	LifecycleEvict LifecycleAction = "evict"

	// LifecycleSleep — a model went to sleep for any reason other
	// than an admission-initiated eviction (idle timeout, manual,
	// shutdown, wake-rollback). `Reason` carries the trigger.
	LifecycleSleep LifecycleAction = "sleep"

	// LifecycleWake — a model woke from sleep. `Related` carries the
	// list of peers that admission evicted to make room (empty if
	// none). `Reason` carries the trigger.
	LifecycleWake LifecycleAction = "wake"

	// LifecycleColdLoad — emitted when a Stopped container is started
	// (cold load). Reasons: "on_demand" (async-503 path), "redeploy"
	// (operator-initiated via /admin/redeploy-member), "redeploy-failed"
	// (rollback), "boot-probe-stopped" (RegisterExternalInstances
	// detected unreachable evict_action:stop container at startup),
	// "failed" (cold-load timeout).
	LifecycleColdLoad LifecycleAction = "cold_load"

	// LifecycleCircuitBreakerTrip — the jukebox-native circuit breaker
	// observed Threshold consecutive 5xx for a Ready model and issued
	// `docker restart` on its container. `Model` is the model;
	// `Reason` describes the trigger ("5xx_consecutive") or failure
	// mode ("docker_restart_failed").
	LifecycleCircuitBreakerTrip LifecycleAction = "circuit_breaker_trip"
)

// LifecycleEvent is the unified shape for every model state transition
// that admission/scheduler decides on. One log line + one counter
// increment per event. Phase 4 cold-load events emit through this same
// shape so dashboards built today survive that PR unchanged.
type LifecycleEvent struct {
	Action   LifecycleAction
	Model    string        // the model whose state changed
	Related  []string      // peers involved (eviction victims, ordering peers, etc.)
	Reason   string        // what triggered it: admission, idle, manual, cold_start, etc.
	GPUs     []int         // GPU set affected
	Duration time.Duration // wall time for the action (0 if not measured)
}

// lifecycleAuditSinkFn is the test-overridable hook shape used to
// observe LifecycleEvent emissions. Aliased so atomic.Pointer can be
// parameterized cleanly.
type lifecycleAuditSinkFn func(ev LifecycleEvent)

// lifecycleAuditSink is set by tests via SetLifecycleAuditSinkForTest.
// When non-nil, LogLifecycleTransition calls the sink AFTER the standard
// log+metric emission so tests can observe the exact event stream —
// useful for asserting "exactly one audit per peer per failed redeploy"
// (Fix 11) without mocking slog.
//
// Concurrency: stored as atomic.Pointer so concurrent reads (production
// code under test) and writes (test setup) are race-detector clean.
var lifecycleAuditSink atomic.Pointer[lifecycleAuditSinkFn]

// LogLifecycleTransition emits the structured audit line + bumps the
// Prometheus counter. Safe to call with empty Related / GPUs slices.
//
// Callers should also keep their existing finer-grained logs
// (admission_evicting, instance_sleeping, etc.) — those carry
// implementation-specific detail this audit line intentionally omits.
// This line is the dashboard-stable shape, the others are the
// debugging-stable shape.
func LogLifecycleTransition(ev LifecycleEvent) {
	reason := ev.Reason
	if reason == "" {
		reason = "unknown"
	}
	slog.Info("lifecycle_transition",
		"action", string(ev.Action),
		"model", ev.Model,
		"related", ev.Related,
		"reason", reason,
		"gpus", ev.GPUs,
		"duration_ms", ev.Duration.Milliseconds(),
	)
	metrics.LifecycleTransitionsTotal.WithLabelValues(string(ev.Action), reason).Inc()
	if fn := lifecycleAuditSink.Load(); fn != nil {
		(*fn)(ev)
	}
}

// gpusForModel returns the configured GPU set for a model, or nil if
// the model is unknown. Used by sleepInstance which doesn't already
// have modelCfg in scope (it's called from many call sites that pass
// only inst + reason). Safe to call with an empty model name.
func (s *Scheduler) gpusForModel(modelName string) []int {
	if s == nil || s.cfg == nil || modelName == "" {
		return nil
	}
	mc, ok := liveModelCfg(s.cfg, modelName)
	if !ok {
		return nil
	}
	if len(mc.GPUs) == 0 {
		return nil
	}
	out := make([]int, len(mc.GPUs))
	copy(out, mc.GPUs)
	return out
}
