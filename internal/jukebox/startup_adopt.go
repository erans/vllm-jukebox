package jukebox

import (
	"context"
	"log/slog"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// ---------------------------------------------------------------------------
// Startup adoption (Issue #5b)
// ---------------------------------------------------------------------------
//
// Background — the bug:
//   Issue #5's ExternalStartMonitor (external_start.go) SIGKILLs any
//   managed peer container observed `running` while admission still
//   records it as Stopped or Sleeping. The intent: catch a bare
//   `docker start vllm-peer` that bypasses jukebox's cross-group GPU
//   eviction path. The unintended consequence: when JUKEBOX ITSELF
//   restarts (operator deploy, image pull, OOM-of-jukebox, anything),
//   every running vllm-* peer triggers the same "admission says
//   Sleeping (because /is_sleeping probe at boot correctly reports
//   true — vLLM's normal slept-L1 resting state), docker says running"
//   condition and is SIGKILLed. Observed 3x in 30 min on 2026-06-14:
//   every jukebox bounce cascade-killed vllm-main, vllm-task,
//   vllm-vision, vllm-embeddings, vllm-reranker — and then
//   KickColdLoad refused with admission_state_or_cooldown_gate,
//   stranding the fleet until manual `docker compose up -d <peer>`
//   recovery for every peer.
//
// The discriminator: jukebox's own boot epoch vs. container.StartedAt.
//
//   - Case A (the bug): container.StartedAt < jukebox bootEpoch.
//     The peer was ALREADY RUNNING when jukebox came up. It is NOT
//     an external start; it is a pre-existing peer surviving the
//     jukebox bounce. ADOPT it — leave the container alone, reconcile
//     jukebox's view of admission state to reality, NO SIGKILL, NO
//     KickColdLoad.
//
//   - Case B (the original Issue #5 fault): container.StartedAt >=
//     jukebox bootEpoch. The container started AFTER jukebox. Either
//     (a) jukebox itself started it via the proper cold-load path —
//     in which case admission would be Ready/Awake and the monitor
//     wouldn't even consider it a candidate — or (b) it was started
//     out-of-band by an operator on a live jukebox. (b) is the real
//     external start; the existing SIGKILL + KickColdLoad behavior
//     is correct.
//
// shouldAdoptOnBoot encodes this decision. Pure function, no docker
// calls, no scheduler-state mutation — it consumes the inspect result
// the caller already paid for and returns a yes/no plus a structured
// reason for logging.
//
// Clock-skew tolerance: container StartedAt comes from the docker
// daemon's clock; jukebox bootEpoch comes from the jukebox process's
// clock. On a single host they share /etc/localtime and ntpd, but a
// freshly-started jukebox container that itself was just docker-run
// could have a clock that drifted ~ms during init. We use a small
// adoption-grace window: containers that started within
// `adoptGraceBeforeBoot` BEFORE bootEpoch are treated as
// pre-existing (the host clock or NTP could trivially explain this).
// This is asymmetric on purpose — we'd rather false-positive adopt a
// container that JUST barely started (worst case: an external start
// that started <1s before jukebox boot is adopted instead of killed,
// which means it gets to live with whatever admission state we infer;
// the periodic state-reconciler will catch any drift) than
// false-negative SIGKILL a healthy peer that the user explicitly
// asked us not to kill 3 times running.
//
// Reciprocally, the grace window also tolerates the case where the
// jukebox process's clock is slightly behind the docker daemon's
// (e.g. container clock-skew at host startup) — a 5s window comfortably
// covers expected single-host NTP drift while staying narrow enough
// that a real external-start triggered seconds after jukebox boot
// still gets caught by the SIGKILL path.

// adoptGraceBeforeBoot widens the adoption window backwards by this
// much before bootEpoch. See comment above for rationale. Conservative
// — 5s is much larger than expected NTP drift on a healthy host but
// still narrow enough that a real bare-`docker start` issued seconds
// after jukebox boot still trips the SIGKILL path.
const adoptGraceBeforeBoot = 5 * time.Second

// adoptDecision captures whether the running peer should be adopted
// (instead of SIGKILLed) and a human-readable reason for logs.
type adoptDecision struct {
	Adopt  bool
	Reason string
	// ParsedStartedAt is the parsed value of containerState.StartedAt,
	// zero-time if parsing failed. Exposed so callers can log it as
	// a uniformly-formatted time.
	ParsedStartedAt time.Time
}

// shouldAdoptOnBoot decides whether a running peer should be ADOPTED
// (Case A — pre-existing across jukebox bounce) rather than SIGKILLed
// as an external start (Case B — bare `docker start` on a live jukebox).
//
// startedAtStr is the docker-daemon-reported State.StartedAt as
// RFC3339Nano. bootEpoch is the scheduler's construction time. If
// startedAtStr cannot be parsed (malformed, zero string, ancient
// docker version), we FAIL-CLOSED: do NOT adopt (preserve the
// original Issue #5 behavior — SIGKILL the suspicious container).
// Fail-closed here means "if we can't tell, behave like the existing
// fix" — that's a strictly lower-risk default than fail-open.
func shouldAdoptOnBoot(startedAtStr string, bootEpoch time.Time) adoptDecision {
	if startedAtStr == "" {
		return adoptDecision{
			Adopt:  false,
			Reason: "missing_started_at",
		}
	}
	// Docker emits time.RFC3339Nano. Try it first, then plain RFC3339
	// as a defensive fallback for older daemon versions / odd builds.
	t, err := time.Parse(time.RFC3339Nano, startedAtStr)
	if err != nil {
		if t2, err2 := time.Parse(time.RFC3339, startedAtStr); err2 == nil {
			t = t2
		} else {
			return adoptDecision{
				Adopt:  false,
				Reason: "unparseable_started_at",
			}
		}
	}
	if t.IsZero() {
		return adoptDecision{
			Adopt:  false,
			Reason: "zero_started_at",
		}
	}
	if bootEpoch.IsZero() {
		// Defense: a Scheduler constructed via a non-standard path
		// might not have bootEpoch set. Treat as "unknown" → fail-closed.
		return adoptDecision{
			Adopt:           false,
			Reason:          "missing_boot_epoch",
			ParsedStartedAt: t,
		}
	}
	// Adopt if the container started BEFORE jukebox booted (plus the
	// grace window for clock-skew). See module-level comment for the
	// rationale on the asymmetric grace.
	adoptCutoff := bootEpoch.Add(adoptGraceBeforeBoot)
	if t.Before(adoptCutoff) {
		return adoptDecision{
			Adopt:           true,
			Reason:          "started_before_jukebox_boot",
			ParsedStartedAt: t,
		}
	}
	return adoptDecision{
		Adopt:           false,
		Reason:          "started_after_jukebox_boot",
		ParsedStartedAt: t,
	}
}

// adoptPreExistingPeer reconciles jukebox's view of a peer's admission
// state to "this container is running and healthy" without killing the
// container. Called by checkOneExternalStart when shouldAdoptOnBoot
// returns Adopt=true.
//
// The schedInstance.state at entry is one of {StateStopped,
// StateSleeping} (those are the only states checkExternalStarts adds
// to its candidate list). We don't blindly flip both to StateReady:
//
//   - If state == StateSleeping: the /is_sleeping boot probe returned
//     true. The container IS in slept-L1 (normal vLLM resting state)
//     and the next consumer request will take the wake path. Keep
//     state at StateSleeping. The admission controller may or may not
//     have been notified at boot probe time (NotifySleep with reason
//     "boot-probe" runs at boot for sleeping probes); we re-issue
//     NotifySleep("boot-adopt") defensively because markSleepingLocked
//     is idempotent and the controller may be in admissionUnknown
//     if the boot probe didn't run yet (TOCTOU vs. the 2s monitor
//     tick).
//
//   - If state == StateStopped: at boot probe time, /health was
//     unreachable AND evict_action: stop, so we seeded state=Stopped.
//     But now the container IS running — the boot probe must have
//     had a transient failure (probes finish under a 10s budget;
//     a fresh container can take longer to respond than that). Flip
//     state back to StateReady and notify admission via NotifyStarted
//     so the budget reflects the resident model.
//
// Returns the new state we settled on (for logging).
func (s *Scheduler) adoptPreExistingPeer(name string, observedState State, modelCfg config.ModelConfig) State {
	switch observedState {
	case StateSleeping:
		// Keep StateSleeping — /is_sleeping=true is the truth, the
		// next consumer will wake. Make sure admission knows.
		// NotifySleep is idempotent (markSleepingLocked guards on
		// already-sleeping).
		if s.admission != nil {
			// Reason "boot-adopt" is intentionally distinct from
			// "boot-probe" so audit grep can separate "we re-affirmed
			// during external_start adoption" from "we observed at
			// the boot health probe pass".
			s.admission.NotifySleep(name, "boot-adopt")
		}
		s.mu.Lock()
		if inst := s.instances[name]; inst != nil {
			// Defense in depth: re-check that we haven't been moved
			// off Sleeping by a concurrent path (the lock release
			// between the snapshot and this Lock() is wide).
			if inst.state == StateStopped || inst.state == StateSleeping {
				inst.state = StateSleeping
			}
		}
		s.mu.Unlock()
		return StateSleeping

	case StateStopped:
		// Container is alive — must have been a transient probe
		// failure at boot. Reconcile to Ready + tell admission the
		// container is started so it accounts for the running VRAM.
		if s.admission != nil {
			s.admission.NotifyStarted(name)
		}
		s.mu.Lock()
		if inst := s.instances[name]; inst != nil {
			if inst.state == StateStopped {
				inst.state = StateReady
				// startedAt is informational; set it to bootEpoch so
				// idle timers don't count the pre-existing uptime as
				// "always been awake" which would trigger immediate
				// idle-sleep on swap-group peers.
				if inst.startedAt.IsZero() {
					inst.startedAt = s.bootEpoch
				}
				if inst.lastUsedAt.IsZero() {
					inst.lastUsedAt = s.bootEpoch
				}
			}
		}
		s.mu.Unlock()
		return StateReady
	}
	// Shouldn't happen — caller only invokes with StateStopped or
	// StateSleeping. Defensive return.
	return observedState
}

// logAndCountAdoption emits the unified WARN log + bumps the metric
// + emits a lifecycle audit. Split out from checkOneExternalStart so
// the adopt path is greppable in one place.
func (s *Scheduler) logAndCountAdoption(ctx context.Context, name, container string, observedState, settledState State, dec adoptDecision) {
	_ = ctx // reserved for future tracing
	slog.Warn("external_start_adopted_pre_existing_peer",
		"model", name,
		"container", container,
		"observed_admission_state", string(observedState),
		"settled_admission_state", string(settledState),
		"reason", dec.Reason,
		"container_started_at", dec.ParsedStartedAt.Format(time.RFC3339Nano),
		"jukebox_boot_epoch", s.bootEpoch.Format(time.RFC3339Nano),
		"action", "adopt_no_sigkill",
		"runbook", "external-start-startup-adopt-Issue-5b",
	)
	metrics.AdmissionStartupAdoptedTotal.WithLabelValues(name).Inc()
	LogLifecycleTransition(LifecycleEvent{
		Action: LifecycleColdLoad, // closest existing audit verb; "adopted"
		// is novel — we reuse ColdLoad with the boot-adopt reason so
		// dashboards counting cold-load events don't undercount the
		// boot reconciliation pass.
		Model:  name,
		Reason: "boot-adopt-pre-existing",
		GPUs:   s.gpusForModel(name),
	})
}
