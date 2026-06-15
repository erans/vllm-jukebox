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
// External-start detection (Issue #5)
// ---------------------------------------------------------------------------
//
// Background: jukebox's cold-load pipeline (coldLoadStoppedMember →
// coldLoadStoppedMemberLocked) is the ONLY admission-aware entry point
// that runs evictCrossGroupGPUContendersLocked before bringing a
// managed peer up. When an operator (or an external process) starts a
// managed peer container directly via `docker start <peer>`, the
// cross-group eviction never fires — the peer's vLLM init proceeds with
// whatever resident peers happen to be holding overlapping GPUs, and
// OOMs at worker init.
//
// Live-observed 2026-06-13: `docker start vllm-vision` on a host where
// `vllm-main` was awake holding 16.49 GiB on GPU 1 → vision exited(1)
// at vLLM init with "Free memory on cuda:1 (5.31/23.56 GiB) less than
// gpu_memory_utilization (0.8, 18.85 GiB)". The cross-group eviction
// helper (fix 1f37d916) was present in scheduler code, but the helper
// is only callable from coldLoadStoppedMemberLocked, which was bypassed.
//
// Fix: a background monitor polls each admission-tracked external
// peer's docker container state every externalStartCheckInterval. When
// it observes a container in `running` state while admission still
// records it as Stopped or Sleeping (i.e. the start did NOT come
// through jukebox's lifecycle), it:
//
//   1. Logs `external_start_detected` at WARN with full context for
//      operator visibility / postmortem.
//   2. Bumps jukebox_external_start_detected_total{model=…}.
//   3. Issues `docker stop -t 0` IMMEDIATELY to halt the misordered
//      init before it OOMs (vLLM's CUDA worker init takes 10-30s, so
//      a 2s poll cycle reliably catches the container while the init
//      is still in-flight or has just finished failing).
//   4. Routes the model through the proper cold-load path
//      (KickColdLoad → coldLoadStoppedMember →
//      coldLoadStoppedMemberLocked → evictCrossGroupGPUContendersLocked
//      → docker start). This time the eviction fires first, freeing
//      contended GPUs, and the cold-load proceeds cleanly.
//
// Why `docker stop -t 0` and not pause/SIGSTOP:
//   - `docker pause` uses cgroup freezer on the entire process tree;
//     pausing mid-CUDA-init wedges the driver in undocumented ways
//     and the container often cannot recover on unpause.
//   - SIGSTOP via `docker kill --signal=STOP` has the same problem
//     (the kernel can't safely freeze a thread holding a CUDA mutex).
//   - `docker stop -t 0` sends SIGKILL immediately. The half-initialized
//     vLLM process dies cleanly; no driver state corruption because
//     CUDA contexts are owned by the process and torn down with it.
//     The subsequent KickColdLoad starts a fresh container with the
//     proper eviction sequencing.
//
// Why not just refuse to route to externally-started peers:
//   - By the time jukebox observes the external start, vLLM's worker
//     init is already racing with the resident peer for VRAM. The
//     OOM happens at init, BEFORE any inbound request. Refusing
//     to route does nothing to prevent the OOM — the container has
//     already crashed.
//
// Cost: 1 `docker inspect` per admission-tracked external model per
// 2s tick. With N=10 models, that's 5 invocations/sec, or ~0.5% CPU
// overhead on a healthy host (each inspect is <100ms). Negligible.

// externalStartCheckInterval is the default poll cadence for the
// external-start monitor. Tunable via SchedulerConfig
// .ExternalStartCheckIntervalSeconds (0 = use default). Lower values
// catch external starts faster but cost more docker inspects.
//
// 2s is short enough that we catch the misordered start during vLLM's
// 10-30s worker init window (so we can SIGKILL before the OOM is
// committed), long enough that the docker-inspect overhead stays
// negligible.
const defaultExternalStartCheckInterval = 2 * time.Second

// externalStartPollInterval is the live cadence read by the monitor
// (allows test overrides without test-only fields on the production
// struct). Backed by atomic.Int64 (nanoseconds) for race-detector
// cleanliness.
var externalStartPollInterval atomic.Int64

func externalStartPollIntervalDuration() time.Duration {
	if v := externalStartPollInterval.Load(); v > 0 {
		return time.Duration(v)
	}
	return defaultExternalStartCheckInterval
}

// ExternalStartMonitor runs the external-start detection loop. Tick
// every externalStartPollIntervalDuration(), scan admission-tracked
// external instances, and react to any container observed running
// while admission still records it as Stopped or Sleeping. Runs until
// ctx is cancelled. Start once from main.go as a goroutine — exactly
// like IdleMonitor.
//
// The first tick fires after one interval (not immediately) so that
// boot-time RegisterExternalInstances has a chance to reconcile state
// before we start probing. Without that, a startup-probe race could
// surface false positives.
func (s *Scheduler) ExternalStartMonitor(ctx context.Context) {
	interval := externalStartPollIntervalDuration()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	slog.Info("external_start_monitor_starting",
		"interval_ms", interval.Milliseconds(),
	)
	for {
		select {
		case <-ctx.Done():
			slog.Info("external_start_monitor_stopped")
			return
		case <-ticker.C:
			s.checkExternalStarts(ctx)
		}
	}
}

// checkExternalStarts performs one scan pass. For each admission-
// tracked external model whose admission state is Stopped or Sleeping,
// we `docker inspect` the container and react if it's running.
//
// Selection criteria: only external lifecycle models with admission
// enabled (ExpectedVRAMMBPerGPU > 0). Internal lifecycle is jukebox-
// managed end-to-end and external_start cannot apply; admission-
// disabled models are intentionally invisible to admission and don't
// participate in the cross-group eviction pipeline.
//
// Snapshot pattern mirrors checkIdle (sleep.go:2563): build the
// candidate list under read lock, release, then iterate without
// holding s.mu. The per-candidate docker work (inspect, stop, kick)
// takes its own locks downstream and must NOT hold s.mu.
func (s *Scheduler) checkExternalStarts(ctx context.Context) {
	if s == nil || s.cfg == nil || s.admission == nil {
		return
	}

	type candidate struct {
		name      string
		modelCfg  config.ModelConfig
		container string
		instState State
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
			// No container name to inspect — operator config bug, but
			// not the monitor's job to surface. Skip cleanly.
			continue
		}
		inst := s.instances[name]
		if inst == nil {
			continue
		}
		// Only Stopped / Sleeping admission state is "should not be
		// running". Ready means jukebox put it there; in-flight
		// transitions (Stopping, Starting) are caller-domain.
		switch inst.state {
		case StateStopped, StateSleeping:
			candidates = append(candidates, candidate{
				name:      name,
				modelCfg:  modelCfg,
				container: modelCfg.Host,
				instState: inst.state,
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
		s.checkOneExternalStart(ctx, c.name, c.modelCfg, c.container, c.instState)
	}
}

// checkOneExternalStart inspects a single managed container and reacts
// if it's running while admission says it should be Stopped/Sleeping.
//
// Split out from checkExternalStarts so tests can exercise the single-
// container code path without driving the full ticker loop.
func (s *Scheduler) checkOneExternalStart(ctx context.Context, name string, modelCfg config.ModelConfig, container string, observedState State) {
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

	// Short ctx for the inspect — docker daemon should respond in <1s
	// on a healthy host; a hung daemon shouldn't stall the monitor.
	inspectCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	st, err := inspectContainerState(inspectCtx, container)
	cancel()
	if err != nil {
		// Best-effort: a transient docker hiccup or a deleted container
		// (operator removed it manually) is not actionable here. Log
		// at DEBUG so /var/log doesn't fill up if a misconfigured model
		// permanently fails inspection.
		slog.Debug("external_start_inspect_failed",
			"model", name,
			"container", container,
			"err", err,
		)
		return
	}

	// Only `running` indicates an external start that bypassed our
	// lifecycle. `restarting` is in-flight (next tick will resolve).
	// `paused` / `exited` / `dead` / `created` / `removing` are all
	// safe — container is not consuming GPU VRAM in any way that
	// requires cross-group eviction.
	if !strings.EqualFold(st.Status, "running") {
		return
	}

	// TOCTOU re-check: did our own lifecycle just start this peer
	// between the snapshot and now? Re-read instance state under the
	// scheduler lock; if it flipped to Ready, jukebox owns this start
	// and we must NOT intervene.
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
	if currentState != StateStopped && currentState != StateSleeping {
		// Already transitioned through proper path. No-op.
		return
	}

	// ONE-SHOT BOOT-ADOPT GUARD: once a peer has been adopted by this
	// jukebox process, skip the adopt path on subsequent ticks. The
	// StateSleeping → StateSleeping adopt branch leaves inst.state
	// unchanged, so the candidate filter re-selects the peer every 2s
	// and the WARN + metric + lifecycle audit re-fire indefinitely
	// (live-observed 2026-06-13 on llm for vllm-main). Without this
	// guard, real cold-load events become indistinguishable from
	// boot-adopt re-runs in dashboards. The bootAdoptedPeers set is
	// populated by adoptPreExistingPeer below.
	s.mu.RLock()
	_, alreadyAdopted := s.bootAdoptedPeers[name]
	s.mu.RUnlock()
	if alreadyAdopted {
		return
	}

	// STARTUP-ADOPT (Issue #5b): if the container's StartedAt is BEFORE
	// jukebox's own bootEpoch, the peer was already running when
	// jukebox itself restarted. This is the common case after every
	// `docker restart jukebox` / deploy: vLLM's normal slept-L1 resting
	// state surfaces as /is_sleeping=true → admission seeded Sleeping
	// at boot probe, then container shows running here → we MUST NOT
	// SIGKILL. Adopt: reconcile admission state to match reality, leave
	// the container alone, no KickColdLoad. See startup_adopt.go for
	// the full design rationale and clock-skew grace window.
	if dec := shouldAdoptOnBoot(st.StartedAt, s.bootEpoch); dec.Adopt {
		settled := s.adoptPreExistingPeer(name, currentState, modelCfg)
		s.logAndCountAdoption(ctx, name, container, currentState, settled, dec)
		s.mu.Lock()
		s.bootAdoptedPeers[name] = struct{}{}
		s.mu.Unlock()
		return
	}

	// Confirmed external start. Log at WARN — operators want this in
	// their postmortem stream when GPU-contention OOMs happen.
	slog.Warn("external_start_detected",
		"model", name,
		"container", container,
		"observed_state", string(observedState),
		"current_state", string(currentState),
		"docker_status", st.Status,
		"started_at", st.StartedAt,
		"jukebox_boot_epoch", s.bootEpoch.Format(time.RFC3339Nano),
		"action", "docker stop -t 0 + kick cold-load",
		"runbook", "external-start-bypasses-cross-group-eviction",
	)
	metrics.AdmissionExternalStartDetectedTotal.WithLabelValues(name).Inc()
	LogLifecycleTransition(LifecycleEvent{
		Action: LifecycleEvict,
		Model:  name,
		Reason: "external-start-detected-stop-for-cold-load",
		GPUs:   s.gpusForModel(name),
	})

	// `docker stop -t 0` sends SIGKILL immediately. We don't want
	// SIGTERM grace here — the vLLM init is racing for VRAM with the
	// resident peer and will OOM if we let it complete; killing fast
	// minimizes the window during which both processes contend.
	stopCtx, stopCancel := context.WithTimeout(ctx, 30*time.Second)
	stopOut, stopErr := runSleepDocker(stopCtx, "docker", "stop", "-t", "0", container)
	stopCancel()
	if stopErr != nil {
		// Stop failed — log loudly. The container may still be running
		// and OOM'ing. There's nothing more this monitor can do; next
		// tick will retry. Don't kick the cold-load: KickColdLoad's
		// docker start would race with the still-running container.
		slog.Error("external_start_docker_stop_failed",
			"model", name,
			"container", container,
			"err", stopErr,
			"output", string(stopOut),
		)
		metrics.AdmissionExternalStartStopFailedTotal.WithLabelValues(name).Inc()
		return
	}

	// Belt-and-suspenders: ensure admission records Stopped before
	// kicking. If admission already said Stopped, NotifyStopped is
	// idempotent (markStoppedLocked is a no-op on already-Stopped).
	// If admission said Sleeping (the container was admission-Sleeping
	// but somehow running, e.g. a prior crash + auto-restart), the
	// transition Sleeping → Stopped is the honest reflection of
	// "container is no longer holding VRAM" (we just SIGKILL'd it).
	s.admission.NotifyStopped(name)
	s.mu.Lock()
	if inst := s.instances[name]; inst != nil {
		inst.state = StateStopped
	}
	s.mu.Unlock()

	// Now route through the proper cold-load path. KickColdLoad spawns
	// a goroutine; the inner coldLoadStoppedMemberLocked will run
	// evictCrossGroupGPUContendersLocked BEFORE docker start, so the
	// next start sees the GPUs freed and proceeds cleanly.
	if ok, reason := s.KickColdLoadWithReason(name); !ok {
		// KickColdLoad refused — emit a SPECIFIC hint per the 4 distinct
		// refusal paths (admission_not_stopped / no_live_config /
		// kick_in_flight / cooldown_active). The legacy
		// `admission_state_or_cooldown_gate` hint conflated all four and
		// masked a self-cancelling tick-loop race for a full session
		// (2026-06-14). The reason enum is the authoritative signal;
		// keep the hint key for grep continuity but populate from reason.
		slog.Warn("external_start_kick_cold_load_refused",
			"model", name,
			"container", container,
			"hint", string(reason),
		)
	}
}
