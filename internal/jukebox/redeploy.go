package jukebox

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"sort"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// MemberRedeployer is the optional interface a Router satisfies when it
// supports the operator-initiated /admin/redeploy-member workflow. The
// HTTP handler type-asserts the configured Router against this
// interface; failure returns 501 to the operator. *Scheduler is the
// only implementation today; *Coordinator does not have the concept of
// a swap-group member, so the verb does not apply in swap mode.
type MemberRedeployer interface {
	RedeployMember(ctx context.Context, name string) (RedeployResult, error)
}

// RedeployResult is the structured summary returned to the operator on
// a successful redeploy. The shape is what the HTTP handler serializes
// directly into the 200 response body.
type RedeployResult struct {
	Model          string   `json:"model"`
	Container      string   `json:"container"`
	Result         string   `json:"result"`
	DurationMS     int64    `json:"duration_ms"`
	EvictedPinned  []string `json:"evicted_pinned"`
	RestoredPinned []string `json:"restored_pinned"`
	// StoppedPeers is the list of OTHER swap-group peers (evict_action:
	// stop, GPU-overlapping) that were `docker stop`'d before the target
	// cold-load to free residual VRAM. Slept-L1 peers leave ~1.5-2.2 GiB
	// per GPU resident — on a tight 24 GiB GPU, two or three of those
	// stacked is enough to OOM a 0.88-util cold-load. We stop them so the
	// cold-load sees the free-memory it needs.
	StoppedPeers []string `json:"stopped_peers,omitempty"`
	// LeftStopped is the list of peers from StoppedPeers that we
	// intentionally did NOT restart after the redeploy. These were
	// evict_action: stop members (rarely-woken by definition — that's
	// the contract) and will async-recover on demand via the standard
	// wake-from-Stopped path (KickColdLoad + 503 + Retry-After). Doing
	// the restart inline would force the operator to wait N×5min cold
	// loads in series; the async path is cleaner.
	LeftStopped []string `json:"left_stopped,omitempty"`
}

// ErrRedeployIneligible is returned by RedeployMember when the target
// model is not eligible for the redeploy-member verb. The handler maps
// this to HTTP 400. Eligibility = admission-tracked AND evict_action:
// stop. Sleep-mode swap-group peers do not need redeploy-member —
// they already cycle via /sleep + /wake_up without container churn,
// and their compose entries typically carry restart: unless-stopped
// which would defeat docker stop. If an operator wants to redeploy a
// sleep-mode peer they can `docker compose restart` it manually.
var ErrRedeployIneligible = errors.New("model is not eligible for redeploy-member")

// ErrRedeployUnknownModel is returned by RedeployMember when the
// target model is not tracked by admission (typo, deleted from config,
// admission disabled). The handler maps this to HTTP 400.
var ErrRedeployUnknownModel = errors.New("model is not tracked by admission")

// dockerCmdFn is the docker-exec function shape used by RedeployMember.
// Defaults to a real exec.CommandContext invocation that returns combined
// stdout+stderr; tests override via SetDockerCmdForTest so the
// redeploy logic can be exercised without spawning docker. Kept local
// to scheduler / redeploy (sleep.go's existing direct exec.Command
// usage is intentionally NOT refactored to use this — per task scope).
type dockerCmdFn func(ctx context.Context, name string, args ...string) ([]byte, error)

// redeployDockerCmd is the package-level docker-exec hook used only by
// RedeployMember. Lives at package scope (not on Scheduler) so the
// Scheduler struct does not need to change for redeploy support; tests
// in this package use SetDockerCmdForTest (see sleep_export_test.go) to
// swap it. Reset between tests by the same setter passing nil.
//
// Concurrency: stored as an atomic.Pointer[dockerCmdFn] so concurrent
// reads (from runDocker) and writes (from SetDockerCmdForTest) are
// race-detector clean. The previous plain-var implementation relied on
// tests running serially; the atomic upgrade removes that fragility
// without changing the prod fast-path (the load is one acquire+nil
// check, identical cost in practice).
var redeployDockerCmd atomic.Pointer[dockerCmdFn]

func defaultDockerCmd(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

// runDocker dispatches to redeployDockerCmd if set (tests), otherwise to
// the real exec.CommandContext. Keeps the indirection in one place so
// the rest of the redeploy code reads cleanly.
func runDocker(ctx context.Context, name string, args ...string) ([]byte, error) {
	if fn := redeployDockerCmd.Load(); fn != nil {
		return (*fn)(ctx, name, args...)
	}
	return defaultDockerCmd(ctx, name, args...)
}

// RedeployMember performs an operator-initiated stop+start of the
// container backing `name`, with pinned-peer host pause/restore so the
// restart does not contend for VRAM with same-GPU peers.
//
// Eligibility (enforced): admission-tracked AND evict_action: stop.
// The evict_action: stop contract requires the operator's compose entry
// to carry restart: "no" (otherwise docker daemon will auto-restart
// the stopped container, defeating the stop+start sequence and
// violating admission's accounting invariant). Sleep-mode swap-group
// peers without evict_action: stop are NOT eligible — they cycle via
// /sleep + /wake_up without container churn and don't need
// redeploy-member. Operators wanting to redeploy a sleep-mode peer
// should `docker compose restart` it manually.
//
// Flow (per Kagi PATCH 16, extended with peer-stop per LIVE-OOM-1):
//   1. Eligibility check: admission-tracked AND evict_action: stop.
//      Otherwise return ErrRedeployIneligible.
//   2. Acquire the global cold-load lock (admission.WithColdLoadLock).
//      Same lock as the wake-from-Stopped path so concurrent cold-loads
//      serialize.
//   3. Inside the lock (lock-free internals):
//      a. Find pinned peers in the same swap_group; if any are awake,
//         sleep them with reason "redeploy-member-host".
//      a2. Find OTHER swap-group peers with evict_action: stop AND GPU
//         overlap with the target. For each that is currently resident
//         (any state other than already-Stopped), call NotifyStopped
//         then docker stop. Why: slept-L1 peers leave ~1.5-2.2 GiB
//         resident per GPU, and on a tight 24 GiB GPU 3-4 such peers
//         stacked is enough to OOM a 0.88-util cold-load. Real
//         observed (4-GPU swap group): main slept-L1 (~1.5 GiB) +
//         moe slept-L1 (~2.2 GiB) + vision slept-L1 (~1.5 GiB) +
//         baseline (~0.7 GiB) = ~6 GiB resident, leaving ~18 GiB free
//         — longctx at util 0.88 wants 20.73 GiB → OOM.
//      b. docker stop <container>  -t 60  (drain SIGTERM grace)
//      c. NotifyStopped(name)      — books reflect Stopped
//      d. docker start <container>
//      e. Poll /is_sleeping until 200 or timeout
//      f. NotifyStarted(name)      — books reflect Sleeping + residual
//      g. Wake each previously-slept pinned peer via
//         performWakeFromInsideColdLoadLock with trigger "redeploy-member-restore".
//      h. Leave the peers from step (a2) Stopped. They will async-
//         recover on demand via the standard wake-from-Stopped path
//         (KickColdLoad + 503 + Retry-After). evict_action: stop is by
//         contract "rarely-woken" — auto-cold-loading them inline would
//         add N×5min serial waits to the operator's request for no
//         benefit (they're not currently serving any traffic — they
//         were Stopped/Slept before the redeploy too).
//
// On failure mid-sequence: best-effort `docker stop` the target +
// NotifyStartFailed(name) so books reflect Stopped. Wake any pinned
// peer we slept (best-effort). Return the error to the operator. Goal:
// leave the system in a recoverable state, never worse than we found.
//
// Long-running — cold-load can be 5-8 minutes. Synchronous from the
// operator's POV. ctx cancellation aborts the in-flight docker call
// but does not roll back partial state.
func (s *Scheduler) RedeployMember(ctx context.Context, name string) (RedeployResult, error) {
	if s == nil {
		return RedeployResult{}, fmt.Errorf("scheduler is nil")
	}
	if s.admission == nil {
		return RedeployResult{}, fmt.Errorf("admission controller not configured; redeploy-member requires admission")
	}
	modelCfg, ok := liveModelCfg(s.cfg, name)
	if !ok {
		return RedeployResult{}, ErrRedeployUnknownModel
	}
	if !modelCfg.AdmissionEnabled() {
		return RedeployResult{}, ErrRedeployUnknownModel
	}

	// Eligibility: evict_action: stop ONLY. Sleep-mode swap-group peers
	// (without evict_action: stop) cycle via /sleep + /wake_up — no
	// container churn needed — and their compose entries typically
	// carry restart: unless-stopped which would race docker stop.
	// Tightening to evict_action: stop matches the only path with a
	// hardened restart: "no" deployment contract.
	if modelCfg.EffectiveEvictAction() != config.EvictActionStop {
		return RedeployResult{}, fmt.Errorf("%w: %q has evict_action=%q (redeploy-member requires evict_action: stop on the target model)",
			ErrRedeployIneligible, name, modelCfg.EffectiveEvictAction())
	}

	if modelCfg.Host == "" {
		return RedeployResult{}, fmt.Errorf("model %q has no host (cannot derive container name)", name)
	}
	container := modelCfg.Host

	s.mu.RLock()
	inst := s.instances[name]
	s.mu.RUnlock()
	if inst == nil {
		return RedeployResult{}, fmt.Errorf("model %q has no registered instance (cannot poll /is_sleeping)", name)
	}

	startPeriod := modelCfg.EffectiveColdLoadTimeout()

	result := RedeployResult{
		Model:     name,
		Container: container,
	}
	var redeployErr error

	s.admission.WithColdLoadLock(func() {
		t0 := s.now()

		// (a) Find pinned peers in the same swap_group + sleep any that are
		// awake so they don't compete for VRAM during the restart.
		pinnedPeers := s.pinnedPeersInSwapGroup(name, modelCfg.SwapGroup)
		var paused []string
		for _, peer := range pinnedPeers {
			s.mu.RLock()
			peerInst := s.instances[peer]
			s.mu.RUnlock()
			if peerInst == nil {
				slog.Warn("redeploy_pinned_peer_no_instance", "peer", peer, "target", name)
				continue
			}
			if peerInst.state != StateReady {
				// Already not awake (sleeping/stopped/etc.) — nothing to pause.
				continue
			}
			peerCfg, ok := liveModelCfg(s.cfg, peer)
			if !ok {
				continue
			}
			slog.Info("redeploy_pausing_pinned_peer", "peer", peer, "target", name)
			if err := s.sleepInstance(ctx, peerInst, peerCfg.EffectiveSleepLevel(), "redeploy-member-host"); err != nil {
				slog.Warn("redeploy_pause_pinned_peer_failed", "peer", peer, "target", name, "err", err)
				// Best-effort: try to restore any peers we already paused
				// before reporting the error.
				s.bestEffortRestorePinned(ctx, paused)
				redeployErr = fmt.Errorf("pause pinned peer %q: %w", peer, err)
				return
			}
			paused = append(paused, peer)
		}
		result.EvictedPinned = append([]string(nil), paused...)

		// (a2) Stop OTHER evict_action: stop swap-group peers whose GPUs
		// overlap the target's. Slept-L1 peers leave ~1.5-2.2 GiB
		// resident per GPU; on a tight 24 GiB GPU, two or three of those
		// stacked is enough to OOM the target's cold-load (real observed
		// failure on the 4-GPU swap group). docker stop fully reclaims
		// the CUDA context.
		//
		// LOCK ORDER (matches Phase 4 contract): NotifyStopped FIRST so
		// admission books reflect the pending stop while we still hold
		// the cold-load lock; THEN docker stop. Concurrent admission
		// decisions during the brief window (we already hold coldLoadMu,
		// so no other wake-from-Stopped or RequestWake-with-eviction can
		// run, but RequestWake without eviction and idle paths can) see
		// the peer as Stopped immediately and won't pick it as an
		// eviction victim.
		//
		// We use the package-level redeployDockerCmd hook (not
		// runSleepDocker) for testability symmetry with the target stop.
		stopPeers := s.evictStopPeersInSwapGroupOverlapping(name, modelCfg)
		var stopped []string
		for _, peer := range stopPeers {
			// Bail cleanly on operator-cancellation BEFORE mutating any
			// admission state for this iteration. Without this, the rest
			// of the loop body runs unconditionally — drainInstance is
			// best-effort against ctx, NotifyStopped + state flip happen
			// regardless, and dockerExec(peerStopCtx, ...) instantly
			// returns context.Canceled because peerStopCtx derives from
			// the cancelled parent. End-state: admission books say the
			// peer is Stopped (zero VRAM), but the container is still
			// running and holding tens of GiB — silent VRAM drift, with
			// only an ERROR-log + audit hint pointing at it.
			//
			// Cancelling here returns the partially-collected `stopped`
			// list (peers we already fully stopped above this iteration
			// are correctly booked); we just don't touch peers we never
			// reached. Outer redeploy returns ctx.Err() to the operator.
			if err := ctx.Err(); err != nil {
				redeployErr = err
				if len(stopped) > 0 {
					result.StoppedPeers = append([]string(nil), stopped...)
					result.LeftStopped = append([]string(nil), stopped...)
				}
				return
			}
			// Re-acquire s.mu for both the instance lookup AND the state
			// check — the previous pattern read peerInst under RLock,
			// RUnlock'd, then state-checked outside the lock, opening a
			// race window where a concurrent idle-suspend or admission
			// eviction could flip peerInst.state between unlock and check
			// (e.g. already-stopping peer flagged not-yet-stopped → double-
			// stop path executes, racing with the stopper).
			s.mu.RLock()
			peerInst := s.instances[peer]
			var peerState State
			if peerInst != nil {
				peerState = peerInst.state
			}
			s.mu.RUnlock()
			if peerInst == nil {
				slog.Warn("redeploy_evict_peer_no_instance", "peer", peer, "target", name)
				continue
			}
			if peerState == StateStopped {
				// Already stopped — nothing to reclaim, nothing to do.
				continue
			}
			peerCfg, ok := liveModelCfg(s.cfg, peer)
			if !ok {
				continue
			}
			if peerCfg.Host == "" {
				slog.Warn("redeploy_evict_peer_no_host", "peer", peer, "target", name)
				continue
			}
			slog.Info("redeploy_stopping_evict_peer",
				"peer", peer, "target", name,
				"peer_state", peerState, "container", peerCfg.Host,
			)
			// Drain in-flight on the peer (best-effort, bounded) so its
			// SIGTERM doesn't abort live generation. Mirrors target stop.
			s.drainInstance(ctx, peerInst)
			// MEDIUM #4 — TOCTOU sub-window fix: dockerExec FIRST, then
			// mutate admission books only on success. The previous
			// ordering (NotifyStopped + state=StateStopped → dockerExec)
			// had a window where ctx cancellation between the mutation
			// and the dockerExec returned context.Canceled while
			// admission's books said the peer was Stopped (zero VRAM)
			// even though the container was still running. End-state:
			// silent VRAM drift with only an ERROR log + audit hint
			// pointing at it.
			//
			// New ordering: dockerExec first. On failure (any cause —
			// ctx.Cancelled, docker daemon error, container missing) we
			// do NOT touch admission state for this peer. The audit
			// "stop-failed-vram-drift-risk" still fires + drift-risk
			// metric bumps so dashboards can alert. On success we apply
			// NotifyStopped + state flip in one short critical section.
			peerStopCtx, peerStopCancel := context.WithTimeout(ctx, 90*time.Second)
			peerStopOut, peerStopErr := s.dockerExec(peerStopCtx, "docker", "stop", "-t", "60", peerCfg.Host)
			peerStopCancel()
			// Always clear draining (drainInstance set it).
			s.mu.Lock()
			peerInst.draining = false
			s.mu.Unlock()
			if peerStopErr != nil {
				// docker stop failed — DO NOT flip admission/state. The
				// container may still be running and holding tens of GiB
				// of VRAM; if we marked it Stopped here every future
				// scheduling decision would treat that VRAM as free and
				// could OOM the next cold-load. Leave admission's view
				// untouched; surface the drift via the audit + metric +
				// ERROR log so an operator can reconcile.
				//
				// ERROR (not WARN) because admission's books and physical
				// state can drift: admission's books are unchanged (peer
				// still Awake/Sleeping), but if docker stop failed because
				// the daemon was transiently unreachable the container's
				// actual state is unknown. Operator scanning for ERROR-
				// level should see this.
				slog.Error("redeploy_evict_peer_stop_failed",
					"peer", peer, "target", name, "container", peerCfg.Host,
					"err", peerStopErr, "output", string(peerStopOut),
				)
				// Audit the inconsistency. Reason "stop-failed-vram-drift-risk"
				// is distinct from "redeploy-member-peer-stop" (success
				// audit, only emitted below) so a grep after a failed
				// redeploy surfaces only the at-risk peers.
				LogLifecycleTransition(LifecycleEvent{
					Action: LifecycleEvict,
					Model:  peer,
					Reason: "stop-failed-vram-drift-risk",
					GPUs:   s.gpusForModel(peer),
				})
				metrics.AdmissionVRAMDriftRiskTotal.WithLabelValues(peer).Inc()
				// Do NOT add this peer to `stopped` — admission books
				// still consider it whatever-it-was, so LeftStopped must
				// not claim we stopped it.
				continue
			}
			// docker stop succeeded — now safe to mutate admission books
			// and scheduler state in one short critical section.
			s.admission.NotifyStopped(peer)
			s.mu.Lock()
			// Centralized Stopped transition + boot-adopt latch clear — a
			// redeploy that stops this peer must not leave a stale adopt
			// latch that would disable Issue #5 rogue-start protection.
			s.setInstanceStoppedLocked(peer, peerInst)
			s.mu.Unlock()
			// Success path: emit the redeploy-member-peer-stop audit only
			// when the docker stop actually succeeded. Previously this
			// audit fired UNCONDITIONALLY after the if-block, which meant
			// a failed stop emitted both the stop-failed-vram-drift-risk
			// audit AND the redeploy-member-peer-stop audit — operators
			// counting lifecycle_transitions_total{action="evict"} saw
			// double events on failure paths.
			LogLifecycleTransition(LifecycleEvent{
				Action: LifecycleEvict,
				Model:  peer,
				Reason: "redeploy-member-peer-stop",
				GPUs:   s.gpusForModel(peer),
			})
			stopped = append(stopped, peer)
		}
		if len(stopped) > 0 {
			result.StoppedPeers = append([]string(nil), stopped...)
		}

		// (b+c) docker stop the target. Idempotent: a stop on an already-
		// stopped container exits 0. We always run it so the post-state
		// is predictable regardless of pre-state (Awake/Sleeping/Stopped).
		// Drain in-flight first (best-effort, bounded) so SIGTERM doesn't
		// abort live generation.
		s.drainInstance(ctx, inst)

		stopCtx, stopCancel := context.WithTimeout(ctx, 90*time.Second)
		stopOut, stopErr := s.dockerExec(stopCtx, "docker", "stop", "-t", "60", container)
		stopCancel()
		if stopErr != nil {
			s.bestEffortRestorePinned(ctx, paused)
			// Peers we stopped in (a2) are LEFT Stopped on rollback too —
			// re-cold-loading them inline would add N×5min serial waits
			// when the operator is already getting an error response.
			// They will async-recover on demand. Mirror the success path
			// by reporting them so the operator can see what state
			// changed even on failure.
			if len(stopped) > 0 {
				result.LeftStopped = append([]string(nil), stopped...)
			}
			redeployErr = fmt.Errorf("docker stop %q: %w (output: %s)", container, stopErr, string(stopOut))
			return
		}

		// Order matters: NotifyStopped FIRST, then flip inst.state.
		//
		// If we flip inst.state to StateStopped before notifying admission,
		// there is a small window where admission still has us as
		// admissionAwake but the scheduler considers us Stopped. A
		// concurrent peer-model wake on shared GPUs could pick this model
		// as an eviction victim and call StopForEviction, which would
		// double-fire NotifyStopped (idempotent — but emits redundant
		// lifecycle audit + log/metric noise) and run a no-op docker
		// stop on an already-stopped container.
		//
		// Tell admission first (drops awake/residual books) so concurrent
		// admission decisions see the correct state, then mirror
		// StopForEviction's scheduler state transition. NotifyStopped is
		// idempotent so the order here is safe even if admission already
		// had us as Stopped (e.g. pre-state was Stopped).
		s.admission.NotifyStopped(name)

		s.mu.Lock()
		// Centralized Stopped transition + boot-adopt latch clear (mirrors
		// StopForEviction). Clearing the self latch on redeploy is correct:
		// the redeploy re-starts the container immediately below, and if
		// that start fails the operator's manual recovery must flow through
		// the health-gate, not be short-circuited by a stale adopt latch.
		s.setInstanceStoppedLocked(name, inst)
		inst.draining = false
		s.mu.Unlock()

		// (d) docker start.
		startCtx, startCancel := context.WithTimeout(ctx, 60*time.Second)
		startOut, startErr := s.dockerExec(startCtx, "docker", "start", container)
		startCancel()
		if startErr != nil {
			s.admission.NotifyStartFailed(name)
			s.bestEffortRestorePinned(ctx, paused)
			if len(stopped) > 0 {
				result.LeftStopped = append([]string(nil), stopped...)
			}
			redeployErr = fmt.Errorf("docker start %q: %w (output: %s)", container, startErr, string(startOut))
			return
		}

		// (e) Poll /is_sleeping until 200 or timeout. Reuses doColdLoad's
		// polling shape — but doColdLoad ALSO runs docker start, which we
		// already did, so we inline the poll here. (Refactoring doColdLoad
		// to split start/poll is out of scope.)
		if err := s.pollUntilSleeping(ctx, inst, startPeriod); err != nil {
			// Half-loaded vLLM: best-effort docker stop so admission's
			// Stopped state and container state stay consistent.
			//
			// CRITICAL ORDERING: only call NotifyStartFailed (which marks
			// admission Stopped → zero VRAM on books) if the cleanup
			// stop actually succeeded. If both fail, the container may
			// still be running and holding tens of GiB. Marking admission
			// Stopped would hide the drift from every future scheduling
			// decision. Surface a hard error and bump the drift-risk
			// metric so an operator can reconcile.
			cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 60*time.Second)
			cleanupOut, cleanupErr := s.dockerExec(cleanupCtx, "docker", "stop", "-t", "30", container)
			cleanupCancel()
			if cleanupErr != nil {
				slog.Error("cold_load_cleanup_stop_failed_books_inconsistent",
					"model", name,
					"container", container,
					"cleanup_err", cleanupErr,
					"cleanup_output", string(cleanupOut),
					"cold_load_err", err,
					"runbook", "manual-reconcile-vram",
				)
				LogLifecycleTransition(LifecycleEvent{
					Action:   LifecycleEvict,
					Model:    name,
					Reason:   "cleanup-stop-failed-vram-drift-risk",
					GPUs:     s.gpusForModel(name),
					Duration: s.now().Sub(t0),
				})
				metrics.AdmissionVRAMDriftRiskTotal.WithLabelValues(name).Inc()
				// DO NOT NotifyStartFailed — admission's books are kept
				// in their pre-call in-flight state. Pinned-peer restore
				// still runs (best-effort) so we don't leave them slept.
				s.bestEffortRestorePinned(ctx, paused)
				if len(stopped) > 0 {
					result.LeftStopped = append([]string(nil), stopped...)
				}
				// Wrap ErrAdmissionVRAMDriftRisk so callers using
				// errors.Is at the boundary can classify this failure:
				//   - mapWakeError (sleep.go cold-load path) routes it
				//     to RejectAdminIntervention with NO Retry-After.
				//   - redeployMemberHandler (admin endpoint) maps it to
				//     HTTP 503 + body code "admin_intervention_required".
				// Keeps sentinel-classification uniform across cold-load
				// failure sites — both surface as terminal/operator-action.
				redeployErr = fmt.Errorf("%w: redeploy poll failed AND cleanup stop failed (container may still be running on GPUs %v): cold_load_err=%v cleanup_err=%v",
					ErrAdmissionVRAMDriftRisk, s.gpusForModel(name), err, cleanupErr)
				return
			}
			s.admission.NotifyStartFailed(name)
			s.bestEffortRestorePinned(ctx, paused)
			if len(stopped) > 0 {
				result.LeftStopped = append([]string(nil), stopped...)
			}
			LogLifecycleTransition(LifecycleEvent{
				Action:   LifecycleColdLoad,
				Model:    name,
				Reason:   "redeploy-failed",
				GPUs:     s.gpusForModel(name),
				Duration: s.now().Sub(t0),
			})
			redeployErr = err
			return
		}

		// (f) Flip admission Stopped → Sleeping (residual back on books).
		s.admission.NotifyStarted(name)

		// HIGH-2: clear any prior cold-load failure cooldown for this
		// model. The /admin/redeploy-member contract is "operator
		// escape hatch — bypasses the cold-load failure cooldown". The
		// REQUEST-PATH gate is already bypassed (RedeployMember never
		// calls KickColdLoad), but if we leave the stale record in
		// coldLoadFailures and the redeployed model later transitions
		// Sleeping → Stopped (e.g. via stop-eviction) inside the
		// original 30s cooldown window, the next request-triggered
		// KickColdLoad would gate on a now-stale failure that the
		// operator just resolved. Delete the entry on success so the
		// post-redeploy state is fully clean.
		s.mu.Lock()
		delete(s.coldLoadFailures, name)
		s.mu.Unlock()

		// Mirror the post-cold-load state in the instance: the container
		// is up and reporting is_sleeping=true, so flip the instance to
		// StateSleeping so the next consumer request takes the wake path.
		s.mu.Lock()
		inst.state = StateSleeping
		s.mu.Unlock()

		dur := s.now().Sub(t0)
		LogLifecycleTransition(LifecycleEvent{
			Action:   LifecycleColdLoad,
			Model:    name,
			Reason:   "redeploy",
			GPUs:     s.gpusForModel(name),
			Duration: dur,
		})
		slog.Info("redeploy_member_cold_load_succeeded",
			"model", name,
			"container", container,
			"duration_ms", dur.Milliseconds(),
			"paused_pinned", paused,
		)

		// (g) Wake each previously-slept pinned peer. Best-effort: a
		// failure here is logged + reported in the result, but the
		// redeploy itself is considered successful (the target is up,
		// admission books are correct).
		//
		// We're inside WithColdLoadLock — use performWakeFromInsideColdLoadLock so
		// a peer that happens to be admissionStopped does not deadlock
		// by re-acquiring coldLoadMu. In practice the validator rejects
		// pinned + evict_action: stop so this is belt-and-suspenders,
		// but it removes a latent reentrancy hazard.
		var restored []string
		for _, peer := range paused {
			s.mu.RLock()
			peerInst := s.instances[peer]
			s.mu.RUnlock()
			if peerInst == nil {
				slog.Warn("redeploy_restore_pinned_peer_skipped", "peer", peer, "reason", "no instance")
				continue
			}
			peerCfg, ok := liveModelCfg(s.cfg, peer)
			if !ok {
				continue
			}
			if err := s.performWakeFromInsideColdLoadLock(ctx, peerInst, peerCfg, "redeploy-member-restore"); err != nil {
				slog.Warn("redeploy_restore_pinned_peer_failed", "peer", peer, "err", err)
				continue
			}
			restored = append(restored, peer)
		}
		result.RestoredPinned = restored
		// (h) The peers we stopped in step (a2) are intentionally LEFT
		// Stopped. They were rarely-woken evict_action: stop members
		// (that's the contract — the whole reason they exist as stop-mode
		// rather than sleep-mode), so the wake-on-demand async-503 path
		// is the right shape: a future consumer request gets 503 +
		// Retry-After, KickColdLoad spawns a background cold-load, the
		// retry lands on the warmed-up peer. No need to burn N×5min of
		// the operator's redeploy window cold-loading peers that nobody
		// is asking for.
		if len(stopped) > 0 {
			result.LeftStopped = append([]string(nil), stopped...)
		}
		result.DurationMS = dur.Milliseconds()
		result.Result = "redeployed"
	})

	if redeployErr != nil {
		return result, redeployErr
	}
	return result, nil
}

// evictStopPeersInSwapGroupOverlapping returns the names of admission-
// tracked peers in the same swap_group as `name` that (a) have
// evict_action: stop AND (b) whose GPU set overlaps `name`'s GPUs.
// Excludes `name` itself. Sorted for determinism.
//
// Used by RedeployMember step (a2) to free residual VRAM left by
// slept-L1 same-GPU peers before the target's cold-load — without this
// step a stack of slept peers (~1.5-2.2 GiB residual each) can OOM the
// cold-load on tight 24 GiB GPUs.
//
// Why limit to evict_action: stop (and not also sleep-mode peers)?
//   - Stop-mode peers carry the hardened restart: "no" deployment
//     contract that makes `docker stop` safe (no auto-restart race).
//   - Stop-mode peers are by contract "rarely-woken" — the async-503
//     wake-from-Stopped path is exactly designed for them, so leaving
//     them Stopped after the redeploy is the right shape.
//   - Sleep-mode peers cycle via /sleep + /wake_up without container
//     churn. Forcibly stopping a sleep-mode peer would defeat its
//     restart: unless-stopped policy (the docker daemon would race the
//     stop with auto-restart) and create messy state. If a sleep-mode
//     peer is leaving too much L1 residual, the deeper fix is to
//     either /sleep with level=2 (full reclaim) or to migrate it to
//     evict_action: stop in the operator's config.
func (s *Scheduler) evictStopPeersInSwapGroupOverlapping(name string, targetCfg config.ModelConfig) []string {
	if targetCfg.SwapGroup == "" {
		return nil
	}
	targetGPUs := make(map[int]struct{}, len(targetCfg.GPUs))
	for _, g := range targetCfg.GPUs {
		targetGPUs[g] = struct{}{}
	}
	var out []string
	for peerName, peerCfg := range s.cfg.Models {
		if peerName == name {
			continue
		}
		if peerCfg.SwapGroup != targetCfg.SwapGroup {
			continue
		}
		if !peerCfg.AdmissionEnabled() {
			continue
		}
		if peerCfg.EffectiveEvictAction() != config.EvictActionStop {
			continue
		}
		// GPU overlap check: at least one GPU in common.
		overlap := false
		for _, g := range peerCfg.GPUs {
			if _, ok := targetGPUs[g]; ok {
				overlap = true
				break
			}
		}
		if !overlap {
			continue
		}
		out = append(out, peerName)
	}
	sort.Strings(out)
	return out
}

// pinnedPeersInSwapGroup returns the names of admission-tracked pinned
// peers in the same swap_group as `name`, sorted for determinism.
// Excludes `name` itself. Returns empty if name has no swap_group.
func (s *Scheduler) pinnedPeersInSwapGroup(name, swapGroup string) []string {
	if swapGroup == "" {
		return nil
	}
	var out []string
	for peerName, peerCfg := range s.cfg.Models {
		if peerName == name {
			continue
		}
		if peerCfg.SwapGroup != swapGroup {
			continue
		}
		if peerCfg.Pinned == nil || !*peerCfg.Pinned {
			continue
		}
		if !peerCfg.AdmissionEnabled() {
			continue
		}
		out = append(out, peerName)
	}
	sort.Strings(out)
	return out
}

// bestEffortRestorePinned wakes a list of previously-paused pinned
// peers on a failure rollback path. Logs failures but does not return
// them as errors — the caller is already reporting a different error.
// Used so we don't leave pinned peers wedged Sleeping after we paused
// them but failed the redeploy itself.
//
// Returns the list of peers we FAILED to re-wake. Callers that need to
// surface this as a structural failure (e.g. the cold-load goroutine —
// see ErrColdLoadPinnedWakeFailed in sleep.go) can act on the slice;
// callers that already own their own error path (e.g. RedeployMember's
// rollbacks) ignore it. Sorted-input order preserved.
//
// LOCK CONTRACT: this is called from inside RedeployMember's
// WithColdLoadLock callback, so it MUST use performWakeFromInsideColdLoadLock —
// the lock-aware wake variant — to avoid a coldLoadMu reentrancy
// deadlock if a peer happens to be admissionStopped.
func (s *Scheduler) bestEffortRestorePinned(ctx context.Context, paused []string) []string {
	var failed []string
	for _, peer := range paused {
		s.mu.RLock()
		peerInst := s.instances[peer]
		s.mu.RUnlock()
		if peerInst == nil {
			continue
		}
		peerCfg, ok := liveModelCfg(s.cfg, peer)
		if !ok {
			continue
		}
		if err := s.performWakeFromInsideColdLoadLock(ctx, peerInst, peerCfg, "redeploy-member-rollback"); err != nil {
			slog.Warn("redeploy_rollback_restore_pinned_failed", "peer", peer, "err", err)
			failed = append(failed, peer)
		}
	}
	return failed
}

// drainInstance waits for in-flight requests on the instance to settle
// or for drainTimeout to elapse, then flips draining/state so a
// subsequent docker stop doesn't abort live generation. Best-effort —
// failure is logged via the bounded context expiring.
func (s *Scheduler) drainInstance(ctx context.Context, inst *schedInstance) {
	if inst == nil {
		return
	}
	drainTimeout := s.cfg.VLLM.DrainTimeout.Duration
	if drainTimeout <= 0 {
		drainTimeout = 60 * time.Second
	}
	s.mu.Lock()
	inst.draining = true
	inst.state = StateStopping
	s.mu.Unlock()
	if inst.inflight.Count() > 0 {
		drainCtx, cancel := context.WithTimeout(ctx, drainTimeout)
		_ = inst.inflight.WaitForDrain(drainCtx)
		cancel()
	}
}

// pollUntilSleeping polls inst's /is_sleeping endpoint until it returns
// {"is_sleeping": true} (the canonical post-cold-load resting state, aka
// slept-L1) or until timeout elapses. Returns nil on success, an error
// wrapping the last poll observation on timeout.
//
// IMPORTANT: just receiving HTTP 200 is NOT sufficient. After `docker
// start`, vLLM's HTTP server can come up before the model finishes
// loading and respond 200 with `{"is_sleeping": false}`. Returning early
// on err==nil flips admission to Sleeping prematurely; a consumer
// request then arrives, /wake_up is dispatched, and vLLM may accept the
// call before the model is actually resident in VRAM — leading to
// confusing timeouts or stale-state generation. We must wait for
// is_sleeping=true, which is vLLM's signal that the model is fully
// loaded into VRAM and resting (post-cold-load slept-L1).
//
// Caller is responsible for already-completed docker start. The poll
// budget should match the cold-load wall-clock (5-8 min for big models).
func (s *Scheduler) pollUntilSleeping(ctx context.Context, inst *schedInstance, timeout time.Duration) error {
	sc, ok := inst.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("instance %q manager does not support IsSleeping (sleep mode required)", inst.model)
	}
	deadline := s.now().Add(timeout)
	pollInterval := coldLoadPollIntervalDuration()
	var lastErr error
	for s.now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
		probeCtx, probeCancel := context.WithTimeout(ctx, 3*time.Second)
		isSleeping, err := sc.IsSleeping(probeCtx)
		probeCancel()
		if err == nil && isSleeping {
			return nil
		}
		if err != nil {
			lastErr = err
		} else {
			// HTTP 200 but is_sleeping=false — server is up but model is
			// still loading. Track this so the timeout error distinguishes
			// "endpoint never responded" from "endpoint kept saying not-yet".
			lastErr = fmt.Errorf("server responded but is_sleeping=false (model still loading)")
		}
	}
	return fmt.Errorf("cold load timeout after %s waiting for is_sleeping=true on %q: %w", timeout, inst.model, lastErr)
}

// dockerExec dispatches to s.dockerCmd if set (tests), otherwise to the
// real docker exec. Keeping the real exec inline (rather than always
// going through a func field) keeps prod allocation/cost identical and
// gives tests a single override point.
func (s *Scheduler) dockerExec(ctx context.Context, name string, args ...string) ([]byte, error) {
	return runDocker(ctx, name, args...)
}
