package jukebox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/vllm"
	"vllm-jukebox/internal/vllmcli"
)

// SleepCapable is the optional interface that an InstanceManager (or the
// swap-mode coordinator's Manager) implements when it supports vLLM's
// sleep-mode API. Both vllm.Manager and vllm.ExternalManager satisfy it.
type SleepCapable interface {
	Sleep(ctx context.Context, level int) error
	Wake(ctx context.Context, timeout time.Duration) error
	IsSleeping(ctx context.Context) (bool, error)
}

// docker exec hook — separate from redeploy.go's redeployDockerCmd by
// design (per redeploy.go:72-74). Adding a new docker call to sleep.go?
// Use runSleepDocker. Adding one to redeploy.go? Use runDocker. Don't
// share — tests rely on the split for separate injection points
// (SetSleepDockerCmdForTest vs SetDockerCmdForTest). Future
// consolidation MAY combine them but requires updating both
// SetDockerCmdForTest and SetSleepDockerCmdForTest in tandem (and the
// integration tests that drive them).
//
// sleepDockerCmdFn is the function shape stored in sleepDockerCmd.
// Aliased so atomic.Pointer can be parameterized cleanly.
type sleepDockerCmdFn func(ctx context.Context, name string, args ...string) ([]byte, error)

// sleepDockerCmd is the package-level docker-exec hook used by the
// sleep.go paths: StopForEviction's docker stop, doColdLoad's docker
// start, and coldLoadStoppedMember's best-effort cleanup docker stop.
// Defaults to a real exec.CommandContext invocation; tests override via
// SetSleepDockerCmdForTest (see sleep_export_test.go). Mirrors the
// redeployDockerCmd pattern used by redeploy.go — same shape so
// integration tests can drive both paths through a single hook.
//
// Concurrency: stored as an atomic.Pointer[sleepDockerCmdFn] so
// concurrent reads (from production code paths and any
// background-goroutine tests) and writes (from SetSleepDockerCmdForTest)
// are race-detector clean.
var sleepDockerCmd atomic.Pointer[sleepDockerCmdFn]

func runSleepDocker(ctx context.Context, name string, args ...string) ([]byte, error) {
	if fn := sleepDockerCmd.Load(); fn != nil {
		return (*fn)(ctx, name, args...)
	}
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
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

// coldLoadPollInterval is how often the cold-load and redeploy paths
// poll /is_sleeping while waiting for vLLM to finish loading the model
// into VRAM (post-cold-load resting state). Test-overridable via
// SetColdLoadPollIntervalForTest (see sleep_export_test.go):
// production cold-loads dominate at 5s; tests reduce to ~10ms so the
// per-test wall-clock isn't gated on a 5s tick.
//
// NOT in config — this is an internal tuning constant tied to vLLM's
// engine startup characteristics, not a per-deployment knob.
//
// LOW #9: stored as atomic.Int64 (nanoseconds) so a future test author
// adding t.Parallel() to a cold-load test can't race the override+read.
// Plain-var was safe today because no test runs cold-load paths in
// parallel, but the atomic upgrade removes the latent footgun without
// changing prod cost (one atomic.Load per poll cycle).
var coldLoadPollInterval atomic.Int64

func init() {
	coldLoadPollInterval.Store(int64(5 * time.Second))
}

// coldLoadPollIntervalDuration returns the current poll interval as a
// time.Duration, masking the atomic.Int64-of-nanoseconds storage shape.
func coldLoadPollIntervalDuration() time.Duration {
	return time.Duration(coldLoadPollInterval.Load())
}

// admissionReason is the reason string sleepInstance receives when the
// sleep was triggered by the admission controller's eviction path.
// Used to suppress the duplicate NotifySleep call (admission updates
// its own budget after SleepForEviction returns).
const admissionReason = "admission"

// SetAdmission attaches an AdmissionController to the scheduler. Call
// once after construction, before any request enters AcquireRoute.
// Passing nil disables admission (legacy behavior). Also wires the
// auto-restore callback so admission can trigger a swap-group peer
// wake when an idle-sleep frees a group slot.
func (s *Scheduler) SetAdmission(a *AdmissionController) {
	s.mu.Lock()
	s.admission = a
	s.mu.Unlock()
	if a != nil {
		a.SetAutoRestoreWake(func(name, reason string) {
			// Auto-restore runs in admission's spawned goroutine. Build a
			// fresh background context — the original idle-sleep's context
			// may already be cancelled by the time we get here.
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			s.mu.RLock()
			inst := s.instances[name]
			s.mu.RUnlock()
			if inst == nil {
				slog.Warn("swap_group_auto_restore_skipped", "model", name, "reason", "no instance registered")
				return
			}
			modelCfg, ok := liveModelCfg(s.cfg, name)
			if !ok {
				slog.Warn("swap_group_auto_restore_skipped", "model", name, "reason", "no model config")
				return
			}
			slog.Info("swap_group_auto_restore_triggered", "model", name, "trigger", reason)
			if err := s.performWake(ctx, inst, modelCfg, reason); err != nil {
				slog.Warn("swap_group_auto_restore_failed", "model", name, "err", err)
			}
		})
	}
}

// SchedulerEvictor adapts *Scheduler to the AdmissionEvictor interface
// so admission can sleep peers without depending on scheduler
// internals. The adapter wraps sleepInstance and intentionally does
// NOT call NotifySleep — admission updates its own bookkeeping after
// SleepForEviction returns (the documented evictor contract).
type SchedulerEvictor struct {
	S *Scheduler
}

func (e *SchedulerEvictor) SleepForEviction(ctx context.Context, victim, reason string) error {
	if e == nil || e.S == nil {
		return fmt.Errorf("scheduler evictor: nil scheduler")
	}
	e.S.mu.RLock()
	inst := e.S.instances[victim]
	e.S.mu.RUnlock()
	if inst == nil {
		return fmt.Errorf("scheduler evictor: victim %q not registered", victim)
	}
	modelCfg, ok := liveModelCfg(e.S.cfg, victim)
	if !ok {
		return fmt.Errorf("scheduler evictor: victim %q config missing", victim)
	}
	if !modelCfg.SleepMode {
		// Admission picked a non-sleep-mode model as a victim. That
		// shouldn't happen — admission only tracks models with
		// AdmissionEnabled() and SleepMode is typically required for
		// them to be evictable — but be defensive.
		return fmt.Errorf("scheduler evictor: victim %q does not have sleep_mode enabled", victim)
	}
	// Pass admissionReason through so sleepInstance knows NOT to
	// call NotifySleep — admission owns its own bookkeeping.
	return e.S.sleepInstance(ctx, inst, modelCfg.EffectiveSleepLevel(), reason)
}

// StopForEviction tears down the victim's container via `docker compose
// stop` for full reclaim of CUDA context + NCCL/V1 buffers — the bytes
// that vLLM /sleep cannot release. Used for evict_action: stop members
// (typically rarely-woken 4-GPU peers) where the cold-load wall on
// next demand (~5 min from page cache) is acceptable.
//
// The contract matches SleepForEviction: admission owns the bookkeeping
// update (markStoppedLocked) AFTER this returns. Implementation MUST
// NOT callback into NotifyStopped — that path is reserved for
// non-admission-driven stops.
//
// Operational requirements (NOT enforced here — deployment concern):
//   - The jukebox container must have docker-cli installed and the
//     docker socket mounted (typically /var/run/docker.sock:ro=false).
//   - The compose project containing victim's container must be reachable
//     from the jukebox container's working dir, OR the container must be
//     stoppable by container name alone (we use `docker stop <name>`
//     rather than `docker compose stop` to avoid the project-dir
//     dependency — same effect since restart: "no" is set on these
//     swap-group members).
//   - The victim's compose entry must have `restart: "no"` (otherwise
//     docker daemon will auto-restart the stopped container, defeating
//     the reclaim).
func (e *SchedulerEvictor) StopForEviction(ctx context.Context, victim, reason string) error {
	if e == nil || e.S == nil {
		return fmt.Errorf("scheduler evictor: nil scheduler")
	}
	e.S.mu.RLock()
	inst := e.S.instances[victim]
	e.S.mu.RUnlock()
	if inst == nil {
		return fmt.Errorf("scheduler evictor: victim %q not registered", victim)
	}
	modelCfg, ok := liveModelCfg(e.S.cfg, victim)
	if !ok {
		return fmt.Errorf("scheduler evictor: victim %q config missing", victim)
	}
	if modelCfg.Host == "" {
		return fmt.Errorf("scheduler evictor: victim %q has no host (cannot derive container name)", victim)
	}
	// Drain in-flight requests against this instance before stopping —
	// docker stop is harder than /sleep. Best-effort with a bounded
	// timeout matching the sleep path's drain.
	drainTimeout := e.S.cfg.VLLM.DrainTimeout.Duration
	if drainTimeout <= 0 {
		drainTimeout = 60 * time.Second
	}
	e.S.mu.Lock()
	inst.draining = true
	prevState := inst.state
	inst.state = StateStopping
	e.S.mu.Unlock()
	defer func() {
		// On error, restore previous state — admission's bookkeeping
		// rollback is its caller's responsibility.
		if r := recover(); r != nil {
			e.S.mu.Lock()
			inst.state = prevState
			inst.draining = false
			e.S.mu.Unlock()
			panic(r)
		}
	}()
	if !inst.inflightIsDrained() {
		drainCtx, cancel := context.WithTimeout(ctx, drainTimeout)
		_ = inst.inflight.WaitForDrain(drainCtx)
		cancel()
	}

	// docker stop <container_name>. Container name = modelCfg.Host (the
	// docker-compose service name, which by convention equals the
	// container_name attribute set in the user's compose file). 60s grace
	// is plenty for vLLM's SIGTERM handler to flush; longer hangs the
	// admission-eviction wall-clock without buying anything.
	stopCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	out, err := runSleepDocker(stopCtx, "docker", "stop", "-t", "60", modelCfg.Host)
	if err != nil {
		e.S.mu.Lock()
		inst.state = prevState
		inst.draining = false
		e.S.mu.Unlock()
		return fmt.Errorf("docker stop %q: %w (output: %s)", modelCfg.Host, err, string(out))
	}

	e.S.mu.Lock()
	inst.state = StateStopped
	inst.draining = false
	e.S.mu.Unlock()

	// Stops are tracked in jukebox_vllm_stops_total; not double-counted
	// in jukebox_sleeps_total here. Previously this site also bumped
	// SleepsTotal{reason="stop:<reason>"} for back-compat — but Grafana
	// panels doing `sum(rate(jukebox_sleeps_total[5m]))` overcounted
	// actual L1/L2 sleeps by the stop-event rate. StopsTotal is the
	// dedicated counter for this lifecycle event.
	metrics.StopsTotal.WithLabelValues(victim, reason).Inc()
	slog.Info("instance_stopped",
		"model", victim,
		"reason", reason,
		"container", modelCfg.Host,
	)
	LogLifecycleTransition(LifecycleEvent{
		Action: LifecycleEvict,
		Model:  victim,
		Reason: "stop:" + reason,
		GPUs:   e.S.gpusForModel(victim),
	})
	return nil
}

// ---------------------------------------------------------------------------
// Scheduler-mode sleep / wake
// ---------------------------------------------------------------------------

// IsModelColdLoading reports whether `name` is currently admissionStopped
// (Router interface). Returns false if admission is unset, if the model
// isn't admission-tracked, or if the model is in any other admission
// state (Awake / Sleeping / WakePending).
func (s *Scheduler) IsModelColdLoading(name string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	a := s.admission
	s.mu.RUnlock()
	if a == nil {
		return false
	}
	return a.IsStopped(name)
}

// KickColdLoad spawns a background goroutine that drives the
// wake-from-Stopped cold-load (Router interface). The goroutine uses
// context.Background() with a 10-minute timeout — it MUST survive the
// inbound request goroutine that triggered it (which exits as soon as
// the 503 + Retry-After response is written).
//
// Concurrency: this method dedupes goroutine spawns via coldLoadKicks.
// Even without that dedupe, coldLoadStoppedMember's TOCTOU re-check
// inside the global cold-load lock guarantees at most one cold-load
// per model actually runs; the map just avoids piling up no-op
// goroutines when many requests for the same Stopped model arrive
// during the ~5min cold-load window.
//
// Returns true if a goroutine was kicked, false if the model isn't
// admissionStopped, isn't registered, has no config, or already has
// a kick in flight. Caller (proxy handler) should ignore the bool —
// the 503 + Retry-After response is the same either way.
// markColdLoadEviction registers `names` as currently mid-eviction
// during an in-flight peer cold-load. Idempotent — re-marking a name
// that is already marked is a no-op. Concurrent cold-loads are
// serialized by coldLoadMu (a single global Mutex), so two marks for
// the same name from different goroutines is structurally impossible
// in practice; the map nonetheless tolerates it.
//
// Caller MUST pair this with a deferred unmarkColdLoadEviction for the
// same set so a panic inside the cold-load body still clears the gate
// (a stale marker would silently 503 every request to those models
// until process restart).
func (s *Scheduler) markColdLoadEviction(names ...string) {
	if s == nil || len(names) == 0 {
		return
	}
	s.coldLoadEvictionMu.Lock()
	defer s.coldLoadEvictionMu.Unlock()
	if s.coldLoadEviction == nil {
		s.coldLoadEviction = make(map[string]struct{}, 8)
	}
	for _, n := range names {
		if n == "" {
			continue
		}
		s.coldLoadEviction[n] = struct{}{}
	}
}

// unmarkColdLoadEviction removes `names` from the in-flight eviction set.
// Called via defer from coldLoadStoppedMember after the WithColdLoadLock
// callback returns (success or failure).
func (s *Scheduler) unmarkColdLoadEviction(names ...string) {
	if s == nil || len(names) == 0 || s.coldLoadEviction == nil {
		return
	}
	s.coldLoadEvictionMu.Lock()
	defer s.coldLoadEvictionMu.Unlock()
	for _, n := range names {
		delete(s.coldLoadEviction, n)
	}
}

// IsInColdLoadEviction implements ColdLoadAware. Returns true while the
// named model is mid-eviction (or is the cold-load target itself)
// during an in-flight cold-load. See ColdLoadAware docstring for the
// gap this closes vs IsModelColdLoading.
func (s *Scheduler) IsInColdLoadEviction(name string) bool {
	if s == nil {
		return false
	}
	s.coldLoadEvictionMu.RLock()
	defer s.coldLoadEvictionMu.RUnlock()
	if s.coldLoadEviction == nil {
		return false
	}
	_, ok := s.coldLoadEviction[name]
	return ok
}

// swapGroupMembersForColdLoad returns every model name in the same
// swap_group as `target`, INCLUDING `target` itself. Used by
// coldLoadStoppedMember to pre-compute the set of models that the
// in-flight cold-load may transiently sleep/stop, so the
// IsInColdLoadEviction gate fast-fails requests for any of them for
// the duration of the cold-load.
//
// Conservative-by-design: includes peers that may already be slept
// (the gate briefly fast-fails them too — harmless given the cold-load
// will only run for ~5-10 min). Excludes models in OTHER swap groups
// (they're physically free to serve in parallel with the cold-load).
//
// Returns just [target] when swapGroup is empty (no swap-group → no
// peer eviction → only the target itself is the eviction subject).
func (s *Scheduler) swapGroupMembersForColdLoad(target, swapGroup string) []string {
	out := []string{target}
	if s == nil || swapGroup == "" {
		return out
	}
	// config.Current() is documented to return nil when SetCurrent has
	// never been called (early startup, certain test harnesses). Guard
	// the deref so we degrade to "just target" instead of panicking the
	// cold-load goroutine. See sleep.go:420 nil-deref report.
	cur := config.Current()
	if cur == nil {
		return out
	}
	for name, m := range cur.Models {
		if name == target {
			continue
		}
		if m.SwapGroup == swapGroup {
			out = append(out, name)
		}
	}
	return out
}

func (s *Scheduler) KickColdLoad(name string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	a := s.admission
	inst := s.instances[name]
	s.mu.RUnlock()
	if a == nil || inst == nil {
		return false
	}
	if !a.IsStopped(name) {
		return false
	}
	modelCfg, ok := liveModelCfg(s.cfg, name)
	if !ok {
		return false
	}

	// Dedupe: at most one outstanding kick per model. coldLoadStoppedMember
	// would no-op a second arrival via the lock + TOCTOU recheck, but
	// avoiding spawning the goroutine entirely is cleaner.
	s.mu.Lock()
	if s.coldLoadKicks[name] {
		s.mu.Unlock()
		return false
	}
	// Cooldown gate (BUG 2b): if the most recent cold-load failure for
	// this model is within coldLoadFailureCooldown, suppress this kick.
	// Without this, a member-container that exits(1) at worker init
	// (e.g. OOM) gets re-kicked instantly on every inbound request —
	// the kick re-acquires the GPU-set lock, re-runs the doomed docker
	// start, and the loop wedges the lock so healthy peers can't be
	// woken. The cooldown is per-model and self-clearing: once the
	// window elapses, the next kick fires unconditionally. Operators
	// can force-recover via /admin/redeploy-member which does not
	// consult this map.
	if rec, ok := s.coldLoadFailures[name]; ok {
		cooldown := coldLoadFailureCooldownDuration()
		if s.now().Sub(rec.at) < cooldown {
			s.mu.Unlock()
			slog.Warn("async_cold_load_suppressed_cooldown",
				"model", name,
				"last_failure_age_ms", s.now().Sub(rec.at).Milliseconds(),
				"cooldown_ms", cooldown.Milliseconds(),
				"last_err", rec.err,
			)
			LogLifecycleTransition(LifecycleEvent{
				Action: LifecycleColdLoad,
				Model:  name,
				Reason: "retry_suppressed_cooldown",
				GPUs:   s.gpusForModel(name),
			})
			// HIGH-4: bump a per-model counter so dashboards can alert
			// on "all my models stuck in cooldown" silent partial
			// outages. Pairs with the slog.Warn line + Lifecycle audit
			// above; the metric is the actionable signal — the log
			// lines are operator-readable detail.
			metrics.ColdLoadCooldownSuppressedTotal.WithLabelValues(name).Inc()
			return false
		}
		// Cooldown expired — clear the stale record so future failures
		// don't compare against ancient timestamps.
		delete(s.coldLoadFailures, name)
	}
	s.coldLoadKicks[name] = true
	s.mu.Unlock()

	slog.Info("async_cold_load_kicked",
		"model", name,
		"trigger", "request_arrived_while_stopped",
		"timeout", 10*time.Minute,
	)

	go func() {
		defer func() {
			s.mu.Lock()
			delete(s.coldLoadKicks, name)
			s.mu.Unlock()
		}()
		// Generous bounded context — moe/longctx cold-loads take ~5min;
		// 10min gives slack for slow disks / page-cache cold reads.
		// Intentionally context.Background() (not the inbound request
		// ctx, which is already cancelled by the time the client got
		// its 503).
		//
		// Mutex-queue note (single-flight cold-load semantics):
		//
		// coldLoadStoppedMember acquires AdmissionController.coldLoadMu
		// — a plain sync.Mutex, NOT ctx-aware. Deliberately so: the
		// lock enforces a single-flight invariant across the entire
		// admission lifetime ("at most one cold-load globally at any
		// time"), and a ctx-aware acquire (e.g. a buffered-1 channel
		// + select) would let queued cold-loads bail mid-line and
		// re-arm without ever running, which defeats the dedupe
		// coldLoadKicks already provides at the goroutine-spawn layer.
		//
		// Wait pattern in practice:
		//   - Typical: 0 prior cold-loads (fast path) or 1 prior (queue
		//     <=5min). Healthy.
		//   - Pathological: N stopped swap-group peers all kicked at
		//     once → Nth waits ~(N-1)*5min on Lock(). On the production
		//     4-GPU topology every swap-group member shares GPU 3, so
		//     even when many peers are Stopped, only one can cold-load
		//     into GPU 3 at a time anyway — the lock is reflecting a
		//     real hardware-serialized resource, not artificial queueing.
		//
		// Termination invariant: ctx is propagated INTO the lock-held
		// section (passed to coldLoadStoppedMember which threads it
		// through docker/start + health polling), so once a goroutine
		// finally acquires the lock its 10min timeout fires and frees
		// the lock for the next waiter. Waiters that ctx-time-out
		// while still queued behind Lock() are wasted work, but the
		// `defer delete(s.coldLoadKicks, name)` above guarantees the
		// per-model kick map is cleared regardless of where in the
		// pipeline the goroutine exits — so a future request can
		// re-kick once the queue drains.
		//
		// Outer ctx timeout MUST outlive the inner /is_sleeping wait
		// (modelCfg.EffectiveColdLoadTimeout) — otherwise the outer
		// fires first and SIGKILLs vllm mid-init regardless of the
		// per-model timeout setting. Use max(model timeout, 10 min
		// historical floor) + 60s grace for cleanup. Live finding
		// 2026-06-10 overnight: longctx with cold_load_timeout_seconds=720
		// was getting killed at ~632s by the hardcoded 10*Minute outer
		// ctx — the configured 720s was unreachable.
		outerTimeout := modelCfg.EffectiveColdLoadTimeout()
		if outerTimeout < 10*time.Minute {
			outerTimeout = 10 * time.Minute
		}
		outerTimeout += 60 * time.Second // cleanup grace
		ctx, cancel := context.WithTimeout(context.Background(), outerTimeout)
		defer cancel()
		if err := s.coldLoadStoppedMember(ctx, inst, modelCfg); err != nil {
			slog.Error("async_cold_load_failed",
				"model", name,
				"err", err,
			)
			// CRIT-1 (architect round-6): when the cold-load failed via
			// the pinned-peer-wake-failed rollback path, the inner
			// coldLoadStoppedMemberLocked has ALREADY force-cleared
			// coldLoadFailures[name] for this model so the operator's
			// next request can immediately retry against a known-
			// inconsistent admission state (pinned default Sleeping +
			// target Stopped). Re-recording the cooldown here would
			// silently overwrite that clear and re-block the operator's
			// escape hatch for 30s. Skip the record in that case;
			// respect the rollback's deliberate clear.
			//
			// Other failure modes (docker start exit, container exited,
			// health-check timeout, VRAM-drift cleanup) are the original
			// "fast-fail loop" scenarios the cooldown was designed for —
			// keep recording for those.
			if errors.Is(err, ErrColdLoadPinnedWakeFailed) {
				return
			}
			// BUG 2b: record failure timestamp so subsequent KickColdLoad
			// invocations within coldLoadFailureCooldown short-circuit
			// instead of re-firing a doomed docker start that wedges the
			// GPU-set lock. Cleared in the success path (just below the
			// defer above is too early — the cold-load may not have
			// completed yet; we use the success branch in
			// coldLoadStoppedMemberLocked instead. The asymmetry is
			// fine: failure-record is set here; success-record is the
			// absence of an entry. coldLoadStoppedMemberLocked deletes
			// any stale failure entry on success so the model is
			// immediately re-kickable if it stops again later.)
			s.mu.Lock()
			s.coldLoadFailures[name] = coldLoadFailureRecord{
				at:  s.now(),
				err: err.Error(),
			}
			s.mu.Unlock()
		}
	}()
	return true
}

// coldLoadStoppedMember is the wake-from-Stopped entry point: acquire
// the global cold-load mutex, TOCTOU-recheck that the model is still
// admissionStopped (a concurrent request may have cold-loaded it
// already while this one was queued on the lock), call the lock-free
// docker work, and finalize via NotifyStarted on success or best-
// effort docker-stop + NotifyStartFailed on failure.
//
// LOCK CONTRACT: callers MUST NOT hold coldLoadMu when invoking this —
// it acquires coldLoadMu via WithColdLoadLock and sync.Mutex is
// non-reentrant. For the in-lock variant (callers already inside a
// WithColdLoadLock callback), see performWakeFromInsideColdLoadLock which skips
// the cold-load entirely.
//
// Returns nil on success — the caller (performWake or redeploy-member)
// proceeds with the normal Sleeping → Awake transition. Returns the
// docker / health-wait error on failure; the caller must surface it
// to the consumer (not swallow it via a sleep rollback).
//
// Per Kagi PATCH 16's failure-rollback guidance: the half-loaded vLLM
// is best-effort `docker stop`'d to restore the only clean accounting
// invariant (Stopped ⇒ zero VRAM ⇒ books accurate). NotifyStartFailed
// (NOT NotifySleep) finalizes admission's view; NotifySleep would
// silently no-op against a non-Awake state and wedge the model.
func (s *Scheduler) coldLoadStoppedMember(ctx context.Context, inst *schedInstance, modelCfg config.ModelConfig) error {
	if s.admission == nil {
		return fmt.Errorf("coldLoadStoppedMember: admission unset (caller bug)")
	}
	if modelCfg.Host == "" {
		return fmt.Errorf("coldLoadStoppedMember: model %q has no host (cannot derive container name)", inst.model)
	}

	var coldLoadErr error
	// Operator visibility for the global cold-load mutex queue: emit a
	// log line BEFORE we attempt to acquire (so a 3am operator can grep
	// "cold_load_queued_behind_lock" to see the wait-list snapshot at
	// the moment the goroutine got to the gate) and another INSIDE the
	// callback at the top (so the same model name appears with a clear
	// "now serving" timestamp). The two-line pair lets `grep model=foo`
	// surface "queued at T, acquired at T+X" so a stuck cold-load is
	// visually distinct from a slowly-progressing one (e.g. N peers all
	// kicked at once → N-th waits ~(N-1)*5min on Lock()).
	// 2026-06-10 (B-1-rev): mark target + swap-group peers as
	// "in cold-load eviction" BEFORE the WithColdLoadLock acquires.
	// This closes the gate from the moment the cold-load is admitted,
	// not just when admission state finally flips to admissionStopped.
	// Without this, a request for any swap-group member that lands
	// during the eviction window can fall through IsModelColdLoading
	// and block on coldLoadMu inside performWake for the full
	// cold-load wall-clock (~600s observed live).
	evictionScope := s.swapGroupMembersForColdLoad(inst.model, modelCfg.SwapGroup)
	s.markColdLoadEviction(evictionScope...)
	defer s.unmarkColdLoadEviction(evictionScope...)

	slog.Info("cold_load_queued_behind_lock",
		"model", inst.model,
		"container", modelCfg.Host,
	)
	s.admission.WithColdLoadLock(func() {
		slog.Info("cold_load_acquired_lock",
			"model", inst.model,
			"container", modelCfg.Host,
		)
		coldLoadErr = s.coldLoadStoppedMemberLocked(ctx, inst, modelCfg)
	})
	return coldLoadErr
}

// evictPeersForColdLoadLocked is the cold-load equivalent of
// RedeployMember steps (a) + (a2): before docker-starting the target
// member, sleep any awake pinned peer in the same swap_group and stop
// any overlapping evict_action:stop peer. Without this step, a
// request-triggered cold-load would race for VRAM with the resident
// peer that holds the GPUs and OOM at worker init.
//
// Live-observed (2026-06-09): request to STOPPED vllm-moe arrived
// while pinned vllm-main (27B) held the GPUs awake. `docker start
// vllm-moe` returned 0, but vLLM hit:
//
//	ValueError: Free memory on device cuda:2 (5.31/23.56 GiB) on
//	startup is less than desired GPU memory utilization (0.8, 18.85 GiB)
//
// and the container exited(1). The fix is to evict the resident peer
// FIRST so the cold-load sees the free VRAM it needs.
//
// LOCK CONTRACT: caller MUST hold coldLoadMu (we're already inside
// coldLoadStoppedMemberLocked's WithColdLoadLock callback). The
// per-peer sleepInstance call internally uses sleepInstance's own
// state machine — it does NOT acquire coldLoadMu, so this is safe.
//
// Returns:
//   - slept: pinned peers we just slept (caller stores for telemetry)
//   - stopped: stop-mode peers we just stopped (caller stores for telemetry)
//   - err: non-nil if a step failed in a way that should abort the
//     cold-load. Best-effort failures (e.g. a peer's sleep flaked) are
//     logged but do NOT return an error — the doColdLoad call will
//     surface the VRAM-shortage failure cleanly if we couldn't free
//     enough.
//
// Symmetric with RedeployMember (which evicts on operator request);
// the redeploy path keeps its own inline implementation for now to
// preserve its existing rollback semantics (restoring pinned peers
// on mid-sequence failure, populating RedeployResult, etc.). Both
// paths share evictStopPeersInSwapGroupOverlapping +
// pinnedPeersInSwapGroup for peer selection.
func (s *Scheduler) evictPeersForColdLoadLocked(ctx context.Context, name string, modelCfg config.ModelConfig) (slept, stopped []string) {
	// (a) Sleep awake pinned peers in the same swap_group. Pinned peers
	// use evict_action: sleep by contract (validator enforces this),
	// so we use sleepInstance — same path the admission evictor uses
	// when it picks a pinned peer as a victim.
	pinnedPeers := s.pinnedPeersInSwapGroup(name, modelCfg.SwapGroup)
	for _, peer := range pinnedPeers {
		// Bail cleanly on ctx cancel (shutdown / 10min cold-load wrapper
		// timeout) BEFORE evicting more peers. Asymmetric with the
		// stop-mode loop below — without this check, a ctx cancelled
		// mid-pinned-loop keeps slept-listing peers that we'll then have
		// to restore via bestEffortRestorePinned (which itself uses the
		// same cancelled ctx and may fail to wake them). Net pre-fix:
		// pinned peers left Sleeping on shutdown / timeout. Mirrors the
		// ctx.Err check at the top of the stop-mode loop on line ~620.
		if err := ctx.Err(); err != nil {
			return slept, stopped
		}
		s.mu.RLock()
		peerInst := s.instances[peer]
		s.mu.RUnlock()
		if peerInst == nil {
			continue
		}
		if peerInst.state != StateReady {
			// Already not awake — nothing to evict here.
			continue
		}
		peerCfg, ok := liveModelCfg(s.cfg, peer)
		if !ok {
			continue
		}
		slog.Info("cold_load_evicting_pinned_peer",
			"target", name, "peer", peer, "container", peerCfg.Host,
		)
		// Reason "cold-load-host" mirrors redeploy's "redeploy-member-host".
		// Does NOT trigger admission's auto-restore (that's reason="idle").
		if err := s.sleepInstance(ctx, peerInst, peerCfg.EffectiveSleepLevel(), "cold-load-host"); err != nil {
			// Best-effort: log and continue. The subsequent docker start
			// will OOM cleanly if we couldn't free enough VRAM; better
			// to surface that as the cold-load failure (which routes
			// through the existing rollback path) than to fail here and
			// leave admission with a half-evicted peer.
			slog.Warn("cold_load_evict_pinned_peer_failed",
				"target", name, "peer", peer, "err", err,
			)
			continue
		}
		slept = append(slept, peer)
	}

	// (a2) Stop overlapping evict_action: stop peers. Slept-L1 peers
	// leave ~1.5-2.2 GiB resident per GPU; on a tight 24 GiB GPU, a
	// stack of those is enough to OOM the cold-load. Mirrors the
	// equivalent loop in RedeployMember.
	stopPeers := s.evictStopPeersInSwapGroupOverlapping(name, modelCfg)
	for _, peer := range stopPeers {
		if err := ctx.Err(); err != nil {
			return slept, stopped
		}
		s.mu.RLock()
		peerInst := s.instances[peer]
		var peerState State
		if peerInst != nil {
			peerState = peerInst.state
		}
		s.mu.RUnlock()
		if peerInst == nil {
			continue
		}
		if peerState == StateStopped {
			// Already stopped — nothing to reclaim.
			continue
		}
		peerCfg, ok := liveModelCfg(s.cfg, peer)
		if !ok {
			continue
		}
		if peerCfg.Host == "" {
			continue
		}
		// Skip slept-L1 peers that contribute zero residual — stopping
		// them would free nothing and only add wall-clock + a docker
		// stop call. This narrows the cold-load's eviction footprint
		// to peers that are ACTUALLY holding VRAM (either awake or
		// slept-with-non-zero residual). Live-observed (2026-06-09)
		// the OOM was driven by 27B awake (~12GB), not by zero-residual
		// siblings. Belt-and-suspenders against a runaway eviction
		// cascade in swap-groups whose members all configure
		// sleep_l1_residual_mb: 0.
		if peerState == StateSleeping && peerCfg.SleepL1ResidualMB == 0 {
			slog.Info("cold_load_skip_zero_residual_peer",
				"target", name, "peer", peer, "peer_state", peerState,
			)
			continue
		}
		slog.Info("cold_load_stopping_evict_peer",
			"target", name, "peer", peer, "container", peerCfg.Host,
			"peer_state", peerState,
		)
		// Drain in-flight on the peer (best-effort, bounded) so SIGTERM
		// doesn't abort live generation.
		s.drainInstance(ctx, peerInst)
		// docker stop FIRST, then mutate admission/state on success
		// (mirrors redeploy's MEDIUM-#4 ordering — see redeploy.go
		// comment "TOCTOU sub-window fix" for the rationale).
		peerStopCtx, peerStopCancel := context.WithTimeout(ctx, 90*time.Second)
		peerStopOut, peerStopErr := runSleepDocker(peerStopCtx, "docker", "stop", "-t", "60", peerCfg.Host)
		peerStopCancel()
		s.mu.Lock()
		peerInst.draining = false
		s.mu.Unlock()
		if peerStopErr != nil {
			slog.Error("cold_load_evict_peer_stop_failed",
				"target", name, "peer", peer, "container", peerCfg.Host,
				"err", peerStopErr, "output", string(peerStopOut),
			)
			LogLifecycleTransition(LifecycleEvent{
				Action: LifecycleEvict,
				Model:  peer,
				Reason: "stop-failed-vram-drift-risk",
				GPUs:   s.gpusForModel(peer),
			})
			metrics.AdmissionVRAMDriftRiskTotal.WithLabelValues(peer).Inc()
			continue
		}
		// docker stop succeeded — flip admission + state in one short
		// critical section.
		s.admission.NotifyStopped(peer)
		s.mu.Lock()
		peerInst.state = StateStopped
		s.mu.Unlock()
		LogLifecycleTransition(LifecycleEvent{
			Action: LifecycleEvict,
			Model:  peer,
			Reason: "cold-load-peer-stop",
			GPUs:   s.gpusForModel(peer),
		})
		stopped = append(stopped, peer)
	}
	return slept, stopped
}

// coldLoadStoppedMemberLocked is the lock-free body of
// coldLoadStoppedMember: it runs the TOCTOU re-check + docker start +
// /is_sleeping poll + best-effort cleanup, but assumes the global
// cold-load mutex is ALREADY held by the caller (Go's sync.Mutex is not
// reentrant). Use this helper from inside a WithColdLoadLock callback —
// e.g. performWake's Fix-5 path that needs to TOCTOU-re-check IsStopped
// under the same lock as the wake itself, without releasing/reacquiring
// in between. Callers outside the lock should use coldLoadStoppedMember
// instead.
func (s *Scheduler) coldLoadStoppedMemberLocked(ctx context.Context, inst *schedInstance, modelCfg config.ModelConfig) error {
	// TOCTOU re-check INSIDE the lock — a concurrent waiter may
	// have already cold-loaded this model while we were queued.
	if !s.admission.IsStopped(inst.model) {
		slog.Info("cold_load_skipped_already_started", "model", inst.model)
		return nil
	}

	// Lock is held; run the docker work lock-free (the helper
	// asserts the lock is held by us — see doColdLoad's docstring).
	container := modelCfg.Host
	startPeriod := modelCfg.EffectiveColdLoadTimeout()
	coldLoadStart := s.now()
	slog.Info("cold_loading_stopped_member",
		"model", inst.model,
		"container", container,
		"timeout", startPeriod,
	)
	// BUG 1: evict swap-group peers BEFORE starting the new member.
	// Without this, the request-triggered cold-load races for VRAM
	// with the resident pinned peer (live-observed 2026-06-09: moe
	// OOM'd at worker init because main held the GPUs awake). The
	// redeploy-member CLI verb already does this; the request-
	// triggered cold-load path was missing it.
	//
	// sleepInstance settles internally (settleAfterSleep) per peer
	// sleep, so we don't need to add another settle here. The docker-
	// stop path doesn't settle, but `docker stop -t 60` already gives
	// vLLM up to 60s of SIGTERM grace before SIGKILL — long enough
	// for the CUDA context to fully tear down before we proceed.
	slept, stopped := s.evictPeersForColdLoadLocked(ctx, inst.model, modelCfg)
	if len(slept) > 0 || len(stopped) > 0 {
		slog.Info("cold_load_peer_eviction_complete",
			"model", inst.model,
			"slept_pinned", slept,
			"stopped_peers", stopped,
		)
	}
	if err := s.doColdLoad(ctx, inst, container, startPeriod); err != nil {
		// BUG 1 rollback: any pinned peers we slept to free VRAM for
		// the (now-failed) cold-load must be re-woken so we don't
		// leave the operator with the pinned default down. We're
		// inside WithColdLoadLock — bestEffortRestorePinned uses
		// performWakeFromInsideColdLoadLock internally so coldLoadMu
		// reentrancy is avoided. Stop-mode peers we stopped are
		// intentionally LEFT stopped: they were rarely-woken
		// evict_action: stop members and the async-503
		// wake-from-Stopped path is the right shape (matches
		// RedeployMember step h).
		pinnedRestoreFailed := s.bestEffortRestorePinned(ctx, slept)
		// CRITICAL — if the pinned-peer restore failed, the system is
		// now in a strictly-worse state than before this cold-load was
		// attempted: pinned default Sleeping + cold-load target Stopped
		// + (without this clear) the 30s cooldown blocks the operator's
		// next retry. The cooldown's purpose is "doomed cold-load,
		// don't wedge the GPU-set lock"; it is NOT a lockout for known-
		// inconsistent admission state.
		//
		// On restore failure: emit a high-severity audit + bump the
		// pinned-wake-failed counter so dashboards alert, AND clear
		// coldLoadFailures[target] so the operator's next request (or
		// /admin/redeploy-member of the pinned peer) can fire
		// immediately. Wrap the returned error with
		// ErrColdLoadPinnedWakeFailed so mapWakeError routes it to
		// RejectAdminIntervention at the request handler.
		if len(pinnedRestoreFailed) > 0 {
			for _, peer := range pinnedRestoreFailed {
				slog.Error("pinned_peer_wake_failed_admission_inconsistent",
					"model", inst.model,
					"pinned_peer", peer,
					"cold_load_err", err,
					"runbook", "manual-reconcile",
				)
				LogLifecycleTransition(LifecycleEvent{
					// HIGH-1 (architect round-6): emit Action=evict (NOT wake)
					// for this event. The wake DID NOT happen — the rollback
					// attempted it and failed. Tagging the audit row with
					// LifecycleWake inflates wake-success dashboards and
					// breaks any alert that filters on action=wake (because
					// the model is in fact actively DOWN). Mirrors the
					// "cleanup-stop-failed-vram-drift-risk" convention used
					// for the cleanup-stop drift event below.
					Action: LifecycleEvict,
					Model:  peer,
					Reason: "pinned-wake-failed-admission-inconsistent",
					GPUs:   s.gpusForModel(peer),
				})
				metrics.AdmissionPinnedWakeFailedTotal.WithLabelValues(peer).Inc()
			}
			// Clear the cooldown entry for the TARGET cold-load model so
			// the operator's escape hatch (next request OR
			// /admin/redeploy-member) isn't blocked by the post-failure
			// cooldown gate while admission is in a known-inconsistent
			// state. The cooldown only applies on "fast-fail loop"
			// regressions, not on operator-recoverable inconsistency.
			s.mu.Lock()
			delete(s.coldLoadFailures, inst.model)
			s.mu.Unlock()
		}
		// Best-effort cleanup: stop the half-loaded container so the
		// admission-Stopped state and the actual container state stay
		// consistent.
		//
		// CRITICAL ORDERING: only mark admission Stopped if the cleanup
		// stop actually succeeded. If both the cold-load AND the cleanup
		// stop fail (daemon hiccup, network blip, OOM during shutdown),
		// the container may STILL BE RUNNING and holding tens of GiB of
		// VRAM. Calling NotifyStartFailed here would mark admission
		// Stopped (zero VRAM on the books), hiding the leak from every
		// future scheduling decision. We'd rather leave admission's
		// books in their pre-call in-flight state and surface a hard
		// failure than tell the rest of the system "this is fine, no
		// VRAM in use here". Operator owns recovery — auto-retry could
		// stomp on a process that's slowly self-recovering.
		stopCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		cleanupOut, cleanupErr := runSleepDocker(stopCtx, "docker", "stop", "-t", "30", container)
		cancel()
		if cleanupErr != nil {
			slog.Error("cold_load_cleanup_stop_failed_books_inconsistent",
				"model", inst.model,
				"container", container,
				"cleanup_err", cleanupErr,
				"cleanup_output", string(cleanupOut),
				"cold_load_err", err,
				"pinned_restore_failed", pinnedRestoreFailed,
				"runbook", "manual-reconcile-vram",
			)
			// Audit a drift-risk event distinct from the success-path
			// "failed" reason so dashboards can alert on this row
			// specifically. Action=evict matches "we attempted to
			// reclaim, but the books may not match reality".
			LogLifecycleTransition(LifecycleEvent{
				Action:   LifecycleEvict,
				Model:    inst.model,
				Reason:   "cleanup-stop-failed-vram-drift-risk",
				GPUs:     s.gpusForModel(inst.model),
				Duration: s.now().Sub(coldLoadStart),
			})
			metrics.AdmissionVRAMDriftRiskTotal.WithLabelValues(inst.model).Inc()
			// DO NOT call NotifyStartFailed here — see above. Surface a
			// hard, wrapped error so the caller knows recovery is
			// outside automation's purview. Wraps ErrAdmissionVRAMDriftRisk
			// so mapWakeError routes this to RejectAdminIntervention with
			// NO Retry-After (vs. the default retry-loopy 503).
			//
			// HIGH-2 (architect round-6): if pinned-restore ALSO failed
			// above (worst-case dual failure: cold-load failed → rollback
			// couldn't re-wake pinned → cleanup-stop ALSO failed), the
			// returned error must carry BOTH sentinels so:
			//   (a) errors.Is(err, ErrColdLoadPinnedWakeFailed) tells the
			//       operator pinned is down (manual redeploy needed), AND
			//   (b) errors.Is(err, ErrAdmissionVRAMDriftRisk) tells them
			//       the half-loaded container may still be on the GPUs.
			// Pre-fix the cleanup-error early-return dropped the pinned-
			// wake-failed signal entirely. errors.Join (Go 1.20+) joins
			// both sentinels into one chain; mapWakeError's switch picks
			// the more-specific ErrColdLoadPinnedWakeFailed when both
			// match (see MED-1 reorder).
			driftErr := fmt.Errorf("%w: cold load failed AND cleanup stop failed (container may still be running on GPUs %v): cold_load_err=%v cleanup_err=%v",
				ErrAdmissionVRAMDriftRisk, s.gpusForModel(inst.model), err, cleanupErr)
			if len(pinnedRestoreFailed) > 0 {
				pinnedErr := fmt.Errorf("%w: pinned peers left Sleeping after cold-load failure (peers=%v)",
					ErrColdLoadPinnedWakeFailed, pinnedRestoreFailed)
				return errors.Join(driftErr, pinnedErr)
			}
			return driftErr
		}
		s.admission.NotifyStartFailed(inst.model)
		LogLifecycleTransition(LifecycleEvent{
			Action:   LifecycleColdLoad,
			Model:    inst.model,
			Reason:   "failed",
			GPUs:     s.gpusForModel(inst.model),
			Duration: s.now().Sub(coldLoadStart),
		})
		// CRITICAL — if the pinned-peer rollback failed above, wrap the
		// cold-load error with ErrColdLoadPinnedWakeFailed so the caller
		// (performWake → tryRouteFromSleep → mapWakeError) routes this
		// to RejectAdminIntervention. Operators MUST be able to tell
		// "cold-load failed cleanly, retry later" from "cold-load failed
		// AND a pinned peer is now down". Use %w on BOTH wraps so
		// errors.Is matches both ErrColdLoadPinnedWakeFailed AND the
		// underlying cause (e.g. ErrColdLoadContainerExited) — callers
		// classifying via errors.Is don't lose the inner sentinel.
		if len(pinnedRestoreFailed) > 0 {
			return fmt.Errorf("%w: pinned peers left Sleeping after cold-load failure (peers=%v): %w",
				ErrColdLoadPinnedWakeFailed, pinnedRestoreFailed, err)
		}
		return err
	}

	// Success: container is up, /is_sleeping returned, vLLM is in
	// the post-load resting state. Flip admission to Sleeping so
	// the normal RequestWake → /wake_up path can take over.
	s.admission.NotifyStarted(inst.model)
	// Mirror the post-cold-load state in the scheduler's instance map.
	// Without this flip, the scheduler still sees StateStopped while
	// admission has flipped to admissionSleeping — so the next
	// AcquireRoute takes the full scheduling path (StateStopped fails
	// tryRouteFromSleep), evictConflicts → drainAndStopInstance
	// discards the instance from s.instances entirely, leaving
	// admission's L1 residual books pointing at a model the scheduler
	// just threw away (double-accounting bug). Mirrors the matching
	// flip in RedeployMember step (f).
	s.mu.Lock()
	inst.state = StateSleeping
	// BUG 2b: clear any stale failure record so a future Stop → Kick
	// cycle is not gated by an ancient failure timestamp.
	delete(s.coldLoadFailures, inst.model)
	s.mu.Unlock()
	dur := s.now().Sub(coldLoadStart)
	slog.Info("cold_load_succeeded",
		"model", inst.model,
		"container", container,
		"duration_ms", dur.Milliseconds(),
	)
	LogLifecycleTransition(LifecycleEvent{
		Action:   LifecycleColdLoad,
		Model:    inst.model,
		Reason:   "on_demand",
		GPUs:     s.gpusForModel(inst.model),
		Duration: dur,
	})
	return nil
}

// ErrColdLoadContainerExited is returned by doColdLoad when the
// member container's docker state is "exited" or "dead" during the
// post-start poll. Surfaces a fast failure instead of waiting the
// full cold_load_timeout for /is_sleeping to return — the HTTP probe
// against a non-running container hits docker-DNS NXDOMAIN (only
// running containers resolve), and the HTTP client cannot distinguish
// "DNS not ready yet" from "container is gone" so it retries up to
// its full timeout. The inspect probe gives us authoritative truth
// from the docker daemon in ~50ms regardless of DNS state.
//
// Live-observed (2026-06-09): moe container OOM'd at worker init and
// exited(1). /is_sleeping HTTP probes got DNS NXDOMAIN; doColdLoad
// waited the full 5min cold-load timeout before failing, and the
// failure re-kicked the cold-load — infinite OOM/retry loop holding
// the GPU-set lock and starving Sparx.
var ErrColdLoadContainerExited = errors.New("cold-load container exited")

// ErrColdLoadPinnedWakeFailed is returned by coldLoadStoppedMemberLocked
// when the cold-load failed AND the rollback step (re-waking pinned
// peers we slept to free VRAM for the now-failed cold-load) also
// failed. Distinct from a clean cold-load failure: clean failure
// leaves the pinned peer Awake; THIS failure leaves the pinned peer
// stuck Sleeping while the cold-load target is Stopped. The pinned-
// default availability invariant is violated and operator intervention
// is required to reconcile (manual /admin/redeploy-member of the
// pinned peer is the standard recovery).
//
// mapWakeError routes this sentinel to RejectAdminIntervention so
// retry-aware SDKs surface it as terminal rather than retry-loop a
// known-inconsistent state. The cold-load goroutine ALSO clears the
// post-failure cooldown for the target model when wrapping this error
// (sleep.go) so the operator's next request — or a manual
// /admin/redeploy-member — can immediately retry without waiting for
// the 30s cooldown to expire.
var ErrColdLoadPinnedWakeFailed = errors.New("cold-load pinned-peer wake failed; admission inconsistent")

// ErrSleepRejectedInflight is returned by sleepInstance when the
// pre-/sleep drain barrier could not bring inst.inflight to zero
// within the configured cumem_drain_timeout_seconds. Defense against
// vllm-project/vllm#45520 — calling /sleep with an in-flight decode
// corrupts the cumem allocator state via cudaErrorIllegalAddress and
// wedges the engine. Callers (admission evictor, idle suspender,
// redeploy-member peer-stop) should treat this as a transient refusal
// and retry on the next admission cycle / idle tick rather than
// escalate to docker stop — the in-flight requests will complete on
// their own and the next /sleep attempt will succeed.
var ErrSleepRejectedInflight = errors.New("sleep rejected: in-flight requests would race cumem PP-broadcast")

// ErrSleepSkippedWakeSettle is returned by sleepInstance when /sleep
// is requested inside the post-wake settle window. Defense against
// vllm-project/vllm#45519 — /wake_up returns 200 before all PP
// workers have settled; a /sleep that lands in that gap re-enters
// cumem mid-settle and wedges the engine. The settle window is
// per-model (wake_settle_seconds; default 2s). Callers should treat
// this as a transient refusal — the next sleep attempt after the
// settle expires will succeed.
var ErrSleepSkippedWakeSettle = errors.New("sleep skipped: still inside post-wake settle window")

// containerState is the subset of `docker inspect` fields jukebox cares
// about for the cold-load fast-fail probe. JSON unmarshaling drops
// every other field. Stored as a struct (not just a string) so future
// signals (ExitCode, OOMKilled, RestartCount) are easy to add.
type containerState struct {
	Status     string // "created" | "running" | "paused" | "restarting" | "removing" | "exited" | "dead"
	ExitCode   int
	OOMKilled  bool
	Error      string
	StartedAt  string
	FinishedAt string
}

// inspectContainerState returns the docker-daemon-reported state of
// `container`. Used by doColdLoad's poll loop (BUG 2a) to fast-fail
// when the member container has exited rather than waiting the full
// cold-load timeout on an HTTP probe that cannot succeed.
//
// Implementation: shells out via runSleepDocker (same hook the rest
// of sleep.go uses — tests inject a fake). Uses `docker inspect -f
// '{{json .State}}'` so the parse target is narrow and stable across
// docker versions.
//
// Returns (state, nil) on success. Returns (_, err) wrapping the
// underlying docker error on failure (container missing, daemon
// unreachable, malformed JSON). Callers in the cold-load path should
// log the error and continue polling — a transient docker daemon
// hiccup is not a reason to fail the cold-load.
func inspectContainerState(ctx context.Context, container string) (containerState, error) {
	out, err := runSleepDocker(ctx, "docker", "inspect", "-f", "{{json .State}}", container)
	if err != nil {
		return containerState{}, fmt.Errorf("docker inspect %q: %w (output: %s)", container, err, string(out))
	}
	var st containerState
	// Trim trailing newline that docker inspect emits — json.Unmarshal
	// tolerates it but doing the trim explicitly keeps the parse
	// failure messages clean.
	trimmed := bytes.TrimSpace(out)
	if err := json.Unmarshal(trimmed, &st); err != nil {
		return containerState{}, fmt.Errorf("docker inspect %q: parse state: %w (output: %s)", container, err, string(trimmed))
	}
	return st, nil
}

// doColdLoad runs `docker start <container>` (re-using the existing
// container instead of `docker compose up` so we don't need the project
// dir mounted) and polls /is_sleeping until it returns 200 or timeout.
//
// CONTRACT: the global cold-load mutex MUST be held by the caller.
// This function does NOT acquire it (Go's sync.Mutex is not reentrant
// — re-acquiring would deadlock). The caller is one of: performWake's
// Stopped branch, the /admin/redeploy-member handler. Both go through
// coldLoadStoppedMember which holds the lock.
func (s *Scheduler) doColdLoad(ctx context.Context, inst *schedInstance, container string, timeout time.Duration) error {
	startCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	startOut, startErr := runSleepDocker(startCtx, "docker", "start", container)
	cancel()
	if startErr != nil {
		return fmt.Errorf("docker start %q: %w (output: %s)", container, startErr, string(startOut))
	}

	// Poll vLLM /is_sleeping (the canonical post-load resting state).
	// vLLM containers come up slept-L1 by default after model load
	// completes; once this endpoint returns 200 AND is_sleeping=true, we
	// know the engine is initialized and ready to receive /wake_up.
	//
	// IMPORTANT: just receiving HTTP 200 is NOT sufficient. After
	// `docker start`, vLLM's HTTP server can come up before the model
	// finishes loading and respond 200 with `{"is_sleeping": false}`.
	// Returning early on err==nil flips admission to Sleeping
	// prematurely; a consumer request then arrives, /wake_up is
	// dispatched, and vLLM may accept the call before the model is
	// resident in VRAM — leading to confusing timeouts or stale-state
	// generation. We must wait for is_sleeping=true. (Mirrors
	// pollUntilSleeping in redeploy.go; commit 3759293's fix MUST
	// propagate here too — the previous `_, err := sc.IsSleeping(...)`
	// silently discarded the bool.)
	deadline := s.now().Add(timeout)
	pollInterval := coldLoadPollIntervalDuration()
	sc, ok := inst.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("instance %q manager does not support IsSleeping (sleep mode required)", inst.model)
	}
	var lastErr error
	for s.now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
		// BUG 2a: fast-fail probe — before issuing the /is_sleeping HTTP
		// request (which goes through docker's embedded DNS and cannot
		// distinguish "container not running yet" from "container exited
		// permanently"), ask the docker daemon directly. If the container
		// is exited/dead, return ErrColdLoadContainerExited immediately
		// instead of waiting the full cold-load timeout on a doomed
		// HTTP poll. Avoids the live-observed 5-minute wedge that held
		// the GPU-set cold-load lock and starved Sparx (2026-06-09).
		//
		// Inspect failures are logged but non-fatal — we fall through to
		// the HTTP poll. A transient docker daemon hiccup shouldn't
		// abort an otherwise-healthy cold-load.
		inspectCtx, inspectCancel := context.WithTimeout(ctx, 3*time.Second)
		st, inspectErr := inspectContainerState(inspectCtx, container)
		inspectCancel()
		if inspectErr == nil {
			switch st.Status {
			case "exited", "dead":
				slog.Error("cold_load_container_exited_fast_fail",
					"model", inst.model,
					"container", container,
					"status", st.Status,
					"exit_code", st.ExitCode,
					"oom_killed", st.OOMKilled,
					"docker_error", st.Error,
					"started_at", st.StartedAt,
					"finished_at", st.FinishedAt,
				)
				return fmt.Errorf("%w: container %q is in state %q (exit_code=%d oom_killed=%v): %s",
					ErrColdLoadContainerExited, container, st.Status, st.ExitCode, st.OOMKilled, st.Error)
			}
			// "created", "running", "paused", "restarting", "removing" —
			// all valid in-flight states; fall through to the /is_sleeping
			// probe to determine model-load progress.
		} else {
			slog.Warn("cold_load_inspect_probe_failed_continuing",
				"model", inst.model,
				"container", container,
				"err", inspectErr,
			)
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
	return fmt.Errorf("cold load timeout after %s waiting for is_sleeping=true on %q: %w", timeout, container, lastErr)
}

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

	// Read the live model config once — both guards (B: wake-settle,
	// A: cumem-drain) consume per-model knobs from it. liveModelCfg
	// may legitimately return ok=false for instances registered
	// outside the active.yaml model set (RegisterExternalInstances,
	// ComfyUI, redeploy-time pre-config-reload races); in that case
	// we fall back to the package defaults (5s drain / 2s settle).
	modelCfg, _ := liveModelCfg(s.cfg, inst.model)

	// Guard B (post-wake settle barrier — defends vllm#45519). If a
	// successful /wake_up completed within the wake-settle window,
	// refuse /sleep. /wake_up returns 200 before all PP workers have
	// finished their cumem settle; a /sleep in that gap re-enters
	// cumem mid-broadcast and wedges the engine. The schedInstance
	// state is NOT mutated on this path, so there's nothing to roll
	// back. Per-model knob: wake_settle_seconds (default 2s).
	settle := modelCfg.EffectiveWakeSettle()
	s.mu.RLock()
	lastWake := inst.lastWakeAt
	s.mu.RUnlock()
	if !lastWake.IsZero() && settle > 0 {
		elapsed := s.now().Sub(lastWake)
		if elapsed < settle {
			metrics.SleepSkippedCooldownTotal.WithLabelValues(inst.model, "wake_settle").Inc()
			slog.Warn("sleep_skipped_wake_settle_cumem_safety",
				"model", inst.model,
				"elapsed_ms", elapsed.Milliseconds(),
				"settle_ms", settle.Milliseconds(),
				"trigger_reason", reason,
				"upstream_ref", "vllm#45519",
			)
			return fmt.Errorf("%w: model %q woken %s ago, settle window is %s",
				ErrSleepSkippedWakeSettle, inst.model, elapsed, settle)
		}
	}

	start := s.now()

	s.mu.Lock()
	inst.draining = true
	prevState := inst.state
	inst.state = StateStopping
	s.mu.Unlock()

	// Guard A (drain barrier — defends vllm#45520). Strict drain wait
	// before the existing best-effort drain. Calling /sleep while
	// requests are mid-decode corrupts the cumem allocator state via
	// cudaErrorIllegalAddress and wedges the engine; the upstream bug
	// has no client-side fix. We refuse to call sc.Sleep() while the
	// in-flight count is non-zero. The per-model
	// cumem_drain_timeout_seconds (default 5s) bounds the wait — if
	// the count is still non-zero after that, restore prevState and
	// surface ErrSleepRejectedInflight to the caller. This deliberately
	// runs BEFORE the existing s.cfg.VLLM.DrainTimeout best-effort
	// drain (which polls without rejecting); the strict guard catches
	// the cumem-unsafe case quickly, the best-effort drain remains as
	// a politeness budget for the no-rejection case where we have a
	// few in-flight requests we can wait out.
	cumemDrainTimeout := modelCfg.EffectiveCumemDrainTimeout()
	if !inst.inflightIsDrained() && cumemDrainTimeout > 0 {
		drainCtxA, cancelA := context.WithTimeout(context.Background(), cumemDrainTimeout)
		drainErr := inst.inflight.WaitForDrain(drainCtxA)
		remaining := inst.inflight.Count()
		cancelA()
		if drainErr != nil || remaining > 0 {
			metrics.SleepRejectedInflightTotal.WithLabelValues(inst.model, reason).Inc()
			slog.Warn("sleep_rejected_inflight_cumem_safety",
				"model", inst.model,
				"in_flight", remaining,
				"timeout_ms", cumemDrainTimeout.Milliseconds(),
				"trigger_reason", reason,
				"upstream_ref", "vllm#45520",
			)
			s.mu.Lock()
			inst.state = prevState
			inst.draining = false
			s.mu.Unlock()
			return fmt.Errorf("%w: model %q has %d in-flight requests after %s drain wait",
				ErrSleepRejectedInflight, inst.model, remaining, cumemDrainTimeout)
		}
	}

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
		// Wedge-recovery probe: vLLM /sleep may have returned 2xx and then
		// the connection dropped before we read the response (e.g. ctx-derived
		// poll cancellation, transient network hiccup). If we naively
		// restore prevState=StateReady but vLLM IS actually sleeping, the
		// proxy will forward requests to a sleeping backend and they will
		// hang.
		//
		// MEDIUM #5: a single probe attempt is brittle — a 3s timeout on a
		// busy box, transient network hiccup, or vLLM reload-window can
		// flake the probe; falling through to rollback when vLLM IS asleep
		// re-wedges the model. Bounded retry: 3 attempts, ~500ms / 1s / 2s
		// backoff, each with its own Background-detached context (the
		// inbound ctx may already be cancelled by the time we get here).
		// On any success that returns isSleeping=true, take the wedge-
		// recovery path. If all 3 error, log "wedge_probe_inconclusive"
		// and proceed with rollback (current optimistic behavior).
		probeBackoffs := []time.Duration{500 * time.Millisecond, 1 * time.Second, 2 * time.Second}
		var probeErrs []error
		var isSleeping bool
		for i, backoff := range probeBackoffs {
			probeCtx, probeCancel := context.WithTimeout(context.Background(), 3*time.Second)
			gotSleeping, probeErr := sc.IsSleeping(probeCtx)
			probeCancel()
			if probeErr == nil {
				isSleeping = gotSleeping
				break
			}
			probeErrs = append(probeErrs, probeErr)
			// Don't sleep after the last attempt.
			if i < len(probeBackoffs)-1 {
				time.Sleep(backoff)
			}
		}
		// If all 3 probes errored, log the inconclusive outcome and fall
		// through to the optimistic rollback (preserves prior behavior on
		// total probe failure).
		if len(probeErrs) == len(probeBackoffs) {
			slog.Warn("wedge_probe_inconclusive_rolling_back_optimistically",
				"model", inst.model,
				"probe_errs", fmt.Sprintf("%v", probeErrs),
			)
		}
		if isSleeping {
			slog.Warn("sleep_api_errored_but_vllm_is_actually_sleeping",
				"model", inst.model,
				"err", err,
			)
			s.mu.Lock()
			inst.state = StateSleeping
			inst.draining = false
			s.mu.Unlock()
			// Mirror the success-path bookkeeping (subset — we don't have a
			// duration to record because the API call errored). Use the
			// dedicated SleepsTotal label "sleep-api-error-but-vllm-asleep"
			// so dashboards can spot wedge-recoveries via a stable metric
			// query (independent of the original trigger).
			//
			// LOW #8: pass the ORIGINAL `reason` to NotifySleep (NOT the
			// dedicated wedge tag) so the auto-restore gate
			// (`reason == "idle"` in admission.NotifySleep) fires when this
			// wedge-recovery happens during an idle-suspend. Without this,
			// an idle-suspend wedge-recovery on a swap-group member silently
			// skipped the highest-priority peer's auto-wake, leaving a
			// pinned helper unintentionally asleep. Emit a separate slog
			// line so the operator-visible wedge marker is preserved
			// independently of NotifySleep's reason.
			metrics.SleepsTotal.WithLabelValues(inst.model, "sleep-api-error-but-vllm-asleep").Inc()
			if reason != admissionReason && s.admission != nil {
				slog.Warn("wedge_recovery_with_original_reason",
					"model", inst.model,
					"reason", reason,
				)
				s.admission.NotifySleep(inst.model, reason)
			}
			return fmt.Errorf("sleep API: %w (recovered: vLLM is actually sleeping)", err)
		}
		// Probes say vLLM is NOT sleeping (or all probes errored): safe
		// to roll back to prevState so callers can fall back to a hard stop.
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
	// Unified audit shape — same line emitted for sleep/evict/wake/cold_load.
	// Phase 4 stop-on-evict will reuse this for cold_load events without
	// changing the metric or log shape. action=evict when this sleep was
	// initiated by admission to free VRAM for a peer wake; otherwise sleep.
	auditAction := LifecycleSleep
	if reason == admissionReason {
		auditAction = LifecycleEvict
	}
	LogLifecycleTransition(LifecycleEvent{
		Action:   auditAction,
		Model:    inst.model,
		Reason:   reason,
		GPUs:     s.gpusForModel(inst.model),
		Duration: dur,
	})
	// Notify the admission controller so its per-GPU budget reflects
	// the freed VRAM. Skip when the sleep was initiated by admission
	// itself — RequestWake updates the budget internally after this
	// returns (the evictor contract). Double-NotifySleep is idempotent
	// (markSleepingLocked guards against it) but skipping is cleaner.
	if reason != admissionReason && s.admission != nil {
		s.admission.NotifySleep(inst.model, reason)
	}
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
//
// Before calling vLLM, consults the admission controller (if attached)
// to ensure the GPU has budget for the awake-footprint of this model.
// Admission may evict lower-priority peers to make room. If admission
// rejects (no feasible eviction set, evictor failed), the wake fails
// without ever touching vLLM.
//
// LOCK CONTRACT: callers MUST NOT hold coldLoadMu when invoking this.
// performWake's wake-from-Stopped branch acquires coldLoadMu internally
// via coldLoadStoppedMember; if a caller already holds it (e.g. inside
// WithColdLoadLock from RedeployMember), use performWakeFromInsideColdLoadLock
// instead — sync.Mutex is non-reentrant and re-acquiring would deadlock.
func (s *Scheduler) performWake(ctx context.Context, inst *schedInstance, modelCfg config.ModelConfig, trigger string) error {
	// Wake-from-Stopped path: if admission has this model marked
	// admissionStopped (because a previous admission cycle picked it as
	// an evict_action: stop victim and `docker stop`'d the container),
	// the consumer-facing /wake_up call below cannot succeed — the
	// container's API server is gone. Cold-load it first (docker compose
	// up + health-wait + NotifyStarted), THEN fall through to the normal
	// /wake_up path.
	//
	// Holds the global cold-load mutex across the entire decision +
	// docker-up + health-wait + wake, so a concurrent request for the
	// same (or a different swap-group) Stopped peer waits behind us
	// instead of contending on GPU 3 (the GPU every swap-group member
	// touches). See AdmissionController.coldLoadMu's docstring for why
	// per-tuple granularity would under-serialize this topology.
	//
	// Fix 5 (TOCTOU): the IsStopped check + the cold-load + the wake
	// MUST run under the same WithColdLoadLock acquisition. The earlier
	// shape — outer IsStopped check, then coldLoadStoppedMember (which
	// acquired+released the lock), then performWakeFromInsideColdLoadLock
	// (which DID NOT hold the lock) — opened a window in which a
	// concurrent eviction-as-stop-victim by another model's RequestWake
	// could flip `name` to admissionStopped between the outer check and
	// the wake. RequestWake has no Stopped-branch; it would mark
	// admissionAwake on the books, the /wake_up call would fail against
	// the now-stopped container, and the rollback NotifySleep would
	// leave admissionSleeping with phantom L1Residual on the per-GPU
	// map AND IsModelColdLoading=false (blocking async-503 recovery).
	//
	// We close the window by acquiring the lock once around BOTH
	// branches: re-check IsStopped after the lock is held, route to
	// the lock-free cold-load body if the model is now-Stopped, and
	// then run the lock-free wake body. If admission isn't configured
	// at all, take the no-lock fast path (admission == nil means
	// untracked-model deployment; we still need the wake to function
	// for swap-mode-backed setups).
	if s.admission == nil {
		return s.performWakeFromInsideColdLoadLock(ctx, inst, modelCfg, trigger)
	}

	var outerErr error
	s.admission.WithColdLoadLock(func() {
		if s.admission.IsStopped(inst.model) {
			// Surface the post-acquire-Stopped observation distinctly so
			// an operator grepping for "wake_observed_stopped_post_lock_acquire"
			// can prove the TOCTOU re-check fired (vs. the outer-check
			// case which logs cold_load_acquired_lock).
			slog.Info("wake_observed_stopped_post_lock_acquire",
				"model", inst.model,
				"trigger", trigger,
			)
			if err := s.coldLoadStoppedMemberLocked(ctx, inst, modelCfg); err != nil {
				outerErr = err
				return
			}
			// On success the model is admissionSleeping; fall through to
			// the wake body which transitions Sleeping → Awake.
		}
		outerErr = s.performWakeFromInsideColdLoadLock(ctx, inst, modelCfg, trigger)
	})
	return outerErr
}

// performWakeFromInsideColdLoadLock is the lock-aware variant of performWake: it
// runs the normal admission-gate + /wake_up + health-wait + post-wake
// bookkeeping sequence WITHOUT consulting admission.IsStopped or
// invoking coldLoadStoppedMember. Use this from inside a
// WithColdLoadLock callback (e.g. RedeployMember step (g),
// bestEffortRestorePinned) so a peer that happens to be admissionStopped
// does not deadlock by re-acquiring coldLoadMu on the same goroutine.
//
// Callers outside any cold-load lock should use performWake — that
// variant handles the Stopped → Sleeping cold-load BEFORE calling this
// helper. The swap-group-auto-restore callback in SetAdmission runs in
// a fresh goroutine OUTSIDE any lock and therefore correctly uses
// performWake (so a Stopped peer can recover via cold-load).
func (s *Scheduler) performWakeFromInsideColdLoadLock(ctx context.Context, inst *schedInstance, modelCfg config.ModelConfig, trigger string) error {
	sc, ok := inst.mgr.(SleepCapable)
	if !ok {
		return fmt.Errorf("instance %q: manager does not support wake", inst.model)
	}

	// Admission gate. No-op when admission is unset or this model isn't
	// admission-tracked.
	var evictedVictims []string
	if s.admission != nil {
		if victims, err := s.admission.RequestWake(ctx, inst.model); err != nil {
			slog.Error("admission_rejected_wake", "model", inst.model, "err", err, "evicted", victims)
			return err
		} else if len(victims) > 0 {
			slog.Info("admission_evicted_for_wake", "model", inst.model, "victims", victims)
			evictedVictims = victims
		}
	}

	start := s.now()
	timeout := modelCfg.EffectiveWakeTimeout()

	if err := sc.Wake(ctx, timeout); err != nil {
		metrics.SleepFailuresTotal.WithLabelValues(inst.model, "wake_api").Inc()
		slog.Error("instance_wake_failed", "model", inst.model, "err", err, "timeout", timeout)
		// Admission has already updated its budget assuming the wake
		// would succeed. Roll back so the GPU budget doesn't think this
		// model is awake when it isn't. NotifySleep is idempotent.
		// Reason "wake-rollback" is NOT "idle" so this does not trigger
		// swap-group auto-restore.
		if s.admission != nil {
			s.admission.NotifySleep(inst.model, "wake-rollback")
		}
		// Fix 6: best-effort restore evicted peers. Without this, a wake
		// that failed after evicting A,B,C left A,B,C asleep (or stopped)
		// with nobody scheduled to wake them — operator-visible as "I
		// asked for T, T failed, AND my pinned helper is now silently
		// asleep." Peers in sleep mode get a best-effort async wake
		// queued; peers in stop mode are LEFT Stopped (auto-cold-load
		// would block this rollback for N×5min) and will async-recover
		// via the standard wake-from-Stopped 503 path on next demand.
		if len(evictedVictims) > 0 {
			slog.Error("wake_failed_with_victims",
				"target", inst.model,
				"victims", evictedVictims,
				"victim_count", len(evictedVictims),
				"err", err,
			)
			s.bestEffortRestoreVictims(ctx, evictedVictims, inst.model)
		}
		return err
	}

	s.mu.Lock()
	inst.state = StateReady
	inst.lastUsedAt = s.now()
	inst.lastWakeAt = s.now()
	s.mu.Unlock()

	dur := s.now().Sub(start)
	metrics.WakesTotal.WithLabelValues(inst.model, trigger).Inc()
	metrics.WakeDurationSeconds.WithLabelValues(inst.model).Observe(dur.Seconds())
	slog.Info("instance_woken",
		"model", inst.model,
		"trigger", trigger,
		"duration_ms", dur.Milliseconds(),
	)
	// Unified audit shape — admission victims (if any) ride along as
	// Related so a dashboard sees "wake X cost the sleep of [Y, Z]" in
	// one row. trigger ("request", "auto-restore", etc.) goes to Reason.
	LogLifecycleTransition(LifecycleEvent{
		Action:   LifecycleWake,
		Model:    inst.model,
		Related:  evictedVictims,
		Reason:   trigger,
		GPUs:     modelCfg.GPUs,
		Duration: dur,
	})
	if s.admission != nil {
		s.admission.NotifyWakeComplete(inst.model)
	}
	return nil
}

// mapWakeError converts a raw wake error to the user-facing error shape.
//
// Fix 7: not every wake error is retryable. The previous implementation
// wrapped EVERY error as RejectSwapInProgress with RetryAfter: 10s,
// which is correct for transient swap conflicts but pathological for
// structural failures (model too large for any eviction set,
// vram-drift-risk operator-intervention condition) — clients see 503 +
// Retry-After:10s and retry forever against an unsolvable condition.
// Introduces three branches keyed on sentinel errors:
//
//   - ErrAdmissionInfeasible — structural; the model genuinely doesn't
//     fit on this GPU set even after evicting everything that CAN be
//     evicted. RejectInsufficient with NO Retry-After so SDKs surface
//     the error to the operator rather than retry-loop.
//   - ErrAdmissionVRAMDriftRisk — admission books may not match
//     physical state because a cleanup docker stop failed; auto-retry
//     could stomp on a recovering process. RejectAdminIntervention
//     with NO Retry-After so SDKs surface the error and an operator
//     reconciles manually (see audit log).
//   - default — genuine in-flight swap conflict or transient evictor
//     failure. Original behavior: RejectSwapInProgress + 10s Retry-After.
func mapWakeError(err error) error {
	switch {
	case errors.Is(err, ErrAdmissionInfeasible):
		return &RejectError{
			Reason:  RejectInsufficient,
			Message: fmt.Sprintf("wake failed (infeasible): %v", err),
		}
	case errors.Is(err, ErrColdLoadPinnedWakeFailed):
		// MED-1 (architect round-6): match the more-specific pinned-wake
		// sentinel BEFORE ErrAdmissionVRAMDriftRisk. The HIGH-2 dual-
		// failure path joins BOTH sentinels into one chain via
		// errors.Join — naive ordering would let VRAMDriftRisk win and
		// silently shadow the pinned-down operator signal in the
		// returned message string. Both routes still go to
		// RejectAdminIntervention, but the message-string difference
		// matters: "pinned-peer wake rollback failed" tells the operator
		// to manually redeploy the peer; "vram-drift-risk" tells them to
		// reconcile the half-loaded container. Both are needed in the
		// dual-failure case and the message itself enumerates both
		// underlying errors when joined.
		return &RejectError{
			Reason:  RejectAdminIntervention,
			Message: fmt.Sprintf("wake failed (pinned-peer wake rollback failed): %v", err),
		}
	case errors.Is(err, ErrAdmissionVRAMDriftRisk):
		return &RejectError{
			Reason:  RejectAdminIntervention,
			Message: fmt.Sprintf("wake failed (vram-drift-risk): %v", err),
		}
	default:
		return &RejectError{
			Reason:     RejectSwapInProgress,
			RetryAfter: 10 * time.Second,
			Message:    fmt.Sprintf("wake failed: %v", err),
		}
	}
}

// bestEffortRestoreVictims walks a list of admission-evicted peers and
// schedules a best-effort restore. Sleep-mode peers get an async
// performWake queued in a fresh goroutine (so the rollback returns
// promptly to the consumer); stop-mode peers are LEFT Stopped (inline
// cold-load would block the rollback for ~5 min per peer with no
// benefit — they were Stopped victims by definition, not actively
// serving traffic). The wake-from-Stopped async-503 path will recover
// them on next demand.
//
// targetModel is the wake target that failed; used purely for logging.
// ctx is the parent context — restorations spawn from it so test
// teardown cancellation cleanly cancels in-flight wakes.
func (s *Scheduler) bestEffortRestoreVictims(ctx context.Context, victims []string, targetModel string) {
	if s == nil || len(victims) == 0 {
		return
	}
	for _, v := range victims {
		victim := v
		victimCfg, ok := liveModelCfg(s.cfg, victim)
		if !ok {
			slog.Warn("wake_rollback_victim_no_config", "victim", victim, "target", targetModel)
			continue
		}
		if victimCfg.EffectiveEvictAction() == config.EvictActionStop {
			// Stop-mode peer: docker stop completed (or container died)
			// during admission's eviction. Restoring inline would block
			// the rollback for ~5min cold-load per peer. Async-503
			// recovery is the right shape.
			slog.Warn("wake_failed_victim_left_stopped",
				"victim", victim,
				"wake_target", targetModel,
			)
			continue
		}
		s.mu.RLock()
		victimInst := s.instances[victim]
		s.mu.RUnlock()
		if victimInst == nil {
			slog.Warn("wake_rollback_victim_no_instance", "victim", victim, "target", targetModel)
			continue
		}
		// Spawn a goroutine — performWake itself acquires coldLoadMu
		// and may block on the lock for some time; we do NOT want to
		// hold up the consumer-facing wake-error return. Use a
		// background context with a generous bound (5 min covers a
		// full cold-load).
		go func() {
			restoreCtx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			if err := s.performWake(restoreCtx, victimInst, victimCfg, "wake-rollback-restore"); err != nil {
				slog.Warn("wake_rollback_restore_failed",
					"victim", victim,
					"wake_target", targetModel,
					"err", err,
				)
			}
		}()
	}
	_ = ctx // unused — restorations run on their own bounded context so the parent cancellation doesn't kill them.
}

// ---------------------------------------------------------------------------
// External-lifecycle bootstrap
// ---------------------------------------------------------------------------

// RegisterExternalInstances walks the config and registers a
// schedInstance for every model with lifecycle: external. These
// instances are seeded in StateReady by default (we assume the
// operator's external vLLM service is up); a startup health probe
// logs a warning if it isn't, but doesn't fail.
//
// If the model has sleep_mode enabled, we ALSO probe /is_sleeping at
// startup. If the external vLLM is currently asleep — e.g. it was put
// to sleep before jukebox restarted — we seed StateSleeping instead of
// StateReady so the first request takes the wake codepath rather than
// proxying directly to a sleeping backend (which accepts the TCP
// connection but cannot serve, causing the request to hang).
//
// Per-instance probes run in parallel: serial probing would stall
// startup by ~3s × N_externals before the HTTP server can come up.
// /health and /is_sleeping for a single instance also run in parallel.
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
	// (~10s) regardless of N. Each probe is best-effort.
	//
	// For each external instance we run TWO probes concurrently:
	//   1. /health (via VerifyReady) — purely advisory; logs a warning
	//      on failure but does NOT change the seeded state.
	//   2. /is_sleeping — only when the model has sleep_mode enabled.
	//      If it returns true, we flip the seeded state from StateReady
	//      to StateSleeping so the first request goes through the wake
	//      codepath. If it errors or returns false, we keep StateReady
	//      (best-effort: a probe error must not fail jukebox startup).
	//
	// Why 10s (not 3s): a freshly-recreated vLLM container's engine can
	// take several seconds to respond to /is_sleeping even when the
	// HTTP server is up — engine startup serializes against worker IPC
	// and cudagraph compilation. A 3s timeout was observed to fail
	// against a healthy-but-newly-started external instance, leaving
	// jukebox seeded as StateReady when reality was StateSleeping. 10s
	// is generous without delaying happy-path startup meaningfully.
	var wg sync.WaitGroup
	for _, p := range todo {
		wg.Add(1)
		go func(p pending) {
			defer wg.Done()
			probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()

			// Run /health and /is_sleeping in parallel for this instance
			// so wall-clock stays bounded by the slowest single probe.
			var inner sync.WaitGroup
			var healthOK bool
			inner.Add(1)
			go func() {
				defer inner.Done()
				if err := p.mgr.VerifyReady(probeCtx, p.name); err != nil {
					slog.Warn("external instance not reachable at startup; will reconcile on first request",
						"model", p.name,
						"url", p.mgr.BaseURL(),
						"err", err,
					)
					return
				}
				healthOK = true
			}()

			// Determine whether the model has sleep_mode enabled. Only
			// then is /is_sleeping meaningful (and only then will vLLM
			// have been started with VLLM_SERVER_DEV_MODE=1 so the
			// endpoint exists at all).
			var sleeping bool
			var sleepProbeOK bool
			modelCfg, hasCfg := liveModelCfg(s.cfg, p.name)
			if hasCfg && modelCfg.SleepMode {
				inner.Add(1)
				go func() {
					defer inner.Done()
					isSleeping, err := vllmcli.IsSleeping(probeCtx, p.mgr.BaseURL())
					if err != nil {
						slog.Warn("is_sleeping probe failed at startup; assuming ready",
							"model", p.name,
							"url", p.mgr.BaseURL(),
							"err", err,
						)
						return
					}
					sleeping = isSleeping
					sleepProbeOK = true
				}()
			}
			inner.Wait()

			// State write happens AFTER both probes complete.
			//
			// Decision matrix:
			//   reachable + /is_sleeping=true  → StateSleeping  (operator
			//       slept it, or a prior jukebox lifetime did; next request
			//       takes the /wake_up path)
			//   reachable + /is_sleeping=false → StateReady     (default;
			//       happy path, container is awake and ready)
			//   UNREACHABLE + evict_action:stop → StateStopped + admission
			//       NotifyStopped (a prior admission cycle or operator
			//       stopped this container; next consumer request hits the
			//       async-503 + KickColdLoad path which `docker start`s
			//       the container before any /wake_up attempt). Without
			//       this branch, the wake path would hit a dead address
			//       and fail with "connection refused" / "no such host".
			//   UNREACHABLE + evict_action:sleep (default) → StateReady
			//       (existing behavior — will reconcile on first request)
			//
			// "Unreachable" means BOTH probes failed. If /health succeeded
			// but /is_sleeping failed, the container is up (just sleep-API
			// hiccup) so we keep StateReady.
			unreachable := !healthOK && (!hasCfg || !modelCfg.SleepMode || !sleepProbeOK)
			isStopEvict := hasCfg && modelCfg.EffectiveEvictAction() == config.EvictActionStop

			switch {
			case sleepProbeOK && sleeping:
				s.mu.Lock()
				inst := s.instances[p.name]
				if inst != nil && inst.state == StateReady {
					inst.state = StateSleeping
				}
				s.mu.Unlock()
				// Keep admission's per-GPU budget honest: if admission
				// is attached and tracks this model, mark it sleeping
				// so it doesn't double-count the awake footprint.
				// NotifySleep is idempotent (markSleepingLocked guards).
				// Reason "boot-probe" is NOT "idle" — boot discovery of
				// a pre-sleeping external instance must not trigger
				// swap-group auto-restore (operator may have intentionally
				// left a different group member as the awake one).
				if s.admission != nil {
					s.admission.NotifySleep(p.name, "boot-probe")
				}
				slog.Info("registered external instance (asleep — wake on first request)",
					"model", p.name,
					"url", p.mgr.BaseURL(),
				)
			case unreachable && isStopEvict:
				// Container is gone (Exited from a prior admission
				// stop-on-evict cycle, operator action, or host reboot).
				// Mark admissionStopped so the next consumer request
				// hits the async-503 + KickColdLoad path which docker-
				// starts the container BEFORE any /wake_up.
				s.mu.Lock()
				inst := s.instances[p.name]
				if inst != nil {
					inst.state = StateStopped
				}
				s.mu.Unlock()
				if s.admission != nil {
					s.admission.NotifyStopped(p.name)
				}
				slog.Warn("external instance unreachable + evict_action=stop → marked admissionStopped",
					"model", p.name,
					"url", p.mgr.BaseURL(),
				)
				// Emit a unified lifecycle_transition for the Unknown →
				// Stopped seed so dashboards counting cold-load events
				// don't undercount peers that started the jukebox
				// process already-stopped (e.g. host reboot recovery).
				LogLifecycleTransition(LifecycleEvent{
					Action: LifecycleColdLoad,
					Model:  p.name,
					Reason: "boot-probe-stopped",
					GPUs:   s.gpusForModel(p.name),
				})
			}
		}(p)
	}
	wg.Wait()

	// Register ComfyUI-lifecycle instances. ComfyUI has different
	// semantics from vLLM-external — there is no /is_sleeping probe to
	// run, and jukebox itself owns the sleeping-state flag — so we seed
	// StateReady and let the first request reconcile if needed. We still
	// run /system_stats as a best-effort reachability log.
	s.registerComfyUIInstances(ctx)

	return nil
}

// registerComfyUIInstances seeds schedInstances for every model with
// lifecycle: comfyui. Unlike the external vLLM path, there is no
// /is_sleeping endpoint to probe (ComfyUI's "loaded" state is tracked
// internally by the ComfyUIManager itself), so we always seed
// StateReady and let the operator/scheduler drive sleep through normal
// channels. Reachability is probed best-effort against /system_stats
// so an unreachable container surfaces a startup warning, matching the
// external-vLLM ergonomics.
func (s *Scheduler) registerComfyUIInstances(ctx context.Context) {
	type pending struct {
		name string
		mgr  *vllm.ComfyUIManager
	}
	var todo []pending

	for name, model := range s.cfg.Models {
		if model.EffectiveLifecycle() != config.LifecycleComfyUI {
			continue
		}
		// Defense in depth — config validation already rejects this.
		if model.Alias != "" {
			continue
		}

		mgr := vllm.NewComfyUIManager(model.Host, model.Port)
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

		s.mu.Lock()
		s.instances[name] = inst
		s.mu.Unlock()
		metrics.RunningInstances.WithLabelValues(name, portLabel).Set(1)
		slog.Info("registered comfyui instance", "model", name, "url", mgr.BaseURL())
		todo = append(todo, pending{name: name, mgr: mgr})
	}

	if len(todo) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, p := range todo {
		wg.Add(1)
		go func(p pending) {
			defer wg.Done()
			probeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			if err := p.mgr.VerifyReady(probeCtx, p.name); err != nil {
				slog.Warn("comfyui instance not reachable at startup; will reconcile on first request",
					"model", p.name,
					"url", p.mgr.BaseURL(),
					"err", err,
				)
			}
		}(p)
	}
	wg.Wait()
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
		modelCfg, ok := liveModelCfg(s.cfg, inst.model)
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

	_, modelCfg, err := liveResolveModel(c.cfg, current)
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
	modelCfg, ok := liveModelCfg(c.cfg, current)
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
