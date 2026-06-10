package jukebox

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// Sentinel errors returned by RequestWake (and adjacent
// admission-driven failure paths) so callers (e.g. sleep.go's
// mapWakeError) can distinguish RETRYABLE failures (swap conflict,
// transient evictor failure) from STRUCTURAL failures (model can't fit
// even with maximum eviction; cleanup-stop-failed drift). Without this
// distinction every error rolled up as a 503 + Retry-After:10s and
// clients retried forever against unsolvable conditions.
//
// Use errors.Is() at the boundary — these are wrapped with %w by the
// production error sites so an outer wrap chain (e.g. "admission for
// 'longctx': %w") still classifies correctly.
var (
	// ErrAdmissionInfeasible — the model is structurally too large for
	// the pinned-adjusted budget on at least one GPU it touches, even
	// after evicting every eligible peer. Retrying will not help; the
	// fix is operational (resize the model, change pin/group config,
	// add VRAM). Maps to a non-retryable HTTP status.
	ErrAdmissionInfeasible = errors.New("admission: no feasible eviction set (model too large for pinned-adjusted budget)")

	// ErrAdmissionVRAMDriftRisk — a cleanup `docker stop` failed after a
	// cold-load failure (or a peer-stop failed mid-redeploy). The
	// container may STILL BE RUNNING and holding tens of GiB of VRAM
	// while admission's books were intentionally NOT updated to
	// "Stopped" (so we don't hide the leak from future scheduling
	// decisions). Retrying is dangerous — auto-retry could stomp on a
	// process that's slowly self-recovering. Operator must reconcile.
	// Maps to RejectAdminIntervention with NO Retry-After.
	ErrAdmissionVRAMDriftRisk = errors.New("admission: vram-drift-risk (cleanup docker stop failed; books may not match physical state)")
)

// Naming convention used in this file:
//
//   - Notify*  — public state-machine boundary methods called by the
//                scheduler when a real lifecycle transition happens
//                (NotifySleep, NotifyStarted, NotifyStopped,
//                NotifyStartFailed). They take the model name, look up
//                the per-model record under a.mu, run any policy
//                checks, and then call into one of the apply primitives.
//
//   - mark*Locked  — pure-arithmetic helpers (markSleepingLocked,
//                markStoppedLocked, markStartedLocked) that mutate a
//                modelAdmissionState's books in place. They assume the
//                caller already holds a.mu, do not consult policy, and
//                are idempotent (safe to call when state is already the
//                target). The "Locked" suffix names the precondition,
//                not the transition.
//
// Treat the Notify* methods as the contract surface and the mark*Locked
// helpers as inlinable budget-bookkeeping primitives. Adding a new
// transition? Wrap it in a Notify* boundary; reach for mark*Locked only
// to express "this transition's effect on the books is identical to
// transition X's."

// AdmissionController enforces per-GPU VRAM budgets across all awake
// models on a host. Before any wake call lands at vLLM, the scheduler
// consults the controller, which either:
//
//   - admits the wake (the model fits given current budgets), or
//   - admits after evicting peers (it picks the smallest set of lower-
//     priority / least-recently-used victims whose combined freed VRAM
//     lets the requested model fit), or
//   - returns an admission error (no eviction can make it fit — e.g.
//     the requested model is larger than the pinned-adjusted budget).
//
// Without this controller, two concurrent wakes on the same GPU could
// sum past the GPU's capacity and OOM at vLLM. The controller is the
// missing coordination point.
//
// The controller knows nothing about how to actually sleep a model —
// callers supply an AdmissionEvictor that performs the drain+sleep+
// settle while the controller holds its admission lock. This keeps
// admission.go free of dependencies on scheduler internals and makes it
// testable with a stub evictor.
type AdmissionController struct {
	// evictor is invoked synchronously when an eviction is required.
	evictor AdmissionEvictor

	// autoRestoreWake, if set, is dispatched in a goroutine after an
	// idle-sleep of a swap-group member to wake the highest-priority
	// sleeping peer in the same group. Wired by the scheduler in
	// SetAdmission so admission stays agnostic of scheduler internals.
	// Reason string is purely informational (metrics label / log line).
	autoRestoreWake func(name, reason string)

	// totalsByGPU is the total VRAM per GPU (MB). Built from the GPU
	// inventory at scheduler startup. Immutable after construction.
	totalsByGPU map[int]int

	// pinnedByGPU is the sum of declared pinned-model footprints per GPU,
	// subtracted from every budget calculation. Pinned models are never
	// in the eviction set. Immutable after construction.
	pinnedByGPU map[int]int

	// mu serializes admission decisions and protects models/awakeByGPU/
	// l1ResidualByGPU. Held across eviction calls so two concurrent wakes
	// can't both decide to evict the same victim.
	mu              sync.Mutex
	models          map[string]*modelAdmissionState
	awakeByGPU      map[int]int
	l1ResidualByGPU map[int]int

	// coldLoadMu serializes any cold-load (docker compose up + health-wait)
	// triggered by the wake-from-Stopped path. A single global mutex —
	// fine-grained per-GPU-set keying would under-serialize the GPU 3
	// overlap (vision {3} vs moe {0,1,2,3}) which is the exact race we
	// have to prevent (concurrent allocations on the shared GPU). Per
	// Kagi PATCH 16: every swap-group member touches GPU 3 in this
	// topology, so any two cold-loads contend for it; per-GPU-sorted
	// granularity buys zero parallelism here. Revisit if a future member
	// has a non-overlapping GPU set.
	//
	// Acquired by RequestWake's wake-from-Stopped branch (or the
	// /admin/redeploy-member handler) and held across the entire
	// cold-load + state transition. The cold-load helper itself is
	// lock-free and asserts the mutex is held by its caller (Go's
	// sync.Mutex is not reentrant — re-acquiring would deadlock).
	coldLoadMu sync.Mutex
}

// AdmissionEvictor is the REQUIRED callback shape the controller uses
// to sleep a victim it has picked. The implementation MUST block until
// the victim's VRAM is fully released (vLLM's /sleep + settle window)
// so the controller's budget math reflects reality before the incoming
// wake proceeds.
//
// CONTRACT: implementations MUST NOT call AdmissionController methods
// (NotifySleep, NotifyWakeComplete, RequestWake, TouchActivity) during
// SleepForEviction. The controller is holding its admission mutex
// across this call and any callback would deadlock. The controller
// updates its own budget bookkeeping after SleepForEviction returns.
//
// Evictors that ALSO want to support `docker stop`-based eviction (for
// models declared evict_action: stop in the catalog) should additionally
// implement AdmissionStopper. The controller type-asserts at the
// per-victim dispatch site; an evictor that does NOT satisfy
// AdmissionStopper will receive SleepForEviction calls even for
// evict_action: stop victims (the controller logs a warning and falls
// back to sleep accounting). This split keeps the interface
// source-compatible for any external implementer of AdmissionEvictor
// that pre-dates the stop-on-evict feature.
type AdmissionEvictor interface {
	SleepForEviction(ctx context.Context, victim, reason string) error
}

// AdmissionStopper is the OPTIONAL extension implemented by evictors
// that can tear a victim's container all the way down (full reclaim of
// CUDA context + NCCL/V1 buffers — the bytes vLLM /sleep cannot release).
// Used for models with EvictAction = config.EvictActionStop.
//
// The implementation must `docker stop` (or compose-stop) the container
// and not return until the stop is observed. Same admission-mutex
// contract as SleepForEviction applies: implementations MUST NOT
// callback into AdmissionController methods.
//
// Evictors that don't implement AdmissionStopper aren't broken — the
// controller falls back to SleepForEviction with a warning log line.
// The fallback is correct (the victim still goes offline VRAM-wise);
// it just keeps an L1 residual on the books that a real stop would
// have reclaimed.
type AdmissionStopper interface {
	StopForEviction(ctx context.Context, victim, reason string) error
}

// admissionModelState is the controller's view of one model.
type modelAdmissionState struct {
	Name string
	GPUs []int
	// ExpectedVRAMMB is the model's awake VRAM footprint, keyed by GPU.
	// Populated from ModelConfig.EffectiveExpectedVRAMMB(gpu) at controller
	// construction so downstream bookkeeping never has to know whether the
	// source was a flat scalar or a per-GPU map. Every GPU in `GPUs` has
	// an entry.
	ExpectedVRAMMB  map[int]int
	L1ResidualMB    int
	Priority        string
	Pinned          bool
	SwapGroup       string
	EvictAction     string // config.EvictActionSleep | config.EvictActionStop
	State           admissionModelState
	LastRequestTime time.Time

	// Booked* fields snapshot the EXACT values used when this model's
	// footprint was added to the controller's per-GPU maps (awakeByGPU,
	// l1ResidualByGPU, pinnedByGPU). Reversal arithmetic in
	// markSleepingLocked / markStoppedLocked / RequestWake (Sleeping →
	// Awake transition) MUST use these snapshots — NOT the live
	// ExpectedVRAMMB / GPUs / L1ResidualMB fields, which may have been
	// mutated in place by a hot-reload RefreshConfig in between booking
	// and reversal.
	//
	// Grandfathering invariant: RefreshConfig replaces ExpectedVRAMMB /
	// GPUs / L1ResidualMB in place but leaves Booked* untouched while
	// the booking is live. Booked* is refreshed only on the NEXT
	// booking transition (e.g. Sleeping → Awake re-snapshots from the
	// then-current live fields).
	//
	// BookedExpectedVRAMMB: per-GPU awake-vram amounts currently
	//   contributing to awakeByGPU (or pinnedByGPU when BookedInPinnedMap).
	//   nil when nothing is booked (e.g. admissionUnknown / admissionStopped).
	// BookedL1ResidualMB: per-GPU residual currently contributing to
	//   l1ResidualByGPU (same on every GPU in BookedL1ResidualGPUs).
	// BookedL1ResidualGPUs: GPUs where the L1 residual is currently
	//   booked. nil when nothing is booked.
	// BookedInPinnedMap: true when BookedExpectedVRAMMB is contributing
	//   to pinnedByGPU (non-swap-group pinned at construction time)
	//   rather than awakeByGPU.
	BookedExpectedVRAMMB map[int]int
	BookedL1ResidualMB   int
	BookedL1ResidualGPUs []int
	BookedInPinnedMap    bool
}

// ExpectedOn returns the model's awake VRAM footprint on the given GPU.
// Defensive default of 0 when the GPU isn't in the model's map (which the
// config validator prevents in production but keeps tests that hand-craft
// states from panicking).
func (m *modelAdmissionState) ExpectedOn(gpu int) int {
	if v, ok := m.ExpectedVRAMMB[gpu]; ok {
		return v
	}
	return 0
}

type admissionModelState int

const (
	admissionUnknown admissionModelState = iota
	admissionAwake
	admissionSleeping
	// admissionStopped — container is `docker compose stop`'d. The model
	// holds zero awake VRAM AND zero L1 residual on its GPUs (full reclaim
	// of CUDA context + NCCL buffers). Restoring it requires `docker
	// compose up` + health-wait + an explicit NotifyStarted call (cold
	// load ~5 min from page cache); /wake_up does NOT apply. Only
	// reachable for models with EvictAction = config.EvictActionStop.
	admissionStopped
)

// NewAdmissionController builds a controller from the model config + a
// GPU inventory snapshot (total MB per GPU). Models without
// AdmissionEnabled() are skipped — their wake is unmediated.
//
// totalsByGPU should reflect REAL GPU capacity (queried via nvidia-smi
// at scheduler startup), not configured caps. Tests supply a synthetic
// map. The controller never re-queries the inventory at runtime.
// SetEvictor swaps the AdmissionEvictor after construction. Used by
// tests that need to wire a real *SchedulerEvictor (which requires the
// already-constructed Scheduler). Production paths set the evictor at
// NewAdmissionController time and never call this. Safe to call only
// before any RequestWake — the controller does NOT re-acquire the
// mutex around this swap.
func (a *AdmissionController) SetEvictor(evictor AdmissionEvictor) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.evictor = evictor
}

// RefreshConfig reconciles the controller's per-model state records
// against a freshly-loaded config (e.g. after an fsnotify hot-reload
// of active.yaml). Used by Scheduler.OnConfigReloaded to keep per-model
// admission decisions in sync with the operator-edited config without
// requiring a process restart.
//
// Grandfathering — IMPORTANT: live per-GPU bookkeeping (awakeByGPU,
// l1ResidualByGPU, pinnedByGPU) is NOT recomputed against the new
// config for currently-booked models. The reversal arithmetic in
// markSleepingLocked / markStoppedLocked / RequestWake uses the
// per-model Booked* snapshots (captured at booking time, never
// rewritten while the booking is live), so a hot-reload of
// expected_vram_mb_* / gpus / sleep_l1_residual_mb mid-booking does
// NOT leak phantom budget on the next reversal.
//
// The NEW ExpectedVRAMMB / GPUs / L1ResidualMB apply on the next
// booking transition (Sleeping/Stopped → Awake re-snapshots from the
// then-current live fields). This matches operator intent: "tuning
// vram doesn't move the running model around; the next wake uses the
// new value."
//
// Adding a model that wasn't admission-tracked before: registered with
// zero current bookkeeping; it is admissionUnknown until the next
// scheduler-observed lifecycle event (wake / sleep / stop). If the
// new model is pinned AND admission-enabled, its footprint is ALSO
// immediately added — to pinnedByGPU when non-swap-group, to
// awakeByGPU when swap-group — and State set to admissionAwake. This
// mirrors NewAdmissionController so a hot-reload that adds a pinned
// model behaves identically to a restart with that model already in
// config.
//
// Removing a model that WAS tracked: dropped from the controller's
// view; any stale bookings on the per-GPU maps ARE clawed back via
// the Booked* snapshots, so the per-GPU maps stay accurate across
// model removal too.
//
// Pinned-flip handling (pinned ↔ unpinned via hot-reload):
//   - flip TRUE → FALSE for a model whose footprint is in pinnedByGPU
//     (non-swap-group): footprint is migrated from pinnedByGPU to
//     awakeByGPU (BookedInPinnedMap cleared). Model stays awake.
//   - flip FALSE → TRUE: not reconciled — the model's footprint stays
//     in awakeByGPU until its next sleep. Pinned arithmetic is
//     deferred to the next booking transition. Operator-tolerable:
//     pinned-flag is a hot-reloadable knob that takes effect on the
//     next wake; documented in the hot-reload matrix.
//
// Field-by-field per-model refresh:
//   - GPUs               replaced (booking grandfathered via snapshot)
//   - ExpectedVRAMMB     replaced (grandfathered booking)
//   - L1ResidualMB       replaced (grandfathered booking)
//   - Priority           replaced (Pinned still forces critical)
//   - Pinned             replaced
//   - SwapGroup          replaced
//   - EvictAction        replaced
//
// State / LastRequestTime / Booked* fields are preserved across
// refresh; Booked* turns over only at the next booking transition.
func (a *AdmissionController) RefreshConfig(cfg *config.Config) {
	if a == nil || cfg == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	seen := map[string]bool{}
	for name, m := range cfg.Models {
		if !m.AdmissionEnabled() {
			continue
		}
		seen[name] = true
		existing, ok := a.models[name]
		if !ok {
			// Newly-admission-tracked model. Register with zero
			// bookkeeping; lifecycle observers (NotifyStarted /
			// NotifySleep / NotifyStopped) will transition it.
			expected := make(map[int]int, len(m.GPUs))
			for _, g := range m.GPUs {
				expected[g] = m.EffectiveExpectedVRAMMB(g)
			}
			st := &modelAdmissionState{
				Name:           name,
				GPUs:           append([]int(nil), m.GPUs...),
				ExpectedVRAMMB: expected,
				L1ResidualMB:   m.SleepL1ResidualMB,
				Priority:       m.EffectivePriority(),
				Pinned:         m.Pinned != nil && *m.Pinned,
				SwapGroup:      m.SwapGroup,
				EvictAction:    m.EffectiveEvictAction(),
				State:          admissionUnknown,
			}
			if st.Pinned {
				st.Priority = config.PriorityCritical
				// Mirror NewAdmissionController: pinned models get
				// immediate booking + Awake state at registration time,
				// snapshotting Booked* so future hot-reload mutations
				// don't corrupt reversal arithmetic.
				st.State = admissionAwake
				booked := make(map[int]int, len(st.GPUs))
				for _, g := range st.GPUs {
					booked[g] = st.ExpectedOn(g)
				}
				st.BookedExpectedVRAMMB = booked
				if st.SwapGroup == "" {
					st.BookedInPinnedMap = true
					for _, g := range st.GPUs {
						a.pinnedByGPU[g] += st.ExpectedOn(g)
					}
				} else {
					for _, g := range st.GPUs {
						a.awakeByGPU[g] += st.ExpectedOn(g)
					}
				}
			}
			a.models[name] = st
			continue
		}
		// In-place field update; State + LastRequestTime + Booked*
		// preserved. The booking on the per-GPU maps stays at the
		// original (snapshotted) values — the new ExpectedVRAMMB /
		// L1ResidualMB / GPUs apply on the NEXT booking transition.
		existing.GPUs = append(existing.GPUs[:0], m.GPUs...)
		newExpected := make(map[int]int, len(m.GPUs))
		for _, g := range m.GPUs {
			newExpected[g] = m.EffectiveExpectedVRAMMB(g)
		}
		existing.ExpectedVRAMMB = newExpected
		existing.L1ResidualMB = m.SleepL1ResidualMB
		newPinned := m.Pinned != nil && *m.Pinned
		// Pinned TRUE → FALSE migration: footprint currently in
		// pinnedByGPU (BookedInPinnedMap=true) moves to awakeByGPU.
		// Reverse-and-re-add using the BookedExpectedVRAMMB snapshot
		// (NOT live ExpectedVRAMMB) so the migration is exactly
		// budget-conservative.
		if existing.Pinned && !newPinned && existing.BookedInPinnedMap {
			for g, v := range existing.BookedExpectedVRAMMB {
				a.pinnedByGPU[g] -= v
				a.awakeByGPU[g] += v
			}
			existing.BookedInPinnedMap = false
		}
		existing.Pinned = newPinned
		existing.SwapGroup = m.SwapGroup
		existing.EvictAction = m.EffectiveEvictAction()
		if existing.Pinned {
			existing.Priority = config.PriorityCritical
		} else {
			existing.Priority = m.EffectivePriority()
		}
	}
	// Drop models that disappeared from the config. Claw back their
	// bookings via the Booked* snapshots — safe because the snapshots
	// describe EXACTLY what was added to the per-GPU maps. Without this,
	// a model removed mid-Awake would leak its footprint until restart.
	for name, m := range a.models {
		if seen[name] {
			continue
		}
		a.reverseAllBookingsLocked(m)
		delete(a.models, name)
	}
	a.publishGauges()
}

// reverseAllBookingsLocked subtracts every active per-GPU booking for m
// using the Booked* snapshots. Used by RefreshConfig when a tracked
// model disappears from the new config. Idempotent (clears the
// snapshots so a second call is a no-op). Caller must hold a.mu.
func (a *AdmissionController) reverseAllBookingsLocked(m *modelAdmissionState) {
	if len(m.BookedExpectedVRAMMB) > 0 {
		if m.BookedInPinnedMap {
			for g, v := range m.BookedExpectedVRAMMB {
				a.pinnedByGPU[g] -= v
			}
		} else {
			for g, v := range m.BookedExpectedVRAMMB {
				a.awakeByGPU[g] -= v
			}
		}
		m.BookedExpectedVRAMMB = nil
		m.BookedInPinnedMap = false
	}
	if m.BookedL1ResidualMB > 0 && len(m.BookedL1ResidualGPUs) > 0 {
		for _, g := range m.BookedL1ResidualGPUs {
			a.l1ResidualByGPU[g] -= m.BookedL1ResidualMB
		}
		m.BookedL1ResidualMB = 0
		m.BookedL1ResidualGPUs = nil
	}
}

func NewAdmissionController(cfg *config.Config, totalsByGPU map[int]int, evictor AdmissionEvictor) *AdmissionController {
	a := &AdmissionController{
		evictor:         evictor,
		totalsByGPU:     map[int]int{},
		pinnedByGPU:     map[int]int{},
		models:          map[string]*modelAdmissionState{},
		awakeByGPU:      map[int]int{},
		l1ResidualByGPU: map[int]int{},
	}
	for g, mb := range totalsByGPU {
		a.totalsByGPU[g] = mb
	}
	for name, m := range cfg.Models {
		if !m.AdmissionEnabled() {
			continue
		}
		expected := make(map[int]int, len(m.GPUs))
		for _, g := range m.GPUs {
			expected[g] = m.EffectiveExpectedVRAMMB(g)
		}
		st := &modelAdmissionState{
			Name:           name,
			GPUs:           append([]int(nil), m.GPUs...),
			ExpectedVRAMMB: expected,
			L1ResidualMB:   m.SleepL1ResidualMB,
			Priority:       m.EffectivePriority(),
			Pinned:         m.Pinned != nil && *m.Pinned,
			SwapGroup:      m.SwapGroup,
			EvictAction:    m.EffectiveEvictAction(),
		}
		// Pinned models default to "always awake, footprint reserved
		// permanently in pinnedByGPU". BUT pinned + swap_group is a
		// different beast: the operator declared that the pinned member
		// can be swapped out by in-group peers, so its footprint must be
		// released when it sleeps. For that case treat it like any other
		// awake admission-tracked model (footprint in awakeByGPU, eligible
		// for sleeping via markSleepingLocked, eligible for re-waking).
		// The Pinned flag is then only relevant for idle-suspend exclusion
		// (sleep.go's checkIdle) and for swap-group auto-restore preference.
		if st.Pinned {
			st.Priority = config.PriorityCritical
			st.State = admissionAwake
			// Snapshot the per-GPU footprint used for the booking so a
			// future hot-reload of expected_vram_mb_* / gpus does NOT
			// retroactively rewrite the reversal arithmetic.
			booked := make(map[int]int, len(st.GPUs))
			for _, g := range st.GPUs {
				booked[g] = st.ExpectedOn(g)
			}
			st.BookedExpectedVRAMMB = booked
			if st.SwapGroup == "" {
				st.BookedInPinnedMap = true
				for _, g := range st.GPUs {
					a.pinnedByGPU[g] += st.ExpectedOn(g)
				}
			} else {
				for _, g := range st.GPUs {
					a.awakeByGPU[g] += st.ExpectedOn(g)
				}
			}
		}
		a.models[name] = st
	}
	return a
}

// SetAutoRestoreWake installs a callback used by NotifySleep to
// auto-wake the highest-priority sleeping peer in the same swap_group
// after a group member auto-sleeps via idle_timeout. Set by the
// scheduler in SetAdmission; passing nil disables auto-restore. Safe
// to call any time before the first NotifySleep.
//
// The callback is dispatched in a fresh goroutine — admission's mutex
// is NOT held during the wake, and the wake codepath (Scheduler.performWake
// → AdmissionController.RequestWake) will re-acquire admission's lock
// normally.
func (a *AdmissionController) SetAutoRestoreWake(fn func(name, reason string)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.autoRestoreWake = fn
}

// TrackedModels returns the names of all models the controller is
// tracking (admission-enabled, including pinned). Used by callers
// that need to know whether a given wake should be mediated.
func (a *AdmissionController) TrackedModels() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, 0, len(a.models))
	for n := range a.models {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// IsTracked reports whether `name` is admission-managed.
func (a *AdmissionController) IsTracked(name string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	_, ok := a.models[name]
	return ok
}

// RequestWake is called by the scheduler BEFORE calling vLLM's
// /wake_up for `name`. It blocks until admission is granted (possibly
// after sleeping one or more peers). Returns the list of victims that
// were evicted (informational, for logging/metrics) and any error.
//
// If `name` is not admission-tracked, RequestWake is a no-op that
// returns immediately — callers should always call this method
// unconditionally and let the controller decide whether to act.
//
// On error: the model is NOT marked awake; callers should NOT proceed
// to /wake_up. Common errors: context cancellation, evictor failure,
// "no feasible eviction set" (the model is larger than the
// pinned-adjusted budget no matter what we evict).
func (a *AdmissionController) RequestWake(ctx context.Context, name string) ([]string, error) {
	start := time.Now()

	a.mu.Lock()
	// NOTE: a.mu is NOT held via `defer Unlock` in this function. The
	// evictor calls below run with the lock RELEASED so a 60s+ evict
	// wall-clock doesn't block concurrent IsStopped / SnapshotBudgets /
	// TouchActivity / NotifyWakeComplete. Every exit path must call
	// a.mu.Unlock() explicitly. See the explicit unlock+reacquire pattern
	// around evictor.SleepForEviction / stopper.StopForEviction below.

	m, ok := a.models[name]
	if !ok {
		// Untracked model — pass through unmediated.
		a.mu.Unlock()
		return nil, nil
	}
	if m.Pinned && m.State == admissionAwake {
		// Pinned + awake = no-op (the common case — pinned models are
		// presumed awake from boot). A pinned model in admissionSleeping
		// state can occur only via swap-group eviction; in that case fall
		// through and do real wake-budget work just like any other model.
		a.mu.Unlock()
		return nil, nil
	}
	if m.State == admissionAwake {
		// Already accounted-for awake (e.g. a duplicate wake call). The
		// fan-out at the scheduler layer already de-dupes concurrent wake
		// requests for the same model, so reaching here means the model
		// was woken by some other path; just refresh the timestamp.
		m.LastRequestTime = time.Now()
		a.mu.Unlock()
		return nil, nil
	}

	// Compute fit per GPU. The model is currently in the L1Residual
	// pool (if it was sleeping) — bringing it awake means moving
	// (ExpectedVRAMMB - L1ResidualMB) from residual to awake on each GPU
	// it touches. So the "additional VRAM needed" on each GPU is the
	// delta. For an unknown-state model (first wake ever), the full
	// ExpectedVRAMMB is needed.
	//
	// `needed` is per-GPU because ExpectedVRAMMB can differ across GPUs
	// (vLLM pipeline-parallel rank asymmetry: e.g. rank 1 carries
	// embeddings + lm_head + sampler, +1-2 GiB heavier than rank 0).
	needed := make(map[int]int, len(m.GPUs))
	for _, g := range m.GPUs {
		n := m.ExpectedOn(g)
		if m.State == admissionSleeping {
			n -= m.L1ResidualMB
		}
		needed[g] = n
	}

	victims, err := a.pickVictims(m, needed)
	if err != nil {
		metrics.AdmissionRejectionsTotal.WithLabelValues(name, "infeasible").Inc()
		a.mu.Unlock()
		return nil, err
	}

	if len(victims) > 0 {
		slog.Info("admission_evicting",
			"model", name,
			"victims", victimNames(victims),
			"needed_mb_by_gpu", needed,
			"gpus", m.GPUs,
		)
	}

	for _, v := range victims {
		// Dispatch sleep vs stop per victim's configured EvictAction.
		// The state-update primitive differs (markSleepingLocked vs
		// markStoppedLocked) but the admission contract is the same:
		// the evictor must not callback into NotifySleep / NotifyStopped.
		//
		// For EvictActionStop victims, we type-assert the evictor to the
		// OPTIONAL AdmissionStopper interface. Evictors that don't
		// implement it (the historical AdmissionEvictor-only shape) fall
		// back to SleepForEviction with a warning + sleep accounting —
		// the model still goes offline VRAM-wise, just with the L1
		// residual that a real stop would have reclaimed. This keeps
		// the interface source-compatible.
		//
		// CRITICAL: a.mu is RELEASED across the evictor call and
		// RE-ACQUIRED to apply the post-evict state mutation. This keeps
		// the 60s+ drain+sleep+settle out of the admission lock so
		// concurrent IsStopped / SnapshotBudgets / TouchActivity calls
		// from the proxy hot path don't pile up behind it. The mark*Locked
		// helpers are idempotent — two concurrent RequestWake calls that
		// pick the same victim during the unlocked window race-then-converge
		// (both call SleepForEviction = idempotent on already-asleep vLLM,
		// both call markSleepingLocked = idempotent on already-Sleeping
		// state). No double-decrement of awakeByGPU.
		switch v.EvictAction {
		case config.EvictActionStop:
			stopper, ok := a.evictor.(AdmissionStopper)
			if !ok {
				slog.Warn("evictor_does_not_support_stop_falling_back_to_sleep",
					"victim", v.Name,
					"evict_action", "stop",
					"requesting_model", name,
				)
				a.mu.Unlock()
				evictErr := a.evictor.SleepForEviction(ctx, v.Name, "admission")
				a.mu.Lock()
				if evictErr != nil {
					metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_failed").Inc()
					a.mu.Unlock()
					return nil, fmt.Errorf("admission for %q: failed to evict %q (stop-not-supported fallback to sleep): %w", name, v.Name, evictErr)
				}
				metrics.AdmissionEvictsTotal.WithLabelValues(name, v.Name).Inc()
				// Use sleep accounting because we performed a sleep, not a stop.
				a.markSleepingLocked(v)
				continue
			}
			a.mu.Unlock()
			stopErr := stopper.StopForEviction(ctx, v.Name, "admission")
			a.mu.Lock()
			if stopErr != nil {
				metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_failed").Inc()
				a.mu.Unlock()
				return nil, fmt.Errorf("admission for %q: failed to stop %q: %w", name, v.Name, stopErr)
			}
			metrics.AdmissionEvictsTotal.WithLabelValues(name, v.Name).Inc()
			a.markStoppedLocked(v)
		default:
			// EvictActionSleep (default).
			a.mu.Unlock()
			evictErr := a.evictor.SleepForEviction(ctx, v.Name, "admission")
			a.mu.Lock()
			if evictErr != nil {
				metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_failed").Inc()
				a.mu.Unlock()
				return nil, fmt.Errorf("admission for %q: failed to evict %q: %w", name, v.Name, evictErr)
			}
			metrics.AdmissionEvictsTotal.WithLabelValues(name, v.Name).Inc()
			a.markSleepingLocked(v)
		}
	}

	// Re-check fit after evictions — a defensive sanity check against
	// arithmetic errors in pickVictims.
	for _, g := range m.GPUs {
		n := needed[g]
		if avail := a.availableMB(g); avail < n {
			metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_insufficient").Inc()
			a.mu.Unlock()
			return victimNames(victims), fmt.Errorf("%w: admission for %q: GPU %d still short %dMB after evicting %d peers (need %d, have %d)",
				ErrAdmissionInfeasible, name, g, n-avail, len(victims), n, avail)
		}
	}

	// Mark model awake and update budgets. If it was sleeping, also
	// drop its residual contribution.
	//
	// Snapshot discipline: reverse the residual using the BookedL1*
	// snapshots (set when the model entered Sleeping), then re-snapshot
	// from the LIVE config (m.GPUs / m.ExpectedVRAMMB / m.L1ResidualMB)
	// for the new awake booking. This is the canonical "next booking
	// transition picks up the hot-reloaded config" point.
	if m.State == admissionSleeping && m.BookedL1ResidualMB > 0 {
		for _, g := range m.BookedL1ResidualGPUs {
			a.l1ResidualByGPU[g] -= m.BookedL1ResidualMB
		}
		m.BookedL1ResidualMB = 0
		m.BookedL1ResidualGPUs = nil
	}
	booked := make(map[int]int, len(m.GPUs))
	for _, g := range m.GPUs {
		v := m.ExpectedOn(g)
		a.awakeByGPU[g] += v
		booked[g] = v
	}
	m.BookedExpectedVRAMMB = booked
	m.BookedInPinnedMap = false
	m.State = admissionAwake
	m.LastRequestTime = time.Now()

	a.publishGauges()
	a.mu.Unlock()
	metrics.AdmissionWaitSeconds.WithLabelValues(name).Observe(time.Since(start).Seconds())
	return victimNames(victims), nil
}

// NotifySleep is called by the scheduler AFTER a successful sleep —
// either an idle-timeout sleep or a manual sleep. (Admission-initiated
// evictions update budgets internally via markSleepingLocked; callers
// MUST NOT call NotifySleep for those — see AdmissionEvictor contract.)
// Updates budgets to reflect the freed VRAM (less the L1 residual).
//
// The reason string distinguishes sleep triggers ("idle", "manual",
// "evict", etc.). Only "idle" sleeps trigger swap-group auto-restore:
// when the just-slept model belongs to a swap_group, the highest-priority
// sleeping peer in the same group is auto-woken via the SetAutoRestoreWake
// callback. Manual / eviction / shutdown sleeps do NOT trigger auto-restore.
//
// Safe to call for untracked models (no-op). Idempotent: double-calls
// or calls on already-sleeping models are no-ops.
func (a *AdmissionController) NotifySleep(name, reason string) {
	a.mu.Lock()

	m, ok := a.models[name]
	if !ok {
		a.mu.Unlock()
		return
	}
	// Non-swap-group pinned models have their footprint in pinnedByGPU
	// and never legitimately transition to sleeping. Guard against an
	// errant NotifySleep underflowing awakeByGPU. Swap-group pinned
	// models DO transition (their footprint lives in awakeByGPU); fall
	// through to markSleepingLocked.
	if m.Pinned && m.SwapGroup == "" {
		a.mu.Unlock()
		return
	}
	a.markSleepingLocked(m)

	// Swap-group auto-restore: when an idle-sleep evicts a group member,
	// auto-wake the highest-priority sleeping peer in the same group so a
	// pinned/default model can come back up after a transient peer idles
	// out. Only fires on "idle" — manual / eviction / shutdown paths don't
	// trigger restoration.
	var restoreName string
	if reason == "idle" && m.SwapGroup != "" && a.autoRestoreWake != nil {
		var best *modelAdmissionState
		for peerName, peer := range a.models {
			if peerName == name {
				continue
			}
			if peer.SwapGroup != m.SwapGroup {
				continue
			}
			if peer.State != admissionSleeping {
				continue
			}
			if best == nil || priorityRank(peer.Priority) > priorityRank(best.Priority) {
				best = peer
			}
		}
		if best != nil {
			restoreName = best.Name
		}
	}
	fn := a.autoRestoreWake
	a.mu.Unlock()

	if restoreName != "" && fn != nil {
		// Dispatch via goroutine: the wake re-enters the scheduler (and
		// transitively this controller) and we must not hold a.mu across
		// that path. Goroutine also keeps NotifySleep's caller (the
		// sleep-completion path in sleep.go) snappy.
		go fn(restoreName, "swap-group-auto-restore")
	}
}

// NotifyStopped is called by the scheduler AFTER a docker compose stop
// completes for a model with EvictAction = stop. (Admission-initiated
// stops update budgets internally via markStoppedLocked; callers MUST
// NOT call NotifyStopped for those — see AdmissionEvictor contract.)
// Reverses both awakeByGPU and l1ResidualByGPU contributions — the
// container's full process tree exited and the CUDA context plus all
// captured allocations + non-pool buffers (NCCL, V1 logits) are gone.
//
// Safe to call for untracked models (no-op). Idempotent: double-calls
// on already-stopped models are no-ops.
//
// Unlike NotifySleep, this does NOT trigger swap-group auto-restore.
// A stopped peer requires docker compose up + health-wait + an explicit
// NotifyStarted to come back; that's a multi-minute operation, and the
// auto-restore semantic ("a transient peer idled, wake the pinned
// default") doesn't fit. Operators wanting to rotate the pinned default
// back into service after a stop event use the redeploy-member CLI verb.
func (a *AdmissionController) NotifyStopped(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	m, ok := a.models[name]
	if !ok {
		return
	}
	a.markStoppedLocked(m)
}

// NotifyStarted is called by the scheduler AFTER a docker compose up
// has produced a HEALTHY container that reports is_sleeping (the
// canonical post-cold-load resting state). Re-adds the L1 residual to
// the per-GPU budget. Callers must not flip the model directly to
// awake via this path — the next consumer request goes through the
// normal RequestWake flow which handles Sleeping → Awake.
//
// Safe to call for untracked models or for models that aren't currently
// Stopped (no-op).
func (a *AdmissionController) NotifyStarted(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	m, ok := a.models[name]
	if !ok {
		return
	}
	a.markStartedLocked(m)
}

// NotifyStartFailed is called by the scheduler AFTER a docker compose
// up attempt has failed (timed-out health-wait, container crashed,
// /is_sleeping never returned 200, etc.). Keeps the model in
// admissionStopped — the half-loaded vLLM has been stopped (best-effort)
// by the caller, so books should reflect zero VRAM use.
//
// CRITICAL: callers must NOT use NotifySleep for cold-load failures.
// NotifySleep → markSleepingLocked silently no-ops when state isn't
// admissionAwake, leaving a Stopped model wedged in the wrong state
// (awake VRAM not booked + future RequestWake assumes a clean Stopped
// → re-attempts cold-load infinitely). Instead this path explicitly
// ensures admissionStopped + zero contribution to either map.
//
// Idempotent: double-calls or calls on already-stopped models are
// no-ops. Safe to call for untracked models.
func (a *AdmissionController) NotifyStartFailed(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	m, ok := a.models[name]
	if !ok {
		return
	}
	// markStoppedLocked is idempotent and handles every prior state
	// (Awake, Sleeping, Stopped, Unknown) by collapsing to zero books.
	a.markStoppedLocked(m)
}

// IsStopped reports whether `name` is currently admissionStopped. Used
// by the wake path's TOCTOU re-check: a request acquires the global
// cold-load lock, then re-asks IsStopped because a concurrent request
// may have already cold-loaded the model while this one was waiting on
// the lock. Returns false for untracked models (their wake is unmediated).
func (a *AdmissionController) IsStopped(name string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	m, ok := a.models[name]
	if !ok {
		return false
	}
	return m.State == admissionStopped
}

// WithColdLoadLock runs fn while holding the global cold-load mutex.
// Any caller that needs to perform a docker compose up + health-wait +
// state transition for a Stopped admission member MUST go through this
// helper — the lock is the single-flight guarantee that prevents two
// concurrent cold-loads from contending for the same GPU set.
//
// fn runs lock-free (Go's sync.Mutex is not reentrant; calling
// WithColdLoadLock from inside fn would deadlock). The contract for fn
// is: "I own the cold-load right now, the admission state machine
// won't see another start-attempt for any swap-group member while I
// run." Inside fn, callers can read state via IsStopped, drive docker
// via os/exec, and finalize via NotifyStarted / NotifyStartFailed.
func (a *AdmissionController) WithColdLoadLock(fn func()) {
	a.coldLoadMu.Lock()
	defer a.coldLoadMu.Unlock()
	fn()
}

// markSleepingLocked is the idempotent state-update primitive used by
// both NotifySleep and RequestWake's eviction loop. Caller must hold
// a.mu.
//
// Snapshot discipline: reverses the awake booking via the model's
// BookedExpectedVRAMMB snapshot (NOT the live ExpectedVRAMMB which a
// hot-reload may have mutated since booking). Then snapshots the new
// L1 residual booking from the LIVE config so the future Sleeping →
// Awake / Sleeping → Stopped reversal also uses a known-correct
// snapshot.
func (a *AdmissionController) markSleepingLocked(m *modelAdmissionState) {
	if m.State != admissionAwake {
		// Already sleeping or never woke. Don't double-count.
		return
	}
	if m.BookedInPinnedMap {
		// Pinned-non-swap-group footprint lives in pinnedByGPU. Reverse
		// using snapshot. (Pinned + swap-group goes through awakeByGPU
		// and falls into the else branch.)
		for g, v := range m.BookedExpectedVRAMMB {
			a.pinnedByGPU[g] -= v
		}
		m.BookedInPinnedMap = false
	} else {
		for g, v := range m.BookedExpectedVRAMMB {
			a.awakeByGPU[g] -= v
		}
	}
	m.BookedExpectedVRAMMB = nil
	// Snapshot new residual booking from live config.
	if m.L1ResidualMB > 0 && len(m.GPUs) > 0 {
		for _, g := range m.GPUs {
			a.l1ResidualByGPU[g] += m.L1ResidualMB
		}
		m.BookedL1ResidualMB = m.L1ResidualMB
		m.BookedL1ResidualGPUs = append(m.BookedL1ResidualGPUs[:0], m.GPUs...)
	}
	m.State = admissionSleeping
	a.publishGauges()
}

// markStoppedLocked is the idempotent state-update primitive used when
// a peer transitions from any state to Stopped (container stopped, full
// reclaim of CUDA context + NCCL buffers). Caller must hold a.mu.
//
// Unlike markSleepingLocked which leaves the L1 residual on the books,
// this fully releases the model's footprint on every GPU it touches.
// Reversal uses the Booked* snapshots — NOT the live ExpectedVRAMMB /
// GPUs / L1ResidualMB — so a hot-reload mid-booking does not leak.
// Restoring requires a docker compose up + health-wait + an explicit
// NotifyStarted call (cold load ~5 min from page cache); /wake_up does
// NOT apply.
func (a *AdmissionController) markStoppedLocked(m *modelAdmissionState) {
	if m.State == admissionStopped {
		return
	}
	if m.State == admissionAwake {
		if m.BookedInPinnedMap {
			for g, v := range m.BookedExpectedVRAMMB {
				a.pinnedByGPU[g] -= v
			}
		} else {
			for g, v := range m.BookedExpectedVRAMMB {
				a.awakeByGPU[g] -= v
			}
		}
		m.BookedExpectedVRAMMB = nil
		m.BookedInPinnedMap = false
	}
	if m.State == admissionSleeping {
		for _, g := range m.BookedL1ResidualGPUs {
			a.l1ResidualByGPU[g] -= m.BookedL1ResidualMB
		}
		m.BookedL1ResidualMB = 0
		m.BookedL1ResidualGPUs = nil
	}
	// admissionUnknown: no books to release.
	m.State = admissionStopped
	a.publishGauges()
}

// markStartedLocked is the idempotent state-update primitive used when
// a Stopped peer's container has come up + reached healthy + reported
// is_sleeping. Caller must hold a.mu. Re-adds the L1 residual (vLLM
// containers are typically slept-L1 immediately after the cold-load
// settle, before any consumer demand reaches them).
//
// Snapshot discipline: snapshots the new residual booking from the
// LIVE config (m.GPUs / m.L1ResidualMB) so reversal in a later
// markSleepingLocked / markStoppedLocked / RequestWake uses a
// known-correct snapshot rather than potentially-mutated live fields.
func (a *AdmissionController) markStartedLocked(m *modelAdmissionState) {
	if m.State != admissionStopped {
		return
	}
	if m.L1ResidualMB > 0 && len(m.GPUs) > 0 {
		for _, g := range m.GPUs {
			a.l1ResidualByGPU[g] += m.L1ResidualMB
		}
		m.BookedL1ResidualMB = m.L1ResidualMB
		m.BookedL1ResidualGPUs = append(m.BookedL1ResidualGPUs[:0], m.GPUs...)
	}
	m.State = admissionSleeping
	a.publishGauges()
}

// NotifyWakeComplete is called by the scheduler AFTER vLLM /wake_up
// returned successfully (i.e. the wake the controller admitted has
// actually finished). Today this is mostly a recency-touch — RequestWake
// already updated budgets — but it gives callers a clear "successful
// wake landed" hook that we can hang post-wake bookkeeping off in
// future.
//
// Safe to call for untracked models (no-op).
func (a *AdmissionController) NotifyWakeComplete(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	m, ok := a.models[name]
	if !ok {
		return
	}
	m.LastRequestTime = time.Now()
}

// TouchActivity records a recent request to `name` for the purposes of
// LRU bookkeeping. Called from the proxy request path so the
// LastRequestTime reflects real traffic, not just wake events.
//
// Safe to call for untracked models (no-op).
func (a *AdmissionController) TouchActivity(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if m, ok := a.models[name]; ok {
		m.LastRequestTime = time.Now()
	}
}

// availableMB is the unbudgeted VRAM on `gpu`, given current awake +
// pinned + L1-residual contributions. Caller must hold a.mu.
func (a *AdmissionController) availableMB(gpu int) int {
	total, ok := a.totalsByGPU[gpu]
	if !ok {
		// GPU not in inventory — treat as unbudgeted. This happens when
		// the operator declared a model on a GPU that nvidia-smi didn't
		// report; the existing scheduler-mode validation catches that
		// earlier, so reaching here is a defensive fallback.
		return 0
	}
	return total - a.pinnedByGPU[gpu] - a.awakeByGPU[gpu] - a.l1ResidualByGPU[gpu]
}

// pickVictims returns the smallest set of evictable peers whose freed
// VRAM lets `m` fit on every GPU it touches. Returns an empty slice if
// no eviction is needed, or an error if no feasible set exists (the
// model is structurally too large for the pinned-adjusted budget on
// some GPU).
//
// `needed` is per-GPU because per-GPU expected footprints can differ
// (vLLM PP rank asymmetry). Likewise, a victim's `freed` contribution
// is per-GPU because the victim itself may have asymmetric per-GPU
// footprints.
//
// Algorithm (greedy):
//  1. For each target GPU, compute the SHORTFALL = needed[g] - available[g].
//  2. Build the candidate pool: awake non-pinned non-critical models
//     that share ≥1 GPU with `m`. PLUS: if `m` has a non-empty
//     SwapGroup, awake peers in the same swap_group are eligible
//     regardless of priority (override the pinned/critical exclusion).
//  3. Sort candidates into two tiers: NON-group candidates first (sorted
//     by priority bucket asc, LRU asc), then SAME-group candidates
//     (sorted the same way). Rationale: same-group eviction is by-design
//     (operator declared the swap), but it's still more disruptive than
//     evicting a normal peer; prefer non-group when both fit the shortfall.
//  4. Walk the sorted candidates, picking each one if it reduces
//     shortfall on any still-short GPU. Subtract its per-GPU freed VRAM
//     on every GPU it occupies. Stop when all target GPUs have
//     shortfall ≤ 0.
//  5. If shortfall remains after exhausting candidates → return error.
//
// Note: this is greedy, not optimal. For the typical home-cluster case
// (≤ 10 models, ≤ 8 GPUs) the difference vs. optimal knapsack is
// negligible and the greedy is far easier to reason about.
//
// Caller must hold a.mu.
func (a *AdmissionController) pickVictims(m *modelAdmissionState, needed map[int]int) ([]*modelAdmissionState, error) {
	shortfall := map[int]int{}
	for _, g := range m.GPUs {
		short := needed[g] - a.availableMB(g)
		if short > 0 {
			shortfall[g] = short
		}
	}
	if len(shortfall) == 0 {
		return nil, nil
	}

	targetSet := map[int]bool{}
	for _, g := range m.GPUs {
		targetSet[g] = true
	}

	// Same-group peers bypass priority rules; track membership so we can
	// (a) include them in the candidate pool and (b) tier them last.
	hasSwapGroup := m.SwapGroup != ""

	nonGroup := []*modelAdmissionState{}
	sameGroup := []*modelAdmissionState{}
	for name, peer := range a.models {
		if name == m.Name {
			continue
		}
		if peer.State != admissionAwake {
			continue
		}
		overlaps := false
		for _, g := range peer.GPUs {
			if targetSet[g] {
				overlaps = true
				break
			}
		}
		if !overlaps {
			continue
		}
		peerSameGroup := hasSwapGroup && peer.SwapGroup == m.SwapGroup
		if peerSameGroup {
			// In-group peer: eligible regardless of priority (even pinned/critical).
			sameGroup = append(sameGroup, peer)
			continue
		}
		// Out-of-group peer: standard rules — pinned/critical are sacred.
		if peer.Pinned || peer.Priority == config.PriorityCritical {
			continue
		}
		nonGroup = append(nonGroup, peer)
	}

	tierSort := func(s []*modelAdmissionState) {
		sort.Slice(s, func(i, j int) bool {
			pi, pj := priorityRank(s[i].Priority), priorityRank(s[j].Priority)
			if pi != pj {
				return pi < pj
			}
			return s[i].LastRequestTime.Before(s[j].LastRequestTime)
		})
	}
	tierSort(nonGroup)
	tierSort(sameGroup)

	// Walk tier 1 (non-group) first; only dip into tier 2 (same-group) if
	// shortfall remains. Concatenation keeps the existing pick loop simple.
	candidates := append(nonGroup, sameGroup...)

	picked := []*modelAdmissionState{}
	for _, c := range candidates {
		stillShort := false
		for _, g := range m.GPUs {
			if shortfall[g] > 0 {
				stillShort = true
				break
			}
		}
		if !stillShort {
			break
		}
		// Does this candidate reduce shortfall on any GPU we still need?
		helps := false
		for _, g := range c.GPUs {
			if shortfall[g] > 0 {
				helps = true
				break
			}
		}
		if !helps {
			continue
		}
		// Pick it — freed VRAM is per-GPU and depends on EvictAction:
		//   sleep: leaves L1 residual on the books (= ExpectedOn(g) - Residual)
		//   stop:  full reclaim (= ExpectedOn(g); container exit drops residual too)
		for _, g := range c.GPUs {
			if shortfall[g] <= 0 {
				continue
			}
			freed := c.ExpectedOn(g) - c.L1ResidualMB
			if c.EvictAction == config.EvictActionStop {
				freed = c.ExpectedOn(g)
			}
			shortfall[g] -= freed
			if shortfall[g] < 0 {
				shortfall[g] = 0
			}
		}
		picked = append(picked, c)
	}

	for g, s := range shortfall {
		if s > 0 {
			return nil, fmt.Errorf("%w: model %q on GPU %d: cannot free %dMB by evicting all eligible peers (best-effort + normal-priority awake models)",
				ErrAdmissionInfeasible, m.Name, g, s)
		}
	}
	return picked, nil
}

// publishGauges updates the per-GPU Prometheus gauges. Caller must
// hold a.mu.
func (a *AdmissionController) publishGauges() {
	for g, total := range a.totalsByGPU {
		label := strconv.Itoa(g)
		metrics.GPUBudgetAwakeMB.WithLabelValues(label).Set(float64(a.awakeByGPU[g] + a.pinnedByGPU[g]))
		metrics.GPUBudgetAvailableMB.WithLabelValues(label).Set(float64(total - a.pinnedByGPU[g] - a.awakeByGPU[g] - a.l1ResidualByGPU[g]))
	}
}

// SnapshotBudgets returns a read-only snapshot of per-GPU budget state
// for observability (e.g. the /status endpoint). The maps are safe to
// keep — they are copies.
type GPUBudgetSnapshot struct {
	GPUID        int
	TotalMB      int
	PinnedMB     int
	AwakeMB      int
	L1ResidualMB int
	AvailableMB  int
}

func (a *AdmissionController) SnapshotBudgets() []GPUBudgetSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]GPUBudgetSnapshot, 0, len(a.totalsByGPU))
	gpus := make([]int, 0, len(a.totalsByGPU))
	for g := range a.totalsByGPU {
		gpus = append(gpus, g)
	}
	sort.Ints(gpus)
	for _, g := range gpus {
		total := a.totalsByGPU[g]
		out = append(out, GPUBudgetSnapshot{
			GPUID:        g,
			TotalMB:      total,
			PinnedMB:     a.pinnedByGPU[g],
			AwakeMB:      a.awakeByGPU[g],
			L1ResidualMB: a.l1ResidualByGPU[g],
			AvailableMB:  total - a.pinnedByGPU[g] - a.awakeByGPU[g] - a.l1ResidualByGPU[g],
		})
	}
	return out
}

// priorityRank maps a priority string to a sort key. Lower = evicted
// first. Unknown priorities map to "normal" (defensive — config
// validation rejects unknown values, so this is unreachable in normal
// operation).
func priorityRank(p string) int {
	switch p {
	case config.PriorityBestEffort:
		return 0
	case config.PriorityNormal:
		return 1
	case config.PriorityCritical:
		return 2
	default:
		return 1
	}
}

func victimNames(victims []*modelAdmissionState) []string {
	out := make([]string, len(victims))
	for i, v := range victims {
		out[i] = v.Name
	}
	return out
}
