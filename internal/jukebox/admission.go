package jukebox

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

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
}

// AdmissionEvictor is the callback shape the controller uses to sleep a
// victim it has picked. The implementation MUST block until the
// victim's VRAM is fully released (vLLM's /sleep + settle window) so
// the controller's budget math reflects reality before the incoming
// wake proceeds.
//
// CONTRACT: implementations MUST NOT call AdmissionController methods
// (NotifySleep, NotifyWakeComplete, RequestWake, TouchActivity) during
// SleepForEviction. The controller is holding its admission mutex
// across this call and any callback would deadlock. The controller
// updates its own budget bookkeeping after SleepForEviction returns.
type AdmissionEvictor interface {
	SleepForEviction(ctx context.Context, victim, reason string) error
}

// admissionModelState is the controller's view of one model.
type modelAdmissionState struct {
	Name            string
	GPUs            []int
	ExpectedVRAMMB  int
	L1ResidualMB    int
	Priority        string
	Pinned          bool
	State           admissionModelState
	LastRequestTime time.Time
}

type admissionModelState int

const (
	admissionUnknown admissionModelState = iota
	admissionAwake
	admissionSleeping
)

// NewAdmissionController builds a controller from the model config + a
// GPU inventory snapshot (total MB per GPU). Models without
// AdmissionEnabled() are skipped — their wake is unmediated.
//
// totalsByGPU should reflect REAL GPU capacity (queried via nvidia-smi
// at scheduler startup), not configured caps. Tests supply a synthetic
// map. The controller never re-queries the inventory at runtime.
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
		st := &modelAdmissionState{
			Name:           name,
			GPUs:           append([]int(nil), m.GPUs...),
			ExpectedVRAMMB: m.ExpectedVRAMMBPerGPU,
			L1ResidualMB:   m.SleepL1ResidualMB,
			Priority:       m.EffectivePriority(),
			Pinned:         m.Pinned != nil && *m.Pinned,
		}
		// Pinned models are forced to critical and presumed awake from
		// the moment the controller boots — they will never sleep and
		// never wake "for the first time".
		if st.Pinned {
			st.Priority = config.PriorityCritical
			st.State = admissionAwake
			for _, g := range st.GPUs {
				a.pinnedByGPU[g] += st.ExpectedVRAMMB
			}
		}
		a.models[name] = st
	}
	return a
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
	defer a.mu.Unlock()

	m, ok := a.models[name]
	if !ok {
		// Untracked model — pass through unmediated.
		return nil, nil
	}
	if m.Pinned {
		// Pinned models are always awake by construction; nothing to do.
		return nil, nil
	}
	if m.State == admissionAwake {
		// Already accounted-for awake (e.g. a duplicate wake call). The
		// fan-out at the scheduler layer already de-dupes concurrent wake
		// requests for the same model, so reaching here means the model
		// was woken by some other path; just refresh the timestamp.
		m.LastRequestTime = time.Now()
		return nil, nil
	}

	// Compute fit per GPU. The model is currently in the L1Residual
	// pool (if it was sleeping) — bringing it awake means moving
	// (ExpectedVRAMMB - L1ResidualMB) from residual to awake on each GPU
	// it touches. So the "additional VRAM needed" on each GPU is the
	// delta. For an unknown-state model (first wake ever), the full
	// ExpectedVRAMMB is needed.
	needed := m.ExpectedVRAMMB
	if m.State == admissionSleeping {
		needed = m.ExpectedVRAMMB - m.L1ResidualMB
	}

	victims, err := a.pickVictims(m, needed)
	if err != nil {
		metrics.AdmissionRejectionsTotal.WithLabelValues(name, "infeasible").Inc()
		return nil, err
	}

	if len(victims) > 0 {
		slog.Info("admission_evicting",
			"model", name,
			"victims", victimNames(victims),
			"needed_mb_per_gpu", needed,
			"gpus", m.GPUs,
		)
	}

	for _, v := range victims {
		if err := a.evictor.SleepForEviction(ctx, v.Name, "admission"); err != nil {
			// Eviction failed. Don't proceed; any already-evicted
			// victims stay slept (we updated their state below as each
			// one succeeded, so the budget already reflects them).
			metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_failed").Inc()
			return nil, fmt.Errorf("admission for %q: failed to evict %q: %w", name, v.Name, err)
		}
		metrics.AdmissionEvictsTotal.WithLabelValues(name, v.Name).Inc()
		// Update budget directly — the evictor contract forbids
		// callbacks into NotifySleep, so admission owns this bookkeeping.
		a.markSleepingLocked(v)
	}

	// Re-check fit after evictions — a defensive sanity check against
	// arithmetic errors in pickVictims.
	for _, g := range m.GPUs {
		if avail := a.availableMB(g); avail < needed {
			metrics.AdmissionRejectionsTotal.WithLabelValues(name, "evict_insufficient").Inc()
			return victimNames(victims), fmt.Errorf("admission for %q: GPU %d still short %dMB after evicting %d peers (need %d, have %d)",
				name, g, needed-avail, len(victims), needed, avail)
		}
	}

	// Mark model awake and update budgets. If it was sleeping, also
	// drop its residual contribution.
	if m.State == admissionSleeping {
		for _, g := range m.GPUs {
			a.l1ResidualByGPU[g] -= m.L1ResidualMB
		}
	}
	for _, g := range m.GPUs {
		a.awakeByGPU[g] += m.ExpectedVRAMMB
	}
	m.State = admissionAwake
	m.LastRequestTime = time.Now()

	a.publishGauges()
	metrics.AdmissionWaitSeconds.WithLabelValues(name).Observe(time.Since(start).Seconds())
	return victimNames(victims), nil
}

// NotifySleep is called by the scheduler AFTER a successful sleep —
// either an idle-timeout sleep or a manual sleep. (Admission-initiated
// evictions update budgets internally via markSleepingLocked; callers
// MUST NOT call NotifySleep for those — see AdmissionEvictor contract.)
// Updates budgets to reflect the freed VRAM (less the L1 residual).
//
// Safe to call for untracked models (no-op). Idempotent: double-calls
// or calls on already-sleeping models are no-ops.
func (a *AdmissionController) NotifySleep(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	m, ok := a.models[name]
	if !ok || m.Pinned {
		return
	}
	a.markSleepingLocked(m)
}

// markSleepingLocked is the idempotent state-update primitive used by
// both NotifySleep and RequestWake's eviction loop. Caller must hold
// a.mu.
func (a *AdmissionController) markSleepingLocked(m *modelAdmissionState) {
	if m.State != admissionAwake {
		// Already sleeping or never woke. Don't double-count.
		return
	}
	for _, g := range m.GPUs {
		a.awakeByGPU[g] -= m.ExpectedVRAMMB
		a.l1ResidualByGPU[g] += m.L1ResidualMB
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
// Algorithm (greedy):
//  1. For each target GPU, compute the SHORTFALL = needed - available.
//  2. Build the candidate pool: awake non-pinned non-critical models
//     that share ≥1 GPU with `m`.
//  3. Sort candidates by (priority bucket asc, LRU asc) — best-effort
//     models first, then normal-priority by LastRequestTime ascending
//     (oldest first).
//  4. Walk the sorted candidates, picking each one if it reduces
//     shortfall on any still-short GPU. Subtract its freed VRAM on
//     every GPU it occupies. Stop when all target GPUs have
//     shortfall ≤ 0.
//  5. If shortfall remains after exhausting candidates → return error.
//
// Note: this is greedy, not optimal. For the typical home-cluster case
// (≤ 10 models, ≤ 8 GPUs) the difference vs. optimal knapsack is
// negligible and the greedy is far easier to reason about.
//
// Caller must hold a.mu.
func (a *AdmissionController) pickVictims(m *modelAdmissionState, needed int) ([]*modelAdmissionState, error) {
	shortfall := map[int]int{}
	for _, g := range m.GPUs {
		short := needed - a.availableMB(g)
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

	candidates := []*modelAdmissionState{}
	for name, peer := range a.models {
		if name == m.Name {
			continue
		}
		if peer.State != admissionAwake {
			continue
		}
		if peer.Pinned || peer.Priority == config.PriorityCritical {
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
		candidates = append(candidates, peer)
	}

	sort.Slice(candidates, func(i, j int) bool {
		pi, pj := priorityRank(candidates[i].Priority), priorityRank(candidates[j].Priority)
		if pi != pj {
			return pi < pj
		}
		return candidates[i].LastRequestTime.Before(candidates[j].LastRequestTime)
	})

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
		// Pick it — its freed VRAM is (ExpectedVRAMMB - L1ResidualMB) on
		// each of its GPUs.
		freed := c.ExpectedVRAMMB - c.L1ResidualMB
		for _, g := range c.GPUs {
			if shortfall[g] > 0 {
				shortfall[g] -= freed
				if shortfall[g] < 0 {
					shortfall[g] = 0
				}
			}
		}
		picked = append(picked, c)
	}

	for g, s := range shortfall {
		if s > 0 {
			return nil, fmt.Errorf("model %q on GPU %d: cannot free %dMB by evicting all eligible peers (best-effort + normal-priority awake models)",
				m.Name, g, s)
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
