// Predictive cold-load — SKELETON ONLY.
//
// This file ships the histogram + predictor goroutine without actually
// firing KickColdLoad. The intent (downstream carry, never
// upstream) is to pre-warm a Stopped model whose `EvictAction = stop`
// containers cost ~5 minutes to wake from page cache, by tracking
// per-model `(hour-of-day, day-of-week)` traffic and pre-firing when
// recent observations say a request is likely in the next window.
//
// The actual KickColdLoad call site is intentionally a TODO + slog line
// in this commit. Step 1 is to ship the metrics so we can observe the
// shape of the predictions in prod and tune `p_threshold` /
// `window_minutes` against real traffic. Step 2 (separate PR) flips the
// gate.
//
// Threading
// ---------
//   - Predictor.Observe() is safe under arbitrary concurrent callers
//     (one mutex). Designed to be called from the request hot path,
//     so it MUST be cheap: a single map lookup + a few integer increments
//     under a per-model lock split.
//   - Predictor.Run(ctx) owns the predictor goroutine. Ticks every
//     `tickInterval`. Read-locks the histogram, computes per-model
//     probability, emits metrics, evaluates the budget gate. Never
//     blocks on disk; persistence is opportunistic (best-effort write
//     once per hour).
//
// Persistence
// -----------
// Optional JSON sidecar at `state_path` (configured via PredictorOptions).
// On startup, Load() reads it; on a tick boundary roughly every hour
// the predictor writes it back atomically (`<path>.tmp` + rename). The
// state file format is intentionally simple — a fresh start (no file)
// is recoverable: the predictor just begins observing fresh and crosses
// the threshold once it has enough evidence.
//
// Decay
// -----
// We don't carry per-day timestamps for every observation; instead, every
// histogram cell ages via exponential decay (default 30-day half-life)
// applied lazily at observation time using the `decayedAt` watermark.
// This is approximate (the decay is computed at the cell's last touch,
// not continuously) but the error vs. true exponential is bounded by
// `tickInterval` and irrelevant for the threshold comparison we actually
// make.

package jukebox

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

const (
	// histBuckets — 7 days * 24 hours = 168 cells per model.
	histBucketsDow  = 7
	histBucketsHour = 24
	histBuckets     = histBucketsDow * histBucketsHour

	// defaultPThreshold — minimum P(request in next window) to fire.
	defaultPThreshold = 0.4

	// defaultMaxPerDay — at most this many predictive warms per model
	// per local-day. 24 ≈ "once an hour"; should never be exceeded with
	// the cooldown set to 30min.
	defaultMaxPerDay = 24

	// defaultCooldownMinutes — after a predictive warm fires, mute the
	// gate for this long even if the probability stays high.
	defaultCooldownMinutes = 30

	// defaultWindowMinutes — forecast horizon. We're asking "will a
	// request arrive in the next N minutes?". Tuned so warm-completion
	// (~5min cold-load) lands inside the window.
	defaultWindowMinutes = 10

	// defaultTickInterval — predictor goroutine wakeup cadence. One
	// minute is the right granularity for an hourly histogram (24×7
	// cells). Going faster wastes CPU on identical buckets.
	defaultTickInterval = 1 * time.Minute

	// defaultDecayHalfLife — observations age out on this half-life.
	// 30 days = a stable monthly pattern dominates; a one-off Sunday
	// 3 a.m. spike from last week barely registers a month later.
	defaultDecayHalfLife = 30 * 24 * time.Hour

	// persistInterval — best-effort histogram-state JSON write cadence.
	persistInterval = 1 * time.Hour

	// matchedRequestWindow — how long after a predictive fire we count
	// the next real Observe() as a "hit" for that fire. Same default
	// as the forecast window — symmetric.
	matchedRequestWindow = time.Duration(defaultWindowMinutes) * time.Minute
)

// histCell holds aggregate counts for one (dow, hour) cell.
//
// `Observed` is the number of (dow, hour) ticks the predictor has
// witnessed for this cell — i.e. the denominator. `Hits` is the count
// of those ticks where at least one request was observed in the
// associated `windowMinutes` slice. Probability ≈ Hits / Observed.
//
// `decayedAt` is the wall-clock at which Observed and Hits were last
// rescaled. On every read or update, we compare `now() - decayedAt` to
// `decayHalfLife` and shrink both counters by `0.5^(elapsed/halflife)`.
type histCell struct {
	Hits      float64   `json:"hits"`
	Observed  float64   `json:"observed"`
	DecayedAt time.Time `json:"decayed_at"`
}

// modelHistogram — per-model 168-cell ring with the decay watermark.
type modelHistogram struct {
	Cells [histBuckets]histCell `json:"cells"`
}

// PredictorOptions tunes the predictor at construction time. Most of
// these are derived from the per-model PredictiveColdLoadConfig values
// when a tick fires; the options here are process-wide overrides used
// almost exclusively by tests (mockable now() + tick cadence) and by
// the persistence path (state_path + decay half-life).
type PredictorOptions struct {
	Now              func() time.Time
	TickInterval     time.Duration
	DecayHalfLife    time.Duration
	StatePath        string // empty = no persistence
	MatchedRequestWindow time.Duration
	OnFire           func(model string) // nil-safe; SKELETON: leaves KickColdLoad as TODO
}

// Predictor is the histogram-driven cold-load pre-warm advisor.
//
// Lifecycle: NewPredictor() → Load() (optional) → go p.Run(ctx) → Stop()
// indirectly via ctx cancel. The Predictor itself does NOT call
// KickColdLoad in this commit. When a per-model gate would fire, the
// `OnFire` callback (if set) is invoked; production wiring leaves it
// nil and the gate just emits metrics + slog.
type Predictor struct {
	cfg *config.Config

	now func() time.Time

	tickInterval         time.Duration
	decayHalfLife        time.Duration
	matchedRequestWindow time.Duration
	statePath            string
	onFire               func(string)

	mu sync.Mutex
	// per-model histograms keyed by canonical model name.
	hist map[string]*modelHistogram

	// per-model post-fire bookkeeping.
	lastFireAt   map[string]time.Time
	firesToday   map[string]int      // resets at local midnight
	firesDay     map[string]string   // YYYY-MM-DD label for firesToday's day
	pendingFire  map[string]time.Time // model → fire timestamp pending hit/miss

	// IsStopped accessor. Pluggable so the predictor can run without a
	// Scheduler (tests). In production wired to (*Scheduler).IsModelColdLoading.
	isStopped func(string) bool
}

// NewPredictor builds a Predictor. `cfg` is the construction-time
// config (used as fallback by liveCfg / liveModelCfg — same pattern as
// the rest of the package). `isStopped` is the gate that tells the
// predictor whether the model is currently in the admissionStopped
// state and therefore eligible for a pre-warm; nil isStopped means
// "always eligible" (test path).
func NewPredictor(cfg *config.Config, isStopped func(string) bool, opts PredictorOptions) *Predictor {
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.TickInterval <= 0 {
		opts.TickInterval = defaultTickInterval
	}
	if opts.DecayHalfLife <= 0 {
		opts.DecayHalfLife = defaultDecayHalfLife
	}
	if opts.MatchedRequestWindow <= 0 {
		opts.MatchedRequestWindow = matchedRequestWindow
	}
	if isStopped == nil {
		isStopped = func(string) bool { return true }
	}
	return &Predictor{
		cfg:                  cfg,
		now:                  opts.Now,
		tickInterval:         opts.TickInterval,
		decayHalfLife:        opts.DecayHalfLife,
		matchedRequestWindow: opts.MatchedRequestWindow,
		statePath:            opts.StatePath,
		onFire:               opts.OnFire,
		hist:                 map[string]*modelHistogram{},
		lastFireAt:           map[string]time.Time{},
		firesToday:           map[string]int{},
		firesDay:             map[string]string{},
		pendingFire:          map[string]time.Time{},
		isStopped:            isStopped,
	}
}

// dowHourBucket returns the bucket index for the (weekday, hour) of t
// in t's local zone. 0..167 with day-major ordering (Sunday=0..23,
// Monday=24..47, ...). Local-time is the right semantic — operator
// schedules ("8am-6pm Mon-Fri") are local, not UTC.
func dowHourBucket(t time.Time) int {
	// time.Weekday: Sunday=0..Saturday=6 — matches our histBucketsDow
	// ordering directly.
	dow := int(t.Weekday())
	hour := t.Hour()
	return dow*histBucketsHour + hour
}

// applyDecay rescales (Hits, Observed) for the elapsed time since
// the cell's DecayedAt. Caller must hold p.mu.
func (p *Predictor) applyDecay(c *histCell, now time.Time) {
	if c.DecayedAt.IsZero() {
		c.DecayedAt = now
		return
	}
	elapsed := now.Sub(c.DecayedAt)
	if elapsed <= 0 {
		return
	}
	if p.decayHalfLife <= 0 {
		c.DecayedAt = now
		return
	}
	factor := math.Pow(0.5, float64(elapsed)/float64(p.decayHalfLife))
	c.Hits *= factor
	c.Observed *= factor
	c.DecayedAt = now
}

// Observe records that a real request landed for `name` at time `t`.
// Cheap: one map lookup + one cell update under p.mu.
//
// Called from the request hot path. SKELETON: production hook is
// deferred to a follow-up PR (see TODO in admission.TouchActivity);
// this method is exercised by tests today.
func (p *Predictor) Observe(name string, t time.Time) {
	if p == nil || name == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	h, ok := p.hist[name]
	if !ok {
		h = &modelHistogram{}
		p.hist[name] = h
	}
	bucket := dowHourBucket(t)
	c := &h.Cells[bucket]
	p.applyDecay(c, t)
	c.Hits += 1.0

	// If this Observe lands inside a pending-fire window, count it as
	// a hit for the metrics histogram + flip the pending-fire to "hit".
	if firedAt, hasFire := p.pendingFire[name]; hasFire {
		dt := t.Sub(firedAt)
		if dt >= 0 && dt <= p.matchedRequestWindow {
			metrics.PredictiveWarmsTotal.WithLabelValues(name, "hit").Inc()
			metrics.PredictiveWarmToFirstRequestSeconds.WithLabelValues(name).Observe(dt.Seconds())
			delete(p.pendingFire, name)
		}
	}
}

// observeWindow records that the predictor witnessed a (dow, hour) tick
// for `name` — i.e., increments the denominator. Caller must hold p.mu.
func (p *Predictor) observeWindowLocked(name string, t time.Time) {
	h, ok := p.hist[name]
	if !ok {
		h = &modelHistogram{}
		p.hist[name] = h
	}
	bucket := dowHourBucket(t)
	c := &h.Cells[bucket]
	p.applyDecay(c, t)
	c.Observed += 1.0
}

// probabilityLocked returns the predicted P(request) for `name` at time
// `t`. -1 sentinel for "no observations" (predictor has no opinion;
// the gate should NOT fire). Caller must hold p.mu.
func (p *Predictor) probabilityLocked(name string, t time.Time) float64 {
	h, ok := p.hist[name]
	if !ok {
		return -1
	}
	bucket := dowHourBucket(t)
	c := &h.Cells[bucket]
	p.applyDecay(c, t)
	if c.Observed < 1.0 {
		return -1
	}
	p_ := c.Hits / c.Observed
	if p_ < 0 {
		return 0
	}
	if p_ > 1 {
		return 1
	}
	return p_
}

// Probability is the test/debug accessor for the most-recent
// P(request | bucket) without firing the gate. Safe to call concurrently.
func (p *Predictor) Probability(name string, t time.Time) float64 {
	if p == nil {
		return -1
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.probabilityLocked(name, t)
}

// effectiveOpts pulls per-model PredictiveColdLoadConfig from the
// LIVE config (so hot-reloads of `enabled` and `p_threshold` take
// effect on the next tick without restart). Returns the resolved
// (PThreshold, MaxPerDay, CooldownMinutes, WindowMinutes, enabled).
func (p *Predictor) effectiveOpts(name string) (pcfg config.PredictiveColdLoadConfig, ok bool) {
	mc, found := liveModelCfg(p.cfg, name)
	if !found {
		return config.PredictiveColdLoadConfig{}, false
	}
	pcfg = mc.PredictiveColdLoad
	if pcfg.PThreshold <= 0 {
		pcfg.PThreshold = defaultPThreshold
	}
	if pcfg.MaxPerDay <= 0 {
		pcfg.MaxPerDay = defaultMaxPerDay
	}
	if pcfg.CooldownMinutes <= 0 {
		pcfg.CooldownMinutes = defaultCooldownMinutes
	}
	if pcfg.WindowMinutes <= 0 {
		pcfg.WindowMinutes = defaultWindowMinutes
	}
	return pcfg, true
}

// dayLabel returns the local-date label used to bucket per-day fire
// counts. Local-time matters — "max 24 fires/day" should bound a
// human-perceived day.
func dayLabel(t time.Time) string {
	return t.Format("2006-01-02")
}

// resetDailyBudgetIfNeededLocked rolls the per-model fires-today
// counter to zero when we've crossed local midnight since the last
// fire. Caller must hold p.mu.
func (p *Predictor) resetDailyBudgetIfNeededLocked(name string, now time.Time) {
	curDay := dayLabel(now)
	if p.firesDay[name] != curDay {
		p.firesDay[name] = curDay
		p.firesToday[name] = 0
	}
}

// evalFireLocked is the gate. Called once per model per tick. Returns
// the outcome label (matching PredictiveWarmsTotal labels). Caller
// must hold p.mu.
func (p *Predictor) evalFireLocked(name string, now time.Time) string {
	pcfg, ok := p.effectiveOpts(name)
	if !ok || !pcfg.Enabled {
		return "" // not tracked — no metrics noise
	}

	prob := p.probabilityLocked(name, now)
	if prob < 0 {
		// No observations yet for this bucket; emit gauge as -1 so
		// dashboards can surface "we have no opinion".
		metrics.PredictedProbability.WithLabelValues(name).Set(-1)
		return ""
	}
	metrics.PredictedProbability.WithLabelValues(name).Set(prob)

	if prob < pcfg.PThreshold {
		return ""
	}

	// Threshold crossed. Now apply the gates.

	if !p.isStopped(name) {
		return "skipped_not_stopped"
	}

	// Cooldown gate.
	if last, hasLast := p.lastFireAt[name]; hasLast {
		cooldown := time.Duration(pcfg.CooldownMinutes) * time.Minute
		if now.Sub(last) < cooldown {
			return "skipped_cooldown"
		}
	}

	// Daily budget gate.
	p.resetDailyBudgetIfNeededLocked(name, now)
	if p.firesToday[name] >= pcfg.MaxPerDay {
		return "skipped_budget"
	}

	// Fire! (skeleton: this does not actually call KickColdLoad — see
	// onFire callback. Production wiring leaves onFire nil; the call
	// site flip lives in the follow-up PR after we observe predictions
	// in prod.)
	p.lastFireAt[name] = now
	p.firesToday[name]++
	p.pendingFire[name] = now

	if p.onFire != nil {
		p.onFire(name)
	}

	slog.Info("predictive_cold_load_would_fire",
		"model", name,
		"probability", prob,
		"threshold", pcfg.PThreshold,
		"window_minutes", pcfg.WindowMinutes,
		"fires_today", p.firesToday[name],
		"max_per_day", pcfg.MaxPerDay,
		"todo", "kick_cold_load_call_deferred_to_followup_pr",
	)

	return "fired"
}

// reapStaleFiresLocked converts pending fires whose window has elapsed
// without a matching Observe into "miss" outcomes. Caller must hold p.mu.
func (p *Predictor) reapStaleFiresLocked(now time.Time) {
	for name, firedAt := range p.pendingFire {
		if now.Sub(firedAt) > p.matchedRequestWindow {
			metrics.PredictiveWarmsTotal.WithLabelValues(name, "miss").Inc()
			delete(p.pendingFire, name)
		}
	}
}

// tick runs one predictor pass: increment denominators, then evaluate
// fire gate for every predictively-tracked model.
func (p *Predictor) tick(now time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()

	cfg := liveCfg(p.cfg)
	if cfg == nil {
		return
	}

	for name := range cfg.Models {
		// Per-tick: bump the denominator for every tracked model. The
		// denominator counts predictor ticks where we asked the
		// question, NOT requests; this normalizes hits/observed to a
		// per-tick rate. Multiply by `windowMinutes / tickIntervalMin`
		// in the consumer if you want a per-window rate; the gate
		// compares P directly against PThreshold so we don't need that
		// scaling here.
		mc, ok := cfg.Models[name]
		if !ok || !mc.PredictiveColdLoad.Enabled {
			continue
		}
		p.observeWindowLocked(name, now)
		outcome := p.evalFireLocked(name, now)
		if outcome != "" {
			metrics.PredictiveWarmsTotal.WithLabelValues(name, outcome).Inc()
		}
	}

	p.reapStaleFiresLocked(now)
}

// Run drives the predictor goroutine. Returns when ctx is cancelled.
// Best-effort persistence on a separate cadence.
func (p *Predictor) Run(ctx context.Context) {
	if p == nil {
		return
	}
	tick := time.NewTicker(p.tickInterval)
	defer tick.Stop()

	persistTick := time.NewTicker(persistInterval)
	defer persistTick.Stop()

	slog.Info("predictor_started",
		"tick_interval_seconds", p.tickInterval.Seconds(),
		"decay_half_life_hours", p.decayHalfLife.Hours(),
		"state_path", p.statePath,
	)

	for {
		select {
		case <-ctx.Done():
			// Best-effort final flush.
			if err := p.persist(); err != nil {
				slog.Warn("predictor_state_final_flush_failed", "err", err)
			}
			slog.Info("predictor_stopped")
			return
		case t := <-tick.C:
			p.tick(t)
		case <-persistTick.C:
			if err := p.persist(); err != nil {
				slog.Warn("predictor_state_persist_failed", "err", err)
			}
		}
	}
}

// persistedState — JSON shape for the on-disk sidecar.
type persistedState struct {
	Version int                        `json:"version"`
	SavedAt time.Time                  `json:"saved_at"`
	Hist    map[string]*modelHistogram `json:"hist"`
}

const persistedStateVersion = 1

// persist writes the histogram state to p.statePath atomically.
// No-op when statePath is empty.
func (p *Predictor) persist() error {
	if p == nil || p.statePath == "" {
		return nil
	}
	p.mu.Lock()
	state := persistedState{
		Version: persistedStateVersion,
		SavedAt: p.now(),
		Hist:    map[string]*modelHistogram{},
	}
	for k, v := range p.hist {
		// Shallow copy is fine — modelHistogram is a value-type array
		// so the json encoder gets a stable snapshot.
		copyHist := *v
		state.Hist[k] = &copyHist
	}
	p.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(p.statePath), 0o755); err != nil {
		return fmt.Errorf("predictor: mkdir state dir: %w", err)
	}
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("predictor: marshal state: %w", err)
	}
	tmp := p.statePath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("predictor: write tmp: %w", err)
	}
	if err := os.Rename(tmp, p.statePath); err != nil {
		return fmt.Errorf("predictor: atomic rename: %w", err)
	}
	return nil
}

// Load reads a previously-persisted state file, if any. Missing file
// is NOT an error — fresh start. Other read/parse errors are returned
// so the operator can surface them in startup logs (a corrupt state
// file should not silently zero out the histogram).
func (p *Predictor) Load() error {
	if p == nil || p.statePath == "" {
		return nil
	}
	data, err := os.ReadFile(p.statePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("predictor: read state: %w", err)
	}
	var state persistedState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("predictor: parse state: %w", err)
	}
	if state.Version != persistedStateVersion {
		// Forward-compat skip; a future version bump can supply a
		// migration. For now we just warn and start fresh.
		slog.Warn("predictor_state_version_mismatch",
			"got", state.Version,
			"want", persistedStateVersion,
			"action", "starting_fresh",
		)
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if state.Hist != nil {
		p.hist = state.Hist
	}
	slog.Info("predictor_state_loaded",
		"models", len(p.hist),
		"saved_at", state.SavedAt,
	)
	return nil
}
