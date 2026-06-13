package jukebox

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// makePredCfg builds a minimal *config.Config carrying one or more
// predictively-enabled models. Validation is intentionally bypassed —
// the predictor only reads the per-model PredictiveColdLoad block and
// never re-validates the rest of the model record, so we don't have to
// satisfy the full ModelConfig invariants here.
func makePredCfg(models map[string]config.PredictiveColdLoadConfig) *config.Config {
	c := &config.Config{
		Models: map[string]config.ModelConfig{},
	}
	for name, pcfg := range models {
		c.Models[name] = config.ModelConfig{
			Path:               "stub",
			GPUs:               []int{0},
			PredictiveColdLoad: pcfg,
		}
	}
	return c
}

// makeFakeClock returns a closure that returns a controllable time. Tests
// use this to inject deterministic (dow, hour) buckets without waiting
// for real wall-clock to advance.
func makeFakeClock(t time.Time) (func() time.Time, *time.Time) {
	cur := t
	return func() time.Time { return cur }, &cur
}

// TestHistogramObserveAndProbability — pure histogram math: observing
// hits + windows in a bucket should produce a probability that converges
// to hits/observed.
func TestHistogramObserveAndProbability(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 30, 0, 0, time.Local) // Friday 14:30 local
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true},
	})

	p := NewPredictor(cfg, nil, PredictorOptions{
		Now:           clock,
		TickInterval:  1 * time.Minute,
		DecayHalfLife: 30 * 24 * time.Hour,
	})

	// Walk 10 ticks of the same (dow, hour) bucket. 4 of them have a
	// real Observe (hit), 6 don't.
	for i := 0; i < 10; i++ {
		// Advance only minutes inside the same hour so dowHourBucket
		// stays constant.
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i%3 == 0 {
			p.Observe("alpha", *cur) // 4 hits at i=0,3,6,9
		}
	}

	prob := p.Probability("alpha", now)
	want := 0.4
	if prob < want-0.05 || prob > want+0.05 {
		t.Fatalf("probability = %v, want ~%v (hits/observed = 4/10)", prob, want)
	}
}

// TestProbabilityNoObservations — fresh model returns -1 sentinel.
func TestProbabilityNoObservations(t *testing.T) {
	clock, _ := makeFakeClock(time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local))
	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true},
	})
	p := NewPredictor(cfg, nil, PredictorOptions{Now: clock})
	if got := p.Probability("alpha", clock()); got != -1 {
		t.Fatalf("Probability with no observations = %v, want -1", got)
	}
	// Unknown model also -1.
	if got := p.Probability("never-seen", clock()); got != -1 {
		t.Fatalf("Probability for unknown model = %v, want -1", got)
	}
}

// TestEvalFireGate_BelowThreshold — probability below threshold means
// the gate must NOT fire.
func TestEvalFireGate_BelowThreshold(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true, PThreshold: 0.9},
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:    clock,
		OnFire: func(string) { fires++ },
	})

	// 10 ticks, only 1 hit → P=0.1 < 0.9.
	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i == 0 {
			p.Observe("alpha", *cur)
		}
	}
	if fires != 0 {
		t.Fatalf("OnFire called %d times below threshold; want 0", fires)
	}
}

// TestEvalFireGate_AboveThreshold — high probability + Stopped + no
// cooldown → onFire MUST be invoked.
func TestEvalFireGate_AboveThreshold(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true, PThreshold: 0.4, MaxPerDay: 10, CooldownMinutes: 30},
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:    clock,
		OnFire: func(name string) { fires++ },
	})

	// Pre-populate the histogram with strong evidence — 8 hits in 10
	// observations for the bucket. Achieved by mixing tick + Observe.
	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 8 {
			p.Observe("alpha", *cur)
		}
	}

	// One more tick — gate should fire because P ≈ 0.8 ≥ 0.4 and no
	// prior fire has run.
	*cur = now.Add(11 * time.Minute)
	p.tick(*cur)
	if fires == 0 {
		t.Fatalf("OnFire never called above threshold; got fires=0")
	}
}

// TestEvalFireGate_Cooldown — once a fire happens, a subsequent fire
// within the cooldown window must be suppressed.
func TestEvalFireGate_Cooldown(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true, PThreshold: 0.4, MaxPerDay: 10, CooldownMinutes: 30},
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:    clock,
		OnFire: func(string) { fires++ },
	})

	// Strong-evidence bucket.
	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 8 {
			p.Observe("alpha", *cur)
		}
	}
	*cur = now.Add(11 * time.Minute)
	p.tick(*cur) // first fire
	firstFires := fires

	// Within cooldown — must NOT fire again.
	*cur = now.Add(20 * time.Minute)
	p.tick(*cur)
	if fires != firstFires {
		t.Fatalf("Fire within cooldown: fires went %d → %d (expected suppression)", firstFires, fires)
	}

	// Past cooldown — fire allowed again.
	*cur = now.Add(50 * time.Minute)
	p.tick(*cur)
	if fires <= firstFires {
		t.Fatalf("Past cooldown should re-fire; fires=%d firstFires=%d", fires, firstFires)
	}
}

// TestEvalFireGate_NotStoppedSuppresses — model that's already running
// must not be re-pre-warmed even at high probability.
func TestEvalFireGate_NotStoppedSuppresses(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true, PThreshold: 0.4},
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return false /* not stopped */ }, PredictorOptions{
		Now:    clock,
		OnFire: func(string) { fires++ },
	})

	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 9 {
			p.Observe("alpha", *cur)
		}
	}
	*cur = now.Add(11 * time.Minute)
	p.tick(*cur)
	if fires != 0 {
		t.Fatalf("Fired %d times when model is not Stopped; want 0", fires)
	}
}

// TestEvalFireGate_DisabledByConfig — predictive_cold_load.enabled=false
// must short-circuit the entire gate.
func TestEvalFireGate_DisabledByConfig(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: false}, // explicitly off
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:    clock,
		OnFire: func(string) { fires++ },
	})

	for i := 0; i < 20; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 18 {
			p.Observe("alpha", *cur)
		}
	}
	if fires != 0 {
		t.Fatalf("Fires=%d for disabled model; want 0", fires)
	}
}

// TestEvalFireGate_DailyBudget — once the daily budget is reached, no
// more fires today regardless of probability.
func TestEvalFireGate_DailyBudget(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {
			Enabled:         true,
			PThreshold:      0.4,
			MaxPerDay:       2,
			CooldownMinutes: 1, // tiny cooldown so we can stack fires
		},
	})

	fires := 0
	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:    clock,
		OnFire: func(string) { fires++ },
	})

	// Build strong evidence.
	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 8 {
			p.Observe("alpha", *cur)
		}
	}

	// Step past evidence-build, advancing 5 min between attempts so
	// cooldown (1 min) doesn't matter, but stay inside the same local
	// day so per-day budget DOES.
	for step := 1; step <= 6; step++ {
		*cur = now.Add(time.Duration(15*step) * time.Minute)
		p.tick(*cur)
	}

	if fires > 2 {
		t.Fatalf("Fires=%d exceeds MaxPerDay=2", fires)
	}
	if fires < 1 {
		t.Fatalf("Fires=%d, expected at least 1 (probability was high enough)", fires)
	}
}

// TestExponentialDecay — old observations should decay so a single
// stale spike doesn't dominate forever.
func TestExponentialDecay(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(t0)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true},
	})

	// Half-life = 1 week so we can advance by exactly 1 week (back to
	// the same dow,hour cell) and observe ~1 half-life of decay
	// without dropping below the "Observed >= 1.0" floor that
	// Probability uses to return a real value.
	p := NewPredictor(cfg, nil, PredictorOptions{
		Now:           clock,
		DecayHalfLife: 7 * 24 * time.Hour,
	})

	// 10 ticks/observations all hits at t0.
	for i := 0; i < 10; i++ {
		*cur = t0.Add(time.Duration(i) * time.Second)
		p.tick(*cur)
		p.Observe("alpha", *cur)
	}
	probInitial := p.Probability("alpha", *cur)
	if probInitial < 0.9 {
		t.Fatalf("initial probability %v should be ~1.0", probInitial)
	}

	// Snapshot raw hit count before the time-jump.
	p.mu.Lock()
	bucket := dowHourBucket(t0)
	hitsBefore := p.hist["alpha"].Cells[bucket].Hits
	p.mu.Unlock()

	// Jump 1 week forward — same (dow, hour) bucket, exactly one
	// half-life of elapsed time.
	*cur = t0.Add(7 * 24 * time.Hour)
	probDecayed := p.Probability("alpha", *cur)
	// Ratio is invariant under proportional decay (both Hits and
	// Observed scale by the same factor).
	if probDecayed < 0.9 {
		t.Fatalf("ratio should be invariant under decay; got %v", probDecayed)
	}

	// Raw counts MUST have shrunk by ~50%.
	p.mu.Lock()
	hitsAfter := p.hist["alpha"].Cells[bucket].Hits
	p.mu.Unlock()
	if hitsAfter >= hitsBefore {
		t.Fatalf("hits did not decay: before=%v after=%v", hitsBefore, hitsAfter)
	}
	if hitsAfter > 0.6*hitsBefore {
		t.Fatalf("decay too small after one half-life: before=%v after=%v", hitsBefore, hitsAfter)
	}
}

// TestPersistRoundTrip — Save → Load round-trips the histogram state.
func TestPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "predictor.json")

	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, _ := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"alpha": {Enabled: true},
	})

	p1 := NewPredictor(cfg, nil, PredictorOptions{Now: clock, StatePath: statePath})
	for i := 0; i < 5; i++ {
		p1.Observe("alpha", now.Add(time.Duration(i)*time.Minute))
	}
	if err := p1.persist(); err != nil {
		t.Fatalf("persist: %v", err)
	}
	if _, err := os.Stat(statePath); err != nil {
		t.Fatalf("state file missing after persist: %v", err)
	}

	p2 := NewPredictor(cfg, nil, PredictorOptions{Now: clock, StatePath: statePath})
	if err := p2.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	p2.mu.Lock()
	defer p2.mu.Unlock()
	got := p2.hist["alpha"]
	if got == nil {
		t.Fatalf("histogram for 'alpha' did not survive round-trip")
	}
	bucket := dowHourBucket(now)
	if got.Cells[bucket].Hits < 1.0 {
		t.Fatalf("hits not preserved across round-trip: %v", got.Cells[bucket].Hits)
	}
}

// TestPersistMissingFile — Load() on a missing state path is NOT an
// error (fresh-start semantics).
func TestPersistMissingFile(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "does-not-exist.json")
	cfg := makePredCfg(nil)
	p := NewPredictor(cfg, nil, PredictorOptions{StatePath: statePath})
	if err := p.Load(); err != nil {
		t.Fatalf("Load() on missing file should not error; got %v", err)
	}
}

// TestRunGoroutineShutdown — Run(ctx) must exit promptly when ctx is
// cancelled. Defends against goroutine leaks in main.go.
func TestRunGoroutineShutdown(t *testing.T) {
	cfg := makePredCfg(nil)
	p := NewPredictor(cfg, nil, PredictorOptions{
		TickInterval: 50 * time.Millisecond,
	})
	ctx, cancel := context.WithCancel(context.Background())

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		p.Run(ctx)
	}()

	cancel()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("Predictor.Run did not exit within 2s of ctx cancel")
	}
}

// TestPendingFireHitMiss — a fire that gets a matching Observe inside
// the matched-request window is a "hit"; one that doesn't is a "miss"
// (reaped on the next tick past the window).
func TestPendingFireHitMiss(t *testing.T) {
	now := time.Date(2026, 6, 12, 14, 0, 0, 0, time.Local)
	clock, cur := makeFakeClock(now)

	cfg := makePredCfg(map[string]config.PredictiveColdLoadConfig{
		"hit-model":  {Enabled: true, PThreshold: 0.4, CooldownMinutes: 1, MaxPerDay: 10},
		"miss-model": {Enabled: true, PThreshold: 0.4, CooldownMinutes: 1, MaxPerDay: 10},
	})

	p := NewPredictor(cfg, func(string) bool { return true }, PredictorOptions{
		Now:                  clock,
		MatchedRequestWindow: 10 * time.Minute,
	})

	// Build evidence for both models.
	for i := 0; i < 10; i++ {
		*cur = now.Add(time.Duration(i) * time.Minute)
		p.tick(*cur)
		if i < 8 {
			p.Observe("hit-model", *cur)
			p.Observe("miss-model", *cur)
		}
	}

	// Fire both.
	*cur = now.Add(11 * time.Minute)
	p.tick(*cur)

	// hit-model gets a real request within 5 min — should clear pending.
	*cur = now.Add(15 * time.Minute)
	p.Observe("hit-model", *cur)

	p.mu.Lock()
	_, hitPending := p.pendingFire["hit-model"]
	_, missPending := p.pendingFire["miss-model"]
	p.mu.Unlock()

	if hitPending {
		t.Fatalf("hit-model still has pending fire after a real Observe inside window")
	}
	if !missPending {
		t.Fatalf("miss-model should still have pending fire (no Observe yet)")
	}

	// Advance past the window — miss-model's pending should be reaped.
	*cur = now.Add(30 * time.Minute)
	p.tick(*cur)

	p.mu.Lock()
	_, missPending = p.pendingFire["miss-model"]
	p.mu.Unlock()
	if missPending {
		t.Fatalf("miss-model pending fire should have been reaped past window")
	}
}

// TestDowHourBucketStability — boundary check for the bucket function:
// every (dow, hour) maps to exactly one cell in [0, 168).
func TestDowHourBucketStability(t *testing.T) {
	seen := map[int]bool{}
	for d := 0; d < 7; d++ {
		for h := 0; h < 24; h++ {
			// time.Date with explicit weekday is awkward; instead pick
			// a known Sunday and offset.
			sunday := time.Date(2026, 6, 7, 0, 0, 0, 0, time.Local) // 2026-06-07 is a Sunday
			ts := sunday.AddDate(0, 0, d).Add(time.Duration(h) * time.Hour)
			b := dowHourBucket(ts)
			if b < 0 || b >= histBuckets {
				t.Fatalf("bucket %d out of range for d=%d h=%d", b, d, h)
			}
			if seen[b] {
				t.Fatalf("collision: bucket %d already seen", b)
			}
			seen[b] = true
		}
	}
	if len(seen) != histBuckets {
		t.Fatalf("expected %d unique buckets, got %d", histBuckets, len(seen))
	}
}
