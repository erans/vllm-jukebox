package jukebox

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// recordingDocker records every runDocker call into a slice; useful
// to assert "exactly N restarts fired" without spawning real docker.
type recordingDocker struct {
	mu    sync.Mutex
	calls [][]string
	err   error
}

func (r *recordingDocker) cmd(_ context.Context, name string, args ...string) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	full := append([]string{name}, args...)
	r.calls = append(r.calls, full)
	return []byte("ok"), r.err
}

func (r *recordingDocker) Count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

// RestartCount returns the number of `docker restart` invocations,
// filtering out `docker inspect` calls (the cold-load grace gate
// runs an inspect before every potential trip; tests that count
// trip-actions want only restarts).
func (r *recordingDocker) RestartCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, call := range r.calls {
		if len(call) >= 2 && call[1] == "restart" {
			n++
		}
	}
	return n
}

func (r *recordingDocker) Last() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		return nil
	}
	out := make([]string, len(r.calls[len(r.calls)-1]))
	copy(out, r.calls[len(r.calls)-1])
	return out
}

func installRecordingDocker(t *testing.T) *recordingDocker {
	t.Helper()
	rd := &recordingDocker{}
	var fn dockerCmdFn = rd.cmd
	redeployDockerCmd.Store(&fn)
	t.Cleanup(func() { redeployDockerCmd.Store(nil) })
	return rd
}

func cbTestCfg(t *testing.T, enabled bool) *config.Config {
	t.Helper()
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"vllm-main": {Path: "/m", Container: "vllm-main"},
			"vllm-task": {Path: "/t", Container: "vllm-task"},
			"alias":     {Alias: "vllm-main"},
			"no-container": {Path: "/n"},
		},
	}
	cfg.Behavior.CircuitBreaker = config.CircuitBreakerConfig{
		Enabled:                enabled,
		Window:                 5,
		Threshold:              3,
		CooldownSeconds:        300,
		CrashLoopMaxTrips:      3,
		CrashLoopWindowSeconds: 1800,
		RestartTimeoutSeconds:  60,
	}
	return cfg
}

// waitFor polls fn until it returns true or 1s elapses.
func waitFor(t *testing.T, fn func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if fn() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("waitFor timeout: %s", msg)
}

// fakeClock is a mutex-protected wall clock for tests that mutate the
// observed time while goroutines spawned by the breaker may read it.
// A plain `now := time.Now()` reassigned by the test would race the
// breaker goroutine's read of the same variable.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock { return &fakeClock{t: time.Now()} }
func (f *fakeClock) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.t
}
func (f *fakeClock) Advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.t = f.t.Add(d)
}

func TestCircuitBreaker_DisabledNoOp(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, false)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	for i := 0; i < 10; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	// Synchronously: trip eval is spawned async. Give it a moment but
	// expect zero calls even after.
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("disabled breaker fired docker: %v", rd.calls)
	}
}

func TestCircuitBreaker_TripsAfterThreshold(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	// Seed cold-load grace latch (Track A fix 2026-06-12) — a 2xx
	// proves the container has serviced traffic, so subsequent 5xx
	// counts as a real wedge rather than a legit cold-load. Without
	// this seed every test expecting a trip would suppress.
	cb.RecordResponse("vllm-main", 200)

	// 2 5xx (below threshold) — no trip.
	cb.RecordResponse("vllm-main", 500)
	cb.RecordResponse("vllm-main", 502)
	time.Sleep(20 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("trip fired below threshold: %v", rd.calls)
	}
	// Third 5xx — should trip.
	cb.RecordResponse("vllm-main", 503)
	waitFor(t, func() bool { return rd.Count() >= 1 }, "trip should fire after 3 consecutive 5xx")
	last := rd.Last()
	want := []string{"docker", "restart", "-t", "5", "vllm-main"}
	if len(last) != len(want) {
		t.Fatalf("unexpected cmd: %v", last)
	}
	for i, a := range want {
		if last[i] != a {
			t.Fatalf("docker arg[%d] got %q want %q", i, last[i], a)
		}
	}
}

func TestCircuitBreaker_2xxResetsStreak(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	cb.RecordResponse("vllm-main", 500)
	cb.RecordResponse("vllm-main", 500)
	cb.RecordResponse("vllm-main", 200) // breaks the streak
	cb.RecordResponse("vllm-main", 500)
	cb.RecordResponse("vllm-main", 500)
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("trip fired despite 200 breaking streak: %v", rd.calls)
	}
}

func TestCircuitBreaker_NotReadyGate(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return false }, time.Now)

	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("trip fired despite not-ready gate: %v", rd.calls)
	}
}

func TestCircuitBreaker_CooldownSuppresses(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	clock := newFakeClock()
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, clock.Now)

	cb.RecordResponse("vllm-main", 200) // seed cold-load grace latch
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	waitFor(t, func() bool { return rd.RestartCount() >= 1 }, "first trip")

	// Bump clock 30s — still within 300s cooldown. The first trip
	// already cleared the model's window, so we need a fresh burst
	// of 3 to attempt a re-trip; the cooldown then suppresses.
	clock.Advance(30 * time.Second)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.RestartCount() != 1 {
		t.Fatalf("trip re-fired inside cooldown: %v", rd.calls)
	}
}

func TestCircuitBreaker_CooldownExpiresThenTripsAgain(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	clock := newFakeClock()
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, clock.Now)

	cb.RecordResponse("vllm-main", 200) // seed cold-load grace latch
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	waitFor(t, func() bool { return rd.Count() >= 1 }, "first trip")

	clock.Advance(301 * time.Second) // past cooldown
	cb.RecordResponse("vllm-main", 200) // re-seed latch (trip-fire cleared it)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	waitFor(t, func() bool { return rd.Count() >= 2 }, "second trip after cooldown")
}

func TestCircuitBreaker_CrashLoopFreezes(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cfg.Behavior.CircuitBreaker.CooldownSeconds = 1
	cfg.Behavior.CircuitBreaker.CrashLoopMaxTrips = 2
	cfg.Behavior.CircuitBreaker.CrashLoopWindowSeconds = 600

	clock := newFakeClock()
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, clock.Now)

	for run := 0; run < 5; run++ {
		cb.RecordResponse("vllm-main", 200) // re-seed latch each iter (trip-fire clears it)
		for i := 0; i < 3; i++ {
			cb.RecordResponse("vllm-main", 500)
		}
		time.Sleep(20 * time.Millisecond)
		clock.Advance(2 * time.Second) // past 1s cooldown each time
	}
	time.Sleep(50 * time.Millisecond)
	// MaxTrips=2 so we should see exactly 2 fires, then suppressions.
	if rd.RestartCount() != 2 {
		t.Fatalf("crash-loop guard didn't freeze: got %d trips, want 2", rd.RestartCount())
	}
}

func TestCircuitBreaker_AliasResolvesToContainer(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	cb.RecordResponse("alias", 200) // seed latch (alias resolves to vllm-main)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("alias", 500)
	}
	waitFor(t, func() bool { return rd.Count() >= 1 }, "alias should trip via resolved container")
	last := rd.Last()
	if last[len(last)-1] != "vllm-main" {
		t.Fatalf("alias didn't resolve to vllm-main container, got %v", last)
	}
}

func TestCircuitBreaker_NoContainerObservesOnly(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	for i := 0; i < 5; i++ {
		cb.RecordResponse("no-container", 500)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("model w/o container shouldn't trip: %v", rd.calls)
	}
}

func TestCircuitBreaker_DryRunDoesntRestart(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cfg.Behavior.CircuitBreaker.DryRun = true
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.RestartCount() != 0 {
		t.Fatalf("dry-run fired docker restart: %v", rd.calls)
	}
}

func TestCircuitBreaker_StatusZeroTreatedAs5xx(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	cb.RecordResponse("vllm-main", 200) // seed latch
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 0)
	}
	waitFor(t, func() bool { return rd.Count() >= 1 }, "status=0 should trip")
}

func TestCircuitBreaker_4xxNeverTrips(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	for i := 0; i < 10; i++ {
		cb.RecordResponse("vllm-main", 404)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 0 {
		t.Fatalf("4xx tripped breaker: %v", rd.calls)
	}
}

func TestCircuitBreaker_ConcurrentRecordSafe(t *testing.T) {
	// Race-detector smoke: many goroutines push concurrently.
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	cb.RecordResponse("vllm-main", 200) // seed cold-load grace latch

	var wg sync.WaitGroup
	var counter int64
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				cb.RecordResponse("vllm-main", 500)
				atomic.AddInt64(&counter, 1)
			}
		}()
	}
	wg.Wait()
	time.Sleep(100 * time.Millisecond)
	// At least one trip should fire; we don't assert exact count
	// because cooldown is 300s — only one fire is expected.
	if rd.Count() < 1 {
		t.Fatalf("concurrent burst didn't trip: %d records, %d calls", counter, rd.Count())
	}
}

// --- Cold-load grace gate tests (Track A fix 2026-06-12) ---
//
// The breaker tripped on legit cold-load during the 2026-06-12 02:00 UTC
// incident: vllm-main was rebooting on a new image, /is_sleeping returned
// connect-refused for ~6min while workers spawned, drift_probe counted
// those refusals as 5xx (status=0), breaker tripped → `docker restart`
// aborted the cold-load mid-flight → infinite loop. Root cause: breaker
// had no way to distinguish "container has never been up yet" from
// "container was up and is now wedged".
//
// Gate: latch per-container on the first 2xx response. A container that
// has not returned a 2xx in its current incarnation gets the trip
// suppressed. The latch CLEARS when the breaker fires `docker restart`
// (the restart starts a fresh cold-load window — see breaker.go).

// TestCircuitBreaker_ColdLoadGrace_NoPriorHealthy_SuppressesTrip
// reproduces the 2026-06-12 02:00Z incident: cold-load in progress,
// every drift_probe returns refused (status 0 = 5xx in breaker). The
// 5xx threshold (3) is crossed but the grace gate must suppress
// because no 2xx has been observed.
func TestCircuitBreaker_ColdLoadGrace_NoPriorHealthy_SuppressesTrip(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	// Cold-load: every drift_probe returns refused.
	for i := 0; i < 5; i++ {
		cb.RecordResponse("vllm-main", 0)
	}
	time.Sleep(80 * time.Millisecond)
	if rd.RestartCount() != 0 {
		t.Fatalf("cold-load grace failed — trip fired with no 2xx observed: %v", rd.calls)
	}
}

// TestCircuitBreaker_ColdLoadGrace_PriorHealthyThenWedge_TripsNormally
// is the inverse: container HAS returned 2xx (latch set). When it
// later returns 5xx burst, the breaker SHOULD trip — that's the real
// wedge signal the breaker exists to catch.
func TestCircuitBreaker_ColdLoadGrace_PriorHealthyThenWedge_TripsNormally(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	// Healthy phase — one 2xx is enough to set the latch.
	cb.RecordResponse("vllm-main", 200)
	// Now wedged: 5xx burst should trip.
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 503)
	}
	waitFor(t, func() bool { return rd.RestartCount() >= 1 }, "healthy→wedged should trip")
}

// TestCircuitBreaker_ColdLoadGrace_TripClearsLatch verifies that after
// the breaker fires `docker restart`, the latch clears so the
// follow-on cold-load (during which /is_sleeping returns refused) does
// NOT re-trip. This is the actual fix for the incident: a single trip
// must not cascade into infinite restart loops while cold-loading.
func TestCircuitBreaker_ColdLoadGrace_TripClearsLatch(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cfg.Behavior.CircuitBreaker.CooldownSeconds = 0 // exercise grace gate, not cooldown
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	// First incarnation: healthy, then wedged → trip fires.
	cb.RecordResponse("vllm-main", 200)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 503)
	}
	waitFor(t, func() bool { return rd.RestartCount() >= 1 }, "initial wedge trip")
	tripsBefore := rd.RestartCount()

	// docker restart fires → cold-load starts → refused/5xx burst.
	// Even though we just observed 2xx pre-trip, the latch must have
	// cleared on trip-fire — so this burst suppresses, not re-trips.
	for i := 0; i < 5; i++ {
		cb.RecordResponse("vllm-main", 0)
	}
	time.Sleep(80 * time.Millisecond)
	if rd.RestartCount() != tripsBefore {
		t.Fatalf("trip-clears-latch failed — re-tripped during follow-on cold-load: %v", rd.calls)
	}
}

// TestCircuitBreaker_ColdLoadGrace_TripClearsLatchThenRehealth verifies
// the full recovery cycle: trip fires (latch cleared) → cold-load
// finishes, container serves 2xx (latch re-sets) → engine wedges again
// → next trip fires. The crash-loop guard handles the final case
// where this pattern repeats too many times.
func TestCircuitBreaker_ColdLoadGrace_TripClearsLatchThenRehealth(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cfg.Behavior.CircuitBreaker.CooldownSeconds = 0
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

	// Trip #1
	cb.RecordResponse("vllm-main", 200)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 503)
	}
	waitFor(t, func() bool { return rd.RestartCount() >= 1 }, "first trip")

	// Cold-load period — refusals suppressed (latch cleared).
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 0)
	}
	time.Sleep(40 * time.Millisecond)
	if rd.RestartCount() != 1 {
		t.Fatalf("cold-load suppressed broke — got %d trips, want 1", rd.RestartCount())
	}

	// Container finishes cold-load → 2xx → latch re-arms.
	cb.RecordResponse("vllm-main", 200)
	// Wedge again → trip #2 should fire.
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 503)
	}
	waitFor(t, func() bool { return rd.RestartCount() >= 2 }, "second trip after re-healthy")
}
