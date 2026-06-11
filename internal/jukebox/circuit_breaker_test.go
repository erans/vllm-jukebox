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

	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	waitFor(t, func() bool { return rd.Count() >= 1 }, "first trip")

	// Bump clock 30s — still within 300s cooldown. The first trip
	// already cleared the model's window, so we need a fresh burst
	// of 3 to attempt a re-trip; the cooldown then suppresses.
	clock.Advance(30 * time.Second)
	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	time.Sleep(50 * time.Millisecond)
	if rd.Count() != 1 {
		t.Fatalf("trip re-fired inside cooldown: %v", rd.calls)
	}
}

func TestCircuitBreaker_CooldownExpiresThenTripsAgain(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	clock := newFakeClock()
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, clock.Now)

	for i := 0; i < 3; i++ {
		cb.RecordResponse("vllm-main", 500)
	}
	waitFor(t, func() bool { return rd.Count() >= 1 }, "first trip")

	clock.Advance(301 * time.Second) // past cooldown
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
		for i := 0; i < 3; i++ {
			cb.RecordResponse("vllm-main", 500)
		}
		time.Sleep(20 * time.Millisecond)
		clock.Advance(2 * time.Second) // past 1s cooldown each time
	}
	time.Sleep(50 * time.Millisecond)
	// MaxTrips=2 so we should see exactly 2 fires, then suppressions.
	if rd.Count() != 2 {
		t.Fatalf("crash-loop guard didn't freeze: got %d trips, want 2", rd.Count())
	}
}

func TestCircuitBreaker_AliasResolvesToContainer(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

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
	if rd.Count() != 0 {
		t.Fatalf("dry-run fired docker: %v", rd.calls)
	}
}

func TestCircuitBreaker_StatusZeroTreatedAs5xx(t *testing.T) {
	rd := installRecordingDocker(t)
	cfg := cbTestCfg(t, true)
	cb := NewCircuitBreaker(func() *config.Config { return cfg },
		func(string) bool { return true }, time.Now)

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
