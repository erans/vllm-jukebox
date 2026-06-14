package jukebox

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/ports"
)

// fakeDecodeMgr is a minimal SleepCapable InstanceManager that returns a
// non-empty BaseURL (the decode probe filters out empty-BaseURL
// instances) and a live PID. The /health/decode responses themselves are
// served by the SetDecodeProbeFnForTest hook, not this struct.
type fakeDecodeMgr struct {
	baseURL string
	pid     int
}

func (m *fakeDecodeMgr) Start(_ context.Context, _ string) (int, error) { return m.pid, nil }
func (m *fakeDecodeMgr) Stop(_ context.Context) error                   { return nil }
func (m *fakeDecodeMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *fakeDecodeMgr) CurrentPID() int                                { return m.pid }
func (m *fakeDecodeMgr) BaseURL() string                                { return m.baseURL }
func (m *fakeDecodeMgr) IsSleeping(_ context.Context) (bool, error)     { return false, nil }
func (m *fakeDecodeMgr) Sleep(_ context.Context, _ int) error           { return nil }
func (m *fakeDecodeMgr) Wake(_ context.Context, _ time.Duration) error  { return nil }

const decodeProbeTestConfig = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
behavior:
  decode_probe_interval: 30s
  circuit_breaker:
    enabled: true
    window: 5
    threshold: 3
    cooldown_seconds: 300
    crash_loop_max_trips: 3
    crash_loop_window_seconds: 1800
    restart_timeout_seconds: 60
models:
  vllm-main:
    lifecycle: external
    host: vllm-main
    port: 8001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    container: vllm-main
`

// makeDecodeProbeScheduler builds a scheduler with a single external
// Ready instance ("vllm-main") wired to a circuit breaker, plus a
// recording docker hook so trips are observable without real docker.
// config.Current must be set so liveModelCfg / liveCfg resolve.
func makeDecodeProbeScheduler(t *testing.T) (*Scheduler, *recordingDocker) {
	t.Helper()
	cfg, err := config.Load([]byte(decodeProbeTestConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	inv := &decodeProbeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	s.EnableCircuitBreaker(func() *config.Config { return config.Current() })

	mgr := &fakeDecodeMgr{baseURL: "http://vllm-main:8001", pid: 4242}
	s.SeedInstanceForTest("vllm-main", 8001, []int{0}, false, StateReady, mgr)

	rd := installRecordingDocker(t)
	return s, rd
}

type decodeProbeInventory struct{ gpus []gpu.GPU }

func (r *decodeProbeInventory) List(_ context.Context) ([]gpu.GPU, error) { return r.gpus, nil }

// waitForRestart polls the recording docker for a `docker restart`
// invocation (the breaker fires the restart on its own goroutine).
func waitForRestart(t *testing.T, rd *recordingDocker) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rd.RestartCount() > 0 {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for docker restart")
}

func assertNoRestart(t *testing.T, rd *recordingDocker) {
	t.Helper()
	// Give any async trip goroutine a chance to (incorrectly) fire.
	time.Sleep(50 * time.Millisecond)
	if n := rd.RestartCount(); n != 0 {
		t.Fatalf("expected no docker restart, got %d (last=%v)", n, rd.Last())
	}
}

// TestDecodeProbe_SustainedStallTripsAfterWindow: 200×N then 503×K.
// With in-flight > 0, the breaker must NOT trip on the first stall
// (transient) but MUST trip once the stall persists across
// sustainedStallIntervals, marking the instance draining + restarting.
func TestDecodeProbe_SustainedStallTripsAfterWindow(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	if !s.SetInstanceInflightForTest("vllm-main", 1) {
		t.Fatal("seed inflight failed")
	}

	var mu sync.Mutex
	status := http.StatusOK
	setStatus := func(v int) { mu.Lock(); status = v; mu.Unlock() }
	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return status, nil
	})()

	ctx := context.Background()

	// Two healthy ticks → no trip, counter stays 0.
	s.CheckDecodeStallsForTest(ctx)
	s.CheckDecodeStallsForTest(ctx)
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("healthy ticks should keep stall count 0, got %d", c)
	}

	// First stall → transient, count=1, NO trip.
	setStatus(http.StatusServiceUnavailable)
	s.CheckDecodeStallsForTest(ctx)
	if c := s.DecodeStallCountForTest("vllm-main"); c != 1 {
		t.Fatalf("first stall should set count=1, got %d", c)
	}
	assertNoRestart(t, rd)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); draining {
		t.Fatal("instance should NOT be draining after a single transient stall")
	}

	// Second consecutive stall → reaches sustainedStallIntervals (2) → TRIP.
	s.CheckDecodeStallsForTest(ctx)
	waitForRestart(t, rd)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); !draining {
		t.Fatal("instance must be draining after a sustained-stall trip")
	}
	// Counter resets after a trip so a recovered-then-re-wedged engine
	// must re-accumulate the full window.
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("stall count should reset to 0 after trip, got %d", c)
	}
	last := rd.Last()
	if len(last) < 3 || last[1] != "restart" || last[len(last)-1] != "vllm-main" {
		t.Fatalf("expected `docker restart ... vllm-main`, got %v", last)
	}
}

// TestDecodeProbe_EndpointAbsentNeverTrips: a 404 (older vLLM without
// /health/decode) must graceful-degrade — no trip, no draining, even
// after many ticks with in-flight > 0.
func TestDecodeProbe_EndpointAbsentNeverTrips(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	s.SetInstanceInflightForTest("vllm-main", 2)

	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		return http.StatusNotFound, nil
	})()

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		s.CheckDecodeStallsForTest(ctx)
	}
	assertNoRestart(t, rd)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); draining {
		t.Fatal("404 endpoint-absent must NOT drain the instance")
	}
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("endpoint-absent must keep stall count 0, got %d", c)
	}
}

// TestDecodeProbe_ConnectionRefusedNeverTrips: a connection-refused
// transport error is a DIFFERENT failure (socket gone, owned by
// TCP-liveness + the 5xx breaker). The decode probe must skip it.
func TestDecodeProbe_ConnectionRefusedNeverTrips(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	s.SetInstanceInflightForTest("vllm-main", 2)

	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		return 0, errors.New("dial tcp 10.0.0.1:8001: connect: connection refused")
	})()

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		s.CheckDecodeStallsForTest(ctx)
	}
	assertNoRestart(t, rd)
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("connection-refused must keep stall count 0, got %d", c)
	}
}

// TestDecodeProbe_AllHealthyNeverTrips: 200 throughout → never trips.
func TestDecodeProbe_AllHealthyNeverTrips(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	s.SetInstanceInflightForTest("vllm-main", 3)

	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		return http.StatusOK, nil
	})()

	ctx := context.Background()
	for i := 0; i < 10; i++ {
		s.CheckDecodeStallsForTest(ctx)
	}
	assertNoRestart(t, rd)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); draining {
		t.Fatal("healthy probes must never drain the instance")
	}
}

// TestDecodeProbe_StallWithNoInflightDoesNotTrip: a stalled
// /health/decode with ZERO in-flight is an idle engine, not the #45094
// wedge (which is by definition running>0 with 0 decode progress). The
// probe must NOT count it toward a trip.
func TestDecodeProbe_StallWithNoInflightDoesNotTrip(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	// No inflight seeded — count stays 0.

	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		return http.StatusServiceUnavailable, nil
	})()

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		s.CheckDecodeStallsForTest(ctx)
	}
	assertNoRestart(t, rd)
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("stall with no inflight must keep count 0, got %d", c)
	}
}

// TestDecodeProbe_TransientStallResetsOnRecovery: stall→recover→stall
// must NOT trip, because the recovery resets the consecutive window.
func TestDecodeProbe_TransientStallResetsOnRecovery(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	s.SetInstanceInflightForTest("vllm-main", 1)

	var mu sync.Mutex
	status := http.StatusServiceUnavailable
	setStatus := func(v int) { mu.Lock(); status = v; mu.Unlock() }
	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return status, nil
	})()

	ctx := context.Background()
	s.CheckDecodeStallsForTest(ctx) // stall, count=1
	if c := s.DecodeStallCountForTest("vllm-main"); c != 1 {
		t.Fatalf("count should be 1, got %d", c)
	}
	setStatus(http.StatusOK)
	s.CheckDecodeStallsForTest(ctx) // recover, count→0
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("recovery should reset count to 0, got %d", c)
	}
	setStatus(http.StatusServiceUnavailable)
	s.CheckDecodeStallsForTest(ctx) // stall again, count=1 (NOT 2)
	assertNoRestart(t, rd)
	if c := s.DecodeStallCountForTest("vllm-main"); c != 1 {
		t.Fatalf("post-recovery stall should restart window at 1, got %d", c)
	}
}
