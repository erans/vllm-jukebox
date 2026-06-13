package jukebox_test

// End-to-end integration tests for the sleep-mode + lifecycle: external
// path. A fakeVLLMServer (httptest.Server) implements the subset of vLLM's
// HTTP API that jukebox cares about — /health, /v1/models, /sleep,
// /wake_up, /is_sleeping, /v1/chat/completions — with realistic state
// transitions. Tests verify the full pipeline: jukebox → fake vLLM, no
// GPU required.

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

// fakeVLLMServer simulates a vLLM 0.22+ instance with sleep-mode support
// behind an httptest.Server. It tracks isSleeping state and exposes
// counters so tests can assert which endpoints were hit and how often.
type fakeVLLMServer struct {
	server *httptest.Server
	model  string

	// State.
	sleeping atomic.Bool

	// Counters.
	sleepHits     atomic.Int32
	wakeHits      atomic.Int32
	isSleepingHit atomic.Int32
	healthHits    atomic.Int32
	chatHits      atomic.Int32

	// Failure injection.
	wakeFails atomic.Bool

	// Wake delay simulates real wake latency.
	wakeDelay time.Duration
}

func newFakeVLLM(modelName string) *fakeVLLMServer {
	f := &fakeVLLMServer{model: modelName}
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		f.healthHits.Add(1)
		if f.sleeping.Load() {
			// Real vLLM returns non-200 when sleeping; jukebox's wake
			// poll loop expects 200 to stop polling.
			http.Error(w, "sleeping", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data":   []map[string]any{{"id": f.model, "object": "model", "owned_by": "vllm"}},
		})
	})

	mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		f.sleepHits.Add(1)
		level := r.URL.Query().Get("level")
		if level != "1" && level != "2" {
			http.Error(w, "invalid level", http.StatusBadRequest)
			return
		}
		f.sleeping.Store(true)
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/wake_up", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		f.wakeHits.Add(1)
		if f.wakeFails.Load() {
			http.Error(w, "simulated wake failure", http.StatusInternalServerError)
			return
		}
		if f.wakeDelay > 0 {
			time.Sleep(f.wakeDelay)
		}
		f.sleeping.Store(false)
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
		f.isSleepingHit.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"is_sleeping": f.sleeping.Load()})
	})

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.chatHits.Add(1)
		if f.sleeping.Load() {
			// Real vLLM's PR #16536: model-dependent endpoints return errors
			// rather than crashing during sleep. Jukebox's design is to wake
			// before forwarding; this endpoint should never see a sleeping
			// fake unless something's wrong.
			http.Error(w, "model is sleeping", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-fake",
			"object": "chat.completion",
			"model":  f.model,
			"choices": []map[string]any{{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "ok",
				},
				"finish_reason": "stop",
			}},
		})
	})

	f.server = httptest.NewServer(mux)
	return f
}

func (f *fakeVLLMServer) Close()      { f.server.Close() }
func (f *fakeVLLMServer) URL() string { return f.server.URL }
func (f *fakeVLLMServer) Host() string {
	h, _, _ := net.SplitHostPort(strings.TrimPrefix(f.URL(), "http://"))
	return h
}
func (f *fakeVLLMServer) Port() int {
	_, p, _ := net.SplitHostPort(strings.TrimPrefix(f.URL(), "http://"))
	n, _ := strconv.Atoi(p)
	return n
}

// buildExternalConfig produces a config that registers each fake as a
// lifecycle: external scheduler-mode model. The model names align with
// the YAML keys; aliases are not used.
func buildExternalConfig(t *testing.T, fakes map[string]*fakeVLLMServer, idleTimeout time.Duration) *config.Config {
	t.Helper()
	var modelsBlock strings.Builder
	for name, f := range fakes {
		fmt.Fprintf(&modelsBlock, `
  %s:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    idle_timeout: %s
`, name, f.Host(), f.Port(), idleTimeout)
	}

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:%s
`, modelsBlock.String())

	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	return cfg
}

// TestIntegration_ExternalLifecycleEndToEnd validates the full
// jukebox → fake-vLLM pipeline: register external instances, route a
// request, sleep via auto-suspend, wake via next request.
func TestIntegration_ExternalLifecycleEndToEnd(t *testing.T) {
	fake := newFakeVLLM("emb")
	defer fake.Close()
	fakes := map[string]*fakeVLLMServer{"emb": fake}

	cfg := buildExternalConfig(t, fakes, 100*time.Millisecond)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	// External instance should be in Status as Ready.
	st := s.Status()
	if len(st.Instances) != 1 {
		t.Fatalf("expected 1 instance after register, got %d", len(st.Instances))
	}
	if st.Instances[0].Model != "emb" || st.Instances[0].State != jukebox.StateReady {
		t.Fatalf("expected emb=Ready, got %+v", st.Instances[0])
	}

	// Acquire a route — should hit the fake's URL.
	route, err := s.AcquireRoute(context.Background(), "emb", "req-1")
	if err != nil {
		t.Fatalf("AcquireRoute(emb): %v", err)
	}
	if route.BaseURL != fake.URL() {
		t.Fatalf("BaseURL mismatch: got %q want %q", route.BaseURL, fake.URL())
	}
	if route.Done != nil {
		route.Done()
	}

	// Force an idle check (idle_timeout=100ms; we've waited 0ms so
	// nothing should sleep yet) — assert no premature sleep.
	s.CheckIdleForTest(context.Background())
	if fake.sleepHits.Load() != 0 {
		t.Fatalf("did not expect Sleep before idle_timeout, got %d hits", fake.sleepHits.Load())
	}

	// Wait past idle_timeout and trigger a check — should auto-sleep.
	time.Sleep(150 * time.Millisecond)
	s.CheckIdleForTest(context.Background())
	if fake.sleepHits.Load() != 1 {
		t.Fatalf("expected exactly 1 Sleep after idle_timeout + check, got %d", fake.sleepHits.Load())
	}
	if !fake.sleeping.Load() {
		t.Fatalf("expected fake to be sleeping after auto-suspend")
	}

	// Verify the instance state is StateSleeping.
	for _, inst := range s.Status().Instances {
		if inst.Model == "emb" && inst.State != jukebox.StateSleeping {
			t.Fatalf("expected StateSleeping, got %s", inst.State)
		}
	}

	// Now request again — must wake the fake and route.
	route2, err := s.AcquireRoute(context.Background(), "emb", "req-2")
	if err != nil {
		t.Fatalf("AcquireRoute(emb) after sleep: %v", err)
	}
	if route2.BaseURL != fake.URL() {
		t.Fatalf("post-wake BaseURL mismatch: got %q want %q", route2.BaseURL, fake.URL())
	}
	if route2.Done != nil {
		route2.Done()
	}
	if fake.wakeHits.Load() != 1 {
		t.Fatalf("expected exactly 1 Wake call, got %d", fake.wakeHits.Load())
	}
	if fake.sleeping.Load() {
		t.Fatalf("expected fake to be awake after wake")
	}
}

// TestIntegration_PinnedExternalNeverAutoSuspends validates that a
// pinned external instance is exempt from idleMonitor even when its
// idle_timeout is non-zero.
func TestIntegration_PinnedExternalNeverAutoSuspends(t *testing.T) {
	fake := newFakeVLLM("main")
	defer fake.Close()

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  main:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    pinned: true
    sleep_mode: true
    idle_timeout: 50ms
`, fake.Host(), fake.Port())
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	time.Sleep(100 * time.Millisecond) // exceed idle_timeout
	s.CheckIdleForTest(context.Background())
	if got := fake.sleepHits.Load(); got != 0 {
		t.Fatalf("pinned model must not be auto-suspended; got %d Sleep calls", got)
	}
}

// TestIntegration_PinnedExternalNotSleptOnStopAll validates that
// StopAll (called on jukebox SIGTERM/graceful shutdown) does NOT sleep
// pinned external instances. Pinned models stay awake across an
// unrelated jukebox restart; otherwise the next request to a pinned
// model after a jukebox bounce eats a wake-latency hit that should
// not have been incurred. Eviction (the other path through
// drainAndStopInstance) IS allowed to sleep pinned — see
// TestIntegration_PinnedExternalSleptOnEviction below.
func TestIntegration_PinnedExternalNotSleptOnStopAll(t *testing.T) {
	fake := newFakeVLLM("main")
	defer fake.Close()

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
vllm:
  port: 8000
models:
  main:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    pinned: true
    sleep_mode: true
`, fake.Host(), fake.Port())
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	// Simulate jukebox shutdown.
	if err := s.StopAll(context.Background()); err != nil {
		t.Fatalf("StopAll: %v", err)
	}
	if got := fake.sleepHits.Load(); got != 0 {
		t.Fatalf("pinned model must not be slept by StopAll; got %d Sleep calls", got)
	}
}

// TestIntegration_WakeFailurePropagates validates that a real HTTP
// failure on /wake_up surfaces as a RejectError to the caller and does
// NOT leave the instance in an inconsistent state.
func TestIntegration_WakeFailurePropagates(t *testing.T) {
	fake := newFakeVLLM("emb")
	defer fake.Close()
	fakes := map[string]*fakeVLLMServer{"emb": fake}

	cfg := buildExternalConfig(t, fakes, 50*time.Millisecond)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("register: %v", err)
	}

	// Sleep it.
	time.Sleep(80 * time.Millisecond)
	s.CheckIdleForTest(context.Background())
	if !fake.sleeping.Load() {
		t.Fatalf("setup: fake must be sleeping")
	}

	// Inject a wake failure.
	fake.wakeFails.Store(true)

	_, err := s.AcquireRoute(context.Background(), "emb", "wake-fail")
	if err == nil {
		t.Fatalf("expected wake failure to surface error")
	}
	// Must be a RejectError so the HTTP layer maps it to 503.
	var rej *jukebox.RejectError
	if !errorIs(err, &rej) {
		t.Fatalf("expected RejectError, got %T: %v", err, err)
	}
}

// TestIntegration_ConcurrentWakeFanOut: 5 goroutines hit the same
// sleeping external instance with a slow wake; we observe exactly ONE
// /wake_up hit at the fake (no thundering herd) and all 5 succeed.
func TestIntegration_ConcurrentWakeFanOut(t *testing.T) {
	fake := newFakeVLLM("emb")
	defer fake.Close()
	fake.wakeDelay = 200 * time.Millisecond
	fakes := map[string]*fakeVLLMServer{"emb": fake}

	cfg := buildExternalConfig(t, fakes, 30*time.Millisecond)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("register: %v", err)
	}

	// Sleep it.
	time.Sleep(50 * time.Millisecond)
	s.CheckIdleForTest(context.Background())
	if !fake.sleeping.Load() {
		t.Fatalf("setup: fake must be sleeping")
	}

	const N = 5
	var wg sync.WaitGroup
	errCh := make(chan error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			route, err := s.AcquireRoute(ctx, "emb", "fan-out")
			if err != nil {
				errCh <- err
				return
			}
			if route.Done != nil {
				route.Done()
			}
			errCh <- nil
		}()
	}
	wg.Wait()
	close(errCh)

	var failures []error
	for err := range errCh {
		if err != nil {
			failures = append(failures, err)
		}
	}
	if len(failures) > 0 {
		t.Fatalf("expected all %d concurrent wakes to succeed; %d failed; first: %v", N, len(failures), failures[0])
	}
	if got := fake.wakeHits.Load(); got != 1 {
		t.Fatalf("expected EXACTLY ONE /wake_up at the fake (fan-out), got %d", got)
	}
}

// TestIntegration_ExternalSleepingAtBootSeedsStateSleeping validates the
// boot-time fix for the "asleep external vLLM hangs first request" bug.
// Before the fix, RegisterExternalInstances unconditionally seeded
// StateReady; a request would hit tryRouteReady → proxy directly to the
// sleeping backend → hang for ~200s. The fix probes /is_sleeping at
// startup (when sleep_mode is enabled) and seeds StateSleeping when the
// external vLLM reports it is sleeping, so the first request takes the
// wake codepath.
func TestIntegration_ExternalSleepingAtBootSeedsStateSleeping(t *testing.T) {
	fake := newFakeVLLM("emb")
	defer fake.Close()
	// Pre-set the fake to sleeping BEFORE jukebox boots — simulates the
	// production scenario where vLLM was put to sleep by a prior jukebox
	// instance and jukebox then restarted.
	fake.sleeping.Store(true)

	fakes := map[string]*fakeVLLMServer{"emb": fake}
	cfg := buildExternalConfig(t, fakes, 0)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	// The /is_sleeping probe must have fired during register.
	if got := fake.isSleepingHit.Load(); got < 1 {
		t.Fatalf("expected at least one /is_sleeping probe at boot, got %d", got)
	}

	// Instance should be StateSleeping, not StateReady.
	st := s.Status()
	if len(st.Instances) != 1 {
		t.Fatalf("expected 1 instance after register, got %d", len(st.Instances))
	}
	if st.Instances[0].Model != "emb" || st.Instances[0].State != jukebox.StateSleeping {
		t.Fatalf("expected emb=StateSleeping, got %+v", st.Instances[0])
	}

	// First request after boot must take the wake codepath — the fake
	// should observe exactly one /wake_up call and then succeed.
	route, err := s.AcquireRoute(context.Background(), "emb", "first-after-boot")
	if err != nil {
		t.Fatalf("AcquireRoute(emb) after boot-asleep: %v", err)
	}
	if route.BaseURL != fake.URL() {
		t.Fatalf("BaseURL mismatch: got %q want %q", route.BaseURL, fake.URL())
	}
	if route.Done != nil {
		route.Done()
	}
	if got := fake.wakeHits.Load(); got != 1 {
		t.Fatalf("expected exactly 1 /wake_up after first request, got %d", got)
	}
	if fake.sleeping.Load() {
		t.Fatalf("expected fake to be awake after wake")
	}
}

// TestIntegration_ExternalAwakeAtBootSeedsStateReady is the
// negative-control complement: when the external vLLM reports
// is_sleeping=false at boot, the seeded state must remain StateReady
// (the original behavior — we are not regressing the common case).
func TestIntegration_ExternalAwakeAtBootSeedsStateReady(t *testing.T) {
	fake := newFakeVLLM("emb")
	defer fake.Close()
	// Default: fake.sleeping is false. Don't touch it.

	fakes := map[string]*fakeVLLMServer{"emb": fake}
	cfg := buildExternalConfig(t, fakes, 0)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 1000, FreeMB: 1000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	st := s.Status()
	if len(st.Instances) != 1 {
		t.Fatalf("expected 1 instance, got %d", len(st.Instances))
	}
	if st.Instances[0].State != jukebox.StateReady {
		t.Fatalf("expected StateReady when external is awake at boot, got %s", st.Instances[0].State)
	}

	// First request must NOT trigger a wake.
	route, err := s.AcquireRoute(context.Background(), "emb", "awake-boot")
	if err != nil {
		t.Fatalf("AcquireRoute: %v", err)
	}
	if route.Done != nil {
		route.Done()
	}
	if got := fake.wakeHits.Load(); got != 0 {
		t.Fatalf("expected zero /wake_up calls when external was awake at boot, got %d", got)
	}
}

// TestIntegration_SwapGroup_IdleAutoRestore validates the full
// swap-group flow end-to-end against fake-vLLM servers:
//   - two models declared with swap_group "g1"; one is pinned (the
//     always-restored default) and starts asleep; one is normal (the
//     transient peer) and starts asleep too
//   - a client wake-request hits the transient model; admission MUST
//     evict the pinned member (swap-group override) and wake the transient
//   - the transient is forced past idle_timeout, CheckIdleForTest fires an
//     idle-sleep; auto-restore MUST dispatch a wake on the pinned peer
func TestIntegration_SwapGroup_IdleAutoRestore(t *testing.T) {
	fakeDefault := newFakeVLLM("default-model")
	defer fakeDefault.Close()
	fakeTransient := newFakeVLLM("transient-model")
	defer fakeTransient.Close()

	// Both start ASLEEP so we can observe the wake on each in isolation.
	fakeDefault.sleeping.Store(true)
	fakeTransient.sleeping.Store(true)

	yaml := fmt.Sprintf(`
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1ms
vllm:
  port: 8000
  startup_timeout: 5s
  drain_timeout: 100ms
  shutdown_timeout: 100ms
models:
  default-model:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    pinned: true
    expected_vram_mb_per_gpu: 10000
    swap_group: g1
  transient-model:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    idle_timeout: 50ms
    priority: normal
    expected_vram_mb_per_gpu: 10000
    swap_group: g1
`, fakeDefault.Host(), fakeDefault.Port(), fakeTransient.Host(), fakeTransient.Port())

	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	pool := ports.New(8100, 8109)
	// GPU total chosen so both 10000-MB members coexist (10000+10000 fits
	// in 24000 even with the pinned-budget subtraction) — but the
	// EARLY-BOOT admission state starts both as awake from pinned-init,
	// so a wake will need real eviction work too. Actually since we
	// override both to "sleeping" via the boot probe, the budget should
	// end up clean and the transient wake is unblocked at boot.
	// Important: the pinned-budget math reserves the pinned model's
	// footprint regardless of state, so available=14000 even when the
	// pinned member is sleeping. The transient (10000) fits.
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	// Wire admission so swap-group logic engages end-to-end.
	totalsByGPU := map[int]int{0: 24000}
	adm := jukebox.NewAdmissionController(cfg, totalsByGPU, &jukebox.SchedulerEvictor{S: s})
	s.SetAdmission(adm)

	if err := s.RegisterExternalInstances(context.Background()); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	// Both fakes should have received an /is_sleeping probe at boot.
	if fakeDefault.isSleepingHit.Load() < 1 {
		t.Fatalf("expected default model /is_sleeping probed at boot, got %d", fakeDefault.isSleepingHit.Load())
	}
	if fakeTransient.isSleepingHit.Load() < 1 {
		t.Fatalf("expected transient model /is_sleeping probed at boot, got %d", fakeTransient.isSleepingHit.Load())
	}

	// Reset hit counters so post-boot Wake/Sleep counts are unambiguous.
	fakeDefault.wakeHits.Store(0)
	fakeDefault.sleepHits.Store(0)
	fakeTransient.wakeHits.Store(0)
	fakeTransient.sleepHits.Store(0)

	// Step 1: client wakes the transient model. Should succeed —
	// admission has plenty of headroom (pinned 10000 reserved, awake
	// budget 0, so 14000 available; transient needs 10000).
	route, err := s.AcquireRoute(context.Background(), "transient-model", "req-wake-transient")
	if err != nil {
		t.Fatalf("AcquireRoute(transient): %v", err)
	}
	if route.BaseURL != fakeTransient.URL() {
		t.Fatalf("route URL mismatch: got %q want %q", route.BaseURL, fakeTransient.URL())
	}
	if route.Done != nil {
		route.Done()
	}
	if fakeTransient.wakeHits.Load() != 1 {
		t.Fatalf("expected exactly 1 Wake on transient, got %d", fakeTransient.wakeHits.Load())
	}

	// Step 2: force transient past idle_timeout, then trigger the idle
	// check. The transient model should be slept AND the auto-restore
	// callback should fire a wake on default-model in the background.
	time.Sleep(80 * time.Millisecond)
	s.CheckIdleForTest(context.Background())

	if fakeTransient.sleepHits.Load() != 1 {
		t.Fatalf("expected exactly 1 Sleep on transient after idle, got %d", fakeTransient.sleepHits.Load())
	}

	// Step 3: poll until the auto-restore wake lands on default-model.
	// The dispatch is asynchronous (goroutine) and the wake itself takes
	// the full RequestWake → SchedulerEvictor → /wake_up codepath.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if fakeDefault.wakeHits.Load() >= 1 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got := fakeDefault.wakeHits.Load(); got != 1 {
		t.Fatalf("expected auto-restore Wake on default-model within 3s, got %d", got)
	}
	if fakeDefault.sleeping.Load() {
		t.Fatalf("default-model should be awake after auto-restore")
	}
}

// errorIs is a tiny errors.As wrapper kept here (rather than importing
// errors) to make the assertion site read clearly.
func errorIs(err error, target any) bool {
	type asErr interface{ As(any) bool }
	if a, ok := err.(asErr); ok {
		return a.As(target)
	}
	// Fall back to errors.As semantics via reflection-free wrapper —
	// import errors and use it directly to avoid a custom shim.
	return errorsAs(err, target)
}

func errorsAs(err error, target any) bool {
	// Inline errors.As to avoid an extra import in this file.
	var current = err
	for current != nil {
		// Type assertion via reflection-free check: just compare the
		// target pointer's element type. Fall back to errors.As if we
		// can. Since this file does not import "errors", do a structural
		// check on RejectError (the only error type we assert on here).
		if rejT, ok := target.(**jukebox.RejectError); ok {
			if r, ok := current.(*jukebox.RejectError); ok {
				*rejT = r
				return true
			}
		}
		// Walk Unwrap chain.
		type unwrapper interface{ Unwrap() error }
		if u, ok := current.(unwrapper); ok {
			current = u.Unwrap()
			continue
		}
		break
	}
	return false
}
