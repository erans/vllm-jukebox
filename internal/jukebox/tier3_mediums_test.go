package jukebox

// Tests for Tier 3 MEDIUM fixes (10-14): metrics double-count, redeploy
// double-fired audit, sleep-API wedge-recovery, admission lock latency,
// and the cold-load poll-interval test override.

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// counterValue extracts the current numeric value of a CounterVec for
// the given label set. Returns 0 if the time series doesn't exist yet.
func counterValue(vec *prometheus.CounterVec, labels ...string) float64 {
	c, err := vec.GetMetricWithLabelValues(labels...)
	if err != nil {
		return 0
	}
	pb := &dto.Metric{}
	if err := c.Write(pb); err != nil {
		return 0
	}
	return pb.GetCounter().GetValue()
}

// ---------------------------------------------------------------------------
// Fix 10 — StopForEviction MUST bump StopsTotal but NOT SleepsTotal.
// ---------------------------------------------------------------------------

// TestFix10_StopForEviction_BumpsOnlyStopsTotal asserts that the
// StopForEviction path increments jukebox_vllm_stops_total but does NOT
// double-bump jukebox_sleeps_total{reason="stop:..."} (the historical
// double-count that overcounted Grafana sleep panels).
func TestFix10_StopForEviction_BumpsOnlyStopsTotal(t *testing.T) {
	yaml := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  fix10target:
    lifecycle: external
    host: vllm-fix10target
    port: 8201
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: g
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 500ms
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 24000}
	inv := newRedeployInventory(totals)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totals, &stubAdmissionEvictor{})
	s.SetAdmission(a)

	mgr := &fakeRedeployMgr{port: 8201}
	mgr.pid.Store(123)
	mgr.isSleeping.Store(false)
	s.SeedInstanceForTest("fix10target", 8201, []int{0}, false, StateReady, mgr)

	// Mock docker so docker stop succeeds.
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	preStops := counterValue(metrics.StopsTotal, "fix10target", "test-reason")
	preSleepsStop := counterValue(metrics.SleepsTotal, "fix10target", "stop:test-reason")
	preSleepsClean := counterValue(metrics.SleepsTotal, "fix10target", "test-reason")

	ev := &SchedulerEvictor{S: s}
	if err := ev.StopForEviction(context.Background(), "fix10target", "test-reason"); err != nil {
		t.Fatalf("StopForEviction: %v", err)
	}

	postStops := counterValue(metrics.StopsTotal, "fix10target", "test-reason")
	postSleepsStop := counterValue(metrics.SleepsTotal, "fix10target", "stop:test-reason")
	postSleepsClean := counterValue(metrics.SleepsTotal, "fix10target", "test-reason")

	if got := postStops - preStops; got != 1 {
		t.Errorf("StopsTotal{model=fix10target,reason=test-reason} expected +1, got +%v", got)
	}
	if got := postSleepsStop - preSleepsStop; got != 0 {
		t.Errorf("SleepsTotal{reason=stop:test-reason} should NOT be bumped (Fix 10 removed the double-count), got +%v", got)
	}
	if got := postSleepsClean - preSleepsClean; got != 0 {
		t.Errorf("SleepsTotal{reason=test-reason} should NOT be bumped on stop path, got +%v", got)
	}
}

// ---------------------------------------------------------------------------
// Fix 11 — peer-stop failure path emits exactly ONE lifecycle audit.
// ---------------------------------------------------------------------------

// recordingAuditSink replaces the default audit-sink hook with one that
// records every emitted LifecycleEvent. Returns an unexport function so
// the test can defer-restore the original sink.
type recordingAuditSink struct {
	mu     sync.Mutex
	events []LifecycleEvent
}

func (r *recordingAuditSink) record(e LifecycleEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *recordingAuditSink) snapshot() []LifecycleEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]LifecycleEvent, len(r.events))
	copy(out, r.events)
	return out
}

// TestFix11_RedeployMember_PeerStopFailure_SingleAuditPerPeer asserts
// that when docker stop on a peer fails, RedeployMember emits exactly
// ONE lifecycle audit per peer (the stop-failed-vram-drift-risk row),
// not TWO (previously the unconditional redeploy-member-peer-stop row
// also fired on the failure path).
func TestFix11_RedeployMember_PeerStopFailure_SingleAuditPerPeer(t *testing.T) {
	yaml := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  target:
    lifecycle: external
    host: vllm-target
    port: 8301
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: g
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 500ms
  peer:
    lifecycle: external
    host: vllm-peer
    port: 8302
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: g
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 500ms
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 24000}
	inv := newRedeployInventory(totals)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totals, &stubAdmissionEvictor{})
	s.SetAdmission(a)

	tgt := &fakeRedeployMgr{port: 8301}
	tgt.pid.Store(0)
	tgt.isSleeping.Store(true)
	s.SeedInstanceForTest("target", 8301, []int{0}, false, StateReady, tgt)
	peer := &fakeRedeployMgr{port: 8302}
	peer.pid.Store(0)
	peer.isSleeping.Store(true)
	s.SeedInstanceForTest("peer", 8302, []int{0}, false, StateReady, peer)

	// Drive admission to Awake for both so peer is considered "currently
	// resident" (any non-Stopped state).
	a.mu.Lock()
	a.markStartedLocked(a.models["target"]) // Stopped → Sleeping
	a.markStartedLocked(a.models["peer"])
	// Force both to Awake by directly mutating (test-only setup).
	for _, name := range []string{"target", "peer"} {
		m := a.models[name]
		if m.State == admissionSleeping {
			a.l1ResidualByGPU[0] -= m.L1ResidualMB
		}
		a.awakeByGPU[0] += m.ExpectedOn(0)
		m.State = admissionAwake
	}
	a.mu.Unlock()

	// Install a recording audit sink.
	rec := &recordingAuditSink{}
	prev := SetLifecycleAuditSinkForTest(rec.record)
	defer SetLifecycleAuditSinkForTest(prev)

	// Docker hook: peer stop fails, target stop+start succeed. We
	// distinguish by container name in the args.
	SetDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		// args = ["stop", "-t", "60", "vllm-peer"] or similar.
		// args = ["start", "vllm-target"]
		for _, a := range args {
			if a == "vllm-peer" {
				return []byte("daemon error"), errors.New("docker stop failed for peer")
			}
		}
		// Flip target.isSleeping=true after start so post-cold-load poll succeeds.
		if len(args) >= 1 && args[0] == "start" {
			tgt.isSleeping.Store(true)
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	if _, err := s.RedeployMember(context.Background(), "target"); err != nil {
		// Redeploy continues past peer-stop failure (best-effort) but the
		// test focuses on audit emission count.
		t.Logf("redeploy returned err (expected continuation): %v", err)
	}

	// Count audits per (peer, action) pair.
	type key struct {
		Model  string
		Action LifecycleAction
		Reason string
	}
	counts := map[key]int{}
	for _, e := range rec.snapshot() {
		counts[key{Model: e.Model, Action: e.Action, Reason: e.Reason}]++
	}

	// Failure-path peer audits — exactly ONE row of stop-failed-vram-drift-risk
	// for "peer", and ZERO rows of redeploy-member-peer-stop for "peer".
	if got := counts[key{Model: "peer", Action: LifecycleEvict, Reason: "stop-failed-vram-drift-risk"}]; got != 1 {
		t.Errorf("expected exactly 1 stop-failed-vram-drift-risk audit for peer, got %d", got)
	}
	if got := counts[key{Model: "peer", Action: LifecycleEvict, Reason: "redeploy-member-peer-stop"}]; got != 0 {
		t.Errorf("expected 0 redeploy-member-peer-stop audits for peer (Fix 11: stop failure must not double-emit), got %d", got)
	}
}

// ---------------------------------------------------------------------------
// Fix 12 — sleep-API errored but vLLM is actually sleeping → wedge-recovery.
// ---------------------------------------------------------------------------

// TestFix12_SleepAPI_Errored_VLLM_IsSleeping_FlipsToSleeping asserts
// that when sc.Sleep returns an error but the post-error /is_sleeping
// probe confirms vLLM IS sleeping, sleepInstance:
//   - flips state to StateSleeping (NOT rolls back to prevState=Ready)
//   - bumps SleepsTotal{reason="sleep-api-error-but-vllm-asleep"}
//   - calls NotifySleep on the admission controller
func TestFix12_SleepAPI_Errored_VLLM_IsSleeping_FlipsToSleeping(t *testing.T) {
	// Spin a fake vLLM where /sleep returns 500 but /is_sleeping returns true
	// — the wedge case. Reuses failingVLLMServer's "sleep-500" scenario
	// which has exactly that shape.
	srv := failingVLLMServer("sleep-500")
	defer srv.Close()

	s, a, _ := makeVLLMFaultScheduler(t, srv.URL, StateReady, admissionAwake)

	preSleepsWedge := counterValue(metrics.SleepsTotal, "moe", "sleep-api-error-but-vllm-asleep")
	preAwakeMB := a.SnapshotBudgets()[0].AwakeMB

	ev := &SchedulerEvictor{S: s}
	err := ev.SleepForEviction(context.Background(), "moe", "test-wedge")
	if err == nil {
		t.Fatalf("expected error from SleepForEviction (sleep API errored), got nil")
	}

	// State: should be StateSleeping (wedge-recovered), NOT StateReady.
	s.mu.RLock()
	got := s.instances["moe"].state
	s.mu.RUnlock()
	if got != StateSleeping {
		t.Errorf("Fix 12: expected wedge-recovery to flip state to StateSleeping, got %v", got)
	}

	// SleepsTotal{reason=sleep-api-error-but-vllm-asleep} bumped by 1.
	postSleepsWedge := counterValue(metrics.SleepsTotal, "moe", "sleep-api-error-but-vllm-asleep")
	if got := postSleepsWedge - preSleepsWedge; got != 1 {
		t.Errorf("Fix 12: expected SleepsTotal{reason=sleep-api-error-but-vllm-asleep}=+1, got +%v", got)
	}

	// admission books — the call was via SchedulerEvictor → sleepInstance with
	// reason="test-wedge" (NOT admissionReason), so sleepInstance does call
	// NotifySleep on success/wedge-recovery. After the wedge-recovery,
	// admission moved from Awake to Sleeping. Awake books should drop.
	postAwakeMB := a.SnapshotBudgets()[0].AwakeMB
	if postAwakeMB >= preAwakeMB {
		t.Errorf("Fix 12: expected admission Awake books to drop after wedge-recovery, pre=%d post=%d", preAwakeMB, postAwakeMB)
	}
}

// ---------------------------------------------------------------------------
// Fix 13 — admission.mu must NOT be held during slow evictor calls.
// ---------------------------------------------------------------------------

// slowStubEvictor blocks SleepForEviction for the supplied duration.
// Used by TestFix13_AdmissionMuNotHeldDuringEviction to detect whether
// concurrent admission reads can proceed while an eviction is in flight.
type slowStubEvictor struct {
	sleepWait time.Duration
}

func (s *slowStubEvictor) SleepForEviction(_ context.Context, _, _ string) error {
	time.Sleep(s.sleepWait)
	return nil
}

func (s *slowStubEvictor) StopForEviction(_ context.Context, _, _ string) error {
	time.Sleep(s.sleepWait)
	return nil
}

// TestFix13_AdmissionMuNotHeldDuringEviction asserts that while a
// RequestWake call is blocked inside the evictor (a 60s-realistic
// drain+sleep+settle on prod), concurrent IsStopped calls return
// immediately rather than blocking on a.mu.
//
// Pre-fix: a.mu was held with `defer Unlock` for the entire RequestWake
// scope, so a 2s evictor stalled every IsStopped call for 2s.
// Post-fix: a.mu is RELEASED across the evictor call, so IsStopped
// returns within ~ms.
func TestFix13_AdmissionMuNotHeldDuringEviction(t *testing.T) {
	// Two models, both on GPU 0, both 4000MB; one Awake, one wants to wake.
	// Total GPU = 6000MB so the wake forces an eviction.
	models := map[string]config.ModelConfig{
		"victim": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
			SleepL1ResidualMB:    100,
			Priority:             config.PriorityNormal,
		},
		"waker": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
			SleepL1ResidualMB:    100,
			Priority:             config.PriorityNormal,
		},
	}
	cfg := makeCfg(models)
	ev := &slowStubEvictor{sleepWait: 2 * time.Second}
	a := NewAdmissionController(cfg, map[int]int{0: 6000}, ev)

	// Drive victim to Awake so it's evictable.
	a.mu.Lock()
	a.awakeByGPU[0] += a.models["victim"].ExpectedOn(0)
	a.models["victim"].State = admissionAwake
	a.mu.Unlock()

	// Kick a RequestWake on a separate goroutine — it'll sit inside the
	// evictor for 2s.
	wakeStarted := make(chan struct{})
	wakeDone := make(chan struct{})
	go func() {
		close(wakeStarted)
		_, _ = a.RequestWake(context.Background(), "waker")
		close(wakeDone)
	}()
	<-wakeStarted

	// Wait briefly for RequestWake to enter the evictor (it has to walk
	// through pickVictims first; ~ms). 50ms is generous.
	time.Sleep(50 * time.Millisecond)

	// While the wake's evictor is asleep, hammer IsStopped from N goroutines.
	// EVERY call must complete in well under the evictor's 2s sleepWait.
	const N = 10
	const probeBudget = 200 * time.Millisecond
	durs := make([]time.Duration, N)
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		i := i
		go func() {
			defer wg.Done()
			start := time.Now()
			_ = a.IsStopped("victim")
			durs[i] = time.Since(start)
		}()
	}
	wg.Wait()

	for i, d := range durs {
		if d > probeBudget {
			t.Errorf("IsStopped[%d] took %v (>%v) — admission.mu was held during eviction (Fix 13 regressed)", i, d, probeBudget)
		}
	}

	// Wait for the wake to complete cleanly.
	select {
	case <-wakeDone:
	case <-time.After(5 * time.Second):
		t.Fatalf("RequestWake did not complete within 5s — possible deadlock from Fix 13 lock-discipline error")
	}
}

// ---------------------------------------------------------------------------
// Fix 14 — coldLoadPollInterval is test-overridable.
// ---------------------------------------------------------------------------

// TestFix14_SetColdLoadPollIntervalForTest_OverrideRestores asserts the
// SetColdLoadPollIntervalForTest helper returns the prior value and
// updates the package-level atomic.
func TestFix14_SetColdLoadPollIntervalForTest_OverrideRestores(t *testing.T) {
	orig := time.Duration(coldLoadPollInterval.Load())
	prev := SetColdLoadPollIntervalForTest(123 * time.Millisecond)
	if prev != orig {
		t.Errorf("expected SetColdLoadPollIntervalForTest to return prior value %v, got %v", orig, prev)
	}
	if got := time.Duration(coldLoadPollInterval.Load()); got != 123*time.Millisecond {
		t.Errorf("expected coldLoadPollInterval=123ms after override, got %v", got)
	}
	// Restore.
	SetColdLoadPollIntervalForTest(orig)
	if got := time.Duration(coldLoadPollInterval.Load()); got != orig {
		t.Errorf("expected coldLoadPollInterval restored to %v, got %v", orig, got)
	}
}

// ---------------------------------------------------------------------------
// Helpers used above.
// ---------------------------------------------------------------------------

// (No-op) silence unused-import diagnostic when an httptest helper is
// referenced only via a tag-guarded path above — actually used below.
var _ = httptest.NewServer
var _ = http.StatusOK
var _ atomic.Int32
var _ = strings.Contains
