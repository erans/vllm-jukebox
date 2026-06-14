package jukebox

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// phantomFakeMgr is a fakeRedeployMgr variant that returns a non-empty
// BaseURL so the post-wake verify probe actually runs against it. The
// probe outcome is injected via SetWakeVerifyProbeForTest in the test
// body; this fake exists only to provide a non-empty BaseURL because
// the production wake path SKIPS the probe when BaseURL=="" (the
// no-URL-no-probe gate that preserves backward compat with existing
// test fakes).
type phantomFakeMgr struct {
	fakeRedeployMgr
	url string
}

func (m *phantomFakeMgr) BaseURL() string { return m.url }

// TestPerformWake_PhantomWakeDetected proves that when /wake_up returns
// 200 (sc.Wake reports nil) but the post-wake verify probe reports a
// wedged engine, the wake path:
//  1. Logs wake_phantom_detected (via the lifecycle audit sink we install).
//  2. Does NOT flip inst.state to Ready (state stays Sleeping).
//  3. Rolls back admission via NotifySleep (so the awake-by-GPU budget
//     doesn't lie about a wedged peer holding VRAM it can't serve).
//  4. Schedules an async RedeployMember for the phantom peer.
//  5. Returns a non-nil error to the caller (so the consumer-facing
//     proxy returns a 503 instead of routing into the wedge with a
//     30s+ timeout).
//
// This is the defense for the vllm-project cumem_tag /wake_up phantom
// failure mode where /health continues to return 200 but every decode
// hangs forever waiting on the shm_broadcast block.
func TestPerformWake_PhantomWakeDetected(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
models:
  vllm-main:
    lifecycle: external
    host: vllm-main
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 0
    wake_verify_timeout_ms: 1000
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 8000}
	s, _, _ := makeRedeployScheduler(t, cfgYAML, nil, totals)
	_ = cfg

	mgr := &phantomFakeMgr{
		fakeRedeployMgr: fakeRedeployMgr{port: 8002},
		url:             "http://phantom-target.test:8002",
	}
	mgr.pid.Store(int64(7000 + 8002))
	mgr.isSleeping.Store(true)
	s.SeedInstanceForTest("vllm-main", 8002, []int{0}, false, StateSleeping, mgr)

	// Inject phantom probe outcome — simulates the cumem wedge where
	// /v1/completions hangs / returns error after /wake_up returned 200.
	probeFired := atomic.Int32{}
	restoreProbe := SetWakeVerifyProbeForTest(func(_ context.Context, baseURL, model string, _ time.Duration) error {
		probeFired.Add(1)
		if !strings.HasPrefix(baseURL, "http://phantom-target.test") {
			t.Errorf("probe received unexpected baseURL %q", baseURL)
		}
		if model != "vllm-main" {
			t.Errorf("probe received unexpected model %q", model)
		}
		return errors.New("simulated wedged engine — shm_broadcast block missing")
	})
	defer restoreProbe()

	// Record that recreate was scheduled. Using the sync-test hook
	// avoids goroutine timing flake (production spawns a 10-min
	// background recreate; tests assert it WAS scheduled, not that
	// it ran to completion).
	recreates := make(chan string, 4)
	restoreRedeploy := SetRedeployMemberAsyncForTest(func(model string) {
		recreates <- model
	})
	defer restoreRedeploy()

	// Audit sink to count emitted phantom-wake lifecycle events.
	var phantomAudits atomic.Int32
	prevAudit := SetLifecycleAuditSinkForTest(func(ev LifecycleEvent) {
		if ev.Action == LifecycleWake && ev.Reason == "phantom-wake-detected" && ev.Model == "vllm-main" {
			phantomAudits.Add(1)
		}
	})
	defer SetLifecycleAuditSinkForTest(prevAudit)

	targetInst := s.instances["vllm-main"]
	targetCfg := s.cfg.Models["vllm-main"]

	err = s.performWake(context.Background(), targetInst, targetCfg, "test-phantom")
	if err == nil {
		t.Fatalf("expected wake to return error after phantom-detection; got nil")
	}
	if !strings.Contains(err.Error(), "phantom wake") {
		t.Errorf("expected error message to mention 'phantom wake'; got %q", err.Error())
	}

	if got := probeFired.Load(); got != 1 {
		t.Errorf("expected verify probe to fire exactly once; got %d", got)
	}

	// State must NOT be Ready — the wake never completed.
	if st, ok := s.InstanceStateForTest("vllm-main"); !ok {
		t.Fatalf("instance lookup failed")
	} else if st == StateReady {
		t.Errorf("expected state != StateReady after phantom-wake; got %s", st)
	}

	// sc.Wake was called once (the /wake_up POST); fake records that.
	if got := mgr.wakeCalls.Load(); got != 1 {
		t.Errorf("expected fake sc.Wake to be called once; got %d", got)
	}

	// Recreate must have been scheduled exactly once for the phantom peer.
	select {
	case got := <-recreates:
		if got != "vllm-main" {
			t.Errorf("recreate scheduled for unexpected model %q (want vllm-main)", got)
		}
	case <-time.After(2 * time.Second):
		t.Errorf("expected RedeployMember to be scheduled for vllm-main after phantom-wake; nothing fired in 2s")
	}

	if got := phantomAudits.Load(); got != 1 {
		t.Errorf("expected exactly one phantom-wake-detected lifecycle audit; got %d", got)
	}
}

// TestPerformWake_HealthyWakeNotPhantom proves that a successful wake
// where the verify probe returns nil follows the normal path:
//  1. inst.state flips to StateReady.
//  2. NotifyWakeComplete fires (no rollback).
//  3. No recreate is scheduled.
//  4. No phantom-wake audit is emitted.
//
// The negative path is just as important as the positive — a probe that
// always-fires-fail would still pass TestPerformWake_PhantomWakeDetected
// but would render every healthy hot-wake a phantom in production
// (false-positive storm forcing 5min cold-loads after every 1-2s hot
// wake). This test guards the false-positive direction.
func TestPerformWake_HealthyWakeNotPhantom(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
models:
  vllm-main:
    lifecycle: external
    host: vllm-main
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 0
    wake_verify_timeout_ms: 1000
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 8000}
	s, _, _ := makeRedeployScheduler(t, cfgYAML, nil, totals)
	_ = cfg

	mgr := &phantomFakeMgr{
		fakeRedeployMgr: fakeRedeployMgr{port: 8002},
		url:             "http://healthy-target.test:8002",
	}
	mgr.pid.Store(int64(7000 + 8002))
	mgr.isSleeping.Store(true)
	s.SeedInstanceForTest("vllm-main", 8002, []int{0}, false, StateSleeping, mgr)

	probeFired := atomic.Int32{}
	restoreProbe := SetWakeVerifyProbeForTest(func(_ context.Context, _, _ string, _ time.Duration) error {
		probeFired.Add(1)
		return nil // healthy decode
	})
	defer restoreProbe()

	// Counter for recreate triggers. Must remain zero for a healthy wake.
	recreates := atomic.Int32{}
	restoreRedeploy := SetRedeployMemberAsyncForTest(func(_ string) {
		recreates.Add(1)
	})
	defer restoreRedeploy()

	// Audit sink to assert that NO phantom-wake event is emitted.
	var phantomAudits atomic.Int32
	var normalWakeAudits atomic.Int32
	prevAudit := SetLifecycleAuditSinkForTest(func(ev LifecycleEvent) {
		if ev.Action != LifecycleWake || ev.Model != "vllm-main" {
			return
		}
		if ev.Reason == "phantom-wake-detected" {
			phantomAudits.Add(1)
		} else {
			normalWakeAudits.Add(1)
		}
	})
	defer SetLifecycleAuditSinkForTest(prevAudit)

	targetInst := s.instances["vllm-main"]
	targetCfg := s.cfg.Models["vllm-main"]

	if err := s.performWake(context.Background(), targetInst, targetCfg, "test-healthy"); err != nil {
		t.Fatalf("expected wake to succeed on healthy probe; got %v", err)
	}

	if got := probeFired.Load(); got != 1 {
		t.Errorf("expected verify probe to fire exactly once; got %d", got)
	}
	if got := mgr.wakeCalls.Load(); got != 1 {
		t.Errorf("expected fake sc.Wake to be called once; got %d", got)
	}
	if st, ok := s.InstanceStateForTest("vllm-main"); !ok {
		t.Fatalf("instance lookup failed")
	} else if st != StateReady {
		t.Errorf("expected state == StateReady after healthy wake; got %s", st)
	}
	if got := recreates.Load(); got != 0 {
		t.Errorf("expected NO recreate to be scheduled on healthy wake; got %d", got)
	}
	if got := phantomAudits.Load(); got != 0 {
		t.Errorf("expected NO phantom-wake audit on healthy wake; got %d", got)
	}
	if got := normalWakeAudits.Load(); got != 1 {
		t.Errorf("expected exactly one normal wake audit; got %d", got)
	}
}

// TestPerformWake_VerifyProbeDisabledByConfig proves that setting
// wake_verify_timeout_ms: -1 in the model config skips the probe
// entirely — preserves the pre-defense behavior for operators who
// explicitly opt out (e.g. non-cumem-sleep models where the phantom
// mode is not reachable). With the probe disabled, even a
// SetWakeVerifyProbeForTest hook that would have flagged phantom is
// NOT consulted; wake proceeds normally.
func TestPerformWake_VerifyProbeDisabledByConfig(t *testing.T) {
	cfgYAML := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 100ms
models:
  vllm-main:
    lifecycle: external
    host: vllm-main
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 0
    wake_verify_timeout_ms: -1
`
	_, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totals := map[int]int{0: 8000}
	s, _, _ := makeRedeployScheduler(t, cfgYAML, nil, totals)

	mgr := &phantomFakeMgr{
		fakeRedeployMgr: fakeRedeployMgr{port: 8002},
		url:             "http://disabled-target.test:8002",
	}
	mgr.pid.Store(int64(7000 + 8002))
	mgr.isSleeping.Store(true)
	s.SeedInstanceForTest("vllm-main", 8002, []int{0}, false, StateSleeping, mgr)

	probeFired := atomic.Int32{}
	restoreProbe := SetWakeVerifyProbeForTest(func(_ context.Context, _, _ string, _ time.Duration) error {
		probeFired.Add(1)
		return errors.New("probe would-have-flagged-phantom but should not be called")
	})
	defer restoreProbe()

	targetInst := s.instances["vllm-main"]
	targetCfg := s.cfg.Models["vllm-main"]
	if got := targetCfg.EffectiveWakeVerifyTimeout(); got != 0 {
		t.Fatalf("expected EffectiveWakeVerifyTimeout==0 for -1 config; got %s", got)
	}

	if err := s.performWake(context.Background(), targetInst, targetCfg, "test-disabled"); err != nil {
		t.Fatalf("expected wake to succeed when probe is disabled; got %v", err)
	}
	if got := probeFired.Load(); got != 0 {
		t.Errorf("expected probe NOT to fire when disabled; got %d", got)
	}
	if st, ok := s.InstanceStateForTest("vllm-main"); !ok || st != StateReady {
		t.Errorf("expected StateReady when probe disabled; got %v ok=%v", st, ok)
	}
}
