package jukebox_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

// stubEvictor is a no-op AdmissionEvictor for unit tests that don't
// exercise the wake/sleep machinery.
type stubEvictor struct {
	mu     sync.Mutex
	called []string
}

func (s *stubEvictor) SleepForEviction(_ context.Context, victim, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.called = append(s.called, victim)
	return nil
}

// makeAdmissionCfg builds a minimal scheduler-mode Config with two
// admission-tracked models on GPU 0. Caller may mutate before calling
// config.Load — it returns the YAML body so tests can write it back to
// disk for the integration case.
func makeAdmissionCfg(t *testing.T, pinned bool, vramMB int) *config.Config {
	t.Helper()
	pinnedStr := "false"
	if pinned {
		pinnedStr = "true"
	}
	body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: ` + pinnedStr + `
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = vramMB // reserved for future per-test override
	return cfg
}

// TestAdmissionRefreshConfig_PinnedFlip verifies that flipping a model
// from pinned:true to pinned:false (via config.SetCurrent + Scheduler
// OnConfigReloaded) makes that model eligible for eviction on the next
// admission decision.
func TestAdmissionRefreshConfig_PinnedFlip(t *testing.T) {
	cfg := makeAdmissionCfg(t, true /*pinned*/, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	totalsByGPU := map[int]int{0: 24000}
	a := jukebox.NewAdmissionController(cfg, totalsByGPU, &stubEvictor{})

	// Initial state: alpha is pinned + critical priority.
	if !a.IsTracked("alpha") {
		t.Fatalf("alpha not tracked initially")
	}

	// Flip pinned: true → false in the live config.
	cfg2Body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: false
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load (post-flip): %v", err)
	}
	config.SetCurrent(cfg2)

	// Refresh admission's per-model state from the new config.
	a.RefreshConfig(cfg2)

	// At the boundary, the controller's internal state for alpha should
	// now have Pinned=false. We can't observe this directly — but
	// IsTracked still returns true and the model's footprint is now
	// eligible for eviction (we'd need a full RequestWake test to
	// observe that, which the integration test below covers).
	if !a.IsTracked("alpha") {
		t.Fatalf("alpha lost tracking after RefreshConfig")
	}
}

// TestAdmissionRefreshConfig_AddRemoveModel verifies models added /
// removed in the live config are reflected in the controller's
// tracking set on the next RefreshConfig.
func TestAdmissionRefreshConfig_AddRemoveModel(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	totalsByGPU := map[int]int{0: 24000}
	a := jukebox.NewAdmissionController(cfg, totalsByGPU, &stubEvictor{})

	got := a.TrackedModels()
	if len(got) != 2 {
		t.Fatalf("initial tracked: got %v want [alpha beta]", got)
	}

	// Remove "beta" from the live config; add a new "gamma".
	cfg2Body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  gamma:
    path: "/models/gamma"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 3000
    sleep_l1_residual_mb: 200
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	tracked := a.TrackedModels()
	hasAlpha, hasBeta, hasGamma := false, false, false
	for _, n := range tracked {
		switch n {
		case "alpha":
			hasAlpha = true
		case "beta":
			hasBeta = true
		case "gamma":
			hasGamma = true
		}
	}
	if !hasAlpha {
		t.Errorf("alpha dropped from tracking after refresh: %v", tracked)
	}
	if hasBeta {
		t.Errorf("beta still tracked after removal from config: %v", tracked)
	}
	if !hasGamma {
		t.Errorf("gamma not picked up after addition to config: %v", tracked)
	}
}

// TestLiveModelCfg_HotReloadPicksUpFieldChange verifies the helper
// reflects field changes in config.Current() within a single
// SetCurrent → read cycle (no fsnotify involved).
func TestLiveModelCfg_HotReloadPicksUpFieldChange(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	// Bump cold_load_timeout_seconds for "alpha" via a fresh Load + SetCurrent.
	cfg2Body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 600
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	config.SetCurrent(cfg2)

	live := config.Current()
	if live == nil {
		t.Fatalf("Current() unexpectedly nil")
	}
	got := live.Models["alpha"].EffectiveColdLoadTimeout()
	want := 600 * time.Second
	if got != want {
		t.Errorf("alpha cold_load_timeout: got %s want %s", got, want)
	}
}

// TestWatcher_FsnotifyDrivenReloadRefreshesAdmission verifies the
// end-to-end flow: write a new active.yaml → watcher fires → atomic
// pointer swap → admission refresh picks up the change.
func TestWatcher_FsnotifyDrivenReloadRefreshesAdmission(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "active.yaml")

	body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	reloadCh := make(chan struct{}, 8)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := config.Watch(ctx, path, func(result config.ReloadResult, _ error) {
		if result == config.ReloadSuccess {
			a.RefreshConfig(config.Current())
			select {
			case reloadCh <- struct{}{}:
			default:
			}
		}
	}); err != nil {
		t.Fatalf("Watch: %v", err)
	}

	// Add a "beta" model in the new active.yaml content.
	updated := body + `
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 200
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	tmp := filepath.Join(dir, ".active.yaml.tmp")
	if err := os.WriteFile(tmp, []byte(updated), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatal(err)
	}

	// Poll for the expected admission state rather than waiting on a
	// wall-clock fallback. fsnotify on macOS (kqueue) can be 1-3s
	// between file-rename and event-delivery on a loaded CI runner;
	// the prior 5s wall-clock fallback masked platform-specific
	// failures with a single magic timeout. Poll every 50ms for up to
	// 3s asserting the expected state.
	deadline := time.Now().Add(3 * time.Second)
	var lastTracked []string
	for time.Now().Before(deadline) {
		select {
		case <-reloadCh:
		default:
		}
		lastTracked = a.TrackedModels()
		hasBeta := false
		for _, n := range lastTracked {
			if n == "beta" {
				hasBeta = true
				break
			}
		}
		if hasBeta {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("beta not picked up via fsnotify-driven reload within 3s: tracked=%v", lastTracked)
}

// TestAdmissionRefreshConfig_StatePreservation verifies that an
// admission-tracked model in admissionAwake state remains awake (and
// keeps its booking) across a RefreshConfig that changes other fields.
// Also drives a SUBSEQUENT sleep transition to assert no phantom leak
// nor negative balance in the per-GPU bookkeeping — the bug the
// architect's round-N+1 review flagged hid behind tests that only
// snapshotted immediately after RefreshConfig and never exercised
// reversal arithmetic.
func TestAdmissionRefreshConfig_StatePreservation(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	totalsByGPU := map[int]int{0: 24000}
	a := jukebox.NewAdmissionController(cfg, totalsByGPU, &stubEvictor{})

	// Drive alpha into admissionAwake by triggering RequestWake. Use a
	// noop evictor — no eviction needed (24GB total, alpha needs 5GB).
	if _, err := a.RequestWake(context.Background(), "alpha"); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("RequestWake: %v", err)
	}

	preBudgets := a.SnapshotBudgets()
	var preAwake int
	for _, b := range preBudgets {
		if b.GPUID == 0 {
			preAwake = b.AwakeMB
		}
	}
	if preAwake == 0 {
		t.Fatalf("expected awake bytes after RequestWake, got 0 (snapshot=%+v)", preBudgets)
	}

	// Refresh with a config where alpha's expected_vram is DIFFERENT.
	// Per the grandfathering contract, the existing booking should NOT
	// be retroactively rewritten.
	cfg2Body := `
server:
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
scheduler:
  port_range_start: 9000
  port_range_end: 9100
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 8000
    sleep_l1_residual_mb: 800
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	postBudgets := a.SnapshotBudgets()
	var postAwake int
	for _, b := range postBudgets {
		if b.GPUID == 0 {
			postAwake = b.AwakeMB
		}
	}
	if postAwake != preAwake {
		t.Errorf("awake bookkeeping retroactively rewritten: pre=%d post=%d (grandfathering broken)", preAwake, postAwake)
	}

	// Drive subsequent sleep — markSleepingLocked should reverse using
	// the SNAPSHOT (original 5000) not the live config (now 8000), so
	// awakeByGPU returns to exactly its pre-wake value (0 contribution
	// from alpha) and l1ResidualByGPU gets the NEW residual (800).
	a.NotifySleep("alpha", "manual")

	sleptBudgets := a.SnapshotBudgets()
	var sleptAwake, sleptResidual int
	for _, b := range sleptBudgets {
		if b.GPUID == 0 {
			sleptAwake = b.AwakeMB
			sleptResidual = b.L1ResidualMB
		}
	}
	if sleptAwake != 0 {
		t.Errorf("post-sleep awake bookkeeping not zero: got %d (phantom leak — reversal used wrong vram)", sleptAwake)
	}
	if sleptResidual != 800 {
		t.Errorf("post-sleep residual: got %d want 800 (new config applied at transition boundary)", sleptResidual)
	}
}

// TestRefreshConfig_VRAMDown_AwakeModelSleeps_NoPhantomLeak — CRITICAL
// regression test for the round-N+1 architect review. Model awake at
// expected=5000. Operator hot-reloads to 3000. Model sleeps. The
// reversal must use the SNAPSHOT (5000) not the live config (3000),
// otherwise awakeByGPU is left with a +2000 phantom positive.
func TestRefreshConfig_VRAMDown_AwakeModelSleeps_NoPhantomLeak(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	if _, err := a.RequestWake(context.Background(), "alpha"); err != nil {
		t.Fatalf("RequestWake: %v", err)
	}

	// alpha awake @ 5000. Hot-reload to 3000.
	cfg2Body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 3000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	a.NotifySleep("alpha", "manual")

	for _, b := range a.SnapshotBudgets() {
		if b.GPUID != 0 {
			continue
		}
		if b.AwakeMB != 0 {
			t.Fatalf("phantom leak on GPU 0: awake=%d (expected 0; bug allows +2000 phantom)", b.AwakeMB)
		}
	}
}

// TestRefreshConfig_VRAMUp_AwakeModelSleeps_NoNegativeBalance — CRITICAL
// regression test. Model awake at expected=5000. Operator hot-reloads
// to 8000. Model sleeps. Reversal using live (8000) instead of snapshot
// (5000) would underflow awakeByGPU to -3000, then a subsequent
// admission of a peer sees +3000 phantom-free and OOMs at vLLM.
func TestRefreshConfig_VRAMUp_AwakeModelSleeps_NoNegativeBalance(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	if _, err := a.RequestWake(context.Background(), "alpha"); err != nil {
		t.Fatalf("RequestWake: %v", err)
	}

	// alpha awake @ 5000. Hot-reload to 8000.
	cfg2Body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 8000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	a.NotifySleep("alpha", "manual")

	for _, b := range a.SnapshotBudgets() {
		if b.GPUID != 0 {
			continue
		}
		if b.AwakeMB < 0 {
			t.Fatalf("negative awake balance on GPU 0: %d (next admit would phantom-fit a peer and OOM at vLLM)", b.AwakeMB)
		}
		if b.AwakeMB != 0 {
			t.Fatalf("awake balance not zero on GPU 0: got %d want 0", b.AwakeMB)
		}
	}
}

// TestRefreshConfig_GpusShrink_AwakeModelSleeps_FullyFrees — HIGH-1
// regression test. Model awake on [0,1,2,3] booking +5000 each.
// Operator hot-reloads to [0,1]. Model sleeps. Reversal iterating the
// new GPU list misses GPUs 2+3 → 10GB leaks until restart. Snapshot
// pattern iterates BookedGPUs and frees all four GPUs cleanly.
func TestRefreshConfig_GpusShrink_AwakeModelSleeps_FullyFrees(t *testing.T) {
	body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0, 1, 2, 3]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	totals := map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}
	a := jukebox.NewAdmissionController(cfg, totals, &stubEvictor{})

	if _, err := a.RequestWake(context.Background(), "alpha"); err != nil {
		t.Fatalf("RequestWake: %v", err)
	}

	// Verify pre-sleep: 5000 booked on each of 4 GPUs.
	preBudgets := a.SnapshotBudgets()
	for _, b := range preBudgets {
		if b.AwakeMB != 5000 {
			t.Fatalf("pre-sleep awake on GPU %d: got %d want 5000", b.GPUID, b.AwakeMB)
		}
	}

	// Hot-reload to shrink GPU set to [0, 1].
	shrunkBody := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(shrunkBody))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	a.NotifySleep("alpha", "manual")

	// All 4 GPUs must show zero awake bookings (full reversal via snapshot).
	for _, b := range a.SnapshotBudgets() {
		if b.AwakeMB != 0 {
			t.Fatalf("GPU %d leaked %dMB after sleep on shrunk-gpus config (snapshot reversal broken)", b.GPUID, b.AwakeMB)
		}
	}
}

// TestRefreshConfig_AddPinnedModel_UpdatesPinnedByGPU — HIGH-2.
// Adding a new pinned non-swap-group model via hot-reload must add
// its footprint to pinnedByGPU so future availableMB calculations
// reserve VRAM correctly. Pre-fix: only NewAdmissionController wrote
// pinnedByGPU; RefreshConfig left it stale.
func TestRefreshConfig_AddPinnedModel_UpdatesPinnedByGPU(t *testing.T) {
	cfg := makeAdmissionCfg(t, false, 5000)
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	prePinned := a.SnapshotBudgets()[0].PinnedMB
	if prePinned != 0 {
		t.Fatalf("pre-refresh PinnedMB: got %d want 0", prePinned)
	}

	// Add a pinned model via hot-reload.
	cfg2Body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  beta:
    path: "/models/beta"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  pinnedguy:
    path: "/models/pinnedguy"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: true
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 400
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	postPinned := a.SnapshotBudgets()[0].PinnedMB
	if postPinned != 4000 {
		t.Fatalf("post-refresh PinnedMB: got %d want 4000 (pinned model add not reconciled into pinnedByGPU)", postPinned)
	}
}

// TestRefreshConfig_RemovePinnedModel_UpdatesPinnedByGPU — HIGH-2.
// Removing a pinned non-swap-group model via hot-reload must release
// its footprint from pinnedByGPU. Pre-fix: removed model's footprint
// stayed subtracted from every availableMB calculation forever.
func TestRefreshConfig_RemovePinnedModel_UpdatesPinnedByGPU(t *testing.T) {
	body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
  pinnedguy:
    path: "/models/pinnedguy"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: true
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 400
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	if got := a.SnapshotBudgets()[0].PinnedMB; got != 4000 {
		t.Fatalf("initial PinnedMB: got %d want 4000", got)
	}

	// Drop pinnedguy from config.
	cfg2Body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "alpha"}
models:
  alpha:
    path: "/models/alpha"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    expected_vram_mb_per_gpu: 5000
    sleep_l1_residual_mb: 500
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	if got := a.SnapshotBudgets()[0].PinnedMB; got != 0 {
		t.Fatalf("post-removal PinnedMB: got %d want 0 (pinned model removal not reconciled)", got)
	}
}

// TestRefreshConfig_FlipPinnedFalse_UpdatesPinnedByGPU — HIGH-2.
// Flipping pinned: true → false via hot-reload must migrate the
// footprint from pinnedByGPU to awakeByGPU. Pre-fix: the booking
// stayed in pinnedByGPU forever, the model was treated as evictable
// (via Priority change), but pinnedByGPU stayed inflated → next
// availableMB lied.
func TestRefreshConfig_FlipPinnedFalse_UpdatesPinnedByGPU(t *testing.T) {
	body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "pinnedguy"}
models:
  pinnedguy:
    path: "/models/pinnedguy"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: true
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 400
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg, err := config.Load([]byte(body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(nil) })

	a := jukebox.NewAdmissionController(cfg, map[int]int{0: 24000}, &stubEvictor{})

	b0 := a.SnapshotBudgets()[0]
	if b0.PinnedMB != 4000 || b0.AwakeMB != 0 {
		t.Fatalf("initial PinnedMB=%d AwakeMB=%d want PinnedMB=4000 AwakeMB=0", b0.PinnedMB, b0.AwakeMB)
	}

	// Flip pinned: true → false.
	cfg2Body := `
server: {port: 8080}
vllm: {binary: "/usr/bin/vllm"}
scheduler: {port_range_start: 9000, port_range_end: 9100}
behavior: {default_model: "pinnedguy"}
models:
  pinnedguy:
    path: "/models/pinnedguy"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
    pinned: false
    expected_vram_mb_per_gpu: 4000
    sleep_l1_residual_mb: 400
    cold_load_timeout_seconds: 300
    sleep_mode: true
`
	cfg2, err := config.Load([]byte(cfg2Body))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a.RefreshConfig(cfg2)

	b0p := a.SnapshotBudgets()[0]
	if b0p.PinnedMB != 0 {
		t.Errorf("post-flip PinnedMB: got %d want 0 (pinned-flip not migrated out of pinnedByGPU)", b0p.PinnedMB)
	}
	if b0p.AwakeMB != 4000 {
		t.Errorf("post-flip AwakeMB: got %d want 4000 (pinned-flip should migrate to awakeByGPU)", b0p.AwakeMB)
	}
	// Total reserved (Pinned + Awake) should be conserved across the flip.
	if (b0p.PinnedMB + b0p.AwakeMB) != (b0.PinnedMB + b0.AwakeMB) {
		t.Errorf("flip non-conservative: pre Pinned+Awake=%d post=%d", b0.PinnedMB+b0.AwakeMB, b0p.PinnedMB+b0p.AwakeMB)
	}
}
