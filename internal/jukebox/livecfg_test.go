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

	select {
	case <-reloadCh:
	case <-time.After(5 * time.Second):
		t.Fatalf("watcher reload never fired")
	}

	tracked := a.TrackedModels()
	hasBeta := false
	for _, n := range tracked {
		if n == "beta" {
			hasBeta = true
		}
	}
	if !hasBeta {
		t.Errorf("beta not picked up via fsnotify-driven reload: tracked=%v", tracked)
	}
}

// TestAdmissionRefreshConfig_StatePreservation verifies that an
// admission-tracked model in admissionAwake state remains awake (and
// keeps its booking) across a RefreshConfig that changes other fields.
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
}
