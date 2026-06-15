package jukebox_test

// Startup-probe tests for RegisterExternalInstances. Specifically the
// "container unreachable at boot" branch — historically this defaulted
// to admissionSleeping with a "will reconcile on first request" warning,
// which is wrong for evict_action: stop members: the next consumer
// request would hit /wake_up against a dead address and return DNS
// failure / connection refused. The fix marks evict_action: stop members
// admissionStopped on unreachable so the proxy's async-503 + KickColdLoad
// path takes over and `docker start`s the container before any wake.

import (
	"context"
	"fmt"
	"net"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

// closedAddrFromHTTPTest returns host+port for a temporary httptest.Server
// that is then immediately Close()'d. Subsequent connection attempts to
// the returned address get "connection refused" — deterministic simulation
// of an Exited container at the same host:port the operator's compose
// declares. Cleaner than picking a random port and praying nothing else
// binds it during the test.
func closedAddrFromHTTPTest(t *testing.T) (host string, port int) {
	t.Helper()
	srv := httptest.NewServer(nil)
	url := srv.URL
	srv.Close()
	hp := strings.TrimPrefix(url, "http://")
	h, p, err := net.SplitHostPort(hp)
	if err != nil {
		t.Fatalf("split host/port from %q: %v", hp, err)
	}
	n, err := strconv.Atoi(p)
	if err != nil {
		t.Fatalf("parse port %q: %v", p, err)
	}
	return h, n
}

// buildStopEvictConfig builds a two-model swap-group config: one pinned
// member (always-awake default) plus one transient evict_action: stop
// peer pointing at the given host:port. The validator requires both for
// evict_action: stop to be accepted.
func buildStopEvictConfig(t *testing.T, transientHost string, transientPort int, pinnedHost string, pinnedPort int) *config.Config {
	t.Helper()
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
  pinned-default:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    pinned: true
    expected_vram_mb_per_gpu: 1000
    swap_group: g1
  transient-stop:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    priority: normal
    expected_vram_mb_per_gpu: 1000
    swap_group: g1
    evict_action: stop
`, pinnedHost, pinnedPort, transientHost, transientPort)
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	return cfg
}

// buildSleepEvictConfig is the negative control: same shape as
// buildStopEvictConfig but transient uses default evict_action (sleep).
// Used to assert we DON'T regress the existing "unreachable → stays
// StateReady, no NotifyStopped" behavior for non-stop-on-evict members.
func buildSleepEvictConfig(t *testing.T, transientHost string, transientPort int, pinnedHost string, pinnedPort int) *config.Config {
	t.Helper()
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
  pinned-default:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    pinned: true
    expected_vram_mb_per_gpu: 1000
    swap_group: g1
  transient-sleep:
    lifecycle: external
    host: "%s"
    port: %d
    gpus: [0]
    min_free_mem_mb_per_gpu: 1
    sleep_mode: true
    sleep_level: 1
    wake_timeout: 3s
    priority: normal
    expected_vram_mb_per_gpu: 1000
    swap_group: g1
`, pinnedHost, pinnedPort, transientHost, transientPort)
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	return cfg
}

// TestRegisterExternalInstances_UnreachableEvictStopMarksAdmissionStopped
// validates the bugfix: when a startup probe fails AND the model has
// evict_action: stop, jukebox marks the model admissionStopped (not
// admissionSleeping/StateReady) so the proxy's async-503 + KickColdLoad
// path will docker-start the container before any /wake_up.
func TestRegisterExternalInstances_UnreachableEvictStopMarksAdmissionStopped(t *testing.T) {
	// Pinned default is a real, reachable fake — only the transient
	// (evict_action: stop) member is unreachable. Mirrors production: the
	// pinned daily-driver is up, the rarely-woken peer is Exited.
	pinnedFake := newFakeVLLM("pinned-default")
	defer pinnedFake.Close()

	transHost, transPort := closedAddrFromHTTPTest(t)

	cfg := buildStopEvictConfig(t, transHost, transPort, pinnedFake.Host(), pinnedFake.Port())
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	totalsByGPU := map[int]int{0: 24000}
	adm := jukebox.NewAdmissionController(cfg, totalsByGPU, &jukebox.SchedulerEvictor{S: s})
	s.SetAdmission(adm)

	// Bound the call so a misbehaving probe path can't hang the test.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.RegisterExternalInstances(ctx); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	if !adm.IsStopped("transient-stop") {
		t.Fatalf("expected transient-stop to be admissionStopped after unreachable boot probe, but IsStopped=false")
	}

	// Negative: the reachable pinned member must NOT be marked stopped.
	if adm.IsStopped("pinned-default") {
		t.Fatalf("pinned-default is reachable; should NOT be admissionStopped")
	}
}

// TestRegisterExternalInstances_UnreachableNonStopStaysSleeping is the
// negative control: when a startup probe fails for a model with the
// DEFAULT evict_action (sleep), we must keep the existing behavior
// (don't flip to admissionStopped — admission stays in its initial
// state, which for non-pinned members is admissionSleeping per
// NewAdmissionController init). The "will reconcile on first request"
// path takes over.
func TestRegisterExternalInstances_UnreachableNonStopStaysSleeping(t *testing.T) {
	pinnedFake := newFakeVLLM("pinned-default")
	defer pinnedFake.Close()

	transHost, transPort := closedAddrFromHTTPTest(t)

	cfg := buildSleepEvictConfig(t, transHost, transPort, pinnedFake.Host(), pinnedFake.Port())
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	totalsByGPU := map[int]int{0: 24000}
	adm := jukebox.NewAdmissionController(cfg, totalsByGPU, &jukebox.SchedulerEvictor{S: s})
	s.SetAdmission(adm)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.RegisterExternalInstances(ctx); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	if adm.IsStopped("transient-sleep") {
		t.Fatalf("expected transient-sleep (default evict_action=sleep) NOT to be admissionStopped on unreachable boot probe; got IsStopped=true")
	}
}

// TestRegisterExternalInstances_PinnedAsleepAutoRestoresOnBoot validates
// the boot-pinned-restore fix: a pinned model found asleep at startup
// (sleepProbeOK=true, sleeping=true) violates the pinned invariant
// ("never auto-sleep on idle; always be ready to serve"). The previous
// behavior left it Sleeping until the first request, surfacing as
// /status reporting pinned=true + state=sleeping while operators expect
// the pinned default to be hot. The fix dispatches a background wake
// from the boot-probe sleeping arm when the model is pinned. Non-pinned
// peers are left Sleeping (operator may have chosen a different swap-
// group member as the awake one). The test starts a pinned fake in the
// SLEEPING state, runs the boot probe, then waits for the fake's
// /wake_up counter to flip from 0 → 1 within a generous timeout. Before
// the fix this loops forever.
func TestRegisterExternalInstances_PinnedAsleepAutoRestoresOnBoot(t *testing.T) {
	pinnedFake := newFakeVLLM("pinned-default")
	defer pinnedFake.Close()
	// Seed the fake as ALREADY sleeping at boot — exactly the scenario
	// the latency-decomp agent observed in production (Qwen3.6-27B
	// state=sleeping immediately after jukebox container start).
	pinnedFake.sleeping.Store(true)

	// Non-pinned swap-group peer (also sleeping) — used to assert we
	// don't regress the explicit "boot-probe is NOT idle, don't auto-
	// restore peers" comment. This peer must STAY sleeping.
	peerFake := newFakeVLLM("transient-sleep")
	defer peerFake.Close()
	peerFake.sleeping.Store(true)

	cfg := buildSleepEvictConfig(t, peerFake.Host(), peerFake.Port(), pinnedFake.Host(), pinnedFake.Port())
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	totalsByGPU := map[int]int{0: 24000}
	adm := jukebox.NewAdmissionController(cfg, totalsByGPU, &jukebox.SchedulerEvictor{S: s})
	s.SetAdmission(adm)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.RegisterExternalInstances(ctx); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	// The pinned-restore wake is dispatched in a goroutine OUTSIDE the
	// 10s probe context, so it runs asynchronously after Register
	// returns. Poll the fake's /wake_up hit counter until it flips,
	// bounded by a generous timeout so a real regression fails fast
	// rather than hanging the test pool.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if pinnedFake.wakeHits.Load() >= 1 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got := pinnedFake.wakeHits.Load(); got < 1 {
		t.Fatalf("pinned-default: expected boot-pinned-restore to call /wake_up at least once within 10s; got wakeHits=%d (regression: pinned model left Sleeping at boot)", got)
	}

	// Negative: the non-pinned swap-group peer must NOT have been
	// auto-restored. The comment block at the boot-probe sleeping arm
	// explicitly says non-pinned + sleeping = operator-chosen, leave it.
	if got := peerFake.wakeHits.Load(); got != 0 {
		t.Fatalf("transient-sleep: expected NO boot-restore for non-pinned peer; got wakeHits=%d (regression: clobbered operator-chosen sleeping peer)", got)
	}
}

// TestRegisterExternalInstances_PinnedUnreachableMarksStopped is the
// Bug #2 (boot-adopt cold-load storm) regression. A PINNED external
// model with the DEFAULT (sleep) evict_action whose container is DOWN at
// boot must be seeded admissionStopped — NOT left at the optimistic
// seeded StateReady, which had admission booking the model's full awake
// VRAM footprint for a dead container. That phantom-awake accounting was
// the precursor to the storm: a later proactive wake cold-loaded the
// pinned model onto a GPU still held by a healthy resident → OOM cascade.
// Seeding Stopped frees the phantom booking and defers recovery to the
// demand-driven async-503 + KickColdLoad path (which now honors the
// never-evict-a-healthy-higher/equal-priority-resident guard).
//
// Contrast with TestRegisterExternalInstances_UnreachableNonStopStaysSleeping:
// a NON-pinned sleep-evict model is intentionally left to reconcile on
// first request; only the pinned case flips to Stopped at boot.
func TestRegisterExternalInstances_PinnedUnreachableMarksStopped(t *testing.T) {
	// Pinned default is UNREACHABLE (closed addr); the transient peer is
	// a reachable fake so we can assert the negative (it is NOT flipped
	// to Stopped by this pinned-only change).
	transientFake := newFakeVLLM("transient-sleep")
	defer transientFake.Close()

	pinnedHost, pinnedPort := closedAddrFromHTTPTest(t)

	cfg := buildSleepEvictConfig(t, transientFake.Host(), transientFake.Port(), pinnedHost, pinnedPort)
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: 24000, FreeMB: 24000}}}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	totalsByGPU := map[int]int{0: 24000}
	adm := jukebox.NewAdmissionController(cfg, totalsByGPU, &jukebox.SchedulerEvictor{S: s})
	s.SetAdmission(adm)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := s.RegisterExternalInstances(ctx); err != nil {
		t.Fatalf("RegisterExternalInstances: %v", err)
	}

	if !adm.IsStopped("pinned-default") {
		t.Fatalf("Bug #2: pinned-default is unreachable at boot; expected admissionStopped (freed phantom-awake booking), got IsStopped=false")
	}
	// Negative: the reachable transient peer must NOT be marked Stopped
	// by the boot probe.
	if adm.IsStopped("transient-sleep") {
		t.Fatalf("Bug #2: transient-sleep is reachable; must NOT be admissionStopped")
	}
}

