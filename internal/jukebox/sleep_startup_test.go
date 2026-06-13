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
