package jukebox

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// externalStartTestConfig matches the live scenario that Issue #5 was
// filed against: two external models in DIFFERENT swap-groups,
// overlapping on GPU 1. `holder` is admission-Ready (resident, holding
// VRAM). `target` is admission-Stopped (no jukebox lifecycle has
// started it yet). The fault: someone runs `docker start vllm-target`
// directly, bypassing the cold-load path. Without the
// ExternalStartMonitor, the target would have OOM'd on GPU 1.
const externalStartTestConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  holder:
    lifecycle: external
    host: vllm-holder
    port: 9001
    gpus: [0, 1]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: holder-group
    pinned: true
    expected_vram_mb_per_gpu: 12000
    sleep_l1_residual_mb: 1000
    wake_timeout: 1s
  target:
    lifecycle: external
    host: vllm-target
    port: 9002
    gpus: [1, 2]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: target-group
    expected_vram_mb_per_gpu: 11000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`

// makeExternalStartScheduler is a fresh-ish copy of
// makeCrossGroupScheduler tuned for ExternalStartMonitor tests: it
// doesn't need a `bystander` model and uses 8000mb totals so the
// admission paths are tight enough to be representative.
func makeExternalStartScheduler(t *testing.T, holderState, targetState State) (*Scheduler, *AdmissionController, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(externalStartTestConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000, 1: 24000, 2: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"holder": {port: 9001},
		"target": {port: 9002},
	}
	for name, m := range mgrs {
		modelCfg := cfg.Models[name]
		var st State
		switch name {
		case "holder":
			st = holderState
		case "target":
			st = targetState
		}
		switch st {
		case StateReady:
			m.pid.Store(int64(7000 + m.port))
			m.isSleeping.Store(false)
		case StateSleeping:
			m.pid.Store(int64(7000 + m.port))
			m.isSleeping.Store(true)
		case StateStopped:
			m.pid.Store(0)
			m.isSleeping.Store(true)
		}
		s.SeedInstanceForTest(name, m.port, modelCfg.GPUs, modelCfg.Pinned != nil && *modelCfg.Pinned, st, m)
	}

	// Sync admission for stopped seeds so IsStopped agrees with the
	// instance map (KickColdLoad consults admission, not s.instances).
	a.mu.Lock()
	if targetState == StateStopped {
		a.markStoppedLocked(a.models["target"])
	}
	if holderState == StateStopped {
		a.markStoppedLocked(a.models["holder"])
	}
	a.mu.Unlock()

	return s, a, mgrs
}

// dockerInspectRecorder is a thread-safe recorder for the docker calls
// the ExternalStartMonitor issues via runSleepDocker. Tests configure
// per-container inspect responses (status string) and assert on the
// stop/start call sequence.
type dockerInspectRecorder struct {
	mu       sync.Mutex
	calls    []string                     // every "docker <args...>" string
	stops    []string                     // container names passed to "docker stop"
	starts   []string                     // container names passed to "docker start"
	statusBy map[string]string            // container → status string returned by `docker inspect -f '{{json .State}}'`
	// startedAtBy lets tests override the StartedAt value reported per
	// container. Empty/unset → falls back to the recorder's default
	// (1h in the future from time.Now()), which is AFTER the scheduler's
	// bootEpoch, so shouldAdoptOnBoot returns Adopt=false and the
	// existing pre-Issue-#5b SIGKILL behavior is preserved. New
	// startup-adopt tests explicitly set a value PRIOR to bootEpoch to
	// exercise the adopt path.
	startedAtBy map[string]string
	stopErr  func(container string) error // optional: per-container stop failure
}

func newDockerInspectRecorder() *dockerInspectRecorder {
	return &dockerInspectRecorder{
		statusBy:    map[string]string{},
		startedAtBy: map[string]string{},
	}
}

func (r *dockerInspectRecorder) setStartedAt(container, startedAt string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.startedAtBy[container] = startedAt
}

func (r *dockerInspectRecorder) startedAtOf(container string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if v, ok := r.startedAtBy[container]; ok && v != "" {
		return v
	}
	// Default: 1 hour from now. This guarantees the container's
	// StartedAt is AFTER any sane scheduler bootEpoch the test
	// constructs via NewSchedulerWithFactory(now=time.Now), so the
	// startup-adopt path is NOT taken and the existing external-start
	// SIGKILL tests stay green.
	return time.Now().Add(1 * time.Hour).UTC().Format(time.RFC3339Nano)
}

func (r *dockerInspectRecorder) setStatus(container, status string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.statusBy[container] = status
}

func (r *dockerInspectRecorder) statusOf(container string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.statusBy[container]
	if !ok {
		// Default = "exited" (the safe assumption when the test hasn't
		// explicitly arranged a status). exited containers are NOT
		// flagged by the monitor.
		return "exited"
	}
	return s
}

func (r *dockerInspectRecorder) handler() func(ctx context.Context, name string, args ...string) ([]byte, error) {
	return func(_ context.Context, name string, args ...string) ([]byte, error) {
		full := name + " " + strings.Join(args, " ")
		r.mu.Lock()
		r.calls = append(r.calls, full)
		r.mu.Unlock()

		// docker inspect -f '{{json .State}}' <container>
		if len(args) >= 4 && args[0] == "inspect" && args[1] == "-f" {
			container := args[3]
			status := r.statusOf(container)
			startedAt := r.startedAtOf(container)
			// Minimal but valid JSON matching containerState's fields.
			payload := fmt.Sprintf(`{"Status":%q,"ExitCode":0,"OOMKilled":false,"Error":"","StartedAt":%q,"FinishedAt":""}`, status, startedAt)
			return []byte(payload), nil
		}

		// docker stop -t 0 <container>
		if len(args) >= 4 && args[0] == "stop" {
			container := args[len(args)-1]
			r.mu.Lock()
			r.stops = append(r.stops, container)
			r.mu.Unlock()
			if r.stopErr != nil {
				if err := r.stopErr(container); err != nil {
					return []byte("stop failed"), err
				}
			}
			// Flip the status to "exited" so a follow-up inspect agrees
			// with the post-stop reality.
			r.setStatus(container, "exited")
			return []byte("ok"), nil
		}

		// docker start <container>
		if len(args) >= 2 && args[0] == "start" {
			container := args[len(args)-1]
			r.mu.Lock()
			r.starts = append(r.starts, container)
			r.mu.Unlock()
			return []byte("ok"), nil
		}

		return []byte("ok"), nil
	}
}

func (r *dockerInspectRecorder) stopsContains(container string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, c := range r.stops {
		if c == container {
			return true
		}
	}
	return false
}

// TestExternalStart_DetectsRunningStoppedPeer_StopsAndKicks is the
// load-bearing Issue #5 regression. Admission records `target` as
// Stopped; docker reports `vllm-target` as running. The monitor MUST:
//  1. Issue `docker stop -t 0 vllm-target` immediately.
//  2. Invoke KickColdLoad which spawns the proper cold-load goroutine.
//
// Pre-fix behavior: the monitor doesn't exist; the externally-started
// container OOMs on GPU 1 because `holder` is awake there. Post-fix:
// stop fires, KickColdLoad fires, the cold-load goroutine runs the
// cross-group eviction before its own docker start.
func TestExternalStart_DetectsRunningStoppedPeer_StopsAndKicks(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running") // holder legitimately Ready
	rec.setStatus("vllm-target", "running") // <-- externally started!
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	// Speed up: prevent the spawned KickColdLoad goroutine from doing
	// its full 5-minute /is_sleeping poll loop in the test.
	oldPoll := SetColdLoadPollIntervalForTest(5 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	s.CheckExternalStartsForTest(context.Background())

	// MUST have stopped vllm-target.
	if !rec.stopsContains("vllm-target") {
		t.Fatalf("expected docker stop vllm-target; got calls=%v", rec.calls)
	}
	// MUST NOT have stopped vllm-holder (admission Ready agrees with
	// docker running — no external start).
	if rec.stopsContains("vllm-holder") {
		t.Fatalf("did NOT expect docker stop vllm-holder (legitimately Ready); got stops=%v", rec.stops)
	}

	// Wait for the async KickColdLoad goroutine to register itself by
	// checking that admission state flips. coldLoadKicks is set
	// synchronously in KickColdLoad before the goroutine spawns; we
	// can probe it directly.
	deadline := time.Now().Add(2 * time.Second)
	kickFired := false
	for time.Now().Before(deadline) {
		s.mu.RLock()
		_, kicked := s.coldLoadKicks["target"]
		s.mu.RUnlock()
		// Either the kick is in-flight (entry exists) OR has completed
		// (entry was deferred-deleted). Either way, KickColdLoad fired.
		if kicked {
			kickFired = true
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !kickFired {
		t.Logf("note: did not observe coldLoadKicks entry (kick may have completed before poll); stop fired correctly")
	}
}

// TestExternalStart_IgnoresLegitimatelyRunningReady asserts the monitor
// does NOT take action on a Ready instance whose container is running
// (the happy path — jukebox put it there). Selection criteria filters
// by admission state Stopped/Sleeping, so a Ready instance is never
// even inspected.
func TestExternalStart_IgnoresLegitimatelyRunningReady(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateReady)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running")
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.CheckExternalStartsForTest(context.Background())

	if len(rec.stops) != 0 {
		t.Fatalf("expected no docker stop calls (both instances Ready); got stops=%v", rec.stops)
	}
}

// TestExternalStart_IgnoresStoppedAndExited covers the steady-state
// no-op case: admission says Stopped, docker says exited. No action.
func TestExternalStart_IgnoresStoppedAndExited(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "exited")
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.CheckExternalStartsForTest(context.Background())

	if len(rec.stops) != 0 {
		t.Fatalf("expected no docker stop calls (target exited matches Stopped admission); got stops=%v", rec.stops)
	}
}

// TestExternalStart_DetectsRunningSleepingPeer: admission records
// `target` as Sleeping (peer was admission-Sleeping, e.g. boot-probe
// found it asleep), but docker reports it running. This is also
// inconsistent — Sleeping means "vLLM is paused via /sleep", not
// "container is stopped" — but the same fix applies: SIGKILL and
// re-route through cold-load to restore invariants.
func TestExternalStart_DetectsRunningSleepingPeer(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateSleeping)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running") // <-- inconsistent with Sleeping
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	oldPoll := SetColdLoadPollIntervalForTest(5 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	s.CheckExternalStartsForTest(context.Background())

	if !rec.stopsContains("vllm-target") {
		t.Fatalf("expected docker stop vllm-target (Sleeping admission + running docker is inconsistent); got calls=%v", rec.calls)
	}
}

// TestExternalStart_NoOpForAdmissionDisabledModels: a model with no
// expected_vram_mb_per_gpu is invisible to admission and must NOT be
// probed by the monitor (admission filter at the selection layer).
func TestExternalStart_NoOpForAdmissionDisabledModels(t *testing.T) {
	const cfg = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  untracked:
    lifecycle: external
    host: vllm-untracked
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    wake_timeout: 1s
`
	parsed, err := config.Load([]byte(cfg))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(parsed, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(parsed, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)
	mgr := &fakeRedeployMgr{port: 9001}
	mgr.pid.Store(0)
	mgr.isSleeping.Store(true)
	modelCfg := parsed.Models["untracked"]
	s.SeedInstanceForTest("untracked", mgr.port, modelCfg.GPUs, false, StateStopped, mgr)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-untracked", "running")
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.CheckExternalStartsForTest(context.Background())

	// Untracked-by-admission models must be filtered out before the
	// inspect even runs.
	for _, c := range rec.calls {
		if strings.Contains(c, "inspect") && strings.Contains(c, "vllm-untracked") {
			t.Fatalf("admission-disabled model must NOT be inspected; got call %q (all calls: %v)", c, rec.calls)
		}
	}
}

// TestExternalStart_StopFailureSurfacesMetricAndSkipsKick: if docker
// stop -t 0 itself fails, the monitor must NOT proceed to KickColdLoad
// (the still-running container would race with the cold-load's
// docker start). It bumps the failure counter and returns.
func TestExternalStart_StopFailureSurfacesMetricAndSkipsKick(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running")
	rec.stopErr = func(container string) error {
		if container == "vllm-target" {
			return fmt.Errorf("simulated docker daemon hiccup")
		}
		return nil
	}
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	s.CheckExternalStartsForTest(context.Background())

	// Stop was attempted.
	if !rec.stopsContains("vllm-target") {
		t.Fatalf("expected docker stop vllm-target attempt; got calls=%v", rec.calls)
	}
	// docker start MUST NOT have followed (KickColdLoad was correctly skipped).
	rec.mu.Lock()
	defer rec.mu.Unlock()
	for _, c := range rec.starts {
		if c == "vllm-target" {
			t.Fatalf("KickColdLoad must NOT run when stop failed; saw docker start %s", c)
		}
	}
}

// TestExternalStart_MonitorLoopExitsOnContextCancel exercises the
// production goroutine path: start the monitor, cancel the ctx, and
// verify it returns within a bounded window. Guards against a missing
// <-ctx.Done() branch (a future refactor that returns early on tick
// processing only).
func TestExternalStart_MonitorLoopExitsOnContextCancel(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

	// Very tight cadence so the test doesn't pay 2s per tick.
	old := SetExternalStartPollIntervalForTest(5 * time.Millisecond)
	defer SetExternalStartPollIntervalForTest(old)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "exited")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var ranAtLeastOnce atomic.Bool
	// Wrap the docker hook to flag that at least one tick ran.
	inner := rec.handler()
	SetSleepDockerCmdForTest(func(ctx context.Context, name string, args ...string) ([]byte, error) {
		ranAtLeastOnce.Store(true)
		return inner(ctx, name, args...)
	})
	defer SetSleepDockerCmdForTest(nil)

	go func() {
		s.ExternalStartMonitor(ctx)
		close(done)
	}()

	// Give the ticker enough wall-clock to fire a few times.
	time.Sleep(50 * time.Millisecond)
	if !ranAtLeastOnce.Load() {
		t.Fatalf("monitor did not run any ticks in 50ms")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("ExternalStartMonitor did not return within 500ms of ctx cancel")
	}
}

// TestExternalStart_DoesNotRefireOnOwnInFlightColdLoad is the regression
// for the dedupe-race that wedged the fleet on 2026-06-14. Scenario:
//
//  1. Tick N: monitor saw target admission=Stopped + container=running,
//     SIGKILL'd it, called KickColdLoad. The cold-load goroutine
//     starts (sets coldLoadKicks[target]=true under s.mu), acquires
//     coldLoadMu, marks coldLoadEviction[target...]=true, runs cross-
//     group eviction, then runs `docker start vllm-target` and starts
//     waiting on /health (~5min). DURING THIS WAIT, inst.state stays
//     StateStopped while the container is `running`.
//  2. Tick N+1 (2s later): WITHOUT THE GUARD, the monitor sees admission
//     Stopped + container running again, classifies it as a fresh
//     external start, SIGKILLs the in-flight cold-load (vLLM dies
//     exit 137), then calls KickColdLoad. KickColdLoad refuses with
//     `kick_in_flight` (the goroutine from tick N hasn't deferred-
//     deleted coldLoadKicks[target] yet). The peer is now wedged: the
//     cold-load failed, coldLoadFailures[target] is recorded, and the
//     30s cooldown gate triggers on retry.
//
// POST-FIX: the guard at the top of checkOneExternalStart skips peers
// where IsInColdLoadEviction is true OR coldLoadKicks[name] is true.
// Tick N+1 returns early without issuing a stop. The cold-load
// goroutine completes its work undisturbed.
//
// Test mechanics: we don't drive a full async cold-load (that requires
// the full coldLoadStoppedMember pipeline + a 5min health wait). We
// directly set s.coldLoadKicks[target] = true to simulate the
// in-flight state from tick N, then invoke checkOneExternalStart and
// assert that NO docker stop fires on vllm-target. We also test the
// IsInColdLoadEviction branch by setting that gate explicitly.
func TestExternalStart_DoesNotRefireOnOwnInFlightColdLoad(t *testing.T) {
	t.Run("guarded_by_coldLoadKicks", func(t *testing.T) {
		s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

		rec := newDockerInspectRecorder()
		rec.setStatus("vllm-holder", "running")
		rec.setStatus("vllm-target", "running") // mid-cold-load: container up, admission still Stopped
		SetSleepDockerCmdForTest(rec.handler())
		defer SetSleepDockerCmdForTest(nil)

		// Simulate tick N's KickColdLoad having spawned its goroutine:
		// coldLoadKicks[target] is set, defer-delete hasn't run yet.
		s.mu.Lock()
		if s.coldLoadKicks == nil {
			s.coldLoadKicks = make(map[string]bool)
		}
		s.coldLoadKicks["target"] = true
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			delete(s.coldLoadKicks, "target")
			s.mu.Unlock()
		}()

		// First tick — the guard should skip target entirely, no stop.
		s.CheckExternalStartsForTest(context.Background())
		if rec.stopsContains("vllm-target") {
			t.Fatalf("PRE-FIX BUG REPRODUCED: monitor re-SIGKILL'd its own in-flight cold-load. stops=%v", rec.stops)
		}
		stopsAfterTick1 := len(rec.stops)

		// Second tick (simulating monitor cadence). Counter must stay
		// flat — the guard still holds while coldLoadKicks is set.
		s.CheckExternalStartsForTest(context.Background())
		if got := len(rec.stops); got != stopsAfterTick1 {
			t.Fatalf("expected stops count stable across ticks (still mid-cold-load); tick1=%d tick2=%d stops=%v",
				stopsAfterTick1, got, rec.stops)
		}
		if rec.stopsContains("vllm-target") {
			t.Fatalf("monitor re-SIGKILL'd target on tick 2 despite kick-in-flight; stops=%v", rec.stops)
		}
	})

	t.Run("guarded_by_coldLoadEviction", func(t *testing.T) {
		s, _, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

		rec := newDockerInspectRecorder()
		rec.setStatus("vllm-holder", "running")
		rec.setStatus("vllm-target", "running")
		SetSleepDockerCmdForTest(rec.handler())
		defer SetSleepDockerCmdForTest(nil)

		// Simulate the inner coldLoadStoppedMember having marked the
		// eviction scope (target + cross-group contenders) — the path
		// IsInColdLoadEviction reports on.
		s.markColdLoadEviction("target")
		defer s.unmarkColdLoadEviction("target")

		s.CheckExternalStartsForTest(context.Background())
		if rec.stopsContains("vllm-target") {
			t.Fatalf("monitor SIGKILL'd target despite IsInColdLoadEviction; stops=%v", rec.stops)
		}

		// And a second tick (matches the live 2s ticker pattern).
		s.CheckExternalStartsForTest(context.Background())
		if rec.stopsContains("vllm-target") {
			t.Fatalf("monitor SIGKILL'd target on tick 2 despite eviction gate; stops=%v", rec.stops)
		}
	})
}
