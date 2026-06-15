package jukebox

import (
	"context"
	"fmt"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// ---------------------------------------------------------------------------
// Startup adoption regression tests (Issue #5b)
// ---------------------------------------------------------------------------
//
// The bug: every `docker restart jukebox` (or recreate, image pull, etc.)
// cascade-killed every running vllm-* peer. Cause: jukebox's boot-time
// /is_sleeping probe correctly observed vLLM's normal slept-L1 resting
// state and seeded admission state = StateSleeping; the
// ExternalStartMonitor then saw {admission=Sleeping, docker=running} and
// SIGKILLed each peer. Then KickColdLoad refused with
// admission_state_or_cooldown_gate, stranding the fleet. Observed 3x in
// 30 min during deploy churn on 2026-06-14.
//
// The fix (startup_adopt.go): if container.StartedAt < jukebox.bootEpoch,
// the peer was already alive when jukebox restarted — ADOPT it
// (reconcile admission, no SIGKILL, no KickColdLoad).
//
// These tests are the load-bearing regression guard. They FAIL pre-fix
// (verified by reverting the shouldAdoptOnBoot call in
// checkOneExternalStart — the SIGKILL fires) and PASS post-fix.

// TestStartupAdopt_PreExistingSleepingPeer_NotKilled is the PRIMARY
// regression test for Issue #5b. Setup mirrors the live failure: a peer
// is in admission state Sleeping (post-/is_sleeping-probe), the docker
// container has been running for an hour (i.e. it pre-dates this
// jukebox process), and ExternalStartMonitor ticks. Pre-fix: SIGKILL.
// Post-fix: adopt — no docker stop, no KickColdLoad, admission state
// stays Sleeping (the wake will happen on the next consumer request).
func TestStartupAdopt_PreExistingSleepingPeer_NotKilled(t *testing.T) {
	// Build a scheduler whose bootEpoch is "now". The recorder will
	// report the peer's StartedAt as 1 hour in the past, which is
	// unambiguously BEFORE bootEpoch — shouldAdoptOnBoot returns
	// Adopt=true.
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateSleeping)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running") // pinned holder, alive
	rec.setStatus("vllm-target", "running") // pre-existing sleeping peer
	// Explicitly set StartedAt 1 hour BEFORE bootEpoch (default is 1h
	// FUTURE, which would not trigger adoption). Use UTC RFC3339Nano.
	preBoot := time.Now().Add(-1 * time.Hour).UTC().Format(time.RFC3339Nano)
	rec.setStartedAt("vllm-target", preBoot)
	rec.setStartedAt("vllm-holder", preBoot)
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	// Cap poll budget — KickColdLoad must NOT fire, so this is mostly
	// defense in depth so we don't burn 5min wall-clock if the test
	// regresses.
	oldPoll := SetColdLoadPollIntervalForTest(5 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	s.CheckExternalStartsForTest(context.Background())

	// PRIMARY ASSERTION: vllm-target must NOT have been stopped.
	// Pre-fix: this assertion FAILS (the SIGKILL fires).
	// Post-fix: PASSES (adopt path taken).
	if rec.stopsContains("vllm-target") {
		t.Fatalf("REGRESSION: pre-existing sleeping peer SIGKILLed at jukebox boot. stops=%v calls=%v", rec.stops, rec.calls)
	}
	if rec.stopsContains("vllm-holder") {
		t.Fatalf("holder must not be stopped either; stops=%v", rec.stops)
	}

	// SECONDARY ASSERTION: KickColdLoad must NOT have been kicked for
	// the adopted peer (the existing test inspects coldLoadKicks to
	// verify kick-fire; we verify kick-NOT-fire).
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		s.mu.RLock()
		_, kicked := s.coldLoadKicks["target"]
		s.mu.RUnlock()
		if kicked {
			t.Fatalf("REGRESSION: KickColdLoad fired for adopted peer (should be no-op)")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// TERTIARY ASSERTION: scheduler still records the peer as Sleeping
	// (we adopt, not transition to Ready). Next consumer hits the wake
	// path normally.
	got, ok := s.InstanceStateForTest("target")
	if !ok {
		t.Fatalf("target instance gone from scheduler")
	}
	if got != StateSleeping {
		t.Fatalf("expected StateSleeping post-adopt, got %v", got)
	}
}

// TestStartupAdopt_PreExistingStoppedPeer_ReconciledSleepModeAware covers
// the second adoption branch: boot probe failed to reach /health AND the
// model has evict_action: stop → boot seeded state=Stopped. Then the
// peer DOES become reachable (the probe was just transiently slow) and
// the external_start_monitor sees {admission=Stopped, docker=running}.
// Pre-fix: this triggered the same SIGKILL fault as the Sleeping case.
// Post-fix: we adopt and reconcile SLEEP_MODE-AWARE (mirroring
// ReprobeStoppedExternal). `target` is sleep_mode:true → it reconciles to
// StateSleeping (NOT StateReady — the supervisor-FAIL fix: marking a
// sleep_mode:true peer Ready while only booking the L1 residual under-books
// its awake VRAM). The next consumer wakes it from Sleeping.
func TestStartupAdopt_PreExistingStoppedPeer_ReconciledSleepModeAware(t *testing.T) {
	s, a, _ := makeExternalStartScheduler(t, StateReady, StateStopped)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running") // alive despite admission=Stopped
	preBoot := time.Now().Add(-30 * time.Minute).UTC().Format(time.RFC3339Nano)
	rec.setStartedAt("vllm-target", preBoot)
	rec.setStartedAt("vllm-holder", preBoot)
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	oldPoll := SetColdLoadPollIntervalForTest(5 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	s.CheckExternalStartsForTest(context.Background())

	if rec.stopsContains("vllm-target") {
		t.Fatalf("REGRESSION: pre-existing stopped peer SIGKILLed at jukebox boot. stops=%v", rec.stops)
	}

	got, ok := s.InstanceStateForTest("target")
	if !ok {
		t.Fatalf("target instance gone from scheduler")
	}
	if got != StateSleeping {
		t.Fatalf("expected StateSleeping post-adopt (sleep_mode:true, was Stopped), got %v", got)
	}
	// Admission should have transitioned out of admissionStopped (now
	// Sleeping with the L1 residual booked, NOT full awake VRAM).
	if a.IsStopped("target") {
		t.Fatalf("expected admission NOT Stopped post-adopt (NotifyStarted should have fired)")
	}
}

// TestStartupAdopt_PostBootExternalStart_StillKilled is the negative
// regression: we must NOT have accidentally disabled the original
// Issue #5 fix. A container whose StartedAt is AFTER jukebox bootEpoch
// AND which fails its /health probe is a genuine rogue external start
// (racing for VRAM at init) — the SIGKILL + KickColdLoad path must
// still fire. (A post-boot start that comes up HEALTHY is adopted by
// the Issue #5c healthy-start path — covered in external_start_test.go.)
func TestStartupAdopt_PostBootExternalStart_StillKilled(t *testing.T) {
	s, _, mgrs := makeExternalStartScheduler(t, StateReady, StateStopped)
	// Unhealthy: the rogue init has not come up on /health, so the
	// healthy-start-adopt path (Issue #5c) declines and we reach SIGKILL.
	mgrs["target"].verifyErr = fmt.Errorf("connection refused: init racing for VRAM")

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running") // <- externally started AFTER jukebox boot
	// Set StartedAt 1 minute IN THE FUTURE (relative to test start) —
	// well after bootEpoch (which was set when makeExternalStartScheduler
	// constructed the scheduler a few µs ago). shouldAdoptOnBoot returns
	// Adopt=false → SIGKILL path stays in force.
	postBoot := time.Now().Add(1 * time.Minute).UTC().Format(time.RFC3339Nano)
	rec.setStartedAt("vllm-target", postBoot)
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	oldPoll := SetColdLoadPollIntervalForTest(5 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(oldPoll)

	s.CheckExternalStartsForTest(context.Background())

	if !rec.stopsContains("vllm-target") {
		t.Fatalf("REGRESSION: real external start NOT SIGKILLed (Issue #5 broken). stops=%v calls=%v", rec.stops, rec.calls)
	}
}

// TestShouldAdoptOnBoot_Matrix is a unit-level table test that locks
// down the decision matrix without going through the scheduler.
func TestShouldAdoptOnBoot_Matrix(t *testing.T) {
	bootEpoch := time.Date(2026, 6, 14, 1, 0, 0, 0, time.UTC)

	cases := []struct {
		name       string
		startedAt  string
		wantAdopt  bool
		wantReason string
	}{
		{
			name:       "empty_started_at_fail_closed",
			startedAt:  "",
			wantAdopt:  false,
			wantReason: "missing_started_at",
		},
		{
			name:       "unparseable_fail_closed",
			startedAt:  "not-a-timestamp",
			wantAdopt:  false,
			wantReason: "unparseable_started_at",
		},
		{
			name:       "zero_value_fail_closed",
			startedAt:  "0001-01-01T00:00:00Z",
			wantAdopt:  false,
			wantReason: "zero_started_at",
		},
		{
			name:       "1h_before_boot_adopt",
			startedAt:  "2026-06-14T00:00:00Z",
			wantAdopt:  true,
			wantReason: "started_before_jukebox_boot",
		},
		{
			name:       "1s_before_boot_adopt",
			startedAt:  "2026-06-14T00:59:59Z",
			wantAdopt:  true,
			wantReason: "started_before_jukebox_boot",
		},
		{
			name:       "exactly_at_boot_within_grace_adopt",
			startedAt:  "2026-06-14T01:00:00Z",
			wantAdopt:  true,
			wantReason: "started_before_jukebox_boot",
		},
		{
			name:       "3s_after_boot_within_grace_adopt",
			startedAt:  "2026-06-14T01:00:03Z",
			wantAdopt:  true,
			wantReason: "started_before_jukebox_boot",
		},
		{
			name:       "10s_after_boot_outside_grace_no_adopt",
			startedAt:  "2026-06-14T01:00:10Z",
			wantAdopt:  false,
			wantReason: "started_after_jukebox_boot",
		},
		{
			name:       "1h_after_boot_no_adopt",
			startedAt:  "2026-06-14T02:00:00Z",
			wantAdopt:  false,
			wantReason: "started_after_jukebox_boot",
		},
		{
			name:       "rfc3339_nano_with_sub_second_adopt",
			startedAt:  "2026-06-14T00:30:00.123456789Z",
			wantAdopt:  true,
			wantReason: "started_before_jukebox_boot",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dec := shouldAdoptOnBoot(tc.startedAt, bootEpoch)
			if dec.Adopt != tc.wantAdopt {
				t.Errorf("Adopt mismatch: got %v want %v (reason=%q)", dec.Adopt, tc.wantAdopt, dec.Reason)
			}
			if dec.Reason != tc.wantReason {
				t.Errorf("Reason mismatch: got %q want %q", dec.Reason, tc.wantReason)
			}
		})
	}
}

// TestShouldAdoptOnBoot_ZeroBootEpochFailClosed defends against a
// future refactor that constructs a Scheduler via a non-standard path
// and forgets to set bootEpoch. We must NOT silently adopt every
// running peer in that case — fail closed.
func TestShouldAdoptOnBoot_ZeroBootEpochFailClosed(t *testing.T) {
	dec := shouldAdoptOnBoot("2026-06-14T00:00:00Z", time.Time{})
	if dec.Adopt {
		t.Fatalf("zero bootEpoch must fail closed (got Adopt=true reason=%q)", dec.Reason)
	}
	if dec.Reason != "missing_boot_epoch" {
		t.Errorf("expected missing_boot_epoch, got %q", dec.Reason)
	}
}

// TestBootEpoch_SetByNewScheduler verifies the field is wired through
// NewSchedulerWithFactory. Defense against a future refactor that
// silently drops the bootEpoch initialization.
func TestBootEpoch_SetByNewScheduler(t *testing.T) {
	cfg, err := config.Load([]byte(externalStartTestConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000, 1: 24000, 2: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)

	before := time.Now().Add(-1 * time.Second)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	after := time.Now().Add(1 * time.Second)

	if s.bootEpoch.IsZero() {
		t.Fatalf("bootEpoch was not set")
	}
	if s.bootEpoch.Before(before) || s.bootEpoch.After(after) {
		t.Errorf("bootEpoch %v out of expected window [%v, %v]", s.bootEpoch, before, after)
	}
}
