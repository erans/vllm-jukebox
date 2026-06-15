package jukebox

import (
	"context"
	"testing"
	"time"

	"vllm-jukebox/internal/metrics"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

// TestStartupAdopt_AdoptFiresExactlyOncePerBoot guards the one-shot
// boot-adopt invariant. Without the bootAdoptedPeers guard in
// checkOneExternalStart, the StateSleeping → StateSleeping branch in
// adoptPreExistingPeer leaves inst.state unchanged, so the candidate
// filter re-selects the peer on every 2s tick and the
// external_start_adopted_pre_existing_peer WARN +
// AdmissionStartupAdoptedTotal counter + lifecycle_transition
// action=cold_load reason=boot-adopt-pre-existing audit re-fire
// indefinitely. Live-observed 2026-06-13 on llm for vllm-main
// (Qwen3.6-27B), polluting the lifecycle audit log and making real
// cold-load events undistinguishable from boot-adopt re-runs.
//
// Pre-fix: counter goes to 5 across 5 ticks. Post-fix: stays at 1.
func TestStartupAdopt_AdoptFiresExactlyOncePerBoot(t *testing.T) {
	s, _, _ := makeExternalStartScheduler(t, StateReady, StateSleeping)

	rec := newDockerInspectRecorder()
	rec.setStatus("vllm-holder", "running")
	rec.setStatus("vllm-target", "running")
	preBoot := time.Now().Add(-1 * time.Hour).UTC().Format(time.RFC3339Nano)
	rec.setStartedAt("vllm-target", preBoot)
	rec.setStartedAt("vllm-holder", preBoot)
	SetSleepDockerCmdForTest(rec.handler())
	defer SetSleepDockerCmdForTest(nil)

	// Snapshot the counter before the test (other tests in this
	// package mutate the same global counter; we assert on delta).
	before := testutil.ToFloat64(
		metrics.AdmissionStartupAdoptedTotal.WithLabelValues("target"))

	// Run 5 consecutive ticks. Pre-fix: each tick re-fires adopt for
	// the StateSleeping peer because adoptPreExistingPeer doesn't
	// change inst.state. Post-fix: the bootAdoptedPeers guard skips
	// the adopt path on ticks 2-5.
	for i := 0; i < 5; i++ {
		s.CheckExternalStartsForTest(context.Background())
	}

	after := testutil.ToFloat64(
		metrics.AdmissionStartupAdoptedTotal.WithLabelValues("target"))
	delta := after - before
	if delta != 1 {
		t.Fatalf("adopt fired %v times across 5 ticks, expected exactly 1 (pre-fix bug fires every tick)", delta)
	}

	// Sanity: the adopted peer must still be in StateSleeping (the
	// adopt path must not have flipped it to Ready, and no SIGKILL
	// path should have run).
	got, ok := s.InstanceStateForTest("target")
	if !ok {
		t.Fatalf("target instance gone from scheduler")
	}
	if got != StateSleeping {
		t.Fatalf("expected StateSleeping after repeated adopt ticks, got %v", got)
	}

	// And no docker stop should have fired (defense — confirms we're
	// neither adopting twice nor falling through to the SIGKILL path).
	if rec.stopsContains("vllm-target") {
		t.Fatalf("REGRESSION: docker stop fired for adopted peer; stops=%v", rec.stops)
	}
}
