package jukebox

import (
	"context"
	"net/http"
	"sync"
	"testing"
)

// TestDecodeProbe_DrainClearsAfterRestartHeals is the regression test for
// the decode-stall-drain-never-cleared wedge (CRIT-2).
//
// BUG (pre-fix): tripDecodeStall sets inst.draining=true and fires the
// circuit-breaker docker-restart, but checkDecodeStalls skipped ALL
// draining instances (`if inst.state != StateReady || inst.draining
// { continue }`). Nothing else ever clears the drain — every routing
// fast-path (AcquireRoute / tryRouteReady / Status) also skips draining
// instances — so once the restarted engine self-heals and answers
// /health/decode=200 again, the model stays black-holed OUT OF ROUTING
// FOREVER (until a full jukebox process restart). A single transient
// #45094 wedge thus permanently removes the model.
//
// FIX: checkDecodeStalls now ALSO probes drained-but-StateReady instances
// (purely to detect recovery), and calls the idempotent
// clearDecodeStallDrain when such an instance probes healthy — returning
// it to routing.
//
// This test drives the full loop: healthy → sustained stall → TRIP
// (draining=true) → engine recovers (probe 200) → next tick must CLEAR
// the drain (draining=false), with the recovery counted.
func TestDecodeProbe_DrainClearsAfterRestartHeals(t *testing.T) {
	s, rd := makeDecodeProbeScheduler(t)
	if !s.SetInstanceInflightForTest("vllm-main", 1) {
		t.Fatal("seed inflight failed")
	}

	var mu sync.Mutex
	status := http.StatusServiceUnavailable
	setStatus := func(v int) { mu.Lock(); status = v; mu.Unlock() }
	defer SetDecodeProbeFnForTest(func(_ context.Context, _ string) (int, error) {
		mu.Lock()
		defer mu.Unlock()
		return status, nil
	})()

	ctx := context.Background()

	// Two consecutive stalls (with in-flight > 0) reach
	// sustainedStallIntervals → TRIP → instance marked draining.
	s.CheckDecodeStallsForTest(ctx) // count=1, transient
	s.CheckDecodeStallsForTest(ctx) // count=2 → trip
	waitForRestart(t, rd)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); !draining {
		t.Fatal("precondition: instance must be draining after a sustained-stall trip")
	}

	// The engine self-heals after the breaker's docker-restart and now
	// answers /health/decode=200 again.
	setStatus(http.StatusOK)

	// PRE-FIX: checkDecodeStalls skips the draining instance, so this tick
	// is a no-op and the instance stays draining forever (FAIL below).
	// POST-FIX: the probe sees the drained-but-Ready instance, observes
	// the healthy probe, and clears the drain (PASS).
	s.CheckDecodeStallsForTest(ctx)

	if draining, ok := s.InstanceDrainingForTest("vllm-main"); !ok {
		t.Fatal("instance vanished")
	} else if draining {
		t.Fatal("drain was NOT cleared after the engine recovered — model is " +
			"permanently black-holed out of routing (the CRIT-2 wedge); " +
			"checkDecodeStalls must probe drained-but-Ready instances and " +
			"clear the drain on a healthy /health/decode probe")
	}

	// clearDecodeStallDrain is idempotent: a second healthy tick on the
	// now-non-draining instance must not error or re-toggle anything, and
	// it must take the normal (non-draining) healthy path keeping count 0.
	s.CheckDecodeStallsForTest(ctx)
	if draining, _ := s.InstanceDrainingForTest("vllm-main"); draining {
		t.Fatal("instance must remain non-draining after recovery")
	}
	if c := s.DecodeStallCountForTest("vllm-main"); c != 0 {
		t.Fatalf("recovered instance must keep stall count 0, got %d", c)
	}
}
