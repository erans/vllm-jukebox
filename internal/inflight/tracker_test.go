package inflight_test

import (
	"context"
	"testing"
	"time"

	"vllm-jukebox/internal/inflight"
)

func TestTracker_WaitForDrainReturnsWhenAllDone(t *testing.T) {
	var tr inflight.Tracker

	done1, _ := tr.Track(context.Background())
	done2, _ := tr.Track(context.Background())

	waitCtx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()

	drained := make(chan error, 1)
	go func() {
		drained <- tr.WaitForDrain(waitCtx)
	}()

	// Still in-flight; should not drain yet.
	select {
	case err := <-drained:
		t.Fatalf("unexpected drain early: %v", err)
	case <-time.After(25 * time.Millisecond):
	}

	done1()
	done2()

	select {
	case err := <-drained:
		if err != nil {
			t.Fatalf("expected nil, got %v", err)
		}
	case <-time.After(250 * time.Millisecond):
		t.Fatalf("timed out waiting for drain")
	}
}

func TestTracker_DoneIsIdempotent(t *testing.T) {
	var tr inflight.Tracker

	done, _ := tr.Track(context.Background())
	done()
	done() // should not underflow or panic

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if err := tr.WaitForDrain(ctx); err != nil {
		t.Fatalf("expected drain, got %v", err)
	}
}

// TestTracker_SealRefusesTrackAndCountMonotonic verifies the drain-barrier
// seal: Track admits normally until Seal(), after which Track refuses
// (ok=false) WITHOUT mutating the count; Seal reports alreadyDrained ==
// (count==0); and Unseal re-opens admission. This is the unit-level proof
// that the seal closes the TOCTOU window in sleepInstance (vllm#45520).
func TestTracker_SealRefusesTrackAndCountMonotonic(t *testing.T) {
	var tr inflight.Tracker

	// Normal admission.
	done1, ok1 := tr.Track(context.Background())
	if !ok1 || done1 == nil {
		t.Fatalf("expected first Track to succeed (ok=true, done!=nil), got ok=%v done!=nil=%v", ok1, done1 != nil)
	}
	if got := tr.Count(); got != 1 {
		t.Fatalf("expected count 1 after one Track, got %d", got)
	}

	// Seal with a non-zero count: alreadyDrained must be false.
	if alreadyDrained := tr.Seal(); alreadyDrained {
		t.Fatalf("expected Seal() alreadyDrained=false with 1 in-flight, got true")
	}

	// Track after Seal must refuse without incrementing.
	done2, ok2 := tr.Track(context.Background())
	if ok2 {
		t.Fatalf("expected Track to refuse (ok=false) after Seal, got ok=true")
	}
	if done2 != nil {
		t.Fatalf("expected refused Track to return nil done func, got non-nil")
	}
	if got := tr.Count(); got != 1 {
		t.Fatalf("expected count to stay at 1 after a refused Track, got %d", got)
	}

	// Releasing the original in-flight drains the (still-sealed) tracker.
	done1()
	if got := tr.Count(); got != 0 {
		t.Fatalf("expected count 0 after done(), got %d", got)
	}

	// Re-sealing an already-empty tracker reports alreadyDrained=true.
	if alreadyDrained := tr.Seal(); !alreadyDrained {
		t.Fatalf("expected Seal() alreadyDrained=true with 0 in-flight, got false")
	}

	// Unseal re-allows Track.
	tr.Unseal()
	done3, ok3 := tr.Track(context.Background())
	if !ok3 || done3 == nil {
		t.Fatalf("expected Track to succeed after Unseal (ok=true, done!=nil), got ok=%v done!=nil=%v", ok3, done3 != nil)
	}
	if got := tr.Count(); got != 1 {
		t.Fatalf("expected count 1 after Unseal+Track, got %d", got)
	}
	done3()
}
