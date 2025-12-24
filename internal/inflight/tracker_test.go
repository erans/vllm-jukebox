package inflight_test

import (
	"context"
	"testing"
	"time"

	"vllm-jukebox/internal/inflight"
)

func TestTracker_WaitForDrainReturnsWhenAllDone(t *testing.T) {
	var tr inflight.Tracker

	done1 := tr.Track(context.Background())
	done2 := tr.Track(context.Background())

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

	done := tr.Track(context.Background())
	done()
	done() // should not underflow or panic

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if err := tr.WaitForDrain(ctx); err != nil {
		t.Fatalf("expected drain, got %v", err)
	}
}
