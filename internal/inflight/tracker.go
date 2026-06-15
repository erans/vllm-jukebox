package inflight

import (
	"context"
	"sync"
)

type Tracker struct {
	mu      sync.Mutex
	count   int64
	sealed  bool
	waiters []chan struct{}

	OnChange func(count int64)
}

// Track admits one in-flight request and returns a done func to release it.
//
// The (done, ok) shape closes the drain-barrier TOCTOU "tracker-seal" race
// (vllm#45520): when the tracker has been Seal()'d, Track refuses admission
// (ok=false, done=nil) WITHOUT incrementing the counter. The seal lives on
// THIS mutex — the only lock atomic across both Track and the drain
// observation in WaitForDrain — so once Seal() returns, the count is
// monotonically non-increasing and WaitForDrain's zero-observation is final.
// A request can no longer Track() into the window between WaitForDrain
// returning 0 and the caller firing /sleep. Callers that never seal (the
// global total tracker, the legacy router) simply discard ok.
func (t *Tracker) Track(_ context.Context) (done func(), ok bool) {
	t.mu.Lock()
	if t.sealed {
		// Refuse admission without mutating the counter — the caller must
		// re-route (e.g. via the wake-coalesced path) rather than land on
		// an instance that is about to /sleep.
		t.mu.Unlock()
		return nil, false
	}
	t.count++
	// Emit the change WHILE holding the lock so the observed sequence of
	// OnChange values can never diverge from the sequence of counter
	// mutations. If we snapshotted under the lock but called notify() after
	// unlocking (the original design), two concurrent Track/done callers
	// could swap their notify order and latch the gauge at a stale value
	// while the true count is something else — a wrong Prometheus gauge
	// under concurrency. OnChange is a single Gauge.Set, so holding the
	// (uncontended-in-the-common-case) lock across it is cheap.
	t.notify(t.count)
	t.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			t.mu.Lock()
			defer t.mu.Unlock()
			if t.count <= 0 {
				t.count = 0
				t.notify(0)
				return
			}

			t.count--
			if t.count == 0 {
				for _, ch := range t.waiters {
					close(ch)
				}
				t.waiters = nil
			}
			t.notify(t.count)
		})
	}, true
}

func (t *Tracker) WaitForDrain(ctx context.Context) error {
	t.mu.Lock()
	if t.count == 0 {
		t.mu.Unlock()
		return nil
	}
	ch := make(chan struct{})
	t.waiters = append(t.waiters, ch)
	t.mu.Unlock()

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *Tracker) Count() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.count
}

// Seal closes the admission gate: subsequent Track calls return ok=false
// without incrementing. It returns alreadyDrained==(count==0) so the caller
// can observe, atomically with the seal, whether the instance was already
// idle the instant admission was closed. Combined with a subsequent
// WaitForDrain, this makes the drain observation final — no request can
// Track() into the post-drain / pre-sleep window. Must be paired with
// Unseal on every exit path so a slept/rolled-back instance can re-admit.
func (t *Tracker) Seal() (alreadyDrained bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.sealed = true
	return t.count == 0
}

// Unseal re-opens the admission gate sealed by Seal. Idempotent.
func (t *Tracker) Unseal() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.sealed = false
}

func (t *Tracker) notify(count int64) {
	if t.OnChange != nil {
		t.OnChange(count)
	}
}
