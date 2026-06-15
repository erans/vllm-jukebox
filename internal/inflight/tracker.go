package inflight

import (
	"context"
	"sync"
)

type Tracker struct {
	mu      sync.Mutex
	count   int64
	waiters []chan struct{}

	OnChange func(count int64)
}

func (t *Tracker) Track(_ context.Context) (done func()) {
	t.mu.Lock()
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
	}
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

func (t *Tracker) notify(count int64) {
	if t.OnChange != nil {
		t.OnChange(count)
	}
}
