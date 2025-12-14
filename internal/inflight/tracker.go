package inflight

import (
	"context"
	"sync"
)

type Tracker struct {
	mu      sync.Mutex
	count   int64
	waiters []chan struct{}
}

func (t *Tracker) Track(_ context.Context) (done func()) {
	t.mu.Lock()
	t.count++
	t.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			t.mu.Lock()
			defer t.mu.Unlock()

			if t.count <= 0 {
				t.count = 0
				return
			}

			t.count--
			if t.count == 0 {
				for _, ch := range t.waiters {
					close(ch)
				}
				t.waiters = nil
			}
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

