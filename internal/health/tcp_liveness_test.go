package health

import (
	"context"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeDialer lets a test script the sequence of dial outcomes.
type fakeDialer struct {
	mu        sync.Mutex
	results   []error // popped front-to-back
	callCount atomic.Int32
}

func (f *fakeDialer) Dial(ctx context.Context, network, addr string) (net.Conn, error) {
	f.callCount.Add(1)
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.results) == 0 {
		return nil, errors.New("fakeDialer: out of scripted results")
	}
	r := f.results[0]
	f.results = f.results[1:]
	if r != nil {
		return nil, r
	}
	// Return a closed pipe end so the probe's conn.Close() is a no-op.
	a, b := net.Pipe()
	_ = b.Close()
	return a, nil
}

func newProbeForTest(t *testing.T, fd *fakeDialer, transitions chan bool) *TCPProbe {
	t.Helper()
	return NewTCPProbe(Config{
		Target:           "127.0.0.1:0",
		Interval:         time.Millisecond, // irrelevant — we call probeOnce directly
		Timeout:          time.Millisecond,
		FailThreshold:    3,
		RecoverThreshold: 1,
		Dialer:           fd.Dial,
		OnTransition: func(alive bool) {
			select {
			case transitions <- alive:
			default:
			}
		},
		Now: time.Now,
	})
}

// Test: 3 consecutive failures demote alive from true to false, 1 success
// recovers. Validates the canonical state machine.
func TestTCPProbe_DemotesAfterFailThreshold(t *testing.T) {
	fd := &fakeDialer{results: []error{
		errors.New("connection refused"),
		errors.New("connection refused"),
		errors.New("connection refused"),
		nil, // recovery
	}}
	transitions := make(chan bool, 4)
	p := newProbeForTest(t, fd, transitions)

	if !p.Alive() {
		t.Fatalf("new probe should start optimistic-alive")
	}

	ctx := context.Background()

	// Two failures — still alive (under threshold).
	p.probeOnce(ctx)
	if !p.Alive() {
		t.Fatalf("alive should remain true after 1 failure")
	}
	p.probeOnce(ctx)
	if !p.Alive() {
		t.Fatalf("alive should remain true after 2 failures (threshold=3)")
	}

	// Third failure — demote.
	p.probeOnce(ctx)
	if p.Alive() {
		t.Fatalf("alive should flip to false after 3 consecutive failures")
	}
	select {
	case got := <-transitions:
		if got != false {
			t.Fatalf("transition channel got %v, want false", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected OnTransition(false) callback")
	}

	// One success — recover.
	p.probeOnce(ctx)
	if !p.Alive() {
		t.Fatalf("alive should flip back to true after 1 success (recover_threshold=1)")
	}
	select {
	case got := <-transitions:
		if got != true {
			t.Fatalf("transition channel got %v, want true", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected OnTransition(true) callback")
	}
}

// Test: a single intermittent failure does NOT flip alive, and the
// fail-counter resets when a probe succeeds — i.e. the failure has to be
// *consecutive*, not cumulative.
func TestTCPProbe_FailCountResetsOnSuccess(t *testing.T) {
	fd := &fakeDialer{results: []error{
		errors.New("transient"),
		errors.New("transient"),
		nil, // resets counter
		errors.New("transient"),
		errors.New("transient"),
	}}
	p := newProbeForTest(t, fd, nil)

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		p.probeOnce(ctx)
	}

	if !p.Alive() {
		t.Fatalf("alive should remain true: counter reset by success between bursts")
	}
}

// Test: when configured with a higher RecoverThreshold, a single success
// after demotion is insufficient to recover.
func TestTCPProbe_RecoverThresholdGreaterThanOne(t *testing.T) {
	fd := &fakeDialer{results: []error{
		errors.New("x"), errors.New("x"), errors.New("x"), // demote
		nil, // not enough
		nil, // now enough (threshold=2)
	}}
	p := NewTCPProbe(Config{
		Target: "127.0.0.1:0", Interval: time.Millisecond, Timeout: time.Millisecond,
		FailThreshold: 3, RecoverThreshold: 2, Dialer: fd.Dial,
	})

	ctx := context.Background()
	for i := 0; i < 3; i++ {
		p.probeOnce(ctx)
	}
	if p.Alive() {
		t.Fatalf("should be demoted")
	}
	p.probeOnce(ctx)
	if p.Alive() {
		t.Fatalf("should still be demoted: 1 success < recover_threshold=2")
	}
	p.probeOnce(ctx)
	if !p.Alive() {
		t.Fatalf("should recover: 2 consecutive successes >= recover_threshold")
	}
}

// Test: Snapshot reports the latest error and counters honestly.
func TestTCPProbe_Snapshot(t *testing.T) {
	wantErr := errors.New("dial fail")
	fd := &fakeDialer{results: []error{wantErr, wantErr}}
	p := newProbeForTest(t, fd, nil)

	ctx := context.Background()
	p.probeOnce(ctx)
	p.probeOnce(ctx)

	alive, failCount, okCount, lastCheck, lastErr := p.Snapshot()
	if !alive {
		t.Fatalf("alive should still be true (threshold=3, only 2 fails)")
	}
	if failCount != 2 {
		t.Fatalf("failCount=%d, want 2", failCount)
	}
	if okCount != 0 {
		t.Fatalf("okCount=%d, want 0", okCount)
	}
	if lastCheck.IsZero() {
		t.Fatalf("lastCheck should be set")
	}
	if !errors.Is(lastErr, wantErr) {
		t.Fatalf("lastErr=%v, want %v", lastErr, wantErr)
	}
}

// Test: defaults are populated and the probe starts optimistic-alive.
func TestTCPProbe_DefaultsApplied(t *testing.T) {
	p := NewTCPProbe(Config{Target: "127.0.0.1:0"})
	if p.cfg.Interval != DefaultInterval {
		t.Fatalf("Interval default not applied")
	}
	if p.cfg.Timeout != DefaultTimeout {
		t.Fatalf("Timeout default not applied")
	}
	if p.cfg.FailThreshold != DefaultFailThreshold {
		t.Fatalf("FailThreshold default not applied")
	}
	if p.cfg.RecoverThreshold != DefaultRecoverThreshold {
		t.Fatalf("RecoverThreshold default not applied")
	}
	if !p.Alive() {
		t.Fatalf("new probe should start alive")
	}
}

// Test: Run respects ctx cancellation and the loop does at least one probe.
func TestTCPProbe_RunHonoursContextCancel(t *testing.T) {
	fd := &fakeDialer{results: []error{nil, nil, nil, nil, nil, nil, nil, nil}}
	p := NewTCPProbe(Config{
		Target: "127.0.0.1:0", Interval: 10 * time.Millisecond,
		Timeout: time.Millisecond, FailThreshold: 3, RecoverThreshold: 1,
		Dialer: fd.Dial,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		p.Run(ctx)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("Run did not exit after ctx cancel")
	}

	if fd.callCount.Load() < 1 {
		t.Fatalf("expected at least one probe call")
	}
}

func TestFormatTarget(t *testing.T) {
	if got := FormatTarget("127.0.0.1", 8000); got != "127.0.0.1:8000" {
		t.Fatalf("FormatTarget=%q, want 127.0.0.1:8000", got)
	}
	if got := FormatTarget("::1", 8000); got != "[::1]:8000" {
		t.Fatalf("FormatTarget v6=%q, want [::1]:8000", got)
	}
}
