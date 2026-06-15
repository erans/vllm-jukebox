package jukebox_test

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/health"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
)

// stubManagerAlwaysReady is the simplest possible Manager — every call to
// Start/VerifyReady succeeds, CurrentPID returns a non-zero value. The
// LegacyRouter tests use it because we want to exercise the post-coord
// liveness gate, not the coordinator's own state machine.
type stubManagerAlwaysReady struct{}

func (stubManagerAlwaysReady) Start(_ context.Context, _ string) (int, error) { return 4242, nil }
func (stubManagerAlwaysReady) Stop(_ context.Context) error                   { return nil }
func (stubManagerAlwaysReady) VerifyReady(_ context.Context, _ string) error  { return nil }
func (stubManagerAlwaysReady) CurrentPID() int                                { return 4242 }

// itoa avoids pulling strconv in for one call.
func itoa(i int) string {
	// Small positive integers only — these tests use ports.
	if i == 0 {
		return "0"
	}
	digits := []byte{}
	for i > 0 {
		digits = append([]byte{byte('0' + i%10)}, digits...)
		i /= 10
	}
	return string(digits)
}

// listenOnEphemeralPort opens a TCP listener bound to a random port and
// returns its port + a cleanup that closes the listener. While the
// listener is open, dials succeed; after Close, dials fail with
// "connection refused".
func listenOnEphemeralPort(t *testing.T) (int, *net.TCPListener) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	tcp := l.(*net.TCPListener)
	port := tcp.Addr().(*net.TCPAddr).Port
	// Accept-and-discard loop so we don't leak the goroutine when the
	// test closes the listener.
	go func() {
		for {
			c, err := tcp.Accept()
			if err != nil {
				return
			}
			_ = c.Close()
		}
	}()
	return port, tcp
}

// drainTransitions waits up to d for the probe Snapshot to report the
// requested alive state. Polls because OnTransition fires async.
func waitForAlive(t *testing.T, p *health.TCPProbe, want bool, d time.Duration) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if p.Alive() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("probe alive=%v after %v, want %v", p.Alive(), d, want)
}

// Test (the headline of this PR): when the upstream listener stops
// accepting connections, the LegacyRouter MUST fast-fail subsequent
// AcquireRoute calls with RejectUpstreamUnreachable + a populated
// RetryAfter, instead of letting the proxy forward into a hung 500.
func TestLegacyRouter_AcquireRoute_DemotesWhenUpstreamPortGoesAway(t *testing.T) {
	port, listener := listenOnEphemeralPort(t)

	// Tight thresholds so the test runs in well under a second.
	cfgYAML := `
vllm:
  port: ` + itoa(port) + `
  tcp_liveness:
    enabled: true
    interval: 20ms
    timeout: 50ms
    fail_threshold: 2
    recover_threshold: 1
    retry_after_seconds: 7
models:
  m:
    path: "/models/m"
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}

	var tr inflight.Tracker
	mgr := stubManagerAlwaysReady{}
	coord := jukebox.NewCoordinator(cfg, mgr, &tr, time.Now)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go coord.Run(ctx)

	router := jukebox.NewLegacyRouter(cfg, coord, &tr)
	router.StartLiveness(ctx)

	// While the listener is up, AcquireRoute returns a usable route.
	route, err := router.AcquireRoute(context.Background(), "m", "req_pre")
	if err != nil {
		t.Fatalf("AcquireRoute pre-shutdown: %v", err)
	}
	if route.Done != nil {
		route.Done()
	}

	// Kill the listener — every subsequent dial returns connection refused.
	if err := listener.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}

	// Wait for the probe to demote the peer (interval=20ms, threshold=2,
	// so ~50-100ms is enough; give 2s slop for slow CI).
	probe := router.LivenessProbeForTest()
	if probe == nil {
		t.Fatalf("expected probe to be running after StartLiveness")
	}
	waitForAlive(t, probe, false, 2*time.Second)

	// AcquireRoute MUST now return RejectUpstreamUnreachable.
	_, err = router.AcquireRoute(context.Background(), "m", "req_post")
	if err == nil {
		t.Fatalf("expected RejectUpstreamUnreachable, got nil")
	}
	var rej *jukebox.RejectError
	if !errors.As(err, &rej) {
		t.Fatalf("expected *RejectError, got %T: %v", err, err)
	}
	if rej.Reason != jukebox.RejectUpstreamUnreachable {
		t.Fatalf("Reason=%q, want %q", rej.Reason, jukebox.RejectUpstreamUnreachable)
	}
	if rej.RetryAfter != 7*time.Second {
		t.Fatalf("RetryAfter=%v, want 7s (from config)", rej.RetryAfter)
	}
}

// Test: after the upstream listener returns, the probe recovers and
// AcquireRoute starts succeeding again. Validates the recovery edge.
func TestLegacyRouter_AcquireRoute_RecoversAfterUpstreamReturns(t *testing.T) {
	// Bind first, hold the port reservation, then close so we can rebind
	// to the same port in a moment. SO_REUSEADDR isn't reliable across
	// platforms, so instead we capture an ephemeral port, close it,
	// rebind a new listener immediately. Brief race window between the
	// close and the rebind is tolerated — the probe's threshold model
	// handles it.
	port, listener := listenOnEphemeralPort(t)
	cfgYAML := `
vllm:
  port: ` + itoa(port) + `
  tcp_liveness:
    enabled: true
    interval: 20ms
    timeout: 50ms
    fail_threshold: 2
    recover_threshold: 1
models:
  m:
    path: "/models/m"
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}
	var tr inflight.Tracker
	coord := jukebox.NewCoordinator(cfg, stubManagerAlwaysReady{}, &tr, time.Now)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go coord.Run(ctx)

	router := jukebox.NewLegacyRouter(cfg, coord, &tr)
	router.StartLiveness(ctx)

	probe := router.LivenessProbeForTest()
	if probe == nil {
		t.Fatalf("probe nil")
	}

	// Take the listener away → probe demotes.
	_ = listener.Close()
	waitForAlive(t, probe, false, 2*time.Second)

	// Bring it back on the same port (best-effort; if rebind fails the
	// test is skipped because the port was reused by another test).
	l2, err := net.Listen("tcp", "127.0.0.1:"+itoa(port))
	if err != nil {
		t.Skipf("could not rebind same port (kernel TIME_WAIT or reuse): %v", err)
	}
	defer l2.Close()
	go func() {
		for {
			c, err := l2.Accept()
			if err != nil {
				return
			}
			_ = c.Close()
		}
	}()
	waitForAlive(t, probe, true, 2*time.Second)

	if _, err := router.AcquireRoute(context.Background(), "m", "req_after_recover"); err != nil {
		t.Fatalf("AcquireRoute after recovery: %v", err)
	}
}

// Test: when liveness is explicitly disabled in config, the gate is a
// no-op even if the upstream port is dead. Preserves backward-compat
// for operators who want to disable the new behaviour.
func TestLegacyRouter_LivenessDisabledOptOut(t *testing.T) {
	cfgYAML := `
vllm:
  port: 1
  tcp_liveness:
    enabled: false
models:
  m:
    path: "/models/m"
`
	cfg, err := config.Load([]byte(cfgYAML))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}
	var tr inflight.Tracker
	coord := jukebox.NewCoordinator(cfg, stubManagerAlwaysReady{}, &tr, time.Now)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go coord.Run(ctx)

	router := jukebox.NewLegacyRouter(cfg, coord, &tr)
	router.StartLiveness(ctx) // should be a no-op
	if router.LivenessProbeForTest() != nil {
		t.Fatalf("expected no probe when tcp_liveness.enabled=false")
	}

	// AcquireRoute should still succeed even though port 1 is unreachable.
	if _, err := router.AcquireRoute(context.Background(), "m", "req"); err != nil {
		t.Fatalf("AcquireRoute with liveness disabled: %v", err)
	}
}

// Compile-time guard: stubManagerAlwaysReady satisfies the Manager iface.
var _ jukebox.Manager = stubManagerAlwaysReady{}
