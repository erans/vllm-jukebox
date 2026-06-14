// Package health provides upstream-liveness probing for routed backends.
//
// Why TCP-level (not HTTP)? When the upstream process wedges on a CUDA
// illegal-memory-access or NCCL deadlock, the kernel may still hold the
// listening socket open while every HTTP request hangs indefinitely.
// Worse, the upstream's own /health endpoint can return 200 while the
// inference loop is dead (see vllm-project/vllm#45097 for the
// /health/decode work). A bare net.DialTimeout against the listening port
// is the cheapest signal that the process is at least answering syscalls.
package health

import (
	"context"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Default tuning. Aggressive enough to fence off a dead upstream in under
// 15s of wall-clock; loose enough that a single flaky DialTimeout (e.g. a
// busy GC pause on the upstream) won't flip the gauge.
const (
	DefaultInterval         = 5 * time.Second
	DefaultTimeout          = 2 * time.Second
	DefaultFailThreshold    = 3
	DefaultRecoverThreshold = 1
)

// Config controls a single TCPProbe.
type Config struct {
	// Target is the host:port to dial (e.g. "127.0.0.1:8000").
	Target string

	// Interval between probe attempts. Defaults to DefaultInterval.
	Interval time.Duration

	// Timeout per dial attempt. Defaults to DefaultTimeout.
	Timeout time.Duration

	// FailThreshold is the number of consecutive failures that must be
	// observed before the probe transitions Alive() from true to false.
	// Defaults to DefaultFailThreshold.
	FailThreshold int

	// RecoverThreshold is the number of consecutive successes required to
	// transition Alive() from false back to true. Defaults to
	// DefaultRecoverThreshold.
	RecoverThreshold int

	// Dialer is injectable for tests. Nil means use a real net.Dialer.
	Dialer func(ctx context.Context, network, addr string) (net.Conn, error)

	// OnTransition, if non-nil, is invoked whenever Alive() flips. Useful
	// for emitting structured log lines or Prometheus state-change counters
	// at the call site. It MUST NOT block.
	OnTransition func(alive bool)

	// Now is the clock function for tests. Nil means time.Now.
	Now func() time.Time
}

// TCPProbe periodically TCP-dials a target and exposes its liveness
// asynchronously. The zero value is not usable; construct with NewTCPProbe.
type TCPProbe struct {
	cfg Config

	// alive is 1 when the probe considers the target alive, 0 otherwise.
	// New probes start alive=1 (optimistic) so request flow is not blocked
	// while we collect the first sample.
	alive atomic.Int32

	// startedAlive captures the initial optimistic state for tests.
	startedAlive bool

	mu        sync.Mutex
	failCount int
	okCount   int
	lastCheck time.Time
	lastError error
}

// NewTCPProbe constructs a probe but does not start its background loop.
// Call Run(ctx) to start probing; it returns when ctx is cancelled.
func NewTCPProbe(cfg Config) *TCPProbe {
	if cfg.Interval <= 0 {
		cfg.Interval = DefaultInterval
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if cfg.FailThreshold <= 0 {
		cfg.FailThreshold = DefaultFailThreshold
	}
	if cfg.RecoverThreshold <= 0 {
		cfg.RecoverThreshold = DefaultRecoverThreshold
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Dialer == nil {
		d := &net.Dialer{}
		cfg.Dialer = d.DialContext
	}
	p := &TCPProbe{cfg: cfg, startedAlive: true}
	p.alive.Store(1)
	return p
}

// Alive returns true if the target is currently considered reachable.
func (p *TCPProbe) Alive() bool {
	return p.alive.Load() == 1
}

// Target returns the configured host:port (useful for logging).
func (p *TCPProbe) Target() string { return p.cfg.Target }

// Snapshot returns the current internal counters; primarily for tests
// and /status surfaces.
func (p *TCPProbe) Snapshot() (alive bool, failCount, okCount int, lastCheck time.Time, lastErr error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.Alive(), p.failCount, p.okCount, p.lastCheck, p.lastError
}

// Run blocks until ctx is cancelled, performing a probe every Interval.
// Safe to call exactly once per TCPProbe.
func (p *TCPProbe) Run(ctx context.Context) {
	// Take one immediate sample so callers don't have to wait Interval
	// before the first signal — important on startup when an upstream
	// might already be down.
	p.probeOnce(ctx)

	t := time.NewTicker(p.cfg.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			p.probeOnce(ctx)
		}
	}
}

func (p *TCPProbe) probeOnce(ctx context.Context) {
	dialCtx, cancel := context.WithTimeout(ctx, p.cfg.Timeout)
	defer cancel()

	conn, err := p.cfg.Dialer(dialCtx, "tcp", p.cfg.Target)
	now := p.cfg.Now()

	if err == nil && conn != nil {
		_ = conn.Close()
	}

	p.mu.Lock()
	p.lastCheck = now
	p.lastError = err
	var transitionTo *bool
	if err == nil {
		p.failCount = 0
		p.okCount++
		if p.alive.Load() == 0 && p.okCount >= p.cfg.RecoverThreshold {
			t := true
			transitionTo = &t
			p.alive.Store(1)
		}
	} else {
		p.okCount = 0
		p.failCount++
		if p.alive.Load() == 1 && p.failCount >= p.cfg.FailThreshold {
			t := false
			transitionTo = &t
			p.alive.Store(0)
		}
	}
	p.mu.Unlock()

	if transitionTo != nil && p.cfg.OnTransition != nil {
		p.cfg.OnTransition(*transitionTo)
	}
}

// FormatTarget joins host and port into the form expected by net.Dial.
// Exists so callers don't have to reach for net.JoinHostPort.
func FormatTarget(host string, port int) string {
	return net.JoinHostPort(host, strconv.Itoa(port))
}
