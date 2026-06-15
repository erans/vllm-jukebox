package jukebox

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/health"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/metrics"
)

type LegacyRouter struct {
	cfg   *config.Config
	coord *Coordinator
	tr    *inflight.Tracker

	// liveness is non-nil when TCP liveness probing is enabled. It is
	// started lazily on the first call to AcquireRoute (or explicitly
	// via StartLiveness) so tests that don't actually run a server
	// don't open background goroutines they never asked for.
	livenessMu sync.Mutex
	liveness   *health.TCPProbe
	livenessOn bool
}

func NewLegacyRouter(cfg *config.Config, coord *Coordinator, tr *inflight.Tracker) *LegacyRouter {
	return &LegacyRouter{cfg: cfg, coord: coord, tr: tr}
}

func (r *LegacyRouter) Status() Status {
	if r == nil || r.coord == nil {
		return Status{State: StateIdle}
	}
	return r.coord.Status()
}

// StartLiveness initialises and starts the TCP liveness probe against the
// configured vllm.port. Safe to call multiple times — subsequent calls
// are no-ops. Pass a long-lived ctx tied to server shutdown.
func (r *LegacyRouter) StartLiveness(ctx context.Context) {
	if r == nil || r.cfg == nil {
		return
	}
	if r.cfg.VLLM.TCPLiveness.Enabled == nil || !*r.cfg.VLLM.TCPLiveness.Enabled {
		return
	}

	r.livenessMu.Lock()
	if r.livenessOn {
		r.livenessMu.Unlock()
		return
	}
	target := health.FormatTarget("127.0.0.1", r.cfg.VLLM.Port)
	probe := health.NewTCPProbe(health.Config{
		Target:           target,
		Interval:         r.cfg.VLLM.TCPLiveness.Interval.Duration,
		Timeout:          r.cfg.VLLM.TCPLiveness.Timeout.Duration,
		FailThreshold:    r.cfg.VLLM.TCPLiveness.FailThreshold,
		RecoverThreshold: r.cfg.VLLM.TCPLiveness.RecoverThreshold,
		OnTransition: func(alive bool) {
			direction := "down"
			gauge := 0.0
			if alive {
				direction = "up"
				gauge = 1.0
			}
			metrics.UpstreamTCPAlive.WithLabelValues(target).Set(gauge)
			metrics.UpstreamTCPTransitionsTotal.WithLabelValues(target, direction).Inc()
			slog.Warn(
				"upstream_tcp_liveness_transition",
				"target", target,
				"alive", alive,
			)
		},
	})
	// Seed the gauge optimistically so dashboards don't show "no data"
	// before the first transition.
	metrics.UpstreamTCPAlive.WithLabelValues(target).Set(1)
	r.liveness = probe
	r.livenessOn = true
	r.livenessMu.Unlock()

	go probe.Run(ctx)
}

// LivenessProbeForTest exposes the internal probe for white-box tests
// in the jukebox_test package. Not part of the public API surface.
func (r *LegacyRouter) LivenessProbeForTest() *health.TCPProbe {
	if r == nil {
		return nil
	}
	r.livenessMu.Lock()
	defer r.livenessMu.Unlock()
	return r.liveness
}

func (r *LegacyRouter) AcquireRoute(ctx context.Context, requestedModel, requestID string) (Route, error) {
	if r == nil || r.cfg == nil || r.coord == nil {
		return Route{}, fmt.Errorf("router not configured")
	}
	if err := r.coord.EnsureModel(ctx, requestedModel, requestID); err != nil {
		return Route{}, err
	}

	// Even if EnsureModel says the coordinator is "ready", the upstream
	// process may have wedged or crashed since it was last verified.
	// Gate the route acquisition on a live TCP probe so we never forward
	// a request to a dead listening port. This is the ONLY signal that
	// catches a process whose /health endpoint hangs alongside its
	// inference loop (vllm CUDA illegal-memory-access wedge, NCCL
	// deadlock, etc.).
	r.livenessMu.Lock()
	probe := r.liveness
	r.livenessMu.Unlock()
	if probe != nil && !probe.Alive() {
		retryAfter := time.Duration(r.cfg.VLLM.TCPLiveness.RetryAfterSeconds) * time.Second
		if retryAfter <= 0 {
			retryAfter = 10 * time.Second
		}
		metrics.SwapRejectionsTotal.WithLabelValues(string(RejectUpstreamUnreachable)).Inc()
		return Route{}, &RejectError{
			Reason:     RejectUpstreamUnreachable,
			RetryAfter: retryAfter,
			Message:    "upstream TCP probe reports peer unreachable",
		}
	}

	var done func()
	if r.tr != nil {
		// Legacy single-instance router never seals its tracker, so ok is
		// always true here — discard it.
		done, _ = r.tr.Track(ctx)
	}

	_, modelCfg, err := liveResolveModel(r.cfg, requestedModel)
	if err != nil {
		return Route{}, err
	}

	return Route{
		BaseURL:       fmt.Sprintf("http://127.0.0.1:%d", r.cfg.VLLM.Port),
		UpstreamModel: modelCfg.Path,
		Done:          done,
	}, nil
}

// LegacyRouter intentionally does NOT implement the optional
// ColdLoadAware interface. Swap mode has no admissionStopped concept
// (the single-instance coordinator does not stop containers; it only
// swaps processes), so the async-503 + KickColdLoad path is structurally
// inapplicable. HTTP handlers type-assert at the callsite — when the
// assertion fails the handler skips the async-503 branch and falls
// through to AcquireRoute, which is exactly the behavior swap mode
// needs (and exactly what the no-op stubs used to fake).
