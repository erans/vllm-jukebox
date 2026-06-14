// Decode-stall probe — DETECTION + RECOVERY half of the
// vllm-project/vllm#45094 PP-cudagraph-split-brain wedge fix.
//
// THE BUG (py-spy-proven 2026-06-14, bundle
// nccl-wedge-forensics-20260614T183050Z.md): on a TP×PP vLLM engine
// running cudagraph_mode:PIECEWISE, a large enough admitted batch shape
// makes pipeline stage 0 replay a captured CUDA graph (with baked-in
// NCCL P2P send/recv shapes) while stage 1 runs the eager path with a
// different effective shape. The cross-stage pipeline P2P never
// rendezvous → all GPUs pin at 100% util @ 0 tok/s forever, CPU futex-
// waits in cudaStreamSynchronize. Crucially the FastAPI process stays
// alive and /health keeps answering 200 (the lie), so neither docker's
// healthcheck nor jukebox's 5xx circuit breaker ever sees a failure —
// the wedge is silent and only `docker rm -f` + recreate recovers.
//
// THE PROBE: PR vllm#45097 added GET /health/decode, which returns
// non-200 specifically when the engine has running requests but is
// making no decode progress (exactly the #45094 signature). jukebox's
// :latest image predates that endpoint being wired into anything, so
// this probe is the consumer. On a SUSTAINED stall it marks the
// instance draining (out of routing) and fires the circuit breaker's
// docker-restart path — turning a silent infinite wedge into a bounded
// ~5-min self-heal.
//
// FALSE-POSITIVE GUARDS (the /health/decode PR documents a long-prefill
// caveat — a legitimately long prefill momentarily shows no decode
// progress):
//   - default interval 30s, NOT 15s (BehaviorConfig.DecodeProbeInterval);
//   - require the stall to persist across >= sustainedStallIntervals
//     (=2) consecutive probes before tripping;
//   - require in-flight > 0 during the stall window (a stall with zero
//     in-flight is just an idle engine, not a wedge);
//   - endpoint-absent (404 / connection-refused) graceful-degrades to a
//     debug log + skip — NEVER trips (older vLLM, or a model whose
//     image lacks the endpoint).
package jukebox

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// sustainedStallIntervals is the number of CONSECUTIVE stalled probes
// required before the probe trips. With the default 30s interval this is
// ~60s of confirmed no-decode-progress, which clears the longest honest
// long-prefill the /health/decode PR warns about while still catching a
// real #45094 wedge promptly.
const sustainedStallIntervals = 2

// decodeProbeHTTPTimeout bounds a single /health/decode GET. A wedged
// engine's FastAPI loop still answers /health/decode quickly (the
// endpoint inspects scheduler counters, it doesn't run inference), so a
// short timeout is correct; a timeout itself is treated as a stall
// signal (the engine is too sick to answer even this).
const decodeProbeHTTPTimeout = 3 * time.Second

// decodeProbeResult classifies one /health/decode GET.
type decodeProbeResult int

const (
	// decodeHealthy: HTTP 200 — engine is decoding (or legitimately idle).
	decodeHealthy decodeProbeResult = iota
	// decodeStalled: a non-200 status that indicates a decode stall
	// (the #45094 signature), OR a transport timeout (engine too sick to
	// answer). Counts toward the sustained-stall window.
	decodeStalled
	// decodeEndpointAbsent: 404 (endpoint not implemented by this vLLM)
	// or connection-refused (socket gone — a DIFFERENT failure the 5xx
	// breaker / TCP-liveness already own). Graceful-degrade: never trips.
	decodeEndpointAbsent
)

// decodeProbeFn is the injectable HTTP shape for the probe so tests can
// drive it without a real server. Returns the HTTP status code, or an
// error for transport failures (the classifier maps connection-refused
// → endpoint-absent and any other transport error → stalled).
type decodeProbeFn func(ctx context.Context, baseURL string) (status int, err error)

// decodeProbeHook is the package-level probe hook. Defaults to a real
// HTTP GET against <baseURL>/health/decode. Tests override via
// SetDecodeProbeFnForTest (see decode_probe_export_test.go). Stored as an
// atomic.Pointer so concurrent reads (the probe goroutine) and writes
// (a parallel test) are race-detector clean.
var decodeProbeHook atomic.Pointer[decodeProbeFn]

// defaultDecodeProbe issues GET <baseURL>/health/decode with a short
// timeout and returns the status code.
func defaultDecodeProbe(ctx context.Context, baseURL string) (int, error) {
	url := strings.TrimRight(baseURL, "/") + "/health/decode"
	reqCtx, cancel := context.WithTimeout(ctx, decodeProbeHTTPTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	_ = resp.Body.Close()
	return resp.StatusCode, nil
}

func runDecodeProbe(ctx context.Context, baseURL string) (int, error) {
	if fn := decodeProbeHook.Load(); fn != nil {
		return (*fn)(ctx, baseURL)
	}
	return defaultDecodeProbe(ctx, baseURL)
}

// classifyDecodeProbe maps a (status, err) probe outcome to a result.
// Connection-refused → endpoint-absent (a different, already-owned
// failure); any other transport error (timeout, EOF) → stalled (the
// engine is too sick to answer). HTTP 404 → endpoint-absent; HTTP 200 →
// healthy; any other status → stalled.
func classifyDecodeProbe(status int, err error) decodeProbeResult {
	if err != nil {
		if isConnectionRefused(err) {
			return decodeEndpointAbsent
		}
		return decodeStalled
	}
	switch {
	case status == http.StatusOK:
		return decodeHealthy
	case status == http.StatusNotFound:
		return decodeEndpointAbsent
	default:
		return decodeStalled
	}
}

func isConnectionRefused(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, net.ErrClosed) {
		return true
	}
	// errors.Is against syscall.ECONNREFUSED would pull in syscall on
	// every platform; a substring match on the wrapped dial error is
	// portable and sufficient for the graceful-degrade decision (the
	// only consequence of a misclassification here is "trip vs skip" on
	// a socket that's down, and a down socket is already caught by the
	// TCP-liveness probe + the 5xx breaker — so skewing toward skip is
	// safe).
	msg := err.Error()
	return strings.Contains(msg, "connection refused") || strings.Contains(msg, "no such host")
}

// DecodeProbeMonitor is the background goroutine launched from
// cmd/jukebox/main.go. It ticks at behavior.decode_probe_interval and
// probes every eligible instance each tick. Returns immediately (no-op)
// when the interval is 0 (disabled) — the caller gates on that too, but
// guarding here keeps the method safe to call unconditionally.
func (s *Scheduler) DecodeProbeMonitor(ctx context.Context) {
	if s == nil {
		return
	}
	interval := s.decodeProbeInterval()
	if interval <= 0 {
		slog.Info("decode_probe_disabled", "reason", "behavior.decode_probe_interval is 0")
		return
	}
	slog.Info("decode_probe_enabled",
		"interval_s", interval.Seconds(),
		"sustained_intervals", sustainedStallIntervals,
	)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.checkDecodeStalls(ctx)
		}
	}
}

// decodeProbeInterval reads the live (hot-reloadable) probe interval.
func (s *Scheduler) decodeProbeInterval() time.Duration {
	cfg := liveCfg(s.cfg)
	if cfg == nil {
		return 0
	}
	return cfg.Behavior.EffectiveDecodeProbeInterval()
}

// decodeStallCounts tracks consecutive-stall counts per model across
// probe ticks. Guarded by mu (the scheduler's main lock) — written only
// from checkDecodeStalls which runs single-threaded on the monitor
// goroutine, but read under lock to stay race-clean with any future
// caller. Lazily initialized.
//
// (Declared as a field on Scheduler via decode_probe_state.go-style
// embedding would be cleaner, but the struct is defined in scheduler.go;
// we attach the map there.)

// checkDecodeStalls probes every eligible external-lifecycle Ready
// instance once and advances/clears its sustained-stall counter.
func (s *Scheduler) checkDecodeStalls(ctx context.Context) {
	// Snapshot eligible instances under the read lock.
	type probeTarget struct {
		model    string
		baseURL  string
		inflight int64
	}
	var targets []probeTarget

	s.mu.RLock()
	for _, inst := range s.instances {
		if inst == nil || inst.mgr == nil {
			continue
		}
		if inst.state != StateReady || inst.draining {
			continue
		}
		if inst.mgr.CurrentPID() == 0 {
			continue
		}
		// External-lifecycle only: the probe drives the circuit breaker's
		// docker-restart, which targets a docker-compose-managed sibling
		// container (ModelConfig.Container). Managed-lifecycle instances
		// are owned by jukebox's own process supervisor and recovered by
		// a different path.
		mc, ok := liveModelCfg(s.cfg, inst.model)
		if !ok || mc.EffectiveLifecycle() != config.LifecycleExternal {
			continue
		}
		baseURL := inst.mgr.BaseURL()
		if baseURL == "" {
			continue
		}
		targets = append(targets, probeTarget{
			model:    inst.model,
			baseURL:  baseURL,
			inflight: inst.inflight.Count(),
		})
	}
	s.mu.RUnlock()

	for _, t := range targets {
		status, err := runDecodeProbe(ctx, t.baseURL)
		result := classifyDecodeProbe(status, err)

		switch result {
		case decodeEndpointAbsent:
			// Graceful degrade — older vLLM without /health/decode, or
			// socket gone (owned by TCP-liveness). Never trip; reset any
			// accumulated stall count so a later real stall starts fresh.
			s.resetDecodeStall(t.model)
			metrics.DecodeStallsTotal.WithLabelValues(t.model, "endpoint_absent").Inc()
			slog.Debug("decode_probe_endpoint_absent",
				"model", t.model,
				"status", status,
				"err", errString(err),
			)
		case decodeHealthy:
			s.resetDecodeStall(t.model)
		case decodeStalled:
			// A stall only counts toward a trip when there is in-flight
			// work — a stalled /health/decode with zero in-flight is just
			// an idle engine, not the #45094 wedge (which is BY DEFINITION
			// running>0 with 0 decode progress).
			if t.inflight <= 0 {
				s.resetDecodeStall(t.model)
				slog.Debug("decode_probe_stall_ignored_no_inflight",
					"model", t.model,
					"status", status,
				)
				continue
			}
			count := s.bumpDecodeStall(t.model)
			slog.Warn("decode_probe_stall_observed",
				"model", t.model,
				"status", status,
				"err", errString(err),
				"inflight", t.inflight,
				"consecutive", count,
				"trip_at", sustainedStallIntervals,
			)
			if count < sustainedStallIntervals {
				metrics.DecodeStallsTotal.WithLabelValues(t.model, "transient").Inc()
				continue
			}
			// Sustained stall: trip.
			s.tripDecodeStall(ctx, t.model)
			s.resetDecodeStall(t.model)
			metrics.DecodeStallsTotal.WithLabelValues(t.model, "tripped").Inc()
		}
	}
}

// tripDecodeStall marks the model's instance draining (out of routing)
// and fires the circuit breaker's docker-restart path. Both are
// best-effort; the order (drain THEN trip) is deliberate so no new
// request is admitted to a confirmed-wedged engine while the restart is
// in flight, and ForceTrip bypasses the breaker's ready-gate so the
// just-drained instance still trips.
func (s *Scheduler) tripDecodeStall(_ context.Context, model string) {
	s.mu.Lock()
	if inst := s.instances[model]; inst != nil {
		inst.draining = true
	}
	cb := s.cb
	s.mu.Unlock()

	slog.Error("decode_probe_trip",
		"model", model,
		"action", "drain + circuit-breaker docker restart",
		"ref", "vllm#45094",
	)
	LogLifecycleTransition(LifecycleEvent{
		Action: LifecycleCircuitBreakerTrip,
		Reason: "decode_stall_45094",
		Model:  model,
	})

	if cb != nil {
		cb.ForceTrip(model)
	}
}

func (s *Scheduler) bumpDecodeStall(model string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.decodeStallCounts == nil {
		s.decodeStallCounts = map[string]int{}
	}
	s.decodeStallCounts[model]++
	return s.decodeStallCounts[model]
}

func (s *Scheduler) resetDecodeStall(model string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.decodeStallCounts != nil {
		delete(s.decodeStallCounts, model)
	}
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
