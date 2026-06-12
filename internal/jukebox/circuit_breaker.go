// Circuit breaker — jukebox-native replacement for the external
// 5xx-watching sidecar. Records every upstream HTTP status the proxy
// returned, and when THRESHOLD consecutive 5xx land for a model the
// breaker `docker restart`s that model's container — IF admission
// reports the model Ready (NOT sleeping/stopped), IF the cooldown has
// elapsed since the last trip, and IF the crash-loop guard hasn't
// frozen the container.
//
// Why in jukebox and not as a sidecar:
//   - Jukebox already knows live admission/sleep/stopped state without
//     a polling round-trip to /status.
//   - Reuses the same `runDocker` exec hook redeploy uses (testable
//     by swapping the package-level atomic.Pointer).
//   - One-process operations posture: no extra systemd unit to babysit.
//
// What the breaker explicitly does NOT do:
//   - Inspect response bodies. Status code only.
//   - Restart on 4xx (caller error, not engine death).
//   - Trip on sleeping/stopped models — those states are
//     jukebox-managed and a "5xx" while stopped is the async-503
//     cold-load contract, not a crash signal.
//   - Race the cold-load lock. The breaker calls docker restart
//     unilaterally; the cold-load lock guards the wake-from-Stopped
//     path which doesn't apply here (an engine in distress is
//     admissionAwake, not admissionStopped).
package jukebox

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
)

// CircuitBreaker observes per-model response status streams and
// restarts misbehaving containers. Safe for concurrent use.
//
// Lifecycle: constructed once at scheduler startup with a snapshot of
// the live config (a pointer is held; reads always go through
// liveCfg() so hot-reload picks up new thresholds without rebuild).
// Hand to RecordResponse from the HTTP handler; the breaker does the
// rest async.
type CircuitBreaker struct {
	// liveCfg returns the current config — wired to config.Current()
	// so behavior.circuit_breaker.* hot-reloads work without a
	// breaker rebuild.
	liveCfg func() *config.Config

	// readyChecker reports whether a given model is currently Ready
	// (engine running, not sleeping/stopped/idle). Wired to the
	// Scheduler's Status() walker. Returns false for unknown models
	// or transient lookup failures — never trips on unclear state.
	readyChecker func(model string) bool

	// now is a clock injection point for tests.
	now func() time.Time

	mu sync.Mutex

	// statuses keeps the most-recent Window response codes per model,
	// newest-first. Trimmed to Window on every push.
	statuses map[string][]int

	// lastTripAt is the most recent trip time per container — used
	// by the cooldown gate.
	lastTripAt map[string]time.Time

	// recentTrips records trip timestamps per container within the
	// crash-loop window. Pruned on every check.
	recentTrips map[string][]time.Time

	// lastHealthyAt records the timestamp of the most recent 2xx
	// response observed per container. Cold-load grace gate: a trip
	// only fires if lastHealthyAt[container] is set AND is more recent
	// than lastTripAt[container] — i.e. we have evidence the container
	// has actually served traffic in its current incarnation, post any
	// prior breaker restart.
	//
	// This is the fix for the 2026-06-12 02:00Z incident where the
	// breaker false-positive-tripped vllm-main's cold-load infinitely
	// because /is_sleeping connect-refused during the 6-12min cold-load
	// looked like a 5xx burst. The timestamp comparison naturally
	// handles the post-restart race (a stale 2xx from the dying engine
	// landing AFTER trip-fire is excluded because its timestamp is
	// earlier than lastTripAt).
	lastHealthyAt map[string]time.Time

	// firstSeenAt records the timestamp of the FIRST response observed
	// for each container (any status). Bounds the cold-load grace gate:
	// after ColdLoadGraceMaxSeconds since first-seen, the grace gate
	// exits even without a prior 2xx — protects against born-broken
	// containers (bad weights / OOM-at-init / missing config) that
	// would otherwise silently suppress every trip forever.
	firstSeenAt map[string]time.Time
}

// NewCircuitBreaker builds a breaker. Both liveCfg and readyChecker
// MUST be non-nil. now=nil falls back to time.Now.
func NewCircuitBreaker(liveCfg func() *config.Config, readyChecker func(model string) bool, now func() time.Time) *CircuitBreaker {
	if now == nil {
		now = time.Now
	}
	return &CircuitBreaker{
		liveCfg:       liveCfg,
		readyChecker:  readyChecker,
		now:           now,
		statuses:      map[string][]int{},
		lastTripAt:    map[string]time.Time{},
		recentTrips:   map[string][]time.Time{},
		lastHealthyAt: map[string]time.Time{},
		firstSeenAt:   map[string]time.Time{},
	}
}

// RecordResponse pushes one upstream response status into the
// breaker's window for `model`. Cheap (one map write + slice append);
// safe to call on every proxied request. Decides synchronously
// whether to trip — the docker restart itself runs on a goroutine
// (it can take ~5s for the SIGTERM grace).
//
// Models with no container configured silently observe-only (so the
// metric still reflects request volume) but never trip — the trip
// has nothing to act on.
//
// Status 0 means "the proxy never reached the upstream" (network
// error, timeout, ctx cancel). Treated as a 5xx — a dead engine is
// the exact failure mode this breaker exists for.
func (cb *CircuitBreaker) RecordResponse(model string, status int) {
	cfg := cb.liveCfg()
	if cfg == nil {
		return
	}
	cbcfg := cfg.Behavior.CircuitBreaker
	if !cbcfg.Enabled {
		return
	}

	// Map status 0 (proxy never reached upstream) to 5xx for the
	// observation bucket so dashboards see it where it belongs.
	bucket := classifyStatus(status)
	metrics.CircuitBreakerObservedTotal.WithLabelValues(model, bucket).Inc()

	// Find the container — empty container = observe-only.
	container := containerForModel(cfg, model)
	if container == "" {
		return
	}

	cb.mu.Lock()
	now := cb.now()
	// Track first-ever response for this container — bounds the
	// cold-load grace gate (see ColdLoadGraceMaxSeconds in config).
	if _, seen := cb.firstSeenAt[container]; !seen {
		cb.firstSeenAt[container] = now
	}
	// Latch the cold-load grace gate: any 2xx proves the container
	// has serviced traffic. The gate compares this timestamp against
	// lastTripAt so a stale 2xx from a dying engine landing AFTER
	// trip-fire (its wall-clock is earlier than the trip) is
	// correctly excluded.
	if status >= 200 && status < 300 {
		cb.lastHealthyAt[container] = now
	}
	statuses := append([]int{status}, cb.statuses[model]...)
	if len(statuses) > cbcfg.Window {
		statuses = statuses[:cbcfg.Window]
	}
	cb.statuses[model] = statuses

	// Trip decision needs Window-many samples + Threshold-many leading 5xx.
	shouldTrip := len(statuses) >= cbcfg.Threshold
	if shouldTrip {
		for i := 0; i < cbcfg.Threshold; i++ {
			if !is5xx(statuses[i]) {
				shouldTrip = false
				break
			}
		}
	}
	cb.mu.Unlock()

	if !shouldTrip {
		return
	}

	// Pre-trip gates and the actual docker restart run async — the
	// hot path (handler return) doesn't block on Ready-check + docker
	// exec round-trips.
	go cb.evaluateAndTrip(model, container, cbcfg)
}

// evaluateAndTrip runs the gate stack (ready, cold-load grace, cooldown,
// crash-loop, dry-run) and either fires the docker restart or records
// the suppression reason. Caller is RecordResponse on its own goroutine.
func (cb *CircuitBreaker) evaluateAndTrip(model, container string, cbcfg config.CircuitBreakerConfig) {
	if !cb.readyChecker(model) {
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "suppressed_not_ready").Inc()
		return
	}

	now := cb.now()
	cooldown := time.Duration(cbcfg.CooldownSeconds) * time.Second
	crashWindow := time.Duration(cbcfg.CrashLoopWindowSeconds) * time.Second
	graceMax := time.Duration(cbcfg.ColdLoadGraceMaxSeconds) * time.Second

	// Cold-load grace gate: a trip is allowed ONLY if either
	//   (a) we have observed a 2xx from this container more recently
	//       than the most recent trip (proves the engine has actually
	//       served traffic in its current incarnation), OR
	//   (b) the grace window has elapsed since we first saw a response
	//       (born-broken safety valve — bad weights / OOM-at-init /
	//       missing config would otherwise silently suppress forever).
	// Both reads happen under the same lock as the lastTripAt read so
	// the gate is consistent.
	cb.mu.Lock()
	lastHealthy := cb.lastHealthyAt[container]
	firstSeen := cb.firstSeenAt[container]
	lastTrip := cb.lastTripAt[container]
	cb.mu.Unlock()

	healthyAfterTrip := !lastHealthy.IsZero() && lastHealthy.After(lastTrip)
	graceExceeded := graceMax > 0 && !firstSeen.IsZero() && now.Sub(firstSeen) > graceMax
	if !healthyAfterTrip && !graceExceeded {
		log.Printf("circuit-breaker: cold-load grace container=%s model=%s — suppressing trip (no 2xx observed in this incarnation, grace window not exceeded)", container, model)
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "suppressed_cold_load").Inc()
		return
	}

	cb.mu.Lock()
	if last, ok := cb.lastTripAt[container]; ok && now.Sub(last) < cooldown {
		cb.mu.Unlock()
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "suppressed_cooldown").Inc()
		return
	}

	// Prune crash-loop window then check.
	trips := cb.recentTrips[container]
	kept := trips[:0]
	cutoff := now.Add(-crashWindow)
	for _, t := range trips {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	cb.recentTrips[container] = kept
	if len(kept) >= cbcfg.CrashLoopMaxTrips {
		cb.mu.Unlock()
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "suppressed_crashloop").Inc()
		log.Printf("circuit-breaker: CRASH-LOOP frozen container=%s model=%s (%d trips in last %s) — operator intervention required",
			container, model, len(kept), crashWindow)
		return
	}

	// Reserve the trip slot before releasing the lock so concurrent
	// goroutines for the same container don't double-fire.
	cb.lastTripAt[container] = now
	cb.recentTrips[container] = append(kept, now)
	// Clear status history so the next 5xx burst counts fresh —
	// otherwise the same stale window can re-trigger immediately
	// after the cooldown elapses without any new failures.
	delete(cb.statuses, model)
	cb.mu.Unlock()

	if cbcfg.DryRun {
		log.Printf("circuit-breaker: [DRY_RUN] would restart container=%s model=%s", container, model)
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "suppressed_dryrun").Inc()
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cbcfg.RestartTimeoutSeconds)*time.Second)
	defer cancel()
	log.Printf("circuit-breaker: TRIP container=%s model=%s — docker restart -t 5", container, model)
	out, err := runDocker(ctx, "docker", "restart", "-t", "5", container)
	if err != nil {
		metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "failed").Inc()
		log.Printf("circuit-breaker: TRIP FAILED container=%s model=%s err=%v out=%q",
			container, model, err, truncate(out, 200))
		return
	}
	metrics.CircuitBreakerTripsTotal.WithLabelValues(model, "fired").Inc()
	log.Printf("circuit-breaker: TRIP FIRED container=%s model=%s out=%q", container, model, truncate(out, 200))
	// No explicit latch-clear needed — the timestamp comparison in the
	// grace gate (lastHealthyAt.After(lastTripAt)) naturally requires a
	// fresh post-restart 2xx before the next trip can fire. This avoids
	// the stale-2xx-races-the-delete window that an explicit clear had.
	LogLifecycleTransition(LifecycleEvent{
		Action: LifecycleCircuitBreakerTrip,
		Reason: "5xx_consecutive",
		Model:  model,
	})
}

// containerForModel resolves the docker container backing the model,
// honoring resolved-name aliasing the same way ResolveModel does.
// Returns empty string when the model is unknown or has no container
// configured.
func containerForModel(cfg *config.Config, model string) string {
	if cfg == nil {
		return ""
	}
	resolved, mc, err := cfg.ResolveModel(model)
	if err != nil {
		return ""
	}
	// Prefer the model's own field; fall back to resolved-name lookup
	// in case the operator only set it on the alias target.
	if mc.Container != "" {
		return mc.Container
	}
	if rc, ok := cfg.Models[resolved]; ok && rc.Container != "" {
		return rc.Container
	}
	return ""
}

func is5xx(status int) bool {
	// Status 0 = proxy never reached upstream (network error / timeout
	// / ctx cancel mid-flight). Same failure category as a 5xx from the
	// breaker's POV.
	if status == 0 {
		return true
	}
	return status >= 500 && status < 600
}

func classifyStatus(status int) string {
	switch {
	case status == 0:
		return "5xx"
	case status >= 500:
		return "5xx"
	case status >= 400:
		return "4xx"
	case status >= 200:
		return "2xx"
	default:
		return "other"
	}
}

func truncate(b []byte, n int) string {
	s := strings.TrimSpace(string(b))
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// --- Scheduler wireup ---

// EnableCircuitBreaker wires the in-proxy circuit breaker into this
// Scheduler. Idempotent — re-calls replace the prior breaker (its
// state is dropped, which is fine: trip/cooldown bookkeeping is per
// breaker and a fresh window is the safest re-init).
//
// liveCfg is normally config.Current — passed in so this package
// stays uncoupled from the config singleton (admission_test, etc.
// don't have config.Current set up).
func (s *Scheduler) EnableCircuitBreaker(liveCfg func() *config.Config) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cb = NewCircuitBreaker(liveCfg, s.isModelReady, s.now)
	// Visibility marker: confirms the cold-load grace gate is the
	// timestamp-based variant (post 2026-06-12 02:00Z incident fix),
	// so an operator re-arming the breaker after a deploy can verify
	// from logs that the new code is live before flipping
	// behavior.circuit_breaker.enabled: true.
	log.Printf("circuit-breaker: ENABLED — cold-load grace gate active (timestamp-based latch + grace TTL)")
}

// RecordResponse implements the ResponseRecorder interface — HTTP
// handlers call this with every upstream response status. No-op when
// the breaker isn't enabled.
func (s *Scheduler) RecordResponse(model string, status int) {
	s.mu.RLock()
	cb := s.cb
	s.mu.RUnlock()
	if cb == nil {
		return
	}
	cb.RecordResponse(model, status)
}

// isModelReady walks Scheduler.Status() looking for the resolved-name
// instance and reports whether it's in StateReady with a live process.
// Used as the breaker's readiness gate. Returns false on any
// resolution failure (unknown model, missing instance, etc.) so
// uncertainty never trips.
func (s *Scheduler) isModelReady(model string) bool {
	cfg := s.cfg
	if cfg == nil {
		return false
	}
	resolved, _, err := cfg.ResolveModel(model)
	if err != nil {
		return false
	}
	st := s.Status()
	for _, inst := range st.Instances {
		if inst.Model != resolved {
			continue
		}
		return inst.State == StateReady && inst.PID != 0 && !inst.Draining
	}
	return false
}

// RecordResponse on LegacyRouter is a no-op. Swap mode owns the
// single vLLM process directly — when it goes south the Coordinator
// state machine catches it (failureCount + restart-on-error), so the
// docker-restart breaker (which targets sibling containers) doesn't
// apply. Satisfies ResponseRecorder so handlers can call it
// uniformly without a swap-vs-scheduler branch.
func (r *LegacyRouter) RecordResponse(model string, status int) {}

