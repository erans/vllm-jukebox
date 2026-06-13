package metrics

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// Counter: requests by model and status
	RequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_requests_total",
			Help: "Total HTTP requests handled by Jukebox.",
		},
		[]string{"model", "status"},
	)

	// Counter: model swaps
	SwapsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_swaps_total",
			Help: "Total model swaps.",
		},
		[]string{"from", "to"},
	)

	// Counter: swap rejections by reason
	SwapRejectionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_swap_rejections_total",
			Help: "Total swap rejections by reason.",
		},
		[]string{"reason"},
	)

	// Histogram: swap duration
	SwapDurationSeconds = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "jukebox_swap_duration_seconds",
			Help:    "Duration of model swaps in seconds.",
			Buckets: prometheus.DefBuckets,
		},
	)

	// Histogram: request duration
	RequestDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_request_duration_seconds",
			Help:    "HTTP request duration in seconds.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// Gauge: current state (1 = active)
	State = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_state",
			Help: "Current vLLM state (1 = active).",
		},
		[]string{"state"},
	)

	// Gauge: in-flight requests
	InFlightRequests = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "jukebox_in_flight_requests",
			Help: "Number of in-flight proxied requests.",
		},
	)

	// Gauge: current model (1 = loaded)
	CurrentModel = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_current_model",
			Help: "Currently loaded model (1 = loaded).",
		},
		[]string{"model"},
	)

	// Gauge: consecutive failures
	ConsecutiveFailures = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "jukebox_consecutive_failures",
			Help: "Consecutive vLLM start/swap failures.",
		},
	)

	// Gauge: running instances (scheduler mode)
	RunningInstances = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_running_instances",
			Help: "Running vLLM instances (1 = running).",
		},
		[]string{"model", "port"},
	)

	// Gauge: per-instance in-flight requests (scheduler mode)
	InstanceInFlightRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_instance_in_flight_requests",
			Help: "In-flight proxied requests per instance.",
		},
		[]string{"model", "port"},
	)

	// Counter: evictions (scheduler mode)
	EvictionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_evictions_total",
			Help: "Total instance evictions.",
		},
		[]string{"model"},
	)

	// Counter: scheduler rejections (scheduler mode)
	ScheduleRejectionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_schedule_rejections_total",
			Help: "Total scheduler placement rejections by reason.",
		},
		[]string{"reason"},
	)

	// Counter: sleep operations (L1/L2 vLLM /sleep API calls only).
	// Container-level docker-stop events are tracked separately in
	// StopsTotal — they are NOT double-counted here. Prior versions
	// bumped this counter with reason="stop:<x>" alongside StopsTotal,
	// causing Grafana panels that summed jukebox_sleeps_total to
	// overcount sleeps by the stop-event rate.
	SleepsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_sleeps_total",
			Help: "Total vLLM /sleep API operations (L1/L2 sleep events only — for container stop events see jukebox_vllm_stops_total). " +
				"Migration: rewrite Grafana queries from sum(rate(jukebox_sleeps_total{reason=~\"stop:.*\"}[5m])) to sum(rate(jukebox_vllm_stops_total[5m])).",
		},
		[]string{"model", "reason"},
	)

	// Counter: stop operations (container-level docker stop, distinct
	// from /sleep). Phase 4 stop-on-evict bumps this. SleepsTotal is
	// NO LONGER double-bumped with reason="stop:<x>" (removed to fix
	// dashboard overcount); StopsTotal is the canonical counter for
	// container-stop lifecycle events. Dashboards that previously read
	// `jukebox_sleeps_total{reason=~"stop:.*"}` must migrate to
	// `jukebox_vllm_stops_total`.
	StopsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_vllm_stops_total",
			Help: "Total container-level stop operations (docker stop) for evict_action: stop members, distinct from vLLM /sleep events.",
		},
		[]string{"model", "reason"},
	)

	// Counter: wake operations
	WakesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_wakes_total",
			Help: "Total vLLM wake operations.",
		},
		[]string{"model", "trigger"},
	)

	// Histogram: sleep duration
	SleepDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_sleep_duration_seconds",
			Help:    "Duration of sleep operations (drain + sleep HTTP call + GPU memory settle).",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// Histogram: wake duration
	WakeDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_wake_duration_seconds",
			Help:    "Duration of wake operations (wake HTTP call + /health poll).",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// Counter: sleep/wake failures
	SleepFailuresTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_sleep_failures_total",
			Help: "Total failures during sleep/wake operations.",
		},
		[]string{"model", "kind"},
	)

	// Counter: admission-initiated evictions (scheduler mode + sleep mode)
	AdmissionEvictsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_admission_evicts_total",
			Help: "Total peers slept by the admission controller to make room for an incoming wake.",
		},
		[]string{"model", "victim"},
	)

	// Counter: admission rejections (no feasible eviction set, evict failed, etc)
	AdmissionRejectionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_admission_rejections_total",
			Help: "Total admission rejections by reason.",
		},
		[]string{"model", "reason"},
	)

	// Counter: VRAM drift-risk events. Bumped when admission's books
	// diverge from physical container state because a best-effort
	// `docker stop` cleanup failed mid-cold-load — the half-loaded
	// container may still be running and holding tens of GiB of VRAM
	// while we cannot safely mark it Stopped. Operators must reconcile
	// manually (the container is in an unknown state — auto-retry could
	// stomp on a recovering process). Alertable: any non-zero rate
	// indicates a real-world drift between admission's view and the
	// running fleet.
	AdmissionVRAMDriftRiskTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_admission_vram_drift_risk_total",
			Help: "Total cold-load failures where the cleanup docker stop ALSO failed, leaving admission's view potentially inconsistent with container state.",
		},
		[]string{"model"},
	)

	// AdmissionPinnedWakeFailedTotal bumps when a cold-load failure
	// rollback could not re-wake a previously-slept pinned peer.
	// Symptom: pinned-default model is left Sleeping despite operator
	// expectation that "pinned ⇒ always Awake". Critical alertable —
	// any non-zero rate means a pinned model is silently down and the
	// admission cooldown was force-cleared so the operator's next
	// request OR /admin/redeploy-member can retry. See runbook
	// "manual-reconcile".
	AdmissionPinnedWakeFailedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_admission_pinned_wake_failed_total",
			Help: "Cold-load rollback could not re-wake a previously-slept pinned peer (pinned-default availability invariant violated).",
		},
		[]string{"model"},
	)

	// ColdLoadCooldownSuppressedTotal bumps when KickColdLoad short-
	// circuits a kick because the model's last cold-load failure is
	// still inside the coldLoadFailureCooldown window. Without this
	// counter the only signal is a slog.Warn line + a lifecycle audit
	// event — dashboards can't track "all my models are stuck in
	// cooldown" as a silent partial outage. Per-model labeled so
	// alerts can fire on a single wedged model.
	ColdLoadCooldownSuppressedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_cold_load_cooldown_suppressed_total",
			Help: "KickColdLoad invocations suppressed because the model is within the post-failure cooldown window.",
		},
		[]string{"model"},
	)

	// Histogram: admission wait time (decision + eviction wall-clock)
	AdmissionWaitSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_admission_wait_seconds",
			Help:    "Time the admission controller spent admitting a wake (includes eviction wall-clock).",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"model"},
	)

	// Gauge: per-GPU awake VRAM (pinned + awake non-pinned, MB)
	GPUBudgetAwakeMB = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_gpu_budget_awake_mb",
			Help: "Per-GPU declared awake VRAM (pinned + awake non-pinned), in MB. From admission controller bookkeeping, NOT live nvidia-smi.",
		},
		[]string{"gpu"},
	)

	// Gauge: per-GPU available VRAM (total - pinned - awake - L1 residual, MB)
	GPUBudgetAvailableMB = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_gpu_budget_available_mb",
			Help: "Per-GPU available VRAM (total - pinned - awake - L1 residual), in MB. From admission controller bookkeeping.",
		},
		[]string{"gpu"},
	)

	// Counter: unified lifecycle transitions (every model state change
	// admission/scheduler decides on). One row per event regardless of
	// which path emitted it. Matches the "lifecycle_transition" log line
	// emitted by jukebox.LogLifecycleTransition. Phase 4 stop-on-evict
	// cold-load events count under action="cold_load" without changing
	// the metric shape — dashboards built today survive that PR.
	LifecycleTransitionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_lifecycle_transitions_total",
			Help: "Model lifecycle transitions (evict, sleep, wake, cold_load) labeled by action and reason.",
		},
		[]string{"action", "reason"},
	)

	// ConfigReloadsTotal counts active.yaml hot-reload attempts. The
	// "result" label is one of:
	//   - "success"            — file re-parsed + validated OK, atomic-swapped in
	//   - "validation_failed"  — yaml parsed but Validate() returned an error
	//   - "parse_failed"       — yaml.Decode (or os.ReadFile) failed
	// Failed reloads leave the previous in-memory config intact, so a
	// non-zero count of failed attempts is operator-actionable but NOT
	// service-affecting. Pair with the slog.Error
	// "active.yaml hot-reload validation failed" line for context.
	ConfigReloadsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_config_reloads_total",
			Help: "active.yaml hot-reload attempts labeled by result (success, validation_failed, parse_failed).",
		},
		[]string{"result"},
	)

	// --- content-aware routing (Stage J overnight build, 2026-06-10) ---

	// ContentRoutingDecisionsTotal counts classifier decisions per inbound
	// chat-completions request. Labels:
	//   - decision: forward / reroute_image / reroute_overflow /
	//     reject_no_vision / reject_too_long
	//   - from: requested served-name
	//   - to:   target served-name (when reroute*; empty otherwise)
	// Designed so a single panel shows reroute volume, kill-switch state,
	// and reject rate without joining multiple counters.
	ContentRoutingDecisionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_content_routing_decisions_total",
			Help: "Content-classifier routing decisions on POST /v1/chat/completions, labeled by decision + source/target served-name.",
		},
		[]string{"decision", "from", "to"},
	)

	// ContentRoutingEstTokensHistogram captures the byte-length-derived
	// token estimate distribution. Useful for tuning
	// LengthOverflowThresholdTokens — if the bulk of traffic sits at 100K
	// and the threshold is 256K, the threshold is set right; if traffic
	// piles up just under 256K, the threshold may be too tight.
	ContentRoutingEstTokensHistogram = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "jukebox_content_routing_est_tokens",
			Help:    "Estimated inbound prompt tokens (bytes / chars-per-token) on POST /v1/chat/completions when content routing is enabled.",
			Buckets: []float64{1024, 4096, 16384, 65536, 131072, 196608, 262144, 393216, 524288},
		},
	)

	// --- circuit breaker (Track A) ---

	// CircuitBreakerObservedTotal counts every response status the
	// breaker observed, labeled by model + outcome bucket (2xx/4xx/5xx).
	CircuitBreakerObservedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_circuit_breaker_observed_total",
			Help: "Responses observed by the circuit breaker per model, labeled by outcome class (2xx/4xx/5xx).",
		},
		[]string{"model", "class"},
	)

	// CircuitBreakerTripsTotal counts trip-decisions, labeled by model
	// and outcome (fired = docker restart issued, suppressed_*, failed).
	CircuitBreakerTripsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_circuit_breaker_trips_total",
			Help: "Circuit breaker trip decisions per model+outcome (fired/suppressed_*/failed).",
		},
		[]string{"model", "outcome"},
	)

	// GPUResidualMB exposes per-GPU L1-sleep residual VRAM from
	// admission bookkeeping.
	GPUResidualMB = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_gpu_residual_mb",
			Help: "Per-GPU L1-sleep residual VRAM (vLLM cumem allocator pool retained after /sleep, MB). From admission bookkeeping, NOT live nvidia-smi.",
		},
		[]string{"gpu"},
	)

	// --- predictive cold-load (skeleton) ---

	PredictiveWarmsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_predictive_warms_total",
			Help: "Predictive cold-load pre-warm decisions made by the histogram predictor.",
		},
		[]string{"model", "outcome"},
	)

	PredictedProbability = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_predicted_probability",
			Help: "Most recent P(request in next window) computed by the predictor for this model. Range [0,1]. -1 sentinel = no observations yet for this bucket.",
		},
		[]string{"model"},
	)

	PredictiveWarmToFirstRequestSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_predictive_warm_to_first_request_seconds",
			Help:    "Wall-clock seconds between a predictive pre-warm decision and the first real request for that model.",
			Buckets: []float64{1, 10, 30, 60, 120, 300, 600, 1200},
		},
		[]string{"model"},
	)

	// Counter: upstream 429 responses rewritten to 503 by the proxy.
	// vLLM emits 429 when its in-engine admission queue is saturated
	// (commonly during wake / cold-start when many concurrent requests
	// arrive before the engine reaches steady state). 429 tells SDKs
	// "you, the client, are rate-limited" — leading to long backoffs or
	// terminal errors. The proxy rewrites these to 503 + Retry-After so
	// well-behaved SDKs retry within their normal budget. A non-zero
	// rate on this counter for a given model is a signal that either
	// (a) the model is being woken under burst load, or (b) the engine
	// has hit its steady-state concurrency limit and needs tuning.
	Upstream429RewrittenTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_upstream_429_rewritten_total",
			Help: "Total upstream 429 (Too Many Requests) responses rewritten by the proxy to 503 + Retry-After.",
		},
		[]string{"model"},
	)

	// Gauge: TCP-liveness verdict for an upstream (1 = alive, 0 = dead).
	// Labelled by the resolved target host:port so multiple upstreams
	// (e.g. scheduler-mode instances) can be tracked independently.
	UpstreamTCPAlive = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_upstream_tcp_alive",
			Help: "Upstream TCP-dial liveness (1 = alive, 0 = consecutive failures past threshold).",
		},
		[]string{"target"},
	)

	// Counter: TCP-liveness state transitions, labelled by direction.
	// `direction="down"` is incremented each time the probe demotes an
	// upstream; `direction="up"` each time it recovers.
	UpstreamTCPTransitionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_upstream_tcp_transitions_total",
			Help: "Upstream TCP-liveness state transitions (alive → dead and back).",
		},
		[]string{"target", "direction"},
	)
)

func init() {
	prometheus.MustRegister(
		RequestsTotal,
		SwapsTotal,
		SwapRejectionsTotal,
		SwapDurationSeconds,
		RequestDurationSeconds,
		State,
		InFlightRequests,
		CurrentModel,
		ConsecutiveFailures,
		RunningInstances,
		InstanceInFlightRequests,
		EvictionsTotal,
		ScheduleRejectionsTotal,
		SleepsTotal,
		StopsTotal,
		WakesTotal,
		SleepDurationSeconds,
		WakeDurationSeconds,
		SleepFailuresTotal,
		AdmissionEvictsTotal,
		AdmissionRejectionsTotal,
		AdmissionVRAMDriftRiskTotal,
		AdmissionPinnedWakeFailedTotal,
		ColdLoadCooldownSuppressedTotal,
		AdmissionWaitSeconds,
		GPUBudgetAwakeMB,
		GPUBudgetAvailableMB,
		LifecycleTransitionsTotal,
		ConfigReloadsTotal,
		ContentRoutingDecisionsTotal,
		ContentRoutingEstTokensHistogram,
		CircuitBreakerObservedTotal,
		CircuitBreakerTripsTotal,
		GPUResidualMB,
		PredictiveWarmsTotal,
		PredictedProbability,
		PredictiveWarmToFirstRequestSeconds,
		Upstream429RewrittenTotal,
		UpstreamTCPAlive,
		UpstreamTCPTransitionsTotal,
	)
}

func ObserveRequest(statusCode int, model string, dur time.Duration) {
	if model == "" {
		model = "unknown"
	}
	RequestsTotal.WithLabelValues(model, strconv.Itoa(statusCode)).Inc()
	RequestDurationSeconds.WithLabelValues(model).Observe(dur.Seconds())
}

func SetState(state string) {
	for _, s := range []string{"idle", "starting", "ready", "stopping", "sleeping", "error"} {
		v := 0.0
		if s == state {
			v = 1.0
		}
		State.WithLabelValues(s).Set(v)
	}
}
