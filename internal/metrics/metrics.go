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

	// Counter: predictive cold-load pre-warms attempted by the histogram
	// predictor. `outcome` is one of:
	//   - "fired"   — predictor crossed threshold AND budgets allowed; in
	//                 the skeleton this is the only path that increments
	//                 (the actual scheduler hook is still a TODO).
	//   - "hit"     — a real request landed within the forecast window
	//                 after a "fired" tick (post-hoc validation).
	//   - "miss"    — forecast window elapsed with no real request (the
	//                 predictor was wrong; the model stayed Stopped).
	//   - "skipped_cooldown"      — gate fired but the per-model cooldown
	//                               window was still active.
	//   - "skipped_budget"        — gate fired but max_per_day was hit.
	//   - "skipped_not_stopped"   — gate fired but model was not Stopped.
	//   - "skipped_disabled"      — predictor disabled for this model.
	PredictiveWarmsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_predictive_warms_total",
			Help: "Predictive cold-load pre-warm decisions made by the histogram predictor.",
		},
		[]string{"model", "outcome"},
	)

	// Gauge: most-recently-computed P(request in next window | hour, dow)
	// for each predictively-tracked model. Sampled per predictor tick.
	PredictedProbability = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_predicted_probability",
			Help: "Most recent P(request in next window) computed by the predictor for this model. Range [0,1]. -1 sentinel = no observations yet for this bucket.",
		},
		[]string{"model"},
	)

	// Histogram: time elapsed between a successful predictive warm (when
	// it eventually fires a cold-load — TODO) and the first real request
	// that landed afterward. The whole point of predictive cold-load: if
	// this histogram clusters near 0 we predicted too late; if it clusters
	// near `window_minutes` * 60 we predicted right; if it has a long tail
	// we are firing too eagerly (misses).
	PredictiveWarmToFirstRequestSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_predictive_warm_to_first_request_seconds",
			Help:    "Wall-clock seconds between a predictive pre-warm decision and the first real request for that model.",
			Buckets: []float64{1, 10, 30, 60, 120, 300, 600, 1200},
		},
		[]string{"model"},
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
		PredictiveWarmsTotal,
		PredictedProbability,
		PredictiveWarmToFirstRequestSeconds,
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
	for _, s := range []string{"idle", "starting", "ready", "stopping", "error"} {
		v := 0.0
		if s == state {
			v = 1.0
		}
		State.WithLabelValues(s).Set(v)
	}
}
