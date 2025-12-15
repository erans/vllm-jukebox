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
