package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	RequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_requests_total",
			Help: "Total HTTP requests handled by Jukebox.",
		},
		[]string{"path", "status", "model"},
	)

	RequestDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_request_duration_seconds",
			Help:    "HTTP request duration in seconds.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"path", "model"},
	)

	SwapsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_swaps_total",
			Help: "Total model swaps completed successfully.",
		},
		[]string{"from", "to"},
	)

	SwapRejectionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "jukebox_swap_rejections_total",
			Help: "Total swap rejections by reason.",
		},
		[]string{"reason"},
	)

	SwapDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "jukebox_swap_duration_seconds",
			Help:    "Duration of model swaps in seconds.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"from", "to"},
	)

	State = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "jukebox_state",
			Help: "Current Jukebox vLLM state (1 = active).",
		},
		[]string{"state"},
	)
)

func init() {
	prometheus.MustRegister(
		RequestsTotal,
		RequestDurationSeconds,
		SwapsTotal,
		SwapRejectionsTotal,
		SwapDurationSeconds,
		State,
	)
}

func ObserveRequest(path string, statusCode int, model string, dur time.Duration) {
	RequestsTotal.WithLabelValues(path, itoa(statusCode), model).Inc()
	RequestDurationSeconds.WithLabelValues(path, model).Observe(dur.Seconds())
}

func ObserveSwap(from, to string, dur time.Duration, err error, rejectReason string) {
	if rejectReason != "" {
		SwapRejectionsTotal.WithLabelValues(rejectReason).Inc()
		return
	}
	if err == nil {
		SwapsTotal.WithLabelValues(from, to).Inc()
		SwapDurationSeconds.WithLabelValues(from, to).Observe(dur.Seconds())
	}
}

func SetState(state string) {
	// Set selected state to 1, others to 0.
	for _, s := range []string{"idle", "starting", "ready", "stopping", "error"} {
		v := 0.0
		if s == state {
			v = 1.0
		}
		State.WithLabelValues(s).Set(v)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [32]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
