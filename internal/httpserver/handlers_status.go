package httpserver

import (
	"sort"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

func statusHandler(cfg *config.Config, coord StatusProvider) fiber.Handler {
	return func(c *fiber.Ctx) error {
		live := config.Current()
		if live == nil {
			live = cfg
		}
		st := jukebox.Status{State: jukebox.StateIdle}
		if coord != nil {
			st = coord.Status()
		}
		formatTime := func(t time.Time) any {
			if t.IsZero() {
				return nil
			}
			return t.Format(time.RFC3339Nano)
		}

		var available []string
		if live != nil {
			available = make([]string, 0, len(live.Models))
			for name := range live.Models {
				available = append(available, name)
			}
			sort.Strings(available)
		}

		var cooldownRemaining int64
		// vllm.swap_cooldown is a top-level field captured at boot —
		// keep using cfg here (NOT live) per the hot-reload matrix.
		if cfg != nil && cfg.VLLM.SwapCooldown.Duration > 0 && !st.LastSwapAt.IsZero() {
			next := st.LastSwapAt.Add(cfg.VLLM.SwapCooldown.Duration)
			remaining := time.Until(next)
			if remaining > 0 {
				cooldownRemaining = int64(remaining.Seconds())
				if cooldownRemaining < 0 {
					cooldownRemaining = 0
				}
			}
		}

		var lastSwap any
		if st.LastSwap != nil {
			lastSwap = fiber.Map{
				"from":                    st.LastSwap.From,
				"to":                      st.LastSwap.To,
				"at":                      st.LastSwap.At.Format(time.RFC3339Nano),
				"duration_ms":             st.LastSwap.Duration.Milliseconds(),
				"triggered_by_request_id": st.LastSwap.TriggeredByRequest,
			}
		}

		var instances []any
		if len(st.Instances) > 0 {
			instances = make([]any, 0, len(st.Instances))
			for _, inst := range st.Instances {
				item := fiber.Map{
					"model":              inst.Model,
					"port":               inst.Port,
					"gpus":               inst.GPUs,
					"pid":                inst.PID,
					"state":              inst.State,
					"pinned":             inst.Pinned,
					"draining":           inst.Draining,
					"in_flight_requests": inst.InFlight,
					"started_at":         formatTime(inst.StartedAt),
					"last_used_at":       formatTime(inst.LastUsedAt),
				}
				instances = append(instances, item)
			}
		}

		return c.JSON(fiber.Map{
			"state": st.State,
			// Sleeping AND Stopped are warm-on-demand: Sleeping wakes via
			// /wake_up (~30s), Stopped wakes via docker start + cold-load
			// (~5min from page cache). Both surfaces should signal
			// "accepting (with latency)" so an operator dashboard / LB
			// status check doesn't false-alarm during routine eviction
			// or redeploy-member windows. Mirrors /health.
			"accepting_requests":              st.State == jukebox.StateReady || st.State == jukebox.StateSleeping || st.State == jukebox.StateStopped,
			"current_model":                   st.CurrentModel,
			"in_flight_requests":              st.InFlight,
			"uptime_seconds":                  st.UptimeSeconds,
			"swap_cooldown_remaining_seconds": cooldownRemaining,
			"last_swap":                       lastSwap,
			"failure_count":                   st.FailureCount,
			"available_models":                available,
			"scheduler_busy":                  st.SchedulerBusy,
			"waiters":                         st.Waiters,
			"instances":                       instances,
		})
	}
}
