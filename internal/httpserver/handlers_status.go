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
		st := jukebox.Status{State: jukebox.StateIdle}
		if coord != nil {
			st = coord.Status()
		}

		var available []string
		if cfg != nil {
			available = make([]string, 0, len(cfg.Models))
			for name := range cfg.Models {
				available = append(available, name)
			}
			sort.Strings(available)
		}

		var cooldownRemaining int64
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

		return c.JSON(fiber.Map{
			"state":                           st.State,
			"accepting_requests":              st.State == jukebox.StateReady,
			"current_model":                   st.CurrentModel,
			"in_flight_requests":              st.InFlight,
			"uptime_seconds":                  st.UptimeSeconds,
			"swap_cooldown_remaining_seconds": cooldownRemaining,
			"last_swap":                       lastSwap,
			"failure_count":                   st.FailureCount,
			"available_models":                available,
		})
	}
}
