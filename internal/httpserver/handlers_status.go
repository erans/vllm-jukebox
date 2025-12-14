package httpserver

import (
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
		}

		return c.JSON(fiber.Map{
			"state":              st.State,
			"accepting_requests": st.State == jukebox.StateReady,
			"current_model":      st.CurrentModel,
			"in_flight_requests": st.InFlight,
			"failure_count":      st.FailureCount,
			"pid":                st.PID,
			"available_models":   available,
		})
	}
}

