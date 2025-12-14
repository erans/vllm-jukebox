package httpserver

import (
	"net/http"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/jukebox"
)

func healthHandler(coord StatusProvider) fiber.Handler {
	return func(c *fiber.Ctx) error {
		st := jukebox.Status{State: jukebox.StateIdle}
		if coord != nil {
			st = coord.Status()
		}

		accepting := st.State == jukebox.StateReady

		code := http.StatusOK
		switch st.State {
		case jukebox.StateStarting, jukebox.StateStopping:
			code = http.StatusServiceUnavailable
		case jukebox.StateError:
			code = http.StatusInternalServerError
		}

		statusText := "healthy"
		if code >= 500 {
			statusText = "unhealthy"
		}

		return c.Status(code).JSON(fiber.Map{
			"status":             statusText,
			"accepting_requests": accepting,
			"vllm": fiber.Map{
				"state":              st.State,
				"model":              st.CurrentModel,
				"pid":                st.PID,
				"uptime_seconds":     st.UptimeSeconds,
				"in_flight_requests": st.InFlight,
			},
		})
	}
}
