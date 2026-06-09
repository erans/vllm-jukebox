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
		// Sleeping AND Stopped instances can be woken on demand, so jukebox
		// accepts requests for them (with higher first-request latency: ~30s
		// for Sleeping → Awake via /wake_up, ~5min for Stopped → Sleeping via
		// docker start + cold-load). Surface "accepting" so upstream LBs
		// don't drain traffic from a jukebox that's actively serving via
		// the async-503 + KickColdLoad recovery loop.
		if st.State == jukebox.StateSleeping || st.State == jukebox.StateStopped {
			accepting = true
		}

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
				"state":          st.State,
				"model":          st.CurrentModel,
				"pid":            st.PID,
				"uptime_seconds": st.UptimeSeconds,
			},
		})
	}
}
