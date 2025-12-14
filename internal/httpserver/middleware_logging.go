package httpserver

import (
	"log/slog"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/metrics"
)

const requestedModelLocal = "requested_model"

func requestLoggingMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		err := c.Next()
		dur := time.Since(start)

		requestID, _ := c.Locals(requestIDHeader).(string)
		model, _ := c.Locals(requestedModelLocal).(string)
		slog.Info(
			"http_request",
			"request_id", requestID,
			"method", c.Method(),
			"path", c.Path(),
			"status", c.Response().StatusCode(),
			"duration_ms", dur.Milliseconds(),
		)

		metrics.ObserveRequest(c.Path(), c.Response().StatusCode(), model, dur)

		return err
	}
}
