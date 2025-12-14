package httpserver

import (
	"log/slog"
	"time"

	"github.com/gofiber/fiber/v2"
)

func requestLoggingMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		err := c.Next()
		dur := time.Since(start)

		requestID, _ := c.Locals(requestIDHeader).(string)
		slog.Info(
			"http_request",
			"request_id", requestID,
			"method", c.Method(),
			"path", c.Path(),
			"status", c.Response().StatusCode(),
			"duration_ms", dur.Milliseconds(),
		)

		return err
	}
}
