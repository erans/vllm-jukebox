package httpserver

import "github.com/gofiber/fiber/v2"

// metricsHandler is intentionally minimal for now (spec marks metrics as optional/future).
func metricsHandler() fiber.Handler {
	return func(c *fiber.Ctx) error {
		return c.Status(501).SendString("metrics not implemented")
	}
}
