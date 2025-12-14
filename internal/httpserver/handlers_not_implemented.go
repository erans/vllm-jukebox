package httpserver

import "github.com/gofiber/fiber/v2"

func notImplementedHandler() fiber.Handler {
	return func(c *fiber.Ctx) error {
		return writeOpenAIError(c, 501, "Not implemented", "not_implemented", "not_implemented")
	}
}
