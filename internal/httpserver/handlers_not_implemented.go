package httpserver

import "github.com/gofiber/fiber/v2"

func notImplementedHandler() fiber.Handler {
	return func(c *fiber.Ctx) error {
		return c.Status(501).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Not implemented",
				"type":    "not_implemented",
				"code":    "not_implemented",
			},
		})
	}
}

