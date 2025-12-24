package httpserver

import (
	"github.com/gofiber/fiber/v2"
)

func writeOpenAIError(c *fiber.Ctx, status int, message, errType, code string) error {
	c.Set("Content-Type", "application/json")
	return c.Status(status).JSON(fiber.Map{
		"error": fiber.Map{
			"message": message,
			"type":    errType,
			"code":    code,
		},
	})
}

func writeAnthropicError(c *fiber.Ctx, status int, errType, message string) error {
	c.Set("Content-Type", "application/json")
	return c.Status(status).JSON(fiber.Map{
		"type": "error",
		"error": fiber.Map{
			"type":    errType,
			"message": message,
		},
	})
}
