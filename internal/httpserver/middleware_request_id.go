package httpserver

import (
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

const requestIDHeader = "X-Request-ID"

func requestIDMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		requestID := c.Get(requestIDHeader)
		if requestID == "" {
			requestID = uuid.NewString()
		}
		c.Locals(requestIDHeader, requestID)
		c.Set(requestIDHeader, requestID)
		return c.Next()
	}
}
