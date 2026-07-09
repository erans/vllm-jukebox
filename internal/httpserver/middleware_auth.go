package httpserver

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/gofiber/fiber/v2"
)

// AuthAPIKey returns middleware that requires Authorization: Bearer <key> when
// expected is non-empty. If expected is empty, auth is disabled and requests pass
// through unchanged.
func AuthAPIKey(expected string) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if expected == "" {
			return c.Next()
		}

		const bearerPrefix = "Bearer "
		header := c.Get("Authorization")
		if !strings.HasPrefix(header, bearerPrefix) {
			return writeOpenAIError(c, http.StatusUnauthorized, "Missing or invalid Authorization header", "invalid_request_error", "missing_auth")
		}

		provided := header[len(bearerPrefix):]
		if subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) != 1 {
			return writeOpenAIError(c, http.StatusUnauthorized, "Invalid API key", "invalid_request_error", "invalid_auth")
		}

		return c.Next()
	}
}
