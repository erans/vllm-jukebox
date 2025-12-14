package httpserver

import "github.com/gofiber/fiber/v2"

type openAIError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func writeOpenAIError(c *fiber.Ctx, status int, message, typ, code string) error {
	var payload openAIError
	payload.Error.Message = message
	payload.Error.Type = typ
	payload.Error.Code = code
	return c.Status(status).JSON(payload)
}
