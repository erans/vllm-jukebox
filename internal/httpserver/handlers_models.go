package httpserver

import (
	"sort"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
)

func listModelsHandler(cfg *config.Config) fiber.Handler {
	created := time.Now().Unix()
	return func(c *fiber.Ctx) error {
		if cfg == nil {
			return c.Status(500).JSON(fiber.Map{
				"error": fiber.Map{
					"message": "server not configured",
					"type":    "internal_error",
					"code":    "config_missing",
				},
			})
		}

		names := make([]string, 0, len(cfg.Models))
		for name := range cfg.Models {
			names = append(names, name)
		}
		sort.Strings(names)

		data := make([]fiber.Map, 0, len(names))
		for _, name := range names {
			data = append(data, fiber.Map{
				"id":       name,
				"object":   "model",
				"created":  created,
				"owned_by": "jukebox",
			})
		}

		return c.JSON(fiber.Map{
			"object": "list",
			"data":   data,
		})
	}
}
