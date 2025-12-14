package httpserver

import (
	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

type StatusProvider interface {
	Status() jukebox.Status
}

type Options struct {
	Config      *config.Config
	Coordinator StatusProvider
}

func NewApp(opts Options) *fiber.App {
	app := fiber.New()

	app.Use(requestIDMiddleware())

	app.Get("/health", healthHandler(opts.Coordinator))
	app.Get("/status", statusHandler(opts.Config, opts.Coordinator))

	return app
}

