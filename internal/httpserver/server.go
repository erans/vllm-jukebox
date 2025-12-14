package httpserver

import (
	"context"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
)

type StatusProvider interface {
	Status() jukebox.Status
}

type Coordinator interface {
	StatusProvider
	EnsureModel(ctx context.Context, requestedModel, requestID string) error
}

type Options struct {
	Config      *config.Config
	Coordinator Coordinator
	InFlight    *inflight.Tracker
}

func NewApp(opts Options) *fiber.App {
	app := fiber.New()

	app.Use(requestIDMiddleware())
	app.Use(requestLoggingMiddleware())

	app.Get("/health", healthHandler(opts.Coordinator))
	app.Get("/status", statusHandler(opts.Config, opts.Coordinator))

	app.Get("/v1/models", listModelsHandler(opts.Config))
	app.Get("/metrics", metricsHandler())

	// Model-switching endpoints.
	app.Post("/v1/responses", switchingProxyHandler(opts))
	app.Post("/v1/chat/completions", switchingProxyHandler(opts))
	app.Post("/v1/completions", switchingProxyHandler(opts))

	// Pass-through endpoints (no switching logic).
	app.Post("/v1/embeddings", passthroughProxyHandler(opts))
	app.Post("/v1/tokenize", passthroughProxyHandler(opts))
	app.Post("/v1/detokenize", passthroughProxyHandler(opts))

	// Explicit unsupported endpoints.
	app.All("/v1/audio/*", notImplementedHandler())
	app.All("/v1/images/*", notImplementedHandler())
	app.All("/v1/files*", notImplementedHandler())
	app.All("/v1/fine-tuning/*", notImplementedHandler())
	app.All("/v1/assistants/*", notImplementedHandler())
	// Fallback for unknown /v1 endpoints: return 501 rather than 404.
	app.All("/v1/*", notImplementedHandler())

	return app
}
