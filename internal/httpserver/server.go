package httpserver

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

type StatusProvider interface {
	Status() jukebox.Status
}

type Options struct {
	Config *config.Config
	Router jukebox.Router
}

func NewApp(opts Options) *fiber.App {
	app := fiber.New()

	app.Use(recover.New())
	app.Use(requestIDMiddleware())
	if opts.Config == nil || opts.Config.Server.LogRequests == nil || *opts.Config.Server.LogRequests {
		app.Use(requestLoggingMiddleware())
	}

	app.Get("/health", healthHandler(opts.Router))

	// Protected routes require bearer auth when server.api_key is set.
	apiKey := ""
	if opts.Config != nil {
		apiKey = opts.Config.Server.APIKey
	}
	auth := AuthAPIKey(apiKey)

	app.Get("/status", auth, statusHandler(opts.Config, opts.Router))
	app.Get("/metrics", auth, metricsHandler())

	app.Get("/v1/models", auth, listModelsHandler(opts.Config))
	app.Post("/v1/responses", auth, switchingProxyHandler(opts))
	app.Post("/v1/chat/completions", auth, switchingProxyHandler(opts))
	app.Post("/v1/completions", auth, switchingProxyHandler(opts))
	app.Post("/v1/embeddings", auth, switchingProxyHandler(opts))
	app.Post("/v1/tokenize", auth, switchingProxyHandler(opts))
	app.Post("/v1/detokenize", auth, switchingProxyHandler(opts))

	// Anthropic endpoints
	app.Post("/v1/messages", auth, anthropicProxyHandler(opts))

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
