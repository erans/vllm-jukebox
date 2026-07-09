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
	protected := app.Group("/", AuthAPIKey(apiKey))

	protected.Get("/status", statusHandler(opts.Config, opts.Router))
	protected.Get("/metrics", metricsHandler())

	protected.Get("/v1/models", listModelsHandler(opts.Config))
	protected.Post("/v1/responses", switchingProxyHandler(opts))
	protected.Post("/v1/chat/completions", switchingProxyHandler(opts))
	protected.Post("/v1/completions", switchingProxyHandler(opts))
	protected.Post("/v1/embeddings", switchingProxyHandler(opts))
	protected.Post("/v1/tokenize", switchingProxyHandler(opts))
	protected.Post("/v1/detokenize", switchingProxyHandler(opts))

	// Anthropic endpoints
	protected.Post("/v1/messages", anthropicProxyHandler(opts))

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
