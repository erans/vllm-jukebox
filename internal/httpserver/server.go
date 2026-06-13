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
	app.Get("/status", statusHandler(opts.Config, opts.Router))

	// Admin endpoints. No auth — see handlers_admin.go's docstring for
	// the loopback-binding contract that makes that safe.
	admin := app.Group("/admin")
	admin.Post("/redeploy-member", redeployMemberHandler(opts.Router))

	app.Get("/v1/models", listModelsHandler(opts.Config))
	app.Get("/metrics", metricsHandler())

	// Model-bearing endpoints (scheduler-routed).
	app.Post("/v1/responses", switchingProxyHandler(opts))
	app.Post("/v1/chat/completions", switchingProxyHandler(opts))
	app.Post("/v1/completions", switchingProxyHandler(opts))
	app.Post("/v1/embeddings", switchingProxyHandler(opts))
	app.Post("/v1/tokenize", switchingProxyHandler(opts))
	app.Post("/v1/detokenize", switchingProxyHandler(opts))
	// vLLM exposes reranking under both /v1/rerank and /rerank depending on
	// the installed serving entrypoint. Route both so jukebox is a transparent
	// proxy regardless of which one the client uses. The model is extracted
	// from the request body, so admission + wake fan-out apply normally.
	app.Post("/v1/rerank", switchingProxyHandler(opts))
	app.Post("/rerank", switchingProxyHandler(opts))

	// Anthropic endpoints
	app.Post("/v1/messages", anthropicProxyHandler(opts))

	// Image-gen: OpenAI shape, translated into a ComfyUI workflow.
	// Lifecycle (wake-on-request, sleep coordination) is driven through
	// the existing Router.AcquireRoute path on the ComfyUI peer
	// (lifecycle: comfyui). Subsumes the old comfy-openai-shim sidecar.
	app.Post("/v1/images/generations", imagesGenerationsHandler(opts))

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
