package httpserver

import (
	"fmt"
	"net/http"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/proxy"
)

func anthropicProxyHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Router == nil {
			return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "server not configured")
		}

		modelName, err := extractModel(c.Body())
		if err != nil {
			return writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		}
		if modelName == "" {
			return writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", "missing required field 'model'")
		}

		c.Locals(requestedModelLocal, modelName)

		_, _, err = opts.Config.ResolveModel(modelName)
		if err != nil {
			return writeAnthropicError(
				c,
				http.StatusBadRequest,
				"invalid_request_error",
				fmt.Sprintf("model %q not found in configuration", modelName),
			)
		}

		requestID, _ := c.Locals(requestIDHeader).(string)
		if requestID == "" {
			requestID = c.Get(requestIDHeader)
		}

		route, err := opts.Router.AcquireRoute(c.UserContext(), modelName, requestID)
		if err != nil {
			return mapAnthropicError(c, err)
		}
		if route.Done != nil {
			defer route.Done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
		})
	}
}

func mapAnthropicError(c *fiber.Ctx, err error) error {
	// For now, map all errors to overloaded or api_error
	// Will be refined in Task 5
	return writeAnthropicError(c, http.StatusServiceUnavailable, "overloaded_error", "model loading in progress, please retry")
}
