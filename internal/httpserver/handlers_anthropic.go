package httpserver

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/jukebox"
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
	var rej *jukebox.RejectError
	if errors.As(err, &rej) {
		if rej.RetryAfter > 0 {
			retryAfterSeconds := int64(rej.RetryAfter.Round(time.Second).Seconds())
			if retryAfterSeconds <= 0 {
				retryAfterSeconds = 1
			}
			c.Set("Retry-After", fmt.Sprintf("%d", retryAfterSeconds))
		}

		if rej.Reason == jukebox.RejectBackoff {
			return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "model in error state, please retry")
		}

		msg := "model loading in progress, please retry"
		switch rej.Reason {
		case jukebox.RejectPinnedConflict:
			msg = "insufficient resources: requested model conflicts with a pinned model"
		case jukebox.RejectNoCapacity:
			msg = "insufficient capacity to start requested model, please retry"
		case jukebox.RejectInsufficient:
			msg = "insufficient GPU resources to start requested model, please retry"
		case jukebox.RejectMinUptime:
			msg = "insufficient GPU resources (min uptime), please retry"
		}
		return writeAnthropicError(c, http.StatusServiceUnavailable, "overloaded_error", msg)
	}

	return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "backend unavailable")
}
