package httpserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/proxy"
)

func switchingProxyHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Router == nil {
			return writeOpenAIError(c, http.StatusInternalServerError, "server not configured", "internal_error", "config_missing")
		}

		modelName, err := extractModel(c.Body())
		if err != nil {
			return writeOpenAIError(c, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_json")
		}
		if modelName == "" {
			return writeOpenAIError(c, http.StatusBadRequest, "missing required field 'model'", "invalid_request_error", "model_not_found")
		}

		c.Locals(requestedModelLocal, modelName)

		// Ensure unknown models fail fast with a 400 (per spec), before touching the coordinator.
		_, _, err = opts.Config.ResolveModel(modelName)
		if err != nil {
			return writeOpenAIError(
				c,
				http.StatusBadRequest,
				fmt.Sprintf("Model %q not found in configuration", modelName),
				"invalid_request_error",
				"model_not_found",
			)
		}

		requestID, _ := c.Locals(requestIDHeader).(string)
		if requestID == "" {
			requestID = c.Get(requestIDHeader)
		}

		route, err := opts.Router.AcquireRoute(c.UserContext(), modelName, requestID)
		if err != nil {
			return mapEnsureError(c, err)
		}
		err = proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
			OnDone:           route.Done,
		})
		// ForwardFiber does not call OnDone on its error paths (request build
		// failure, client.Do failure, buffered read failure). The old
		// `defer route.Done()` covered those returns; with OnDone wired only
		// into the success path, an error here would leak the in-flight slot
		// and block future drains/swaps. On success the proxy already fired
		// (and nil'd) OnDone, so route.Done is a no-op there; on error it
		// never fired, so we release exactly once. The inflight tracker's
		// done func uses sync.Once, making this idempotent.
		if err != nil && route.Done != nil {
			route.Done()
		}
		return err
	}
}

func extractModel(body []byte) (string, error) {
	if len(body) == 0 {
		return "", nil
	}
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", err
	}
	return payload.Model, nil
}

func mapEnsureError(c *fiber.Ctx, err error) error {
	var rej *jukebox.RejectError
	if errors.As(err, &rej) {
		retryAfterSeconds := int64(0)
		if rej.RetryAfter > 0 {
			retryAfterSeconds = int64(rej.RetryAfter.Round(time.Second).Seconds())
			if retryAfterSeconds <= 0 {
				retryAfterSeconds = 1
			}
			c.Set("Retry-After", fmt.Sprintf("%d", retryAfterSeconds))
		}

		// Treat backoff as a 500 (spec: error state returns 500) but include Retry-After.
		if rej.Reason == jukebox.RejectBackoff {
			return writeOpenAIError(c, http.StatusInternalServerError, "Backend in error state, please retry", "internal_error", "backend_error")
		}
		msg := "Model switch in progress, please retry"
		switch rej.Reason {
		case jukebox.RejectPinnedConflict:
			msg = "Insufficient resources: requested model conflicts with a pinned model"
		case jukebox.RejectNoCapacity:
			msg = "Insufficient capacity to start requested model, please retry"
		case jukebox.RejectInsufficient:
			msg = "Insufficient GPU resources to start requested model, please retry"
		case jukebox.RejectMinUptime:
			msg = "Insufficient GPU resources (min uptime), please retry"
		}
		return writeOpenAIError(c, http.StatusServiceUnavailable, msg, "service_unavailable", "model_switching")
	}

	return writeOpenAIError(c, http.StatusInternalServerError, "Backend unavailable", "internal_error", "backend_unavailable")
}

func extractModelNameFromError(err error) string {
	msg := err.Error()
	// Best-effort; errors from config look like: model "x" not found
	const prefix = "model "
	i := strings.Index(msg, prefix)
	if i == -1 {
		return ""
	}
	rest := msg[i+len(prefix):]
	rest = strings.TrimSpace(rest)
	rest = strings.Trim(rest, "\"")
	return rest
}
