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
		if opts.Config == nil || opts.Coordinator == nil {
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
		_, modelCfg, err := opts.Config.ResolveModel(modelName)
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

		if err := opts.Coordinator.EnsureModel(c.UserContext(), modelName, requestID); err != nil {
			return mapEnsureError(c, err)
		}

		var done func()
		if opts.InFlight != nil {
			done = opts.InFlight.Track(c.UserContext())
			defer done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          fmt.Sprintf("http://127.0.0.1:%d", opts.Config.VLLM.Port),
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    modelCfg.Path,
			RequestID:        requestID,
		})
	}
}

func passthroughProxyHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Coordinator == nil {
			return writeOpenAIError(c, http.StatusInternalServerError, "server not configured", "internal_error", "config_missing")
		}

		st := opts.Coordinator.Status()
		c.Locals(requestedModelLocal, st.CurrentModel)
		if st.State != jukebox.StateReady {
			c.Set("Retry-After", "5")
			return writeOpenAIError(c, http.StatusServiceUnavailable, "Model switch in progress, please retry", "service_unavailable", "model_switching")
		}

		requestID, _ := c.Locals(requestIDHeader).(string)
		if requestID == "" {
			requestID = c.Get(requestIDHeader)
		}

		var done func()
		if opts.InFlight != nil {
			done = opts.InFlight.Track(c.UserContext())
			defer done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:   fmt.Sprintf("http://127.0.0.1:%d", opts.Config.VLLM.Port),
			RequestID: requestID,
		})
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
			return writeOpenAIError(c, http.StatusInternalServerError, "vLLM in error state, please retry", "internal_error", "vllm_error")
		}
		return writeOpenAIError(c, http.StatusServiceUnavailable, "Model switch in progress, please retry", "service_unavailable", "model_switching")
	}

	return writeOpenAIError(c, http.StatusInternalServerError, "vLLM unavailable", "internal_error", "vllm_unavailable")
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
