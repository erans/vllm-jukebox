package httpserver

import (
	"errors"
	"encoding/json"
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
			return c.Status(http.StatusInternalServerError).JSON(fiber.Map{
				"error": fiber.Map{
					"message": "server not configured",
					"type":    "internal_error",
					"code":    "config_missing",
				},
			})
		}

		modelName, err := extractModel(c.Body())
		if err != nil {
			return c.Status(http.StatusBadRequest).JSON(fiber.Map{
				"error": fiber.Map{
					"message": err.Error(),
					"type":    "invalid_request_error",
					"code":    "invalid_json",
				},
			})
		}
		if modelName == "" {
			return c.Status(http.StatusBadRequest).JSON(fiber.Map{
				"error": fiber.Map{
					"message": "missing required field 'model'",
					"type":    "invalid_request_error",
					"code":    "model_not_found",
				},
			})
		}

		c.Locals(requestedModelLocal, modelName)

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
			RequestID:        requestID,
		})
	}
}

func passthroughProxyHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Coordinator == nil {
			return c.Status(http.StatusInternalServerError).JSON(fiber.Map{
				"error": fiber.Map{
					"message": "server not configured",
					"type":    "internal_error",
					"code":    "config_missing",
				},
			})
		}

		st := opts.Coordinator.Status()
		c.Locals(requestedModelLocal, st.CurrentModel)
		if st.State != jukebox.StateReady {
			c.Set("Retry-After", "5")
			return c.Status(http.StatusServiceUnavailable).JSON(fiber.Map{
				"error": fiber.Map{
					"message": "Model switch in progress, please retry",
					"type":    "service_unavailable",
					"code":    "model_switching",
				},
			})
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

		return c.Status(http.StatusServiceUnavailable).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Model switch in progress, please retry",
				"type":    "service_unavailable",
				"code":    "model_switching",
			},
		})
	}

	msg := err.Error()
	if strings.Contains(msg, "model") && strings.Contains(msg, "not found") {
		return c.Status(http.StatusBadRequest).JSON(fiber.Map{
			"error": fiber.Map{
				"message": fmt.Sprintf("Model %q not found in configuration", extractModelNameFromError(err)),
				"type":    "invalid_request_error",
				"code":    "model_not_found",
			},
		})
	}

	return c.Status(http.StatusInternalServerError).JSON(fiber.Map{
		"error": fiber.Map{
			"message": "vLLM unavailable",
			"type":    "internal_error",
			"code":    "vllm_unavailable",
		},
	})
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
