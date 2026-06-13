package httpserver

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
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

		liveCfg := config.Current()
		if liveCfg == nil {
			liveCfg = opts.Config
		}
		_, _, err = liveCfg.ResolveModel(modelName)
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

		// Async-503 contract for wake-from-Stopped — see
		// switchingProxyHandler for the full rationale. Mirror the OpenAI
		// handler's behavior, but emit Anthropic's error envelope AND
		// Anthropic's overloaded status code: 529 (not 503).
		//
		// 529 is what real Anthropic upstream returns for overloaded_error
		// (their documented contract). Strict Anthropic SDK clients key
		// off the status code, not just the envelope; returning 503 here
		// would slip past lenient clients (e.g. some reverse proxies) but trip stricter
		// ones. The OpenAI sibling endpoint stays on 503 — that's the
		// correct OpenAI-protocol code for "service unavailable / try again".
		//
		// Type-assert to the OPTIONAL ColdLoadAware interface. Routers
		// that don't implement it (e.g. LegacyRouter for swap mode) skip
		// this branch entirely and fall through to AcquireRoute.
		if cl, ok := opts.Router.(jukebox.ColdLoadAware); ok && cl.IsModelColdLoading(modelName) {
			cl.KickColdLoad(modelName)
			// Retry-After: 60 (NOT 300). See OpenAI sibling handler
			// (handlers_proxy.go) for full rationale: OpenAI SDK retry
			// budget caps near ~8s, Anthropic ~60s, intermediate proxies may have their own
			// deadlines. 300s caused clients to surface user-visible
			// errors instead of retrying. Clients re-poll at 60s, get
			// more 503s if still cold-loading, eventually land on warm
			// Sleeping. Strict Anthropic SDK clients honor Retry-After
			// even on 529s.
			c.Set("Retry-After", "60")
			// 529 = Anthropic's overloaded_error code (not in stdlib http.StatusXxx).
			return writeAnthropicError(
				c,
				529,
				"overloaded_error",
				"model is cold-starting (~5 min for large models), please retry",
			)
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

		// Non-retryable reasons must surface as an Anthropic error.type
		// that retry-aware SDKs treat as TERMINAL. Anthropic SDKs treat
		// "overloaded_error" as RETRYABLE and will back-off-retry against
		// the header — but our header correctly omits Retry-After for
		// these reasons, so the JSON type was contradicting the header
		// and clients (Claude SDK, anthropic-python) would retry-loop
		// against an unsolvable condition. "api_error" is treated as
		// non-retryable. The "Retry-After" header is set above only when
		// rej.RetryAfter > 0; for the non-retryable reasons below the
		// RejectError carries no Retry-After so the header is absent.
		switch rej.Reason {
		case jukebox.RejectInsufficient:
			// Structural infeasibility: model exceeds the pinned-adjusted
			// GPU budget no matter what we evict. Operator must reconfigure.
			return writeAnthropicError(c, http.StatusServiceUnavailable, "api_error",
				"model exceeds available GPU budget; reconfigure or add capacity")
		case jukebox.RejectAdminIntervention:
			// Admission books may have drifted from physical container
			// state because a cleanup docker stop failed. Operator must
			// reconcile (see audit log + runbook).
			return writeAnthropicError(c, http.StatusServiceUnavailable, "api_error",
				"resource state inconsistent; operator intervention required (see audit log)")
		}

		msg := "model loading in progress, please retry"
		switch rej.Reason {
		case jukebox.RejectPinnedConflict:
			msg = "insufficient resources: requested model conflicts with a pinned model"
		case jukebox.RejectNoCapacity:
			msg = "insufficient capacity to start requested model, please retry"
		case jukebox.RejectMinUptime:
			msg = "insufficient GPU resources (min uptime), please retry"
		}
		return writeAnthropicError(c, http.StatusServiceUnavailable, "overloaded_error", msg)
	}

	return writeAnthropicError(c, http.StatusInternalServerError, "api_error", "backend unavailable")
}
