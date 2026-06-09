package httpserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/metrics"
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

		// Resolve against the live config so a mid-flight active.yaml
		// reload that adds / removes a model is reflected immediately.
		liveCfg := config.Current()
		if liveCfg == nil {
			liveCfg = opts.Config
		}

		// === Stage J — content-aware routing (kill-switch default off) ===
		//
		// Only triggers on POST /v1/chat/completions (this handler is only
		// wired to chat / completions / responses; embeddings/rerank/
		// tokenize endpoints use the same dispatcher but the classifier
		// returns DecisionForward in zero time when ContentRouting is off,
		// AND a request without messages[].content[].image_url won't match
		// any reroute rule, so /embeddings traffic with content routing
		// ON is also harmless).
		//
		// The classifier reads the body once via json.Unmarshal — same
		// cost as extractModel above. When ContentRouting is false, the
		// classifier short-circuits at the first config check and adds
		// no measurable latency.
		originalModel := modelName
		rewroteByContent := false
		isChatPath := strings.HasSuffix(c.OriginalURL(), "/v1/chat/completions")
		if isChatPath && liveCfg.Behavior.ContentRouting {
			cls := ClassifyContent(c.Body(), modelName, liveCfg)
			if cls.EstTokens > 0 {
				metrics.ContentRoutingEstTokensHistogram.Observe(float64(cls.EstTokens))
			}
			switch cls.Decision {
			case DecisionForward:
				metrics.ContentRoutingDecisionsTotal.WithLabelValues("forward", originalModel, "").Inc()
			case DecisionRerouteImage:
				metrics.ContentRoutingDecisionsTotal.WithLabelValues("reroute_image", originalModel, cls.NewModel).Inc()
				modelName = cls.NewModel
				rewroteByContent = true
			case DecisionRerouteOverflow:
				metrics.ContentRoutingDecisionsTotal.WithLabelValues("reroute_overflow", originalModel, cls.NewModel).Inc()
				modelName = cls.NewModel
				rewroteByContent = true
			case DecisionRejectNoVision:
				metrics.ContentRoutingDecisionsTotal.WithLabelValues("reject_no_vision", originalModel, "").Inc()
				return writeOpenAIError(
					c,
					http.StatusServiceUnavailable,
					cls.Reason+" — configure behavior.vision_model",
					"service_unavailable",
					"vision_model_unavailable",
				)
			case DecisionRejectTooLong:
				metrics.ContentRoutingDecisionsTotal.WithLabelValues("reject_too_long", originalModel, "").Inc()
				return writeOpenAIError(
					c,
					http.StatusBadRequest,
					cls.Reason,
					"invalid_request_error",
					"context_length_exceeded",
				)
			}
		}

		// Ensure unknown models fail fast with a 400 (per spec), before touching the coordinator.
		// Resolve against the live config so a mid-flight active.yaml
		// reload that adds / removes a model is reflected immediately.
		liveCfg := config.Current()
		if liveCfg == nil {
			liveCfg = opts.Config
		}
		_, _, err = liveCfg.ResolveModel(modelName)
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

		// Async-503 contract for wake-from-Stopped (per Kagi PATCH 16).
		// If admission has the model in admissionStopped (its container is
		// `docker compose stop`'d because a previous admission cycle picked
		// it as an evict_action: stop victim), the synchronous wake path
		// would block this request for ~5 min (moe/longctx) — way past
		// Bifrost / downstream HTTP-client timeouts. Kick the cold-load on
		// a background goroutine (using context.Background() with a
		// generous timeout — NOT the inbound ctx which dies when the
		// client gives up) and return 503 + Retry-After: 60 immediately.
		// The client retries against another 503 ("still warming") or hits
		// a Sleeping → Awake fast path once cold-load completes.
		//
		// admissionSleeping (the other not-Awake state) is NOT 503'd —
		// /wake_up takes ~10-30s and fits comfortably in client timeouts.
		//
		// Type-assert to the OPTIONAL ColdLoadAware interface. Routers
		// that don't implement it (e.g. LegacyRouter for swap mode) skip
		// this branch entirely and fall through to AcquireRoute, which
		// still works correctly — swap mode has no admissionStopped state
		// so the async-503 path is structurally inapplicable there.
		if cl, ok := opts.Router.(jukebox.ColdLoadAware); ok && cl.IsModelColdLoading(modelName) {
			cl.KickColdLoad(modelName)
			// Retry-After: 60 (NOT 300). Rationale:
			//   - OpenAI Python SDK caps its retry budget around ~8s and
			//     gives up entirely on Retry-After > a few minutes.
			//   - Anthropic SDK respects up to ~60s, then surfaces the
			//     error to the caller.
			//   - Bifrost / other proxies enforce their own deadlines.
			// 300s caused most SDK clients to surface user-visible errors
			// instead of retrying. With 60s, clients re-poll every minute,
			// see more 503s while the model is still cold-loading, and
			// eventually land on the warm Sleeping path on a later retry —
			// which is the whole point of the async-503 contract.
			c.Set("Retry-After", "60")
			return writeOpenAIError(
				c,
				http.StatusServiceUnavailable,
				"Model is cold-starting (~5 min for large models). Please retry.",
				"service_unavailable",
				"warming_up",
			)
		}

		route, err := opts.Router.AcquireRoute(c.UserContext(), modelName, requestID)
		if err != nil {
			return mapEnsureError(c, err)
		}
		if route.Done != nil {
			defer route.Done()
		}

		// When the classifier rewrote the target, force response-rewriting
		// so the client sees the original requested model name in the
		// completion. UpstreamModel rewrites the REQUEST body to the new
		// served-name so vLLM accepts it; RequestedModel + RewriteModelName
		// rewrites the RESPONSE back to the originally-requested name so
		// the client never sees the swap.
		//
		// External-lifecycle gotcha: route.UpstreamModel is sourced from
		// modelCfg.Path which is EMPTY for `lifecycle: external` models
		// (vllm-main, vllm-vision, etc — they're remote URLs, no on-disk
		// path). When the classifier rewrote, we MUST set UpstreamModel
		// explicitly to the rewritten served-name; otherwise the body
		// rewrite in proxy.RewriteJSONModel skips (the cond is
		// `UpstreamModel != ""`) and vision sees the OLD model name in
		// the body → 404 NotFoundError.
		rewriteModelName := opts.Config.Behavior.RewriteModelName
		requestedForRewrite := modelName
		upstreamForBody := route.UpstreamModel
		if rewroteByContent {
			rewriteModelName = true
			requestedForRewrite = originalModel
			if upstreamForBody == "" {
				upstreamForBody = modelName // the rewritten served-name
			}
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: rewriteModelName,
			RequestedModel:   requestedForRewrite,
			UpstreamModel:    upstreamForBody,
			RequestID:        requestID,
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
			return writeOpenAIError(c, http.StatusInternalServerError, "Backend in error state, please retry", "internal_error", "backend_error")
		}
		msg := "Model switch in progress, please retry"
		switch rej.Reason {
		case jukebox.RejectPinnedConflict:
			msg = "Insufficient resources: requested model conflicts with a pinned model"
		case jukebox.RejectNoCapacity:
			msg = "Insufficient capacity to start requested model, please retry"
		case jukebox.RejectInsufficient:
			// Non-retryable: structural infeasibility (model exceeds the
			// pinned-adjusted GPU budget no matter what we evict). The
			// header omits Retry-After so retry-aware SDKs don't loop.
			// Body must NOT contradict by saying "please retry".
			msg = "Model exceeds available GPU budget; reconfigure or add capacity"
		case jukebox.RejectAdminIntervention:
			// Non-retryable: admission books may have drifted from
			// physical container state because a cleanup `docker stop`
			// failed. Operator intervention is required (see audit log).
			msg = "Resource state inconsistent; operator intervention required (see audit log)"
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
