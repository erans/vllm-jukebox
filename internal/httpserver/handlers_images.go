package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/jukebox"
)

// imagesGenerationsHandler implements OpenAI POST /v1/images/generations
// by translating to a ComfyUI workflow. Lifecycle (wake-on-request,
// admission, sleep coordination) is driven through the existing
// Router.AcquireRoute path — for a `lifecycle: comfyui` peer this
// implicitly calls ComfyUIManager.Wake which probes /system_stats
// until ComfyUI is reachable.
//
// Returns the OpenAI shape on success:
//
//	{"created": <unix-ts>, "data": [{"b64_json": "<base64-PNG>"}]}
//
// Errors are wrapped in the standard writeOpenAIError envelope using
// the status/code suggested by the ImageGenerate translator.
func imagesGenerationsHandler(opts Options) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if opts.Config == nil || opts.Router == nil {
			return writeOpenAIError(c, http.StatusInternalServerError,
				"server not configured", "internal_error", "config_missing")
		}

		var req jukebox.ImageGenRequest
		if len(c.Body()) > 0 {
			if err := json.Unmarshal(c.Body(), &req); err != nil {
				return writeOpenAIError(c, http.StatusBadRequest,
					"invalid JSON body: "+err.Error(),
					"invalid_request_error", "invalid_json")
			}
		}
		if strings.TrimSpace(req.Model) == "" {
			return writeOpenAIError(c, http.StatusBadRequest,
				"missing required field 'model'",
				"invalid_request_error", "model_not_found")
		}
		if strings.TrimSpace(req.Prompt) == "" {
			return writeOpenAIError(c, http.StatusBadRequest,
				"missing required field 'prompt'",
				"invalid_request_error", "invalid_request")
		}
		if req.N == 0 {
			req.N = 1
		}

		modelName := req.Model
		c.Locals(requestedModelLocal, modelName)

		// Resolve against the live config so a mid-flight active.yaml
		// reload that adds / removes a model is reflected immediately.
		liveCfg := config.Current()
		if liveCfg == nil {
			liveCfg = opts.Config
		}
		_, modelCfg, err := liveCfg.ResolveModel(modelName)
		if err != nil {
			return writeOpenAIError(c, http.StatusBadRequest,
				"Model not found in configuration: "+modelName,
				"invalid_request_error", "model_not_found")
		}
		if modelCfg.EffectiveLifecycle() != config.LifecycleComfyUI {
			// /v1/images/generations is only valid against ComfyUI peers
			// today. If a caller asks for a vllm model here, surface a
			// 400 rather than silently routing into the workflow translator
			// (which would 502 with a confusing ComfyUI error).
			return writeOpenAIError(c, http.StatusBadRequest,
				"Model "+modelName+" is not a ComfyUI peer; /v1/images/generations requires lifecycle: comfyui",
				"invalid_request_error", "unsupported_model")
		}

		requestID, _ := c.Locals(requestIDHeader).(string)
		if requestID == "" {
			requestID = c.Get(requestIDHeader)
		}

		// AcquireRoute drives lifecycle: wakes the ComfyUI peer if it
		// was sleeping (POST /free'd), bumps the LRU, and returns the
		// http://comfyui:8188 base URL. Done is invoked when this
		// request finishes so admission can release any held capacity.
		route, err := opts.Router.AcquireRoute(c.UserContext(), modelName, requestID)
		if err != nil {
			return mapEnsureError(c, err)
		}
		if route.Done != nil {
			defer route.Done()
		}
		baseURL := route.BaseURL
		if baseURL == "" {
			// Defensive: a comfyui-lifecycle peer should always
			// resolve to its host:port. If it doesn't, derive from
			// config so we still hit ComfyUI rather than blank-URL.
			baseURL = fmt.Sprintf("http://%s:%d", modelCfg.Host, modelCfg.Port)
		}

		// Bound the upstream wallclock generously. Cold-load on ComfyUI
		// (NFS reload) is ~60-90s per Kagi research, sampling itself
		// is fast under Lightning (~3-5s for 1024x1024 on a 4090) plus
		// VAE decode + PNG encode. 180s leaves headroom for queueing.
		ctx, cancel := context.WithTimeout(c.UserContext(), 180*time.Second)
		defer cancel()

		png, err := jukebox.ImageGenerate(ctx, jukebox.ImageGenOptions{
			BaseURL:      baseURL,
			PollTimeout:  150 * time.Second,
			PollInterval: 1 * time.Second,
			HTTPTimeout:  30 * time.Second,
		}, req)
		if err != nil {
			var ige *jukebox.ImageGenError
			if errors.As(err, &ige) {
				status := ige.Status
				if status == 0 {
					status = http.StatusBadGateway
				}
				return writeOpenAIError(c, status, ige.Message,
					"upstream_error", ige.Code)
			}
			return writeOpenAIError(c, http.StatusBadGateway, err.Error(),
				"upstream_error", "image_gen_failed")
		}

		resp := jukebox.EncodeImageDataURIPayload(png, time.Now().Unix())
		return c.Status(http.StatusOK).JSON(resp)
	}
}
