package jukebox

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ImageGenRequest mirrors the OpenAI POST /v1/images/generations body.
// We accept the minimum fields the smoke + Open-WebUI/Sparx consumers
// send; extra fields are ignored.
type ImageGenRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n"`
	Size           string `json:"size"`
	ResponseFormat string `json:"response_format"`
}

// ImageGenResponse is the OpenAI POST /v1/images/generations response.
type ImageGenResponse struct {
	Created int64           `json:"created"`
	Data    []ImageGenDatum `json:"data"`
}

type ImageGenDatum struct {
	B64JSON string `json:"b64_json"`
}

// ImageGenError is returned by ImageGenerate so HTTP handlers can map
// the underlying failure to the right OpenAI error envelope + status.
type ImageGenError struct {
	Status  int    // suggested HTTP status to surface
	Code    string // OpenAI error.code field
	Message string
}

func (e *ImageGenError) Error() string { return e.Message }

func newImageGenErr(status int, code, msg string) *ImageGenError {
	return &ImageGenError{Status: status, Code: code, Message: msg}
}

// ComfyUIImageGenDefaults captures the workflow knobs the SaveImage
// handler needs. Hardcoded for now — matches the deleted
// comfy-openai-shim defaults (commit 6caf50b, llm/comfy-openai-shim/app.py).
//
// If we ever want per-model tuning, plumb these through ModelConfig as
// an optional WorkflowDefaults sub-block; until then keep it simple.
const (
	defaultDiffusionModel = "qwen_image_fp8_e4m3fn.safetensors"
	defaultTextEncoder    = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
	defaultVAEName        = "qwen_image_vae.safetensors"
	defaultLightningLoRA  = "Qwen-Image-Lightning-8steps-V1.0.safetensors"
	defaultShift          = 3.1
	defaultSamplerSteps   = 8
	defaultCFG            = 1.0
	defaultSamplerName    = "euler"
	defaultScheduler      = "simple"

	// defaultImageW/H mirror the deleted shim — 1024x1024 when the
	// client omits size or sends garbage.
	defaultImageW = 1024
	defaultImageH = 1024

	// minImageDim / maxImageDim clamp + snap to multiples of 8 (Qwen-Image
	// VAE requirement).
	minImageDim = 64
	maxImageDim = 2048
)

// ImageGenOptions selects the ComfyUI instance + per-request timeouts.
// baseURL is the http://host:port that points at ComfyUI itself — for
// `lifecycle: comfyui` peers this is what Router.AcquireRoute returns
// via Route.BaseURL.
type ImageGenOptions struct {
	BaseURL string

	// PollTimeout caps how long we wait for ComfyUI to materialize the
	// SaveImage output. Defaults to 120s if zero.
	PollTimeout time.Duration

	// PollInterval is the gap between /history checks. Defaults to 1s.
	PollInterval time.Duration

	// HTTPTimeout is the per-call timeout for /prompt, /history, and
	// /view sub-requests. Defaults to 30s.
	HTTPTimeout time.Duration
}

// ImageGenerate is the workflow translator: it builds an 11-node
// Qwen-Image-Lightning workflow, POSTs to ComfyUI's /prompt endpoint,
// polls /history until the SaveImage output is materialized, fetches
// the PNG via /view, and returns the bytes base64-encoded. It does NOT
// drive lifecycle — the caller is responsible for having called
// Router.AcquireRoute first so ComfyUI is awake (the call wakes the
// peer via ComfyUIManager.Wake before returning a Route).
//
// Returns a *ImageGenError on failure so the HTTP handler can surface
// an appropriate OpenAI error envelope + status. Returns nil error and
// non-empty bytes on success.
func ImageGenerate(ctx context.Context, opts ImageGenOptions, req ImageGenRequest) ([]byte, error) {
	if strings.TrimSpace(opts.BaseURL) == "" {
		return nil, newImageGenErr(http.StatusInternalServerError, "config_missing",
			"image-gen called with empty base URL — model is not a ComfyUI peer?")
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return nil, newImageGenErr(http.StatusBadRequest, "invalid_request_error",
			"missing required field 'prompt'")
	}
	if req.N != 0 && req.N != 1 {
		return nil, newImageGenErr(http.StatusBadRequest, "invalid_request_error",
			"n must be 1 (jukebox does one image per call)")
	}

	if opts.PollTimeout == 0 {
		opts.PollTimeout = 120 * time.Second
	}
	if opts.PollInterval == 0 {
		opts.PollInterval = 1 * time.Second
	}
	if opts.HTTPTimeout == 0 {
		opts.HTTPTimeout = 30 * time.Second
	}

	width, height := parseImageSize(req.Size)

	seed, err := randomSeed()
	if err != nil {
		return nil, newImageGenErr(http.StatusInternalServerError, "internal_error",
			fmt.Sprintf("seed entropy: %v", err))
	}
	workflow := buildQwenImageWorkflow(req.Prompt, width, height, seed)

	client := &http.Client{Timeout: opts.HTTPTimeout}
	base := strings.TrimRight(opts.BaseURL, "/")

	// === Submit ===
	clientID := uuid.NewString()
	submitBody, err := json.Marshal(map[string]any{
		"prompt":    workflow,
		"client_id": clientID,
	})
	if err != nil {
		return nil, newImageGenErr(http.StatusInternalServerError, "internal_error",
			fmt.Sprintf("marshal workflow: %v", err))
	}
	submitReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		base+"/prompt", bytes.NewReader(submitBody))
	if err != nil {
		return nil, newImageGenErr(http.StatusInternalServerError, "internal_error",
			fmt.Sprintf("build /prompt request: %v", err))
	}
	submitReq.Header.Set("Content-Type", "application/json")

	submitResp, err := client.Do(submitReq)
	if err != nil {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_unreachable",
			fmt.Sprintf("ComfyUI POST /prompt: %v", err))
	}
	defer submitResp.Body.Close()
	submitRespBody, _ := io.ReadAll(submitResp.Body)
	if submitResp.StatusCode == http.StatusServiceUnavailable {
		// ComfyUI sometimes 503s during a reload; bubble up so the
		// caller's retry/backoff (or Bifrost) can re-issue.
		return nil, newImageGenErr(http.StatusServiceUnavailable, "warming_up",
			fmt.Sprintf("ComfyUI /prompt 503: %s", truncateString(string(submitRespBody), 256)))
	}
	if submitResp.StatusCode < 200 || submitResp.StatusCode >= 300 {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_error",
			fmt.Sprintf("ComfyUI /prompt %d: %s", submitResp.StatusCode,
				truncateString(string(submitRespBody), 256)))
	}
	var submitData struct {
		PromptID string `json:"prompt_id"`
	}
	if err := json.Unmarshal(submitRespBody, &submitData); err != nil || submitData.PromptID == "" {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_error",
			fmt.Sprintf("ComfyUI /prompt returned no prompt_id: %s",
				truncateString(string(submitRespBody), 256)))
	}

	// === Poll /history ===
	deadline := time.Now().Add(opts.PollTimeout)
	ticker := time.NewTicker(opts.PollInterval)
	defer ticker.Stop()

	for {
		// Quick gate so we don't sleep through context cancellation.
		select {
		case <-ctx.Done():
			return nil, newImageGenErr(http.StatusGatewayTimeout, "timeout",
				fmt.Sprintf("client cancelled while polling prompt_id %s", submitData.PromptID))
		default:
		}
		if time.Now().After(deadline) {
			return nil, newImageGenErr(http.StatusGatewayTimeout, "timeout",
				fmt.Sprintf("ComfyUI did not produce output for prompt_id %s within %s",
					submitData.PromptID, opts.PollTimeout))
		}

		img, errStatus, errCode, errMsg := pollHistoryOnce(ctx, client, base, submitData.PromptID)
		if errMsg != "" {
			return nil, newImageGenErr(errStatus, errCode, errMsg)
		}
		if img != nil {
			// === Fetch PNG via /view ===
			png, fetchErr := fetchViewBytes(ctx, client, base, img)
			if fetchErr != nil {
				return nil, fetchErr
			}
			return png, nil
		}

		select {
		case <-ticker.C:
		case <-ctx.Done():
			return nil, newImageGenErr(http.StatusGatewayTimeout, "timeout",
				fmt.Sprintf("client cancelled while polling prompt_id %s", submitData.PromptID))
		}
	}
}

// EncodeImageDataURIPayload base64-encodes the PNG bytes and wraps
// them in the OpenAI ImageGenResponse envelope. Created defaults to
// time.Now() if zero.
func EncodeImageDataURIPayload(png []byte, created int64) ImageGenResponse {
	if created == 0 {
		created = time.Now().Unix()
	}
	return ImageGenResponse{
		Created: created,
		Data: []ImageGenDatum{
			{B64JSON: base64.StdEncoding.EncodeToString(png)},
		},
	}
}

// pollHistoryOnce GETs /history/<prompt_id> and inspects the response.
// Returns (img, 0, "", "") with img != nil when an output is ready.
// Returns (nil, 0, "", "") when there is nothing yet (caller polls again).
// Returns (nil, status, code, msg) on a hard ComfyUI error.
type historyImage struct {
	Filename  string
	Subfolder string
	Typ       string
}

func pollHistoryOnce(ctx context.Context, client *http.Client, base, promptID string) (*historyImage, int, string, string) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		base+"/history/"+url.PathEscape(promptID), nil)
	if err != nil {
		return nil, http.StatusInternalServerError, "internal_error",
			fmt.Sprintf("build /history request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		// Transient — caller will retry. Don't fail the whole request
		// on a single missed poll.
		return nil, 0, "", ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		// Treat any non-200 as "not ready"; ComfyUI returns {} as 200
		// when the job is queued but not done.
		return nil, 0, "", ""
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, "", ""
	}
	var hist map[string]struct {
		Status struct {
			StatusStr string `json:"status_str"`
		} `json:"status"`
		Outputs map[string]struct {
			Images []struct {
				Filename  string `json:"filename"`
				Subfolder string `json:"subfolder"`
				Type      string `json:"type"`
			} `json:"images"`
		} `json:"outputs"`
	}
	if err := json.Unmarshal(body, &hist); err != nil {
		return nil, 0, "", ""
	}
	entry, ok := hist[promptID]
	if !ok {
		return nil, 0, "", ""
	}
	if entry.Status.StatusStr == "error" {
		return nil, http.StatusBadGateway, "upstream_error",
			fmt.Sprintf("ComfyUI workflow errored for prompt_id %s: %s",
				promptID, truncateString(string(body), 1024))
	}
	for _, nodeOut := range entry.Outputs {
		if len(nodeOut.Images) == 0 {
			continue
		}
		img := nodeOut.Images[0]
		typ := img.Type
		if typ == "" {
			typ = "output"
		}
		return &historyImage{
			Filename:  img.Filename,
			Subfolder: img.Subfolder,
			Typ:       typ,
		}, 0, "", ""
	}
	return nil, 0, "", ""
}

func fetchViewBytes(ctx context.Context, client *http.Client, base string, img *historyImage) ([]byte, *ImageGenError) {
	q := url.Values{}
	q.Set("filename", img.Filename)
	q.Set("subfolder", img.Subfolder)
	q.Set("type", img.Typ)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		base+"/view?"+q.Encode(), nil)
	if err != nil {
		return nil, newImageGenErr(http.StatusInternalServerError, "internal_error",
			fmt.Sprintf("build /view request: %v", err))
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_unreachable",
			fmt.Sprintf("ComfyUI GET /view: %v", err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_error",
			fmt.Sprintf("ComfyUI /view %d: %s", resp.StatusCode,
				truncateString(string(body), 256)))
	}
	png, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_error",
			fmt.Sprintf("read /view body: %v", err))
	}
	if len(png) == 0 {
		return nil, newImageGenErr(http.StatusBadGateway, "upstream_error",
			"ComfyUI /view returned empty body")
	}
	return png, nil
}

// parseImageSize handles "WxH" strings like "512x512". On invalid /
// empty input it falls back to defaultImageW x defaultImageH. Output
// is clamped to [minImageDim, maxImageDim] and snapped down to the
// nearest multiple of 8 (Qwen-Image VAE requirement).
func parseImageSize(s string) (int, int) {
	if strings.TrimSpace(s) == "" {
		return defaultImageW, defaultImageH
	}
	parts := strings.SplitN(strings.ToLower(s), "x", 2)
	if len(parts) != 2 {
		return defaultImageW, defaultImageH
	}
	w, errW := strconv.Atoi(strings.TrimSpace(parts[0]))
	h, errH := strconv.Atoi(strings.TrimSpace(parts[1]))
	if errW != nil || errH != nil || w <= 0 || h <= 0 {
		return defaultImageW, defaultImageH
	}
	return snapDim(w), snapDim(h)
}

func snapDim(v int) int {
	if v < minImageDim {
		v = minImageDim
	}
	if v > maxImageDim {
		v = maxImageDim
	}
	return (v / 8) * 8
}

func randomSeed() (int64, error) {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, err
	}
	return int64(binary.BigEndian.Uint32(b[:])), nil
}

// buildQwenImageWorkflow returns the ComfyUI API-format graph for one
// image. Mirrors the deleted comfy-openai-shim app.py (commit 6caf50b)
// node-for-node: 11 nodes, 8-step Lightning, cfg=1.0, ModelSamplingAuraFlow
// shift=3.1.
func buildQwenImageWorkflow(prompt string, width, height int, seed int64) map[string]any {
	return map[string]any{
		"1": map[string]any{
			"class_type": "VAELoader",
			"inputs": map[string]any{
				"vae_name": defaultVAEName,
			},
		},
		"2": map[string]any{
			"class_type": "CLIPLoader",
			"inputs": map[string]any{
				"clip_name": defaultTextEncoder,
				"type":      "qwen_image",
				"device":    "default",
			},
		},
		"3": map[string]any{
			"class_type": "UNETLoader",
			"inputs": map[string]any{
				"unet_name":    defaultDiffusionModel,
				"weight_dtype": "default",
			},
		},
		"4": map[string]any{
			"class_type": "LoraLoaderModelOnly",
			"inputs": map[string]any{
				"model":          []any{"3", 0},
				"lora_name":      defaultLightningLoRA,
				"strength_model": 1.0,
			},
		},
		"5": map[string]any{
			"class_type": "ModelSamplingAuraFlow",
			"inputs": map[string]any{
				"model": []any{"4", 0},
				"shift": defaultShift,
			},
		},
		"6": map[string]any{
			"class_type": "CLIPTextEncode",
			"inputs": map[string]any{
				"clip": []any{"2", 0},
				"text": prompt,
			},
		},
		"7": map[string]any{
			"class_type": "CLIPTextEncode",
			"inputs": map[string]any{
				"clip": []any{"2", 0},
				"text": "",
			},
		},
		"8": map[string]any{
			"class_type": "EmptySD3LatentImage",
			"inputs": map[string]any{
				"width":      width,
				"height":     height,
				"batch_size": 1,
			},
		},
		"9": map[string]any{
			"class_type": "KSampler",
			"inputs": map[string]any{
				"model":         []any{"5", 0},
				"positive":      []any{"6", 0},
				"negative":      []any{"7", 0},
				"latent_image":  []any{"8", 0},
				"seed":          seed,
				"steps":         defaultSamplerSteps,
				"cfg":           defaultCFG,
				"sampler_name":  defaultSamplerName,
				"scheduler":     defaultScheduler,
				"denoise":       1.0,
			},
		},
		"10": map[string]any{
			"class_type": "VAEDecode",
			"inputs": map[string]any{
				"samples": []any{"9", 0},
				"vae":     []any{"1", 0},
			},
		},
		"11": map[string]any{
			"class_type": "SaveImage",
			"inputs": map[string]any{
				"images":          []any{"10", 0},
				"filename_prefix": "jukebox",
			},
		},
	}
}

func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
