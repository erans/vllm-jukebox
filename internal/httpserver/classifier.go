package httpserver

import (
	"encoding/json"

	"vllm-jukebox/internal/config"
)

// ContentDecision is the outcome of classifying an inbound chat-completions
// request body against the active routing policy.
type ContentDecision int

const (
	// DecisionForward — the request is fine as-is. Emit no metric. Most
	// requests land here; this is the cheap path.
	DecisionForward ContentDecision = iota
	// DecisionRerouteImage — request has image content but the requested
	// model is not in multimodal_capable_models. Caller should rewrite the
	// model name to the configured vision target.
	DecisionRerouteImage
	// DecisionRerouteOverflow — request's estimated token count exceeds the
	// requested model's max_model_len. Caller should rewrite to the
	// overflow_model target.
	DecisionRerouteOverflow
	// DecisionRejectNoVision — request has image content, model is not
	// multimodal-capable, AND no vision target is configured. Caller should
	// return a clear 503/4xx rather than silently 200 with hallucination.
	DecisionRejectNoVision
	// DecisionRejectTooLong — request's estimated token count exceeds the
	// long-context cap (overflow_model max_model_len). No backend can serve
	// it. Caller should return a clear 4xx.
	DecisionRejectTooLong
)

func (d ContentDecision) String() string {
	switch d {
	case DecisionForward:
		return "forward"
	case DecisionRerouteImage:
		return "reroute_image"
	case DecisionRerouteOverflow:
		return "reroute_overflow"
	case DecisionRejectNoVision:
		return "reject_no_vision"
	case DecisionRejectTooLong:
		return "reject_too_long"
	default:
		return "unknown"
	}
}

// ClassifyResult is the structured output of ClassifyContent, ready to drive
// the proxy handler's routing decision.
type ClassifyResult struct {
	Decision  ContentDecision
	NewModel  string // when DecisionReroute*; empty otherwise
	Reason    string // human-readable for logs/responses
	EstTokens int    // body-bytes / chars-per-token (cheap estimate, ±25%)
	HasImage  bool
}

// chatRequestProbe is the minimal JSON shape we walk to detect image content
// blocks. We accept content as either a string (no images possible) or as an
// array of typed parts (image_url, text, etc.). Anything else (numbers,
// objects, malformed) is treated as no-image — the underlying vLLM will
// still see the original body and reject it normally.
type chatRequestProbe struct {
	Model    string                  `json:"model"`
	Messages []chatRequestProbeEntry `json:"messages"`
}

type chatRequestProbeEntry struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// hasImageContentBlock walks message content[].type fields looking for
// "image_url". Uses json.RawMessage so we don't pay for full content unmarshal
// when the field is a plain string (the common case).
func hasImageContentBlock(messages []chatRequestProbeEntry) bool {
	for _, m := range messages {
		if len(m.Content) == 0 || m.Content[0] == '"' {
			continue // string content cannot carry an image block
		}
		// content is an array of typed parts
		var parts []struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(m.Content, &parts); err != nil {
			continue // malformed; let the upstream reject it
		}
		for _, p := range parts {
			if p.Type == "image_url" || p.Type == "input_image" {
				return true
			}
		}
	}
	return false
}

// ClassifyContent inspects the inbound chat-completions request body and
// returns a routing decision.
//
// Cheap path: if cfg.Behavior.ContentRouting is false, returns DecisionForward
// without any JSON walk — zero added latency for the kill-switch case.
//
// Full path: parses messages once (we accept the cost — typical body is
// <1MB), checks for image content blocks, computes a byte-based token
// estimate, applies the configured rules, returns a structured decision.
//
// Token estimate is len(body) * EstTokensCharsPerToken^-1, rounded up. JSON
// chat (with role markers, escapes, tool definitions) skews ~3 chars/tok in
// English vs the naive 4 chars/tok — defaulting to 3.3 (configurable). For
// the 256K threshold this gives ±25% accuracy, which is fine for a coarse
// "should we reroute?" decision.
//
// Decision precedence: image_url → length_overflow → forward. An image
// request that's also too long for vision goes to DecisionRejectTooLong only
// if the vision target itself can't fit it; otherwise it routes to vision.
func ClassifyContent(body []byte, requestedModel string, cfg *config.Config) ClassifyResult {
	r := ClassifyResult{Decision: DecisionForward}
	if cfg == nil || !cfg.Behavior.ContentRouting {
		return r
	}
	if len(body) == 0 {
		return r
	}

	// Estimate tokens up-front. Cheap, no JSON parse needed.
	cpt := cfg.Behavior.EstTokensCharsPerToken
	if cpt <= 0 {
		cpt = 3.3 // calibrated for English JSON chat with tool defs
	}
	r.EstTokens = int(float64(len(body))/cpt + 0.5)

	// Parse just enough to walk content blocks.
	var probe chatRequestProbe
	if err := json.Unmarshal(body, &probe); err != nil {
		// Malformed JSON — let the upstream's existing 400 path handle it.
		return r
	}
	r.HasImage = hasImageContentBlock(probe.Messages)

	// Rule P1: image content + non-MM target → vision reroute (or reject).
	if r.HasImage {
		if isMultimodalCapable(requestedModel, cfg.Behavior.MultimodalCapableModels) {
			// Target is MM-capable, no reroute needed; fall through to
			// length check (vision still has a max-len limit).
		} else {
			vision := cfg.Behavior.VisionModel
			if vision == "" {
				r.Decision = DecisionRejectNoVision
				r.Reason = "request contains image content but target model is not multimodal-capable and no vision_model is configured"
				return r
			}
			r.Decision = DecisionRerouteImage
			r.NewModel = vision
			r.Reason = "image content; rerouting to vision model"
			return r
		}
	}

	// Rule P2: length overflow.
	threshold := cfg.Behavior.LengthOverflowThresholdTokens
	if threshold > 0 && r.EstTokens > threshold {
		hardCap := cfg.Behavior.LengthHardCapTokens
		if hardCap > 0 && r.EstTokens > hardCap {
			r.Decision = DecisionRejectTooLong
			r.Reason = "estimated token count exceeds long-context capacity"
			return r
		}
		overflow := cfg.Behavior.OverflowModel[requestedModel]
		if overflow == "" {
			// No overflow target configured for this model — forward as-is
			// and let the upstream reject (vLLM will return its own
			// "input is too long for max_model_len" error).
			return r
		}
		r.Decision = DecisionRerouteOverflow
		r.NewModel = overflow
		r.Reason = "prompt exceeds requested model context; rerouting to long-context model"
		return r
	}

	return r
}

func isMultimodalCapable(name string, list []string) bool {
	for _, m := range list {
		if m == name {
			return true
		}
	}
	return false
}
