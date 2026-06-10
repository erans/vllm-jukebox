package httpserver

import (
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
)

// table-driven coverage of the classifier across all 5 decisions + 5 guard
// cases. The classifier is in-process — these tests do NOT spin up a fake
// Fiber app; that's the handler integration test's job (handlers_proxy_test).

func newClassifierCfg(routing bool) *config.Config {
	return &config.Config{
		Models: map[string]config.ModelConfig{
			"text-model":   {Path: "/models/text"},
			"vision-model": {Path: "/models/vision"},
			"long-model":   {Path: "/models/long"},
		},
		Behavior: config.BehaviorConfig{
			ContentRouting:                routing,
			MultimodalCapableModels:       []string{"vision-model"},
			VisionModel:                   "vision-model",
			OverflowModel:                 map[string]string{"text-model": "long-model"},
			LengthOverflowThresholdTokens: 100, // intentionally low for testing
			LengthHardCapTokens:           1000,
			EstTokensCharsPerToken:        3.3,
		},
	}
}

func TestClassifyContent_KillSwitchOff_AlwaysForward(t *testing.T) {
	cfg := newClassifierCfg(false)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("kill-switch off must always forward, got %v", got.Decision)
	}
	// And no token estimate was computed (early-out), so EstTokens=0.
	if got.EstTokens != 0 {
		t.Errorf("kill-switch off should skip estimate, got EstTokens=%d", got.EstTokens)
	}
}

func TestClassifyContent_PlainText_Forward(t *testing.T) {
	cfg := newClassifierCfg(true)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"hello world"}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("plain text must forward, got %v (%s)", got.Decision, got.Reason)
	}
	if got.HasImage {
		t.Errorf("plain text must not flag HasImage")
	}
}

func TestClassifyContent_ImageToMMCapable_Forward(t *testing.T) {
	cfg := newClassifierCfg(true)
	body := []byte(`{"model":"vision-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`)
	got := ClassifyContent(body, "vision-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("image to MM-capable model must forward, got %v", got.Decision)
	}
	if !got.HasImage {
		t.Errorf("classifier missed image_url in MM-capable forward path")
	}
}

func TestClassifyContent_ImageToTextModel_RerouteVision(t *testing.T) {
	cfg := newClassifierCfg(true)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionRerouteImage {
		t.Fatalf("image to text-model must reroute, got %v (%s)", got.Decision, got.Reason)
	}
	if got.NewModel != "vision-model" {
		t.Errorf("expected NewModel=vision-model, got %q", got.NewModel)
	}
}

func TestClassifyContent_ImageNoVisionConfigured_Reject(t *testing.T) {
	cfg := newClassifierCfg(true)
	cfg.Behavior.VisionModel = "" // remove vision target
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}}]}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionRejectNoVision {
		t.Fatalf("image+no vision must reject, got %v (%s)", got.Decision, got.Reason)
	}
	if got.NewModel != "" {
		t.Errorf("reject decisions must not carry a NewModel, got %q", got.NewModel)
	}
}

func TestClassifyContent_LengthOverflow_Reroute(t *testing.T) {
	cfg := newClassifierCfg(true)
	// threshold is 100 tokens at 3.3 chars/tok = 330 chars. Build a 400-char body.
	long := strings.Repeat("x", 400)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"` + long + `"}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionRerouteOverflow {
		t.Fatalf("long prompt must reroute, got %v est=%d", got.Decision, got.EstTokens)
	}
	if got.NewModel != "long-model" {
		t.Errorf("expected NewModel=long-model, got %q", got.NewModel)
	}
}

func TestClassifyContent_HardCap_Reject(t *testing.T) {
	cfg := newClassifierCfg(true)
	// hard cap is 1000 tokens at 3.3 chars/tok = 3300 chars. Build a 4000-char body.
	long := strings.Repeat("x", 4000)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"` + long + `"}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionRejectTooLong {
		t.Fatalf("over-hard-cap prompt must reject, got %v est=%d", got.Decision, got.EstTokens)
	}
}

func TestClassifyContent_OverflowNotConfigured_Forward(t *testing.T) {
	cfg := newClassifierCfg(true)
	cfg.Behavior.OverflowModel = nil // no overflow target for any model
	long := strings.Repeat("x", 400)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"` + long + `"}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("long prompt + no overflow target must forward (let upstream reject), got %v", got.Decision)
	}
}

func TestClassifyContent_MalformedJSON_Forward(t *testing.T) {
	cfg := newClassifierCfg(true)
	body := []byte(`{this is not json`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("malformed JSON must forward (upstream emits 400), got %v", got.Decision)
	}
}

// FALSE-POSITIVE GUARD — the classifier must not flag the literal string
// "image_url" embedded in TEXT content. This is the case where a
// substring scan would mis-route, and is why the classifier walks
// content[].type proper.
func TestClassifyContent_ImageURLStringInText_DoesNotReroute(t *testing.T) {
	cfg := newClassifierCfg(true)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"You can use image_url content blocks to attach images."}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("literal 'image_url' in text must NOT reroute (substring scan would FP here), got %v", got.Decision)
	}
	if got.HasImage {
		t.Errorf("classifier mis-flagged HasImage on text containing the literal string")
	}
}

func TestClassifyContent_EmptyBody_Forward(t *testing.T) {
	cfg := newClassifierCfg(true)
	got := ClassifyContent(nil, "text-model", cfg)
	if got.Decision != DecisionForward {
		t.Fatalf("empty body must forward, got %v", got.Decision)
	}
}

func TestClassifyContent_NilConfig_Forward(t *testing.T) {
	body := []byte(`{"model":"x","messages":[{"role":"user","content":[{"type":"image_url"}]}]}`)
	got := ClassifyContent(body, "x", nil)
	if got.Decision != DecisionForward {
		t.Fatalf("nil cfg must forward, got %v", got.Decision)
	}
}

// Image present + over hard cap: the image rule fires FIRST per the docstring's
// precedence. (Hard-cap on the OVERFLOW path; image reroute is unconditional
// when target is non-MM.)
func TestClassifyContent_ImageWinsOverLength(t *testing.T) {
	cfg := newClassifierCfg(true)
	// 400 chars > overflow threshold but < hard cap. Image content present.
	long := strings.Repeat("x", 400)
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x"}},{"type":"text","text":"` + long + `"}]}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	if got.Decision != DecisionRerouteImage {
		t.Fatalf("image precedence over length, got %v", got.Decision)
	}
	if got.NewModel != "vision-model" {
		t.Errorf("expected vision reroute, got NewModel=%q", got.NewModel)
	}
}

// Token estimate sanity — 990 chars at 3.3 cpt = 300 tokens (below threshold 100? wait)
// Actually we want to verify the estimate math. With cpt=3.3, 330 chars ≈ 100 tokens.
func TestClassifyContent_TokenEstimateMath(t *testing.T) {
	cfg := newClassifierCfg(true)
	cfg.Behavior.LengthOverflowThresholdTokens = 999999 // disable overflow rule
	body := []byte(`{"model":"text-model","messages":[{"role":"user","content":"` + strings.Repeat("x", 330) + `"}]}`)
	got := ClassifyContent(body, "text-model", cfg)
	// total body length is well over 330 (includes JSON overhead).
	if got.EstTokens < 100 {
		t.Errorf("estimate too low for ~370-char body: got %d", got.EstTokens)
	}
}
