package vllmcli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// VerifyModelLoaded calls GET /v1/models on the vLLM backend and confirms
// that the expected model id (or its underlying path) is present.
func VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	url := strings.TrimRight(baseURL, "/") + "/v1/models"
	client := &http.Client{}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("GET /v1/models returned %s", resp.Status)
	}

	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return err
	}

	for _, m := range decoded.Data {
		if m.ID == expectedID || (expectedPath != "" && m.ID == expectedPath) {
			return nil
		}
	}
	return fmt.Errorf("expected model %q not found in /v1/models", expectedID)
}

// VerifyForwardPass issues a minimal real generation against the backend's
// OpenAI-compatible /v1/completions endpoint and confirms a non-empty
// completion comes back from a genuine DECODE step.
//
// This exists because /health and /v1/models only prove the HTTP server and
// model registry are up — they do NOT exercise the inference engine. After a
// cumem sleep/wake cycle the engine can be silently corrupted (e.g. a CUDA
// graph replaying against remapped device pointers): /health returns 200 and
// /v1/models lists the model, but a real forward pass crashes with an
// illegal-address error. Routing user traffic in that window yields a 500 to
// the client.
//
// CRITICAL — the probe MUST force at least one DECODE step, not just a prefill.
// The post-wake crash fires on the decode cudagraph FULL-replay path (the
// uniform_decode batch), NOT on prefill. A max_tokens=1 completion returns
// after the prefill step alone, so the decode cudagraph is never invoked and a
// poisoned engine sails through the probe — then the next real request (which
// does decode) crashes. To guarantee a decode iteration we request
// max_tokens>=2 (the 2nd token can only be produced by a decode step after the
// 1st prefill-produced token) AND set min_tokens>=2 + ignore_eos so the engine
// cannot short-circuit by emitting EOS on token 1. The prompt is deliberately
// short so the whole probe stays cheap.
func VerifyForwardPass(ctx context.Context, baseURL, expectedID string) error {
	url := strings.TrimRight(baseURL, "/") + "/v1/completions"

	reqBody, err := json.Marshal(map[string]any{
		"model":  expectedID,
		"prompt": "ok",
		// >=2 tokens forces at least one decode step (the crash path) after
		// the initial prefill. min_tokens + ignore_eos prevent the engine
		// from stopping at token 1 via an early EOS, which would skip decode.
		"max_tokens":  2,
		"min_tokens":  2,
		"ignore_eos":  true,
		"temperature": 0,
		"stream":      false,
	})
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		// Connection reset / refused / context deadline: a wedged or crashed
		// engine looks exactly like this. Treat as verification failure.
		return fmt.Errorf("forward-pass probe transport error: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024))

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("forward-pass probe POST /v1/completions returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var decoded struct {
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return fmt.Errorf("forward-pass probe: decode /v1/completions response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return fmt.Errorf("forward-pass probe: /v1/completions returned no choices (engine produced no output)")
	}

	// A poisoned engine can return a 200 with an empty completion (no tokens
	// actually decoded) or a missing/garbage finish_reason. Require real output
	// AND a sane terminal reason so the empty/half-dead case is not accepted as
	// a pass. For our 2-token min request a healthy engine finishes with
	// "length" (hit max_tokens); "stop" is also accepted for robustness.
	choice := decoded.Choices[0]
	if strings.TrimSpace(choice.Text) == "" {
		return fmt.Errorf("forward-pass probe: /v1/completions returned empty text (engine decoded no tokens)")
	}
	switch choice.FinishReason {
	case "stop", "length":
		// healthy terminal states
	default:
		return fmt.Errorf("forward-pass probe: /v1/completions finish_reason %q is not a healthy terminal state (want stop|length)", choice.FinishReason)
	}

	return nil
}
