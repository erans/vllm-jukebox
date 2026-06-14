package vllmcli

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// VerifyWakeWithProbe issues a tiny single-token /v1/completions request
// against baseURL to confirm that an apparently-successful /wake_up has
// actually produced a working engine (and is not a "phantom wake" where
// /wake_up returns 200 but the worker processes are wedged).
//
// Background: vllm-project/vllm cumem_tag wake_up has been observed
// returning 200 from the API server while one or more PP/TP workers
// have hit "CUDA Error: invalid argument at cumem_allocator.cpp:169"
// during create_and_map. /health continues to return 200 (the FastAPI
// process is alive) but every subsequent inference request hangs forever
// because the shm_broadcast block is gone. The PR upstream that adds
// /health/decode (#45097) catches one slice of this; this probe catches
// the rest by exercising the actual decode path end-to-end with the
// smallest possible payload.
//
// Design choices:
//   - max_tokens: 1, temperature: 0 — minimizes compute; the goal is
//     "did decode return at all", not "is generation good".
//   - Unique prompt with crypto/rand suffix — defeats the prefix-cache
//     fast path so the probe really exercises the engine, not a cached
//     trie hit. A wedged engine fails on the cache miss path the same
//     way a real consumer request would.
//   - Timeout bound by ctx + the timeout parameter. The probe is a
//     defense BEFORE we route the user request, so the tighter the
//     budget the better — caller passes ~5s.
//   - Returns nil only on HTTP 2xx with at least one choice in the
//     response. Anything else (timeout, 5xx, malformed JSON, empty
//     choices) is treated as a failed verification: caller MUST treat
//     this as a phantom wake.
//
// Note: this function does NOT call /health/decode — that's a complementary
// upstream signal already polled separately during the standard wake
// health-wait loop. This probe specifically catches the "lying /health"
// failure mode where /health is fine but decode is wedged.
func VerifyWakeWithProbe(ctx context.Context, baseURL, model string, timeout time.Duration) error {
	if timeout <= 0 {
		return fmt.Errorf("VerifyWakeWithProbe: timeout must be > 0")
	}
	if strings.TrimSpace(baseURL) == "" {
		return fmt.Errorf("VerifyWakeWithProbe: baseURL is empty")
	}
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("VerifyWakeWithProbe: model is empty")
	}

	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// 8 bytes of randomness = 16 hex chars = enough entropy to defeat
	// prefix-cache collisions even under heavy probe load.
	var nonce [8]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		// crypto/rand failure is essentially unheard of on Linux;
		// surface it as a probe failure rather than silently degrading
		// to a fixed prompt that would hit the prefix cache.
		return fmt.Errorf("VerifyWakeWithProbe: generate nonce: %w", err)
	}
	prompt := "wake-verify-" + hex.EncodeToString(nonce[:])

	body, err := json.Marshal(struct {
		Model       string  `json:"model"`
		Prompt      string  `json:"prompt"`
		MaxTokens   int     `json:"max_tokens"`
		Temperature float64 `json:"temperature"`
		Stream      bool    `json:"stream"`
	}{
		Model:       model,
		Prompt:      prompt,
		MaxTokens:   1,
		Temperature: 0,
		Stream:      false,
	})
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: marshal body: %w", err)
	}

	u := strings.TrimRight(baseURL, "/") + "/v1/completions"
	req, err := http.NewRequestWithContext(probeCtx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: POST /v1/completions: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("VerifyWakeWithProbe: POST /v1/completions returned %s", resp.Status)
	}

	var decoded struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Text         string `json:"text"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return fmt.Errorf("VerifyWakeWithProbe: decode response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return fmt.Errorf("VerifyWakeWithProbe: response had zero choices (phantom wake?)")
	}
	// Don't require non-empty text — max_tokens:1 with temperature:0 on
	// some models legitimately emits an empty string (e.g. EOS as the
	// first sampled token). The fact that decode completed AT ALL is
	// what proves the engine isn't wedged.
	return nil
}
