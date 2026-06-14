package vllmcli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Sleep asks vLLM to enter sleep mode at the given level.
//
//   - Level 1 (L1): offload weights to CPU RAM, discard KV cache. GPU
//     memory is released via CUDA managed memory; wake restores from RAM.
//   - Level 2 (L2): discard weights and KV cache entirely. GPU memory is
//     fully freed; wake requires reloading weights from disk.
//
// vLLM must be started with --enable-sleep-mode and the
// VLLM_SERVER_DEV_MODE=1 environment variable for /sleep to be exposed.
// Returns nil on 2xx, otherwise an error wrapping the HTTP status.
func Sleep(ctx context.Context, baseURL string, level int) error {
	u := fmt.Sprintf("%s/sleep?level=%d", strings.TrimRight(baseURL, "/"), level)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST /sleep: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("POST /sleep returned %s", resp.Status)
	}
	return nil
}

// Wake POSTs /wake_up, then polls /health every 500ms until a 200 response
// arrives or timeout elapses. Returns nil on success. The timeout governs
// the total elapsed time including both the POST and the health polling.
func Wake(ctx context.Context, baseURL string, timeout time.Duration) error {
	if timeout <= 0 {
		return fmt.Errorf("Wake: timeout must be > 0")
	}
	deadline := time.Now().Add(timeout)
	wakeCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	wakeURL := strings.TrimRight(baseURL, "/") + "/wake_up"
	req, err := http.NewRequestWithContext(wakeCtx, http.MethodPost, wakeURL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST /wake_up: %w", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("POST /wake_up returned %s", resp.Status)
	}

	// Poll /health until 200 OK or deadline.
	healthURL := strings.TrimRight(baseURL, "/") + "/health"
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		hreq, err := http.NewRequestWithContext(wakeCtx, http.MethodGet, healthURL, nil)
		if err != nil {
			return err
		}
		hresp, herr := http.DefaultClient.Do(hreq)
		if herr == nil {
			status := hresp.StatusCode
			_ = hresp.Body.Close()
			if status == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ticker.C:
			continue
		case <-wakeCtx.Done():
			return fmt.Errorf("wake timed out after %s waiting for /health=200: %w", timeout, wakeCtx.Err())
		}
	}
}

// IsSleeping queries vLLM's /is_sleeping endpoint and parses the
// {"is_sleeping": bool} response.
func IsSleeping(ctx context.Context, baseURL string) (bool, error) {
	u := strings.TrimRight(baseURL, "/") + "/is_sleeping"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return false, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("GET /is_sleeping: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return false, fmt.Errorf("GET /is_sleeping returned %s", resp.Status)
	}
	var payload struct {
		IsSleeping bool `json:"is_sleeping"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return false, fmt.Errorf("decode /is_sleeping response: %w", err)
	}
	return payload.IsSleeping, nil
}
