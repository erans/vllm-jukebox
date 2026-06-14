package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// comfyUIFreeRequest is the body POSTed to ComfyUI's /free endpoint.
// Both flags set true tells ComfyUI to unload all currently-loaded
// models and release VRAM. ComfyUI has only one freeing mode — there
// is no equivalent of vLLM's sleep_level — so we hardcode the body.
type comfyUIFreeRequest struct {
	UnloadModels bool `json:"unload_models"`
	FreeMemory   bool `json:"free_memory"`
}

// ComfyUIManager is a sibling of ExternalManager for ComfyUI sleep/wake
// semantics. Like ExternalManager, it does NOT own the process; an
// operator runs ComfyUI as a separate container and jukebox proxies
// requests through it. Unlike ExternalManager it does NOT delegate to
// vllmcli — ComfyUI speaks a different protocol:
//
//   - Sleep: POST /free with {"unload_models":true,"free_memory":true}.
//     ComfyUI does not accept a "sleep level"; there is only one freeing
//     mode. The level argument to Sleep is therefore ignored.
//   - Wake: there is no explicit wake call. The next POST /prompt to
//     ComfyUI implicitly reloads any model that wasn't previously
//     loaded. Our Wake() is therefore a state-only flip plus an
//     optional reachability probe so callers get a quick error if
//     ComfyUI itself is unreachable.
//   - IsSleeping: ComfyUI does not expose a sleeping/loaded probe.
//     Jukebox tracks sleeping state in memory; we report the in-process
//     flag rather than lying about probing the engine.
//
// All state mutations go through Sleep / Wake under the manager's
// mutex. There is no manual setter — callers must drive state through
// the sleep/wake API surface.
type ComfyUIManager struct {
	baseURL string
	client  *http.Client

	mu       sync.RWMutex
	sleeping bool
}

// NewComfyUIManager constructs a ComfyUIManager pointing at the given
// host:port. Like ExternalManager, only http:// is supported.
func NewComfyUIManager(host string, port int) *ComfyUIManager {
	u := url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("%s:%d", host, port),
	}
	return &ComfyUIManager{
		baseURL: u.String(),
		client:  &http.Client{},
	}
}

// BaseURL returns http://host:port for the external ComfyUI service.
func (m *ComfyUIManager) BaseURL() string { return m.baseURL }

// CurrentPID returns the same non-zero sentinel as ExternalManager.
// Jukebox does not own the ComfyUI process; the sentinel keeps the
// "pid==0 implies dead instance" guard in the scheduler from
// false-flagging this manager.
func (m *ComfyUIManager) CurrentPID() int { return externalSentinelPID }

// Start is a no-op for ComfyUI instances — jukebox does not own the
// process. Returns the sentinel pid so callers that record it observe
// the same value CurrentPID would report.
func (m *ComfyUIManager) Start(_ context.Context, _ string) (int, error) {
	return externalSentinelPID, nil
}

// Stop is a no-op. Jukebox does not own the ComfyUI process; the
// operator owns its container lifecycle.
func (m *ComfyUIManager) Stop(_ context.Context) error { return nil }

// VerifyReady probes /system_stats. ComfyUI returns 200 with a JSON
// payload describing the device/system whenever the server is up — it
// is the closest analogue to vLLM's /health. The expectedModel
// argument is ignored: ComfyUI doesn't load a single "current model",
// it loads what each workflow asks for on demand.
func (m *ComfyUIManager) VerifyReady(ctx context.Context, _ string) error {
	return m.probeSystemStats(ctx)
}

// Sleep POSTs /free with both flags set. The level argument is ignored
// (ComfyUI has only one freeing mode). On success the sleeping flag
// is set to true; subsequent IsSleeping calls reflect the new state.
// On HTTP error the flag is left unchanged.
func (m *ComfyUIManager) Sleep(ctx context.Context, _ int) error {
	body, err := json.Marshal(comfyUIFreeRequest{
		UnloadModels: true,
		FreeMemory:   true,
	})
	if err != nil {
		return fmt.Errorf("comfyui: marshal /free body: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(m.baseURL, "/")+"/free", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("comfyui: build /free request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("comfyui: POST /free: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("comfyui: POST /free: unexpected status %d", resp.StatusCode)
	}

	m.mu.Lock()
	m.sleeping = true
	m.mu.Unlock()
	return nil
}

// Wake is mostly a state flip — ComfyUI auto-wakes on the next
// POST /prompt, so there's nothing to POST here. We still attempt a
// /system_stats probe so the caller gets a clean error if ComfyUI is
// unreachable, rather than silently flipping state and 5xx'ing on the
// next prompt. The probe is bounded by timeout; if it never succeeds
// we return an error and leave the sleeping flag intact.
func (m *ComfyUIManager) Wake(ctx context.Context, timeout time.Duration) error {
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := m.probeSystemStats(probeCtx); err != nil {
		return fmt.Errorf("comfyui: wake reachability probe: %w", err)
	}
	m.mu.Lock()
	m.sleeping = false
	m.mu.Unlock()
	return nil
}

// IsSleeping returns the in-memory state flag. ComfyUI has no
// /is_sleeping endpoint; jukebox is the source of truth for whether
// this instance has been told to sleep. Document this honestly in
// reviews / dashboards: a fresh ComfyUIManager reports sleeping=false
// even if ComfyUI's own internal state has unloaded all models.
func (m *ComfyUIManager) IsSleeping(_ context.Context) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sleeping, nil
}

// probeSystemStats GETs /system_stats and returns nil iff the response
// is 2xx. Used by both VerifyReady (no timeout — caller controls) and
// Wake (timeout bounded by caller's wake timeout).
func (m *ComfyUIManager) probeSystemStats(ctx context.Context) error {
	u := strings.TrimRight(m.baseURL, "/") + "/system_stats"
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return err
		}
		resp, err := m.client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return nil
			}
		}
		select {
		case <-ticker.C:
			continue
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
