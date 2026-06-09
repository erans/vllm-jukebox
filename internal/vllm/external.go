package vllm

import (
	"context"
	"fmt"
	"net/url"
	"time"

	"vllm-jukebox/internal/vllmcli"
)

// externalSentinelPID is returned by ExternalManager.CurrentPID(). It is
// any non-zero value so the scheduler's "ready but pid==0 → error" guard
// (see Coordinator.Status / Scheduler.tryRouteReady) doesn't false-flag
// externally-managed instances. Choose 1 because it's clearly not a real
// process and unlikely to collide with anything jukebox itself reasons
// about.
const externalSentinelPID = 1

// ExternalManager satisfies the same InstanceManager interface as the
// process-owning Manager but does NOT spawn or stop a vLLM process. It
// is used for lifecycle: external models, where an operator owns the
// vLLM container (typically as a separate docker-compose service) and
// jukebox only proxies requests and — when sleep_mode is enabled —
// sends sleep/wake HTTP calls.
//
// The external vLLM service must be started with both --enable-sleep-mode
// and VLLM_SERVER_DEV_MODE=1 for sleep-mode operations to function.
// Without those, Sleep/Wake/IsSleeping will return errors from vllmcli.
type ExternalManager struct {
	baseURL string
}

// NewExternalManager constructs an ExternalManager pointing at the
// given host:port. Host may be a DNS name (e.g. another compose
// service) or an IP. The constructed BaseURL uses http://; jukebox
// does not currently support https for the external endpoint.
func NewExternalManager(host string, port int) *ExternalManager {
	u := url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("%s:%d", host, port),
	}
	return &ExternalManager{baseURL: u.String()}
}

// BaseURL returns http://host:port for the external vLLM service.
func (m *ExternalManager) BaseURL() string { return m.baseURL }

// CurrentPID returns a non-zero sentinel. ExternalManager does not own
// a process; the sentinel exists purely so jukebox state checks that
// gate on pid != 0 ("is the process actually running?") don't fail
// against externally-managed instances.
func (m *ExternalManager) CurrentPID() int { return externalSentinelPID }

// Start is a no-op for external instances — jukebox does not own the
// process. Returns the sentinel pid so callers that record it observe
// the same value CurrentPID would report.
func (m *ExternalManager) Start(_ context.Context, _ string) (int, error) {
	return externalSentinelPID, nil
}

// Stop is a no-op for external instances. Jukebox is not authorized to
// stop a process it doesn't own; if the operator wants to shut down the
// external service they do it via their own compose / orchestration.
func (m *ExternalManager) Stop(_ context.Context) error { return nil }

// VerifyReady checks that the external service answers /health. Unlike
// Manager.VerifyReady, it does NOT cross-check the loaded model against
// the configured path — the external service was launched by the
// operator with whatever model they chose, and jukebox accepts that
// declaration at face value.
func (m *ExternalManager) VerifyReady(ctx context.Context, _ string) error {
	return WaitForHealth(ctx, m.baseURL)
}

// Sleep delegates to vllmcli.Sleep against the external service.
func (m *ExternalManager) Sleep(ctx context.Context, level int) error {
	return vllmcli.Sleep(ctx, m.baseURL, level)
}

// Wake delegates to vllmcli.Wake against the external service.
func (m *ExternalManager) Wake(ctx context.Context, timeout time.Duration) error {
	return vllmcli.Wake(ctx, m.baseURL, timeout)
}

// IsSleeping delegates to vllmcli.IsSleeping against the external service.
func (m *ExternalManager) IsSleeping(ctx context.Context) (bool, error) {
	return vllmcli.IsSleeping(ctx, m.baseURL)
}
