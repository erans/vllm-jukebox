package jukebox

import (
	"context"
	"fmt"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/inflight"
)

type LegacyRouter struct {
	cfg   *config.Config
	coord *Coordinator
	tr    *inflight.Tracker
}

func NewLegacyRouter(cfg *config.Config, coord *Coordinator, tr *inflight.Tracker) *LegacyRouter {
	return &LegacyRouter{cfg: cfg, coord: coord, tr: tr}
}

func (r *LegacyRouter) Status() Status {
	if r == nil || r.coord == nil {
		return Status{State: StateIdle}
	}
	return r.coord.Status()
}

func (r *LegacyRouter) AcquireRoute(ctx context.Context, requestedModel, requestID string) (Route, error) {
	if r == nil || r.cfg == nil || r.coord == nil {
		return Route{}, fmt.Errorf("router not configured")
	}
	if err := r.coord.EnsureModel(ctx, requestedModel, requestID); err != nil {
		return Route{}, err
	}

	var done func()
	if r.tr != nil {
		done = r.tr.Track(ctx)
	}

	_, modelCfg, err := liveResolveModel(r.cfg, requestedModel)
	if err != nil {
		return Route{}, err
	}

	return Route{
		BaseURL:       fmt.Sprintf("http://127.0.0.1:%d", r.cfg.VLLM.Port),
		UpstreamModel: modelCfg.Path,
		Done:          done,
	}, nil
}

// LegacyRouter intentionally does NOT implement the optional
// ColdLoadAware interface. Swap mode has no admissionStopped concept
// (the single-instance coordinator does not stop containers; it only
// swaps processes), so the async-503 + KickColdLoad path is structurally
// inapplicable. HTTP handlers type-assert at the callsite — when the
// assertion fails the handler skips the async-503 branch and falls
// through to AcquireRoute, which is exactly the behavior swap mode
// needs (and exactly what the no-op stubs used to fake).
