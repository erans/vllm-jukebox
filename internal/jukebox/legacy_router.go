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

	_, modelCfg, err := r.cfg.ResolveModel(requestedModel)
	if err != nil {
		return Route{}, err
	}

	return Route{
		BaseURL:       fmt.Sprintf("http://127.0.0.1:%d", r.cfg.VLLM.Port),
		UpstreamModel: modelCfg.Path,
		Done:          done,
	}, nil
}
