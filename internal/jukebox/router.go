package jukebox

import "context"

type Route struct {
	BaseURL       string
	UpstreamModel string
	Done          func()
}

type Router interface {
	Status() Status
	AcquireRoute(ctx context.Context, requestedModel, requestID string) (Route, error)
}

// PowerController is the subset of gpu.PowerManager used by the coordinator
// and scheduler. Defined here so tests can inject a fake without nvidia-smi.
type PowerController interface {
	ApplyModelLimits(ctx context.Context, gpus []int, powerLimit *int, powerLimits map[int]int) error
	RevertModelLimits(ctx context.Context, gpus []int) error
}
