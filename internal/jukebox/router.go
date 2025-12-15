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

