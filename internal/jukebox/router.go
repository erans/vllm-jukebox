package jukebox

import "context"

type Route struct {
	BaseURL       string
	UpstreamModel string
	Done          func()
}

// Router is the REQUIRED interface every routing implementation
// provides. Both LegacyRouter (swap mode) and Scheduler (scheduler mode)
// satisfy it.
//
// Routers that ALSO support async cold-load (the wake-from-Stopped
// 503 + Retry-After contract introduced by Phase 4) should additionally
// implement ColdLoadAware. Handlers type-assert at the callsite;
// routers that don't implement ColdLoadAware skip the async-503 path
// entirely and fall through to AcquireRoute as before. This keeps the
// Router interface source-compatible for any external implementer that
// pre-dates the cold-load feature.
type Router interface {
	Status() Status
	AcquireRoute(ctx context.Context, requestedModel, requestID string) (Route, error)
}

// ColdLoadAware is the OPTIONAL extension implemented by routers that
// can asynchronously cold-load a Stopped model (admissionStopped state:
// the container is `docker compose stop`'d because a previous admission
// cycle picked it as an evict_action: stop victim).
//
// HTTP handlers type-assert the Router to ColdLoadAware. When the
// assertion succeeds AND IsModelColdLoading reports true for the
// requested model, the handler calls KickColdLoad and returns 503
// (or 529 for Anthropic) + Retry-After immediately — avoiding a
// ~5-minute synchronous block on the inbound request that would blow
// past Bifrost / downstream HTTP-client timeouts. When the assertion
// fails (the router doesn't implement ColdLoadAware), the handler
// skips the async path and falls through to AcquireRoute, which still
// works correctly — it just may take seconds-to-minutes synchronously
// for big models.
//
// The Scheduler (scheduler mode) implements this. The LegacyRouter
// (swap mode) does NOT — swap mode has no admissionStopped concept
// (the single-instance coordinator does not stop containers; it only
// swaps processes), so the Stopped state is unreachable there and the
// async-503 path is structurally inapplicable.
type ColdLoadAware interface {
	// IsModelColdLoading reports whether `name` is currently in the
	// admissionStopped state (its container is `docker compose stop`'d
	// because admission picked it as an evict_action: stop victim) and
	// therefore a request for it must NOT take the synchronous wake path.
	// Returns false for unknown models, for models that aren't tracked
	// by admission, and when admission is unset (legacy mode).
	IsModelColdLoading(name string) bool

	// KickColdLoad spawns a background goroutine that cold-loads
	// `name` (`docker start` + health-wait + admission bookkeeping
	// flip back to admissionSleeping). The goroutine uses
	// context.Background() with a generous timeout — it is intentionally
	// NOT tied to the inbound request context (which gets cancelled when
	// the client gives up on its 503 + Retry-After response). Concurrent
	// KickColdLoad calls for the same model are safe: only one cold-load
	// runs at a time thanks to the global cold-load lock + TOCTOU
	// re-check inside coldLoadStoppedMember. Returns true if a cold-load
	// goroutine was kicked off, false if the model isn't actually
	// admissionStopped (caller bug — should have checked IsModelColdLoading
	// first) or if the model/instance can't be located.
	KickColdLoad(name string) bool
}
