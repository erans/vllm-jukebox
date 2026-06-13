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

// ResponseRecorder is the OPTIONAL extension implemented by routers that
// observe upstream HTTP response statuses for the per-model circuit
// breaker. Scheduler-mode (which owns docker containers) implements it;
// LegacyRouter (swap-mode) does not — swap mode uses an external
// process supervisor, so the docker-restart breaker doesn't apply.
type ResponseRecorder interface {
	RecordResponse(model string, status int)
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
// past downstream HTTP-client timeouts. When the assertion
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

	// IsInColdLoadEviction reports whether `name` is currently being
	// slept or stopped as part of an in-flight peer cold-load (i.e. it
	// is a swap-group peer of a model whose cold-load is mid-flight).
	//
	// This is the COMPANION gate to IsModelColdLoading. The latter
	// returns true ONLY when admission state == admissionStopped (i.e.
	// the model whose container is `docker compose stop`'d). The former
	// closes the gap during the brief window where:
	//   - a peer's cold-load has acquired coldLoadMu,
	//   - admission has begun evicting `name` to make room for the peer,
	//   - but `name`'s admission state is still admissionAwake/Sleeping
	//     (NotifySleep hasn't fired yet because the sleep is still in
	//     progress or has just completed).
	//
	// During that window, a request for `name` would otherwise fall
	// through the IsModelColdLoading fast-fail gate and block on
	// `performWake` which itself blocks on coldLoadMu — observed live
	// 2026-06-10 as a 600s hang on a priority=critical model.
	//
	// Returns false for unknown models and when no cold-load is active.
	IsInColdLoadEviction(name string) bool
}
