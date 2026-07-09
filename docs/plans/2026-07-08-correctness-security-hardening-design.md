# Correctness & security hardening (#1–#5) — design

**Date:** 2026-07-08
**Status:** Design accepted; ready for implementation plan.
**Scope:** Fix the five critical/high issues from the adversarial code review. Hardening items #6–#15 (upstream timeouts, rate limiting, response buffering caps, self-proxy config check, path traversal, header reflection, waiter-channel cleanup, power-limit upper bounds) are explicitly out of scope.

## Motivation

An adversarial review (two competing reviewers) converged on five issues that break the core streaming workload, leak resources, or expose the server by default. This change fixes all five as one cohesive "make it safe and correct" effort. Each fix is scoped to the minimal blast radius.

The issues, in priority order:

1. **SSE in-flight tracking** (Critical): streaming responses release the in-flight counter on handler return, not on stream completion, so swaps/evictions kill vLLM mid-stream.
2. **No auth, default `0.0.0.0`** (High): the listener binds all interfaces by default with no authentication on any endpoint.
3. **`tryRouteReady` TOCTOU** (High): the locked re-check is decorative; a route is returned even when the instance is concurrently being evicted/stopped.
4. **Power-limit lifecycle** (High): swap reverts the *new* model's GPUs (not the old model's); failed Start/Verify paths never revert applied limits.
5. **Scheduler crash-replace leak** (High): replacing a crashed instance only marks the old one `draining` and drops it — no stop, no port release, no metric clear.

## Decisions (confirmed during brainstorm)

| # | Decision |
|---|----------|
| #1 | Lifecycle ownership moves into the proxy via a new `ForwardOptions.OnDone` field. Handler stops doing `defer route.Done()`. Proxy calls `OnDone` on the SSE writer completion path and after the buffered `c.Send` path. |
| #2 | Default bind → `127.0.0.1`. New optional `server.api_key` field. If set, a new `authAPIKey` middleware requires `Authorization: Bearer <key>` on inference + `/status` + `/metrics`. Unset = no auth (backward compatible). |
| #3 | `tryRouteReady` gates its return on the locked re-check (including `CurrentPID()`); returns `false` if the re-check fails. Caller fallthrough is already correct. |
| #4 | (a) `doSwap` resolves the *old* model's config and reverts *its* GPUs. (b) Add power reverts to failed-`Start` and failed-`VerifyReady` branches in both coordinator and scheduler. |
| #5 | Inline cleanup of the replaced `old` instance outside the lock (stop, revert power, clear metrics, release port) — no map `delete` since the new instance owns the key. `drainAndStopInstance` is left untouched. |

## Components touched

| Component | Change |
|-----------|--------|
| `internal/proxy/proxy.go` | Add `OnDone func()` to `ForwardOptions`; call it on both completion paths; guarantee single invocation. |
| `internal/httpserver/handlers_proxy.go` | Drop `defer route.Done()`; pass `OnDone: route.Done` into `ForwardOptions`. |
| `internal/httpserver/handlers_anthropic.go` | Same lifecycle change if it also forwards via `ForwardFiber`. |
| `internal/jukebox/scheduler.go` | `tryRouteReady` gate; crash-replace inline cleanup; power revert on failed Start/Verify. |
| `internal/jukebox/coordinator.go` | `doSwap` reverts old model's GPUs; power revert on failed Start/Verify. |
| `internal/config/config.go` | Default host `127.0.0.1`; add `APIKey` field + validation. |
| `internal/httpserver/middleware_auth.go` (new) | `authAPIKey` middleware. |
| `internal/httpserver/server.go` | Conditionally mount `authAPIKey` on protected routes. |

## Design details

### #1 — SSE in-flight lifecycle

Root cause: Fiber/fasthttp runs the `SetBodyStreamWriter` callback *after* the handler returns, so any `defer route.Done()` in the handler fires while the stream is still open. The handler cannot know in advance whether a request will be SSE (the upstream's response `Content-Type` decides), so it cannot special-case it. The proxy, which sees the response, must own the lifecycle.

```
handler: route := AcquireRoute(...)            // Track++ (count=N)
handler: ForwardFiber(c, {OnDone: route.Done})
  proxy: upstream responds → Content-Type: text/event-stream
  proxy: SetBodyStreamWriter(func(w) {
             defer resp.Body.Close()
             ... stream ...
             defer OnDone()                     // count=N-1, only after stream ends
             _ = w.Flush()
         }); return nil
handler returns (count still N while streaming)   ← no defer Done here
... later, stream completes → OnDone fires (count=N-1)
```

Buffered path: after `c.Send(data)` returns, call `OnDone()` before `ForwardFiber` returns.

**Single-invocation guarantee:** `OnDone` must fire exactly once even if the stream writer panics. Implementation: nil-check-and-nil-out (`if opts.OnDone != nil { opts.OnDone(); opts.OnDone = nil }`) inside both completion paths, which also makes the SSE `defer` safe against a panic recovery in the buffered path and vice versa. If `route.Done == nil` (some test paths construct routes without it), the proxy skips.

Coordinator `WaitForDrain` and scheduler `drainAndStopInstance`/`inflight.WaitForDrain` now observe the real count and will not kill vLLM while a stream is active.

### #2 — Default bind + optional API key

- `config.go` default: `c.Server.Host == ""` → `"127.0.0.1"` (was `"0.0.0.0"`). Operators who want network exposure set `host: "0.0.0.0"` explicitly.
- New field: `Server.APIKey string` (`yaml:"api_key"`).
- Validation: reject empty/whitespace-only `api_key` (operator must omit the field or set a real key). A present-but-empty key is a startup error.
- New `internal/httpserver/middleware_auth.go`: `authAPIKey(expected string) fiber.Handler` returns 401 `application/json` (OpenAI-style error) when `Authorization` is missing, malformed, or mismatched. Constant-time compare.
- `server.go`: if `cfg.Server.APIKey != ""`, mount `authAPIKey` on the model-bearing routes (`/v1/*`), `/status`, and `/metrics`. `/health` and `/v1/models` remain open (health checks and model discovery; model names are not treated as secret since the config is operator-controlled). When `api_key` is unset, no middleware is mounted and behavior is identical to today.

### #3 — `tryRouteReady` gate

Current code (scheduler.go ~319): the locked re-check only guards the `lastUsedAt` bump; `return ..., true` is unconditional. Fix: move the return inside the locked re-check.

```go
now := s.now()
s.mu.Lock()
if inst2 := s.instances[resolvedModelName]; inst2 != nil && inst2 == inst &&
    inst.state == StateReady && !inst.draining && inst.mgr != nil && inst.mgr.CurrentPID() != 0 {
    inst.lastUsedAt = now
    s.mu.Unlock()
    return s.routeForInstance(ctx, inst, upstreamModel), true
}
s.mu.Unlock()
return Route{}, false
```

Caller fallthrough (`AcquireRoute`): on `false`, it acquires the scheduling permit, double-checks `tryRouteReady` again under the permit, then starts a new instance. This is exactly the right behavior when the instance is concurrently being evicted — no caller changes needed.

### #4 — Power-limit lifecycle

**(a) Swap reverts old model's GPUs.** In `doSwap`, before stopping the old model, resolve the *old* model's config from `c.currentModel` (under `c.mu`) and call `RevertModelLimits(ctx, oldCfg.GPUs)`. The current code mistakenly uses `modelCfg.GPUs` (the *new* model), leaving the old model's GPUs pinned at elevated power indefinitely.

**(b) Revert on failed Start/Verify (both modes).**
- Coordinator `doSwap`: after `ApplyModelLimits` and before the eventual `Start`/`VerifyReady`, if either fails, call `RevertModelLimits(ctx, modelCfg.GPUs)` before returning the error.
- Scheduler `AcquireRoute` start path: after `ApplyModelLimits`, if `Start` fails (release port, revert power, return) or `VerifyReady` fails (stop, release port, revert power, return).

Failure-path reverts are best-effort (`_ =`) matching existing style; the original Start/Verify error is the returned value, and a revert error is ignored (optionally logged if a logger is in scope — keep simple, ignore for v1).

Invariant enforced: any time jukebox stops owning a model's GPUs, that model's power limits are reverted to defaults.

### #5 — Scheduler crash-replace cleanup

Current code (scheduler.go ~285): only marks `old.draining = true` and drops it from the map — no `Stop()`, no `ports.Release`, no metric clear. Each crash→restart leaks one port; the process (if still alive) is never killed.

Fix: inside the lock, capture `old` and install the new instance. Outside the lock, if `old != nil`, run inline cleanup:

```go
s.mu.Lock()
old := s.instances[resolvedName]
s.instances[resolvedName] = inst
s.mu.Unlock()

if old != nil {
    old.draining = true
    old.state = StateStopping
    if old.mgr != nil {
        stopCtx, cancel := context.WithTimeout(context.Background(), stopTimeout)
        _ = old.mgr.Stop(stopCtx)
        cancel()
    }
    if s.powerMgr != nil {
        _ = s.powerMgr.RevertModelLimits(context.Background(), old.gpus)
    }
    portLabel := strconv.Itoa(old.port)
    metrics.RunningInstances.WithLabelValues(old.model, portLabel).Set(0)
    metrics.InstanceInFlightRequests.WithLabelValues(old.model, portLabel).Set(0)
    if old.port != 0 {
        s.ports.Release(old.port)
    }
}
```

Note: this deliberately does **not** `delete(s.instances, ...)` — the new instance already occupies that key, and `drainAndStopInstance`'s own `delete` would clobber the new instance. `drainAndStopInstance` (used by eviction/shutdown) is left unchanged. Cleanup runs synchronously before returning the new route so the port is freed immediately.

## Error handling & edge cases

- **#1**: `OnDone` fires exactly once via nil-check-and-nil-out; `route.Done == nil` test paths are skipped. A panic in the stream writer must not skip `OnDone` — the `defer OnDone()` placement (after streaming work, before final flush) ensures it runs even on a panic recovered by Fiber's `recover()` middleware.
- **#4**: revert is best-effort and never masks the original Start/Verify error.
- **#5**: synchronous cleanup before returning the route; bounded by ShutdownTimeout. Crash-replace is rare, so blocking is acceptable and the port is freed immediately for reuse.
- **#2**: `api_key` unset → no middleware → identical to today. Empty-string `api_key` → startup validation error.

## Testing

- **#1 (highest value)**: new `internal/proxy/proxy_test.go`. Fake upstream returns `text/event-stream` and holds the stream open; assert a counter decremented by `OnDone` is still >0 while the stream is open and hits 0 only after the writer completes. Buffered-path test asserting `OnDone` fires after `c.Send`.
- **#3**: scheduler test racing `tryRouteReady` against `drainAndStopInstance`; assert no route returned for a concurrently-stopped instance (falls through to scheduling).
- **#4**: coordinator test asserting old-model GPUs reverted on swap (using a fake power manager); both-modes tests asserting power reverted on failed Start and failed Verify.
- **#5**: scheduler test simulating a crashed instance (`CurrentPID()==0`), triggering replace, asserting old port released and old `Stop` called.
- **#2**: config test for new default host + `api_key` validation; middleware test asserting 401 without/wrong key, 200 with correct key, pass-through when unset.

## Out of scope

#6–#15 from the review: upstream timeout, concurrency cap / rate limiting, `server.port == vllm.port` config check, unbounded response buffering cap, `Content-Length` header asymmetry, log-file path traversal, `X-Request-ID` reflection, inflight waiter-channel cleanup, power-limit upper bounds. These are deferred to a separate hardening pass.
