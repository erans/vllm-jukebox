# B-6-rev: rerank /v1/rerank hangs 180s on auto-wake under main heavy load

**Status**: trace complete, implementation deferred to in-PR iteration. Read this first; comment for direction; the Go change lands as a commit on this branch.

## Symptom

When `vllm-reranker` is L1-sleeping AND main is busy serving 3+ concurrent heavy `/v1/chat/completions`, a `/v1/rerank` to jukebox **hangs 180s and the reranker never auto-wakes**. Manual `/wake_up` on reranker (bypassing jukebox) returns 200 in **226 ms** and a subsequent `/v1/rerank` to the woken reranker returns 200 in **57 ms**. So vllm-side wake works fine.

Measured 2026-06-10 Phase B-3:
- B1.5 (main idle, reranker sleeping): rerank via jukebox succeeded in ~30s including auto-wake.
- B3 (main busy with 3 heavy reqs, reranker sleeping): rerank via jukebox hung 180s with EMPTY response.

## Initial hypothesis ruled out

`coldLoadMu` (admission.go:133) only guards docker-compose-up + health-wait cold-loads. The reranker wake is a vllm `/wake_up` API call, NOT a cold-load. With `s.admission == nil` (the running config has no `expected_vram_mb_per_gpu` on reranker, so `AdmissionEnabled()` is false → `main.go:290` skips `SetAdmission`), `performWake` takes the `s.admission == nil` branch at sleep.go:1477 and skips `WithColdLoadLock` entirely. So the cold-load lock is NOT the gate.

## Root cause (traced)

`/v1/rerank` routes identically to `/v1/chat/completions`: same handler (`switchingProxyHandler`, handlers_proxy.go:19), same `AcquireRoute` path. The classifier (handlers_proxy.go:56-94) is gated on `isChatPath` so rerank skips it. No rerank-specific gate exists.

The wake path is **only reached** if `inst.state == StateSleeping` (sleep.go:1374). For `lifecycle: external` models, state transitions to `StateSleeping` only via:
1. Boot-time `/is_sleeping` probe in `RegisterExternalInstances` (sleep.go:1887, runs once at start).
2. `Scheduler.checkIdle` calling `sleepInstance` when `inst.inflight.Count()==0 && idle > idle_timeout` (sleep.go:2042-2083, ticker every 30s).

**There is no periodic `/is_sleeping` reconcile loop.** If the reranker is L1-sleeping but jukebox's bookkeeping still says `StateReady` (because the reranker slept via a path `checkIdle` missed — out-of-band `/sleep` call, container restart that lost in-memory state, or a wake-then-sleep cycle that completed between checkIdle ticks while main's heavy work delayed something), `tryRouteReady` (scheduler.go:472) returns success and `proxy.ForwardFiber` POSTs `/v1/rerank` straight to the sleeping vllm. vLLM hangs ~60s on rerank to a slept model; client retries → 180s observed.

B1.5 succeeded because the reranker happened to be in sync (state genuinely Ready, slept fresh by `checkIdle` between probe and request). B3 failed because main's heavy in-flight work delayed `checkIdle`'s next tick AND the reranker had transitioned via a path that didn't update state.

## Proposed fix — drift-resistant probe in `tryRouteReady`

For sleep-capable external instances whose `lastUsedAt` is stale (>30s), do a fast `/is_sleeping` probe (500ms timeout) before committing to the Ready route. If sleeping, flip state to `StateSleeping` and fall through to `tryRouteFromSleep`.

### Diff sketch (scheduler.go, inside `tryRouteReady` after the lock re-check ~line 486)

```go
// Drift-recovery: external sleep-capable instances can be slept out-of-band
// (manual /sleep, container restart, missed checkIdle tick under load).
// If lastUsedAt is stale and the instance supports sleep, probe
// /is_sleeping before committing to a Ready route — otherwise we proxy to
// a sleeping vLLM and the request hangs ~60s on /v1/rerank or
// /v1/embeddings (no auto-wake on those endpoints in vllm).
if sc, ok := inst.mgr.(SleepCapable); ok && s.now().Sub(inst.lastUsedAt) > 30*time.Second {
    probeCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
    sleeping, perr := sc.IsSleeping(probeCtx)
    cancel()
    if perr == nil && sleeping {
        s.mu.Lock()
        inst.state = StateSleeping
        s.mu.Unlock()
        return Route{}, false  // caller falls through to tryRouteFromSleep
    }
}
```

### File-level anchors

| file | line | change |
|---|---|---|
| `internal/jukebox/scheduler.go` | ~486 (tryRouteReady, after lock re-check) | add probe block above |

Existing `SleepCapable` interface should have `IsSleeping(ctx) (bool, error)` (sleep.go has `IsSleeping` probes via vllmcli — verify the exact interface method name in-PR).

## Why 30s

- Shorter than `checkIdle`'s 30s tick — so a tick that fires between probe and request still serves cleanly via the Ready path with no probe overhead.
- Long enough that hot paths (back-to-back requests) skip the probe entirely (zero added latency on the request-bursts case).
- 500ms timeout caps worst-case added latency when reranker is genuinely unreachable.

## Test plan (to be added with implementation)

- Unit test: set `lastUsedAt` >30s in past, mock IsSleeping → true, verify state flips + returns `(_, false)`.
- Live re-run of Phase B-3 repro: re-sleep reranker, fire 3 heavy main + concurrent rerank → rerank wakes in <30s + completes in <60s total (was 180s timeout, EMPTY).

## Risk

- The probe adds a 500ms-max latency to requests on stale instances. Small; only affects sleep-capable models with idle gaps > 30s.
- If `/is_sleeping` itself starts hanging (unrelated vllm bug), the probe ctx caps at 500ms — we ignore the error and proceed with Ready path (no worse than today).
- State write needs the same lock that `tryRouteReady` already takes. Confirm no double-acquire deadlock.

## References

- Trace by `general-purpose` subagent 2026-06-10.
- Phase B-3 live evidence: 180s EMPTY rerank response while main concurrent-busy; manual `/wake_up` → 226ms; manual `/v1/rerank` → 57ms.
- Metrics: `jukebox_admission_wait_seconds{model="mxbai-rerank-base-v2"}` shows 3 events at ≤5ms — admission was instant; the hang is downstream in proxy.

Generated with Claude Code
