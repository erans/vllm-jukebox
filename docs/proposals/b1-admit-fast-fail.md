# B-1-rev: priority-aware fast-fail during swap-group eviction

**Status**: trace complete, implementation deferred to in-PR iteration. Read this first; comment for direction; the Go change lands as a commit on this branch.

## Symptom

A request to a `priority: critical` model (`Qwen3.6-27B`/main) arrives during a peer cold-load (`Qwen3.6-27B-512K`/longctx) and blocks for **600s** with no response, instead of fast-failing 503 in <1s like other in-flight cold-load requests do.

Measured 2026-06-10 Phase A: curl `model=Qwen3.6-27B` while longctx was cold-loading → `http=000 time=600s`. Logs show other requests during the same window get 503 in 43-47ms — only this one hung.

## Root cause (traced)

State machine during a longctx cold-load:

1. `coldLoadStoppedMember(longctx)` acquires `coldLoadMu` (admission.go:133, single global `sync.Mutex`).
2. `evictPeersForColdLoadLocked` (sleep.go:612) calls `sleepInstance(main, "cold-load-host")` (sleep.go:649).
3. `sleepInstance` walks main: `StateReady` → `StateStopping` + `draining=true` → drain → `sc.Sleep()` → poll `/is_sleeping` → `StateSleeping` (sleep.go:1185-1323).
4. **Admission-side, NotifySleep does NOT fire** during the eviction transition for `reason="cold-load-host"`. So `admission` state of main remains `admissionAwake` until after the sleep settles. (Even then it goes to `admissionSleeping`, never `admissionStopped`.)

The fast-fail gate at **handlers_proxy.go:133** checks `cl.IsModelColdLoading(modelName)`, which delegates to `Scheduler.IsModelColdLoading` (sleep.go:312) → `a.IsStopped(name)` (admission.go:910) — **true only when admission state == `admissionStopped`**.

Since main is `admissionAwake` (or `admissionSleeping`) during the eviction, the gate misses it. Request falls through to `AcquireRoute` (scheduler.go:308) → `tryRouteFromSleep` (sleep.go:1371). If main's `state == StateSleeping` at that moment, `performWake` is invoked, which at **sleep.go:1482** runs `s.admission.WithColdLoadLock(func() { ... })`. `coldLoadMu` is held by the in-flight longctx cold-load. The request goroutine blocks on `Lock()` for the entire longctx cold-load wall-clock (up to ~10 min outer cap). That's the 600s + http=000 observation.

## Proposed fix

Introduce a scheduler-side **`coldLoadInProgress map[string]struct{}`** populated synchronously at `WithColdLoadLock` entry and cleared on exit. Membership = the cold-load target + every peer the cold-load will evict (slept + stopped). Expose via a new method on the `ColdLoadAware` interface; check it at the fast-fail gate.

### File-level anchors

| file | line | change |
|---|---|---|
| `internal/jukebox/router.go` | ~48 (ColdLoadAware interface) | add `IsInColdLoadEviction(name string) bool` |
| `internal/jukebox/scheduler.go` | struct decl | add `coldLoadInProgress map[string]struct{}` + a small mutex |
| `internal/jukebox/sleep.go` | ~565 (coldLoadStoppedMember, just inside `WithColdLoadLock` cb) | populate set with target + `pinnedPeersInSwapGroup` + `evictStopPeersInSwapGroupOverlapping`; defer clear |
| `internal/jukebox/sleep.go` | new method | `IsInColdLoadEviction(name) bool` |
| `internal/httpserver/handlers_proxy.go` | 133 | extend gate: `if cl.IsModelColdLoading(modelName) \|\| cl.IsInColdLoadEviction(modelName) { ... 503 warming_up ... }` |

### Why scheduler-side and not admission-side

The eviction sequence is owned by `Scheduler.coldLoadStoppedMember`. It already knows the full peer set before any sleep/stop calls fire. Putting the marker there closes the window the moment the cold-load begins, not after admission state catches up.

The alternative — listening for `StateStopping` on individual instances — races on the brief `Ready → Stopping` transition window and would miss the very requests this fix is meant to catch.

### Why not just preempt longctx

Preemption-of-best-effort-by-critical is the **better** long-term fix but architecturally bigger (needs a clean cold-load abort path, not a wait-for-completion). Ship fast-fail as the **floor** in this PR; track preemption as a follow-up.

## Test plan (to be added with implementation)

- Unit test for `coldLoadInProgress` set: populate via `WithColdLoadLock`, verify `IsInColdLoadEviction` returns true for target + peers, false after callback returns.
- Integration test simulating a request arriving mid-eviction → 503 in <100ms.
- Live re-run of Phase A repro: trigger longctx cold-load, fire `model=Qwen3.6-27B`, confirm 503 in <1s (was 600s).

## Risk

- The lock around the new map must NOT be `coldLoadMu` — that's the lock we're avoiding blocking on. Use a separate `sync.RWMutex` (read-mostly access from the handler hot path).
- Clearing on early-return from the callback: use `defer` immediately after population, before the cold-load body, so any panic in the body still clears the set.

## References

- Trace by `general-purpose` subagent 2026-06-10.
- Phase A live evidence: 600s `http=000` on `model=Qwen3.6-27B` during longctx cold-load.
- Existing 43-47ms 503 path: `instance_sleeping → cold_load_host → cold_load_acquired_lock` log sequence shows fast-fail works AFTER admission flips; this PR closes the pre-flip window.

Generated with Claude Code
