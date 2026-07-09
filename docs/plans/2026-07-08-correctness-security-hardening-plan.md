# Correctness & Security Hardening (#1–#5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five critical/high issues from the adversarial review — SSE in-flight tracking, default-bind + optional API-key auth, `tryRouteReady` TOCTOU, power-limit lifecycle, and scheduler crash-replace port/process leak — as one cohesive correctness+security pass.

**Architecture:** Lifecycle ownership for streaming responses moves from the HTTP handler into the proxy via a new `ForwardOptions.OnDone` callback (#1). A new optional `server.api_key` enables bearer-auth middleware on protected routes; the default bind changes to `127.0.0.1` (#2). The scheduler's `tryRouteReady` gates its return on the locked re-check (#3). Power-limit reverts use the correct (old) model's GPUs on swap and are added to all failure paths (#4). The crash-replace path inline-cleans the old instance (#5). A minimal `PowerController` interface is introduced in the jukebox package so power behavior is testable without `nvidia-smi`.

**Tech Stack:** Go 1.24, gofiber/fiber/v2, gopkg.in/yaml.v3, standard `log/slog`.

**Spec:** `docs/plans/2026-07-08-correctness-security-hardening-design.md`

## Global Constraints

- Tests are colocated with source (`*_test.go`), package-named `*_test` where the existing files use that convention (e.g. `package jukebox_test`).
- Errors wrapped with `fmt.Errorf("context: %w", err)` where appropriate; failure-path power reverts are best-effort (`_ =`) matching existing style.
- Run `go vet ./...` and the relevant tests after each task. Run `make check` (fmt-check + vet + tests) before the final task.
- Commit after each task with the exact message shown.
- Hardening items #6–#15 are OUT OF SCOPE — do not implement them.

## File Structure

| File | Responsibility | Create/Modify |
|------|----------------|---------------|
| `internal/proxy/proxy.go` | Add `OnDone` to `ForwardOptions`; call it on both completion paths; guarantee single invocation. | Modify |
| `internal/proxy/proxy_test.go` | E2E proxy lifecycle test (SSE + buffered) proving `OnDone` timing. | Create |
| `internal/httpserver/handlers_proxy.go` | Drop `defer route.Done()`; pass `OnDone: route.Done`. | Modify |
| `internal/httpserver/handlers_anthropic.go` | Same lifecycle change. | Modify |
| `internal/jukebox/router.go` | Add `PowerController` interface (Apply/Revert). | Modify |
| `internal/jukebox/coordinator.go` | Use `PowerController`; revert old model's GPUs on swap; revert on failed Start/Verify. | Modify |
| `internal/jukebox/scheduler.go` | Use `PowerController`; `tryRouteReady` gate; crash-replace cleanup; revert on failed Start/Verify. | Modify |
| `internal/jukebox/coordinator_test.go` | Add fake `PowerController`; tests for swap-revert-old-model + failure-path reverts. | Modify |
| `internal/jukebox/scheduler_test.go` | Add `startErr`/`verifyErr` to fake instance; tests for tryRouteReady race, crash-replace, failure-path reverts. | Modify |
| `internal/config/config.go` | Default host `127.0.0.1`; add `APIKey` field + validation. | Modify |
| `internal/config/config_test.go` | Tests for new default host + api_key validation. | Modify |
| `internal/httpserver/middleware_auth.go` | `authAPIKey` middleware. | Create |
| `internal/httpserver/middleware_auth_test.go` | Auth middleware tests. | Create |
| `internal/httpserver/server.go` | Conditionally mount `authAPIKey` on protected routes. | Modify |
| `internal/httpserver/server_test.go` | Tests asserting middleware mounted only when key set. | Modify |

---

### Task 1: SSE in-flight lifecycle — proxy owns `OnDone`

**Files:**
- Modify: `internal/proxy/proxy.go` (struct `ForwardOptions`; func `ForwardFiber`)
- Test: `internal/proxy/proxy_test.go` (create)

**Interfaces:**
- Produces: new field `ForwardOptions.OnDone func()`. `ForwardFiber` calls it exactly once — on the SSE writer completion path and after the buffered `c.Send` path.

- [ ] **Step 1: Write the failing test — SSE path holds `OnDone` until stream completes**

Create `internal/proxy/proxy_test.go`:

```go
package proxy_test

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"vllm-jukebox/internal/proxy"
)

// Test that OnDone is NOT called while an SSE stream is still open, and IS
// called once the stream writer completes. This is the regression test for the
// critical bug where the handler's defer route.Done() fired on handler return.
func TestForwardFiber_SSE_OnDoneHeldUntilStreamCompletes(t *testing.T) {
	streamStarted := make(chan struct{})
	streamRelease := make(chan struct{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: chunk1\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		close(streamStarted)
		<-streamRelease // hold the stream open
		fmt.Fprintf(w, "data: chunk2\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	var doneCount int32
	onDone := func() { atomic.AddInt32(&doneCount, 1) }

	app := fiber.New()
	app.Use(recover.New())
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL: upstream.URL,
			OnDone:  onDone,
		})
	})

	// Drive the request in a goroutine; fasthttp's stream writer runs after the
	// handler returns, but we need the request to progress far enough to start
	// the stream. Use a real http.Client against the fiber app.
	listenAddr := "127.0.0.1:0"
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go app.Listener(ln)
	defer app.Shutdown()
	base := "http://" + ln.Addr().String()

	client := &http.Client{}
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("Accept", "text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()

	select {
	case <-streamStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("stream never started")
	}

	// While the stream is open, OnDone must NOT have fired.
	if got := atomic.LoadInt32(&doneCount); got != 0 {
		t.Fatalf("OnDone fired while stream open: %d", got)
	}

	close(streamRelease)
	io.Copy(io.Discard, resp.Body) // drain to let the writer finish

	// After the stream completes, OnDone must fire exactly once.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if atomic.LoadInt32(&doneCount) == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := atomic.LoadInt32(&doneCount); got != 1 {
		t.Fatalf("OnDone not fired exactly once after stream: %d", got)
	}
}

// Buffered path: OnDone fires after c.Send, before ForwardFiber returns.
func TestForwardFiber_Buffered_OnDoneAfterSend(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	var doneCount int32
	onDone := func() { atomic.AddInt32(&doneCount, 1) }

	app := fiber.New()
	app.Use(recover.New())
	app.Post("/v1/chat/completions", func(c *fiber.Ctx) error {
		if err := proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL: upstream.URL,
			OnDone:  onDone,
		}); err != nil {
			return err
		}
		// OnDone must have fired by the time ForwardFiber returns for buffered.
		if got := atomic.LoadInt32(&doneCount); got != 1 {
			t.Fatalf("OnDone not fired after buffered send: %d", got)
		}
		return nil
	})

	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	go app.Listener(ln)
	defer app.Shutdown()
	base := "http://" + ln.Addr().String()

	resp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
}
```

Add the missing imports (`io`, `net`, `time`) to the test file's import block.

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/proxy/ -run TestForwardFiber_SSE_OnDoneHeldUntilStreamCompletes -v`
Expected: FAIL — `OnDone` is currently never called by the proxy at all (the handler calls it), so `doneCount` stays 0 after the stream and the test times out / fails the final assertion.

- [ ] **Step 3: Implement `OnDone` in the proxy**

In `internal/proxy/proxy.go`, add the field to `ForwardOptions`:

```go
type ForwardOptions struct {
	BaseURL           string
	RewriteModelName  bool
	RequestedModel    string
	UpstreamModel     string
	RequestID         string
	AdditionalHeaders map[string]string
	Timeout           time.Duration
	OnDone            func()
}
```

Add a helper near the top of the file (after the imports/`ForwardOptions`):

```go
// callOnce invokes f and nils it out so it can only fire once, even across the
// SSE writer closure and the buffered path. Safe if f is nil.
func callOnce(f *func()) {
	if f != nil && *f != nil {
		(*f)()
		*f = nil
	}
}
```

In `ForwardFiber`, for the SSE branch replace the `SetBodyStreamWriter(func(w *bufio.Writer) { ... })` block so `OnDone` fires in a `defer` inside the stream writer after the work completes:

```go
	if strings.Contains(strings.ToLower(contentType), "text/event-stream") {
		onDone := &opts.OnDone
		c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
			defer resp.Body.Close()
			defer callOnce(onDone)
			if opts.RewriteModelName && opts.RequestedModel != "" {
				_ = RewriteSSEModel(resp.Body, w, opts.RequestedModel)
			} else {
				_, _ = copyWithFlush(resp.Body, w)
			}
			_ = w.Flush()
		})
		return nil
	}
```

For the buffered path, after the final `return c.Send(data)`, call `OnDone` before returning. Replace:

```go
	return c.Send(data)
}
```

with:

```go
	callOnce(&opts.OnDone)
	return c.Send(data)
}
```

(Note: `callOnce` is called before `c.Send` so that even if `c.Send` were to panic, the count is released — but `c.Send` does not block on the client, so ordering before vs. after is not load-bearing for the buffered path. Keeping it before matches the "release when we're done producing the response" semantics.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./internal/proxy/ -v`
Expected: PASS for both SSE and buffered tests.

- [ ] **Step 5: Commit**

```bash
git add internal/proxy/proxy.go internal/proxy/proxy_test.go
git commit -m "fix(proxy): release in-flight on response completion, not handler return"
```

---

### Task 2: Wire `OnDone` into the HTTP handlers

**Files:**
- Modify: `internal/httpserver/handlers_proxy.go`
- Modify: `internal/httpserver/handlers_anthropic.go`

**Interfaces:**
- Consumes: `ForwardOptions.OnDone` from Task 1.

- [ ] **Step 1: Update `switchingProxyHandler`**

In `internal/httpserver/handlers_proxy.go`, remove the `defer route.Done()` and pass `OnDone` instead. Replace:

```go
		if route.Done != nil {
			defer route.Done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
		})
```

with:

```go
		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
			OnDone:           route.Done,
		})
```

- [ ] **Step 2: Update `anthropicProxyHandler`**

In `internal/httpserver/handlers_anthropic.go`, the same change. Replace:

```go
		if route.Done != nil {
			defer route.Done()
		}

		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
		})
```

with:

```go
		return proxy.ForwardFiber(c, proxy.ForwardOptions{
			BaseURL:          route.BaseURL,
			RewriteModelName: opts.Config.Behavior.RewriteModelName,
			RequestedModel:   modelName,
			UpstreamModel:    route.UpstreamModel,
			RequestID:        requestID,
			OnDone:           route.Done,
		})
```

- [ ] **Step 3: Run handler tests and vet**

Run: `go vet ./internal/httpserver/ && go test ./internal/httpserver/ -v`
Expected: PASS (existing handler tests should still pass; `route.Done` may now be nil for some test routers but the proxy nil-guards it).

- [ ] **Step 4: Commit**

```bash
git add internal/httpserver/handlers_proxy.go internal/httpserver/handlers_anthropic.go
git commit -m "fix(httpserver): pass route.Done as proxy OnDone instead of handler defer"
```

---

### Task 3: `tryRouteReady` TOCTOU — gate the return on the locked re-check

**Files:**
- Modify: `internal/jukebox/scheduler.go` (func `tryRouteReady`)
- Test: `internal/jukebox/scheduler_test.go`

**Interfaces:**
- Produces: `tryRouteReady` returns `false` when the instance is concurrently stopped/drained (instead of returning a route to a dying instance).

- [ ] **Step 1: Write the failing test — a concurrently-drained instance returns no route**

Append to `internal/jukebox/scheduler_test.go`:

```go
func TestScheduler_TryRouteReady_ReturnsFalseWhenConcurrentlyStopped(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`)

	ctrl := &fakeInstanceController{}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMemoryMB: 100000, FreeMemoryMB: 50000}}}

	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, func(port int, cuda string) jukebox.InstanceManager {
		return &fakeInstance{ctrl: ctrl, port: port}
	}, nil)

	ctx := context.Background()

	// Start model "a" so it becomes Ready.
	route, err := s.AcquireRoute(ctx, "a", "req1")
	if err != nil {
		t.Fatalf("AcquireRoute: %v", err)
	}
	route.Done()

	// Manually mark the instance draining+stopping concurrently, simulating an
	// eviction that won the race between tryRouteReady's RLock read and its Lock.
	s.InstancesForTest()["a"].draining = true
	s.InstancesForTest()["a"].state = jukebox.StateStopping

	// tryRouteReady must return false now (instance is being stopped).
	if _, ok := s.TryRouteReadyForTest(ctx, "a", "/models/a"); ok {
		t.Fatalf("expected tryRouteReady to return false for a concurrently-stopped instance")
	}
}
```

This test needs two test-helpers exposed from the `jukebox` package. Add them to `internal/jukebox/scheduler.go` (these are test-only accessors; the existing codebase already exposes state for tests — check whether similar helpers exist and follow that pattern; if none, add them):

In `internal/jukebox/scheduler.go` add at the end of the file:

```go
// InstancesForTest exposes the instance map for tests. Not safe for concurrent
// use outside tests.
func (s *Scheduler) InstancesForTest() map[string]*schedInstance {
	return s.instances
}

// TryRouteReadyForTest exposes tryRouteReady for tests.
func (s *Scheduler) TryRouteReadyForTest(ctx context.Context, resolvedModelName, upstreamModel string) (Route, bool) {
	return s.tryRouteReady(ctx, resolvedModelName, upstreamModel)
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/jukebox/ -run TestScheduler_TryRouteReady_ReturnsFalseWhenConcurrentlyStopped -v`
Expected: FAIL — `tryRouteReady` currently returns `true` unconditionally, so `ok` is true.

(Note: the test uses `ports.New(8100, 8109)` to construct the pool — the real constructor is `ports.New`, not `ports.NewPool`. Fix the test's constructor call accordingly.)

- [ ] **Step 3: Gate the return in `tryRouteReady`**

In `internal/jukebox/scheduler.go`, replace the body of `tryRouteReady` (the `now := s.now()` block through the final return) with:

```go
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if inst2 := s.instances[resolvedModelName]; inst2 != nil && inst2 == inst &&
		inst.state == StateReady && !inst.draining && inst.mgr != nil && inst.mgr.CurrentPID() != 0 {
		inst.lastUsedAt = now
		return s.routeForInstance(ctx, inst, upstreamModel), true
	}
	return Route{}, false
```

Note this changes the lock from RLock+Lock to a single Lock (the re-check and the decision must be atomic). The RLock-then-Lock pattern was the source of the TOCTOU. `routeForInstance` calls `inst.inflight.Track(ctx)` which takes the tracker's own mutex (not `s.mu`), so no deadlock.

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./internal/jukebox/ -run TestScheduler_TryRouteReady_ReturnsFalseWhenConcurrentlyStopped -v`
Expected: PASS.

- [ ] **Step 5: Run the full scheduler test suite to confirm no regression**

Run: `go test ./internal/jukebox/ -v`
Expected: PASS (all scheduler tests).

- [ ] **Step 6: Commit**

```bash
git add internal/jukebox/scheduler.go internal/jukebox/scheduler_test.go
git commit -m "fix(scheduler): gate tryRouteReady on locked re-check to avoid TOCTOU"
```

---

### Task 4: Introduce `PowerController` interface for testability

**Files:**
- Modify: `internal/jukebox/router.go`
- Modify: `internal/jukebox/coordinator.go`
- Modify: `internal/jukebox/scheduler.go`

**Interfaces:**
- Produces: `jukebox.PowerController` interface with `ApplyModelLimits` and `RevertModelLimits`. `*gpu.PowerManager` satisfies it. Coordinator/scheduler field type changes from `*gpu.PowerManager` to `PowerController`.

- [ ] **Step 1: Define the interface**

In `internal/jukebox/router.go`, add:

```go
// PowerController is the subset of gpu.PowerManager used by the coordinator
// and scheduler. Defined here so tests can inject a fake without nvidia-smi.
type PowerController interface {
	ApplyModelLimits(ctx context.Context, gpus []int, powerLimit *int, powerLimits map[int]int) error
	RevertModelLimits(ctx context.Context, gpus []int) error
}
```

Add `"context"` to `router.go`'s imports if not already present.

- [ ] **Step 2: Change the coordinator field type**

In `internal/jukebox/coordinator.go`, change:

```go
	powerMgr *gpu.PowerManager
```

to:

```go
	powerMgr PowerController
```

The constructors `NewCoordinator` and `NewCoordinatorWithPower` currently take `powerMgr *gpu.PowerManager`. Change `NewCoordinatorWithPower`'s parameter to `powerMgr PowerController`. `NewCoordinator` passes `nil`, which is fine. `*gpu.PowerManager` continues to satisfy the interface at the call sites in `cmd/jukebox/main.go`, so no caller changes are required — verify with `go build ./...`.

- [ ] **Step 3: Change the scheduler field type**

In `internal/jukebox/scheduler.go`, change:

```go
	powerMgr *gpu.PowerManager
```

to:

```go
	powerMgr PowerController
```

Change `NewSchedulerWithFactory`'s `powerMgr *gpu.PowerManager` parameter to `powerMgr PowerController`. Build to confirm `cmd/jukebox/main.go` still compiles.

- [ ] **Step 4: Build and run existing tests**

Run: `go build ./... && go vet ./... && go test ./internal/jukebox/ -v`
Expected: PASS — behavior unchanged; only types widened to an interface.

- [ ] **Step 5: Commit**

```bash
git add internal/jukebox/router.go internal/jukebox/coordinator.go internal/jukebox/scheduler.go
git commit -m "refactor(jukebox): introduce PowerController interface for testability"
```

---

### Task 5: Power-limit lifecycle — coordinator reverts old model's GPUs + failure paths

**Files:**
- Modify: `internal/jukebox/coordinator.go` (func `doSwap`)
- Test: `internal/jukebox/coordinator_test.go`

**Interfaces:**
- Consumes: `PowerController` from Task 4.
- Produces: `doSwap` reverts the *old* model's GPUs on swap; reverts applied limits when `Start` or `VerifyReady` fails.

- [ ] **Step 1: Write the failing test — swap reverts OLD model's GPUs**

Append to `internal/jukebox/coordinator_test.go`. First add a fake power controller near the top of the file (after `fakeManager`):

```go
type fakePowerController struct {
	mu          sync.Mutex
	applied     []appliedLimit
	revertedGPU []int
}

type appliedLimit struct {
	gpus        []int
	powerLimit  *int
	powerLimits map[int]int
}

func (f *fakePowerController) ApplyModelLimits(_ context.Context, gpus []int, powerLimit *int, powerLimits map[int]int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.applied = append(f.applied, appliedLimit{gpus: append([]int(nil), gpus...), powerLimit: powerLimit, powerLimits: powerLimits})
	return nil
}

func (f *fakePowerController) RevertModelLimits(_ context.Context, gpus []int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.revertedGPU = append(f.revertedGPU, gpus...)
	return nil
}
```

Then the test:

```go
func TestCoordinator_DoSwap_RevertsOldModelGPUs(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  a:
    path: "/models/a"
    gpus: [0, 1]
    power_limit: 300
  b:
    path: "/models/b"
    gpus: [2, 3]
    power_limit: 250
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	var tr inflight.Tracker
	mgr := &fakeManager{}
	power := &fakePowerController{}
	clock := func() time.Time { return time.Date(2025, 12, 14, 0, 0, 0, 0, time.UTC) }
	c := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, clock, power)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "a", "req_1"); err != nil {
		t.Fatalf("EnsureModel a: %v", err)
	}

	// Swap to model b. The revert on swap must target a's GPUs [0,1], not b's [2,3].
	if err := c.EnsureModel(context.Background(), "b", "req_2"); err != nil {
		t.Fatalf("EnsureModel b: %v", err)
	}

	power.mu.Lock()
	defer power.mu.Unlock()
	if len(power.revertedGPU) == 0 {
		t.Fatalf("expected a revert on swap, got none")
	}
	// The first revert (during the swap) must be for a's GPUs [0,1].
	first := power.revertedGPU[0]
	if len(first) != 2 || first[0] != 0 || first[1] != 1 {
		t.Fatalf("expected revert of old model GPUs [0 1], got %v", first)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/jukebox/ -run TestCoordinator_DoSwap_RevertsOldModelGPUs -v`
Expected: FAIL — `doSwap` currently reverts `modelCfg.GPUs` (the new model b's GPUs [2,3]), so the test sees `[2 3]` not `[0 1]`.

- [ ] **Step 3: Fix `doSwap` to revert the old model's GPUs**

In `internal/jukebox/coordinator.go`, in `doSwap`, the block that stops the old model currently resolves `_, modelCfg, err := c.cfg.ResolveModel(model)` (new model) at the top of the function and reverts `modelCfg.GPUs`. Change the stop-and-revert block to resolve the *old* model's config.

Find the block:

```go
	// If we already have a running model, stop it after draining in-flight.
	if st := c.Status(); st.State == StateStopping || st.State == StateReady {
		if c.tr != nil {
			drainCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.DrainTimeout.Duration)
			err := c.tr.WaitForDrain(drainCtx)
			cancel()
			if err != nil && !errors.Is(err, context.DeadlineExceeded) {
				return err
			}
		}

		stopCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		stopErr := c.mgr.Stop(stopCtx)
		cancel()
		if stopErr != nil {
			return stopErr
		}

		// Revert power limits after stop (if we have GPUs configured)
		if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
	}
```

Replace with (resolving the old model's config from `c.currentModel`):

```go
	// If we already have a running model, stop it after draining in-flight.
	if st := c.Status(); st.State == StateStopping || st.State == StateReady {
		// Resolve the OLD model's config to revert ITS GPUs (not the new model's).
		var oldGPUs []int
		c.mu.RLock()
		oldName := c.currentModel
		c.mu.RUnlock()
		if oldName != "" {
			if _, oldCfg, err := c.cfg.ResolveModel(oldName); err == nil {
				oldGPUs = oldCfg.GPUs
			}
		}

		if c.tr != nil {
			drainCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.DrainTimeout.Duration)
			err := c.tr.WaitForDrain(drainCtx)
			cancel()
			if err != nil && !errors.Is(err, context.DeadlineExceeded) {
				return err
			}
		}

		stopCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		stopErr := c.mgr.Stop(stopCtx)
		cancel()
		if stopErr != nil {
			return stopErr
		}

		// Revert power limits for the OLD model's GPUs after stop.
		if c.powerMgr != nil && len(oldGPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), oldGPUs)
		}
	}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./internal/jukebox/ -run TestCoordinator_DoSwap_RevertsOldModelGPUs -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test — revert on failed Start and failed Verify**

Append to `internal/jukebox/coordinator_test.go`:

```go
func TestCoordinator_StartFailure_RevertsPowerLimits(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  a:
    path: "/models/a"
    gpus: [0]
    power_limit: 300
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var tr inflight.Tracker
	mgr := &fakeManager{startErr: errors.New("boom")}
	power := &fakePowerController{}
	c := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, func() time.Time { return time.Date(2025, 12, 14, 0, 0, 0, 0, time.UTC) }, power)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "a", "req_1"); err == nil {
		t.Fatalf("expected start error")
	}
	power.mu.Lock()
	defer power.mu.Unlock()
	if len(power.revertedGPU) == 0 {
		t.Fatalf("expected power revert on failed start, got %v", power.revertedGPU)
	}
}

func TestCoordinator_VerifyFailure_RevertsPowerLimits(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  a:
    path: "/models/a"
    gpus: [0]
    power_limit: 300
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var tr inflight.Tracker
	mgr := &fakeManager{verifyErr: errors.New("verify boom")}
	power := &fakePowerController{}
	c := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, func() time.Time { return time.Date(2025, 12, 14, 0, 0, 0, 0, time.UTC) }, power)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "a", "req_1"); err == nil {
		t.Fatalf("expected verify error")
	}
	power.mu.Lock()
	defer power.mu.Unlock()
	if len(power.revertedGPU) == 0 {
		t.Fatalf("expected power revert on failed verify, got %v", power.revertedGPU)
	}
}
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `go test ./internal/jukebox/ -run 'TestCoordinator_(Start|Verify)Failure_RevertsPowerLimits' -v`
Expected: FAIL — `doSwap` does not revert on Start/Verify failure today.

- [ ] **Step 7: Add reverts to the coordinator failure paths**

In `internal/jukebox/coordinator.go`, in `doSwap`, the Start/Verify block currently reads:

```go
	if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
		if err := c.powerMgr.ApplyModelLimits(context.Background(), modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			return err
		}
	}

	startCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	_, err = c.mgr.Start(startCtx, model)
	cancel()
	if err != nil {
		return err
	}

	verifyCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	err = c.mgr.VerifyReady(verifyCtx, model)
	cancel()
	if err != nil {
		// Ensure we don't leave a partially-started vLLM process around if verification fails.
		stopCtx, stopCancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		_ = c.mgr.Stop(stopCtx)
		stopCancel()
		return err
	}
```

Replace with (adding power reverts to both failure paths):

```go
	if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
		if err := c.powerMgr.ApplyModelLimits(context.Background(), modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			return err
		}
	}

	startCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	_, err = c.mgr.Start(startCtx, model)
	cancel()
	if err != nil {
		if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
		return err
	}

	verifyCtx, cancel := context.WithTimeout(context.Background(), c.cfg.VLLM.StartupTimeout.Duration)
	err = c.mgr.VerifyReady(verifyCtx, model)
	cancel()
	if err != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), c.cfg.VLLM.ShutdownTimeout.Duration)
		_ = c.mgr.Stop(stopCtx)
		stopCancel()
		if c.powerMgr != nil && len(modelCfg.GPUs) > 0 {
			_ = c.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
		return err
	}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `go test ./internal/jukebox/ -v`
Expected: PASS (all coordinator tests including the three new ones).

- [ ] **Step 9: Commit**

```bash
git add internal/jukebox/coordinator.go internal/jukebox/coordinator_test.go
git commit -m "fix(coordinator): revert old model power limits on swap; revert on failed start/verify"
```

---

### Task 6: Power-limit lifecycle — scheduler failure paths

**Files:**
- Modify: `internal/jukebox/scheduler.go` (func `AcquireRoute` start path)
- Test: `internal/jukebox/scheduler_test.go`

**Interfaces:**
- Consumes: `PowerController` from Task 4; needs `startErr`/`verifyErr` on the fake instance.

- [ ] **Step 1: Add failure injection to the fake instance**

In `internal/jukebox/scheduler_test.go`, add fields to `fakeInstance`:

```go
type fakeInstance struct {
	ctrl *fakeInstanceController
	port int

	mu  sync.Mutex
	pid int

	startedModel string
	startErr     error
	verifyErr    error
}
```

Update `Start` to return `startErr` when set (after the block check, before setting pid):

```go
func (f *fakeInstance) Start(ctx context.Context, modelName string) (int, error) {
	f.ctrl.mu.Lock()
	ch := f.ctrl.startBlocks[modelName]
	f.ctrl.mu.Unlock()

	if ch != nil {
		select {
		case <-ch:
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}
	if f.startErr != nil {
		return 0, f.startErr
	}
	f.mu.Lock()
	f.pid = 123
	f.startedModel = modelName
	f.mu.Unlock()
	return 123, nil
}
```

Update `VerifyReady` to return `verifyErr`:

```go
func (f *fakeInstance) VerifyReady(_ context.Context, _ string) error {
	if f.verifyErr != nil {
		return f.verifyErr
	}
	return nil
}
```

- [ ] **Step 2: Write the failing test — scheduler reverts power on failed Start**

Append to `internal/jukebox/scheduler_test.go`:

```go
func TestScheduler_StartFailure_RevertsPowerLimits(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    power_limit: 300
`)

	ctrl := &fakeInstanceController{}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMemoryMB: 100000, FreeMemoryMB: 50000}}}
	power := &fakePowerControllerSched{}

	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, func(port int, cuda string) jukebox.InstanceManager {
		return &fakeInstance{ctrl: ctrl, port: port, startErr: errors.New("boom")}
	}, power)

	_, err := s.AcquireRoute(context.Background(), "a", "req1")
	if err == nil {
		t.Fatalf("expected start error")
	}
	if len(power.revertedGPU) == 0 {
		t.Fatalf("expected power revert on failed start, got %v", power.revertedGPU)
	}
	if pool.AvailableForTest() != 10 { // port must be released
		t.Fatalf("expected port released after failed start")
	}
}

func TestScheduler_VerifyFailure_RevertsPowerLimits(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    power_limit: 300
`)

	ctrl := &fakeInstanceController{}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMemoryMB: 100000, FreeMemoryMB: 50000}}}
	power := &fakePowerControllerSched{}

	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, func(port int, cuda string) jukebox.InstanceManager {
		return &fakeInstance{ctrl: ctrl, port: port, verifyErr: errors.New("verify boom")}
	}, power)

	_, err := s.AcquireRoute(context.Background(), "a", "req1")
	if err == nil {
		t.Fatalf("expected verify error")
	}
	if len(power.revertedGPU) == 0 {
		t.Fatalf("expected power revert on failed verify, got %v", power.revertedGPU)
	}
}
```

Add the fake power controller at the top of the test file only if it does not already exist. `fakePowerController` is defined in `coordinator_test.go` (Task 5) in the same `jukebox_test` package, so it is already visible here — reuse it, do not redefine it. The references in the tests below use `&fakePowerController{}`.

Also add a port-pool test accessor. In `internal/ports/pool.go`, add (the struct fields are `start`, `end`, `used map[int]bool`, `mu sync.Mutex`):

```go
// AvailableForTest returns the count of free ports. For tests only.
func (p *Pool) AvailableForTest() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.end - p.start + 1 - len(p.used)
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `go test ./internal/jukebox/ -run 'TestScheduler_(Start|Verify)Failure_RevertsPowerLimits' -v`
Expected: FAIL — scheduler does not revert power on Start/Verify failure.

- [ ] **Step 4: Add reverts to the scheduler failure paths**

In `internal/jukebox/scheduler.go`, in the `AcquireRoute` start path, the block currently reads:

```go
	// Apply model power limits if configured
	if s.powerMgr != nil {
		if err := s.powerMgr.ApplyModelLimits(ctx, modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			s.ports.Release(port)
			return Route{}, err
		}
	}

	startCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	_, startErr := mgr.Start(startCtx, resolvedName)
	cancel()
	if startErr != nil {
		s.ports.Release(port)
		return Route{}, startErr
	}

	verifyCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	verifyErr := mgr.VerifyReady(verifyCtx, resolvedName)
	cancel()
	if verifyErr != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), s.cfg.VLLM.ShutdownTimeout.Duration)
		_ = mgr.Stop(stopCtx)
		stopCancel()
		s.ports.Release(port)
		return Route{}, verifyErr
	}
```

Replace with:

```go
	// Apply model power limits if configured
	if s.powerMgr != nil {
		if err := s.powerMgr.ApplyModelLimits(ctx, modelCfg.GPUs, modelCfg.PowerLimit, modelCfg.PowerLimits); err != nil {
			s.ports.Release(port)
			return Route{}, err
		}
	}

	startCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	_, startErr := mgr.Start(startCtx, resolvedName)
	cancel()
	if startErr != nil {
		s.ports.Release(port)
		if s.powerMgr != nil {
			_ = s.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
		return Route{}, startErr
	}

	verifyCtx, cancel := context.WithTimeout(ctx, s.cfg.VLLM.StartupTimeout.Duration)
	verifyErr := mgr.VerifyReady(verifyCtx, resolvedName)
	cancel()
	if verifyErr != nil {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), s.cfg.VLLM.ShutdownTimeout.Duration)
		_ = mgr.Stop(stopCtx)
		stopCancel()
		s.ports.Release(port)
		if s.powerMgr != nil {
			_ = s.powerMgr.RevertModelLimits(context.Background(), modelCfg.GPUs)
		}
		return Route{}, verifyErr
	}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `go test ./internal/jukebox/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/jukebox/scheduler.go internal/jukebox/scheduler_test.go internal/ports/pool.go
git commit -m "fix(scheduler): revert power limits on failed start/verify"
```

---

### Task 7: Scheduler crash-replace port/process leak

**Files:**
- Modify: `internal/jukebox/scheduler.go` (the replace block in `AcquireRoute`, ~line 285)
- Test: `internal/jukebox/scheduler_test.go`

**Interfaces:**
- Consumes: `PowerController` from Task 4.

- [ ] **Step 1: Write the failing test — crash-replace releases old port and stops old instance**

Append to `internal/jukebox/scheduler_test.go`:

```go
func TestScheduler_CrashReplace_StopsOldAndReleasesPort(t *testing.T) {
	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8100
  port_range_end: 8109
  min_instance_uptime: 1s
vllm:
  port: 8000
  startup_timeout: 5s
models:
  a:
    path: "/models/a"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
`)

	ctrl := &fakeInstanceController{}
	pool := ports.New(8100, 8109)
	inv := &fakeInventory{gpus: []gpu.GPU{{Index: 0, TotalMemoryMB: 100000, FreeMemoryMB: 50000}}}

	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, func(port int, cuda string) jukebox.InstanceManager {
		return &fakeInstance{ctrl: ctrl, port: port}
	}, nil)

	ctx := context.Background()
	route, err := s.AcquireRoute(ctx, "a", "req1")
	if err != nil {
		t.Fatalf("AcquireRoute: %v", err)
	}
	route.Done()
	oldPort := s.InstancesForTest()["a"].port

	// Simulate a crash: PID goes to 0 while state still Ready in the map.
	s.InstancesForTest()["a"].mu.Lock()
	s.InstancesForTest()["a"].pid = 0
	s.InstancesForTest()["a"].mu.Unlock()

	// A new request must replace the crashed instance with a fresh one.
	route2, err := s.AcquireRoute(ctx, "a", "req2")
	if err != nil {
		t.Fatalf("AcquireRoute(replace): %v", err)
	}
	route2.Done()

	// The old port must have been released back to the pool (available count restored).
	if got := pool.AvailableForTest(); got != 10 {
		t.Fatalf("expected old port released (10 available), got %d", got)
	}
	// The old instance's process must have been stopped (stop recorded).
	ctrl.mu.Lock()
	if len(ctrl.stopOrder) == 0 {
		ctrl.mu.Unlock()
		t.Fatalf("expected old instance to be stopped, got stopOrder=%v", ctrl.stopOrder)
	}
	ctrl.mu.Unlock()
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/jukebox/ -run TestScheduler_CrashReplace_StopsOldAndReleasesPort -v`
Expected: FAIL — today the old instance is only marked `draining=true` and dropped; its port is never released (so `AvailableForTest() < 10`) and `stopOrder` is empty.

- [ ] **Step 3: Implement the inline cleanup of the old instance**

In `internal/jukebox/scheduler.go`, replace the replace block:

```go
	s.mu.Lock()
	// If a previous instance exists for this model (e.g., crashed), replace it.
	if old := s.instances[resolvedName]; old != nil {
		// Best-effort cleanup of the old instance without blocking.
		old.draining = true
	}
	s.instances[resolvedName] = inst
	s.mu.Unlock()

	return s.routeForInstance(ctx, inst, modelCfg.Path), nil
```

with:

```go
	s.mu.Lock()
	// If a previous instance exists for this model (e.g., crashed), capture it
	// for cleanup. Do NOT delete from the map here — the new instance owns the
	// key now; drainAndStopInstance's own delete would clobber it.
	old := s.instances[resolvedName]
	s.instances[resolvedName] = inst
	s.mu.Unlock()

	if old != nil {
		old.draining = true
		old.state = StateStopping
		if old.mgr != nil {
			stopTimeout := s.cfg.VLLM.ShutdownTimeout.Duration
			if stopTimeout <= 0 {
				stopTimeout = 30 * time.Second
			}
			stopCtx, stopCancel := context.WithTimeout(context.Background(), stopTimeout)
			_ = old.mgr.Stop(stopCtx)
			stopCancel()
		}
		if s.powerMgr != nil {
			_ = s.powerMgr.RevertModelLimits(context.Background(), old.gpus)
		}
		oldPortLabel := strconv.Itoa(old.port)
		metrics.RunningInstances.WithLabelValues(old.model, oldPortLabel).Set(0)
		metrics.InstanceInFlightRequests.WithLabelValues(old.model, oldPortLabel).Set(0)
		if old.port != 0 {
			s.ports.Release(old.port)
		}
	}

	return s.routeForInstance(ctx, inst, modelCfg.Path), nil
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./internal/jukebox/ -run TestScheduler_CrashReplace_StopsOldAndReleasesPort -v`
Expected: PASS.

- [ ] **Step 5: Run the full jukebox suite**

Run: `go test ./internal/jukebox/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/jukebox/scheduler.go internal/jukebox/scheduler_test.go
git commit -m "fix(scheduler): stop old instance and release its port on crash-replace"
```

---

### Task 8: Default bind `127.0.0.1` + optional `server.api_key` config

**Files:**
- Modify: `internal/config/config.go` (struct `ServerConfig`; func `applyDefaults`; func `Validate`)
- Test: `internal/config/config_test.go`

**Interfaces:**
- Produces: `Server.APIKey string`; default `Server.Host = "127.0.0.1"`; empty-but-present `api_key` is a startup error.

- [ ] **Step 1: Write the failing tests**

Append to `internal/config/config_test.go`:

```go
func TestConfig_DefaultHostIsLocalhost(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Server.Host != "127.0.0.1" {
		t.Fatalf("expected default host 127.0.0.1, got %q", cfg.Server.Host)
	}
}

func TestConfig_ExplicitHostPreserved(t *testing.T) {
	cfg, err := config.Load([]byte(`
server:
  host: "0.0.0.0"
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Server.Host != "0.0.0.0" {
		t.Fatalf("expected explicit host preserved, got %q", cfg.Server.Host)
	}
}

func TestConfig_APIKeyAccepted(t *testing.T) {
	cfg, err := config.Load([]byte(`
server:
  api_key: "secret-token"
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Server.APIKey != "secret-token" {
		t.Fatalf("expected api_key parsed, got %q", cfg.Server.APIKey)
	}
}

func TestConfig_EmptyAPIKeyRejected(t *testing.T) {
	_, err := config.Load([]byte(`
server:
  api_key: ""
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err == nil {
		t.Fatalf("expected error for empty api_key")
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./internal/config/ -run 'TestConfig_(DefaultHost|ExplicitHost|APIKey|EmptyAPIKey)' -v`
Expected: FAIL — default host is `0.0.0.0`, no `APIKey` field.

- [ ] **Step 3: Implement the config changes**

In `internal/config/config.go`, add `APIKey` to `ServerConfig`:

```go
type ServerConfig struct {
	Host         string   `yaml:"host"`
	Port         int      `yaml:"port"`
	ReadTimeout  Duration `yaml:"read_timeout"`
	WriteTimeout Duration `yaml:"write_timeout"`
	LogRequests  *bool    `yaml:"log_requests"`
	APIKey       string   `yaml:"api_key"`
}
```

In `applyDefaults`, change the default host:

```go
	if c.Server.Host == "" {
		c.Server.Host = "127.0.0.1"
	}
```

In `Validate`, add (near the top, after the existing port validations):

```go
	if strings.TrimSpace(c.Server.APIKey) == "" && c.Server.APIKey != "" {
		return fmt.Errorf("server.api_key must not be empty/whitespace; omit the field to disable auth")
	}
```

Add `"strings"` to the imports if not present.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/config/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): default bind to 127.0.0.1; add optional server.api_key"
```

---

### Task 9: `authAPIKey` middleware

**Files:**
- Create: `internal/httpserver/middleware_auth.go`
- Test: `internal/httpserver/middleware_auth_test.go`

**Interfaces:**
- Produces: `authAPIKey(expected string) fiber.Handler`.

- [ ] **Step 1: Write the failing tests**

Create `internal/httpserver/middleware_auth_test.go`:

```go
package httpserver_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/httpserver"
)

func newAuthApp(key string) *fiber.App {
	app := fiber.New()
	app.Use(httpserver.AuthAPIKey(key))
	app.Get("/protected", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})
	return app
}

func TestAuthAPIKey_MissingHeader_401(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}
}

func TestAuthAPIKey_WrongKey_401(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}
}

func TestAuthAPIKey_CorrectKey_200(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}
}

func TestAuthAPIKey_MalformedHeader_401(t *testing.T) {
	app := newAuthApp("secret")
	for _, h := range []string{"secret", "Token secret", "Bearer"} {
		req := httptest.NewRequest("GET", "/protected", strings.NewReader(""))
		req.Header.Set("Authorization", h)
		resp, err := app.Test(req)
		if err != nil {
			t.Fatalf("test: %v", err)
		}
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("expected 401 for header %q, got %d", h, resp.StatusCode)
		}
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./internal/httpserver/ -run TestAuthAPIKey -v`
Expected: FAIL — `httpserver.AuthAPIKey` is not defined.

- [ ] **Step 3: Implement the middleware**

Create `internal/httpserver/middleware_auth.go`:

```go
package httpserver

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/gofiber/fiber/v2"
)

// AuthAPIKey returns middleware that requires `Authorization: Bearer <key>` when
// expected is non-empty. If expected is empty, the middleware is a no-op pass-
// through (auth is disabled). Compare is constant-time.
func AuthAPIKey(expected string) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if expected == "" {
			return c.Next()
		}
		h := c.Get("Authorization")
		const prefix = "Bearer "
		if !strings.HasPrefix(h, prefix) {
			return writeOpenAIError(c, http.StatusUnauthorized, "Missing or invalid Authorization header", "invalid_request_error", "missing_auth")
		}
		provided := h[len(prefix):]
		if subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) != 1 {
			return writeOpenAIError(c, http.StatusUnauthorized, "Invalid API key", "invalid_request_error", "invalid_auth")
		}
		return c.Next()
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/httpserver/ -run TestAuthAPIKey -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/httpserver/middleware_auth.go internal/httpserver/middleware_auth_test.go
git commit -m "feat(httpserver): add optional bearer auth middleware"
```

---

### Task 10: Mount auth middleware on protected routes

**Files:**
- Modify: `internal/httpserver/server.go`
- Test: `internal/httpserver/server_test.go`

**Interfaces:**
- Consumes: `AuthAPIKey` from Task 9; `cfg.Server.APIKey` from Task 8.

- [ ] **Step 1: Inspect the existing server test to match its style**

Run: `sed -n '1,60p' internal/httpserver/server_test.go`
Note how `NewApp` is constructed and how requests are made (it uses `app.Test` or `httptest`). Match that style in the new test.

- [ ] **Step 2: Write the failing test — protected route requires key when set**

Append to `internal/httpserver/server_test.go` (adjust constructor args to match the real `Options`; if `Options` needs a `Router`, pass `nil` and have the handler return a 500 for the proxy path — the auth check must run before the proxy handler reaches `Router`):

```go
func TestServer_AuthMiddleware_MountedWhenAPIKeySet(t *testing.T) {
	cfg, err := config.Load([]byte(`
server:
  api_key: "secret"
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	app := NewApp(Options{Config: cfg})

	// /status without key -> 401
	req := httptest.NewRequest("GET", "/status", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 on /status without key, got %d", resp.StatusCode)
	}

	// /status with key -> not 401 (it may be 200 or another status; auth passed)
	req2 := httptest.NewRequest("GET", "/status", nil)
	req2.Header.Set("Authorization", "Bearer secret")
	resp2, err := app.Test(req2)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp2.StatusCode == http.StatusUnauthorized {
		t.Fatalf("expected auth to pass with correct key, got 401")
	}
}

func TestServer_AuthMiddleware_NotMountedWhenAPIKeyUnset(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  m:
    path: "/models/m"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	app := NewApp(Options{Config: cfg})

	req := httptest.NewRequest("GET", "/status", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode == http.StatusUnauthorized {
		t.Fatalf("expected no auth when api_key unset, got 401")
	}
}
```

Add `"net/http"` and `"vllm-jukebox/internal/config"` to the test imports if missing.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `go test ./internal/httpserver/ -run TestServer_AuthMiddleware -v`
Expected: FAIL — no auth is mounted today, so `/status` returns a non-401 even without a key (the second test passes trivially, but the first test fails because no 401 is returned).

- [ ] **Step 4: Mount the middleware in `NewApp`**

In `internal/httpserver/server.go`, mount `AuthAPIKey` on the protected groups when `api_key` is set. After the `app.Use(requestIDMiddleware())` block and before registering the protected routes, add a guard. Replace the route-registration section:

```go
	app.Get("/health", healthHandler(opts.Router))
	app.Get("/status", statusHandler(opts.Config, opts.Router))

	app.Get("/v1/models", listModelsHandler(opts.Config))
	app.Get("/metrics", metricsHandler())

	// Model-bearing endpoints (scheduler-routed).
	app.Post("/v1/responses", switchingProxyHandler(opts))
	app.Post("/v1/chat/completions", switchingProxyHandler(opts))
	app.Post("/v1/completions", switchingProxyHandler(opts))
	app.Post("/v1/embeddings", switchingProxyHandler(opts))
	app.Post("/v1/tokenize", switchingProxyHandler(opts))
	app.Post("/v1/detokenize", switchingProxyHandler(opts))

	// Anthropic endpoints
	app.Post("/v1/messages", anthropicProxyHandler(opts))

	// Explicit unsupported endpoints.
	app.All("/v1/audio/*", notImplementedHandler())
	app.All("/v1/images/*", notImplementedHandler())
	app.All("/v1/files*", notImplementedHandler())
	app.All("/v1/fine-tuning/*", notImplementedHandler())
	app.All("/v1/assistants/*", notImplementedHandler())
	// Fallback for unknown /v1 endpoints: return 501 rather than 404.
	app.All("/v1/*", notImplementedHandler())

	return app
```

with:

```go
	app.Get("/health", healthHandler(opts.Router))

	// Protected routes: require bearer auth when server.api_key is set.
	apiKey := ""
	if opts.Config != nil {
		apiKey = opts.Config.Server.APIKey
	}
	protected := app.Group("/", AuthAPIKey(apiKey))

	protected.Get("/status", statusHandler(opts.Config, opts.Router))
	protected.Get("/metrics", metricsHandler())

	protected.Get("/v1/models", listModelsHandler(opts.Config))
	protected.Post("/v1/responses", switchingProxyHandler(opts))
	protected.Post("/v1/chat/completions", switchingProxyHandler(opts))
	protected.Post("/v1/completions", switchingProxyHandler(opts))
	protected.Post("/v1/embeddings", switchingProxyHandler(opts))
	protected.Post("/v1/tokenize", switchingProxyHandler(opts))
	protected.Post("/v1/detokenize", switchingProxyHandler(opts))
	protected.Post("/v1/messages", anthropicProxyHandler(opts))

	// Explicit unsupported endpoints (not behind auth — they are 501 stubs).
	app.All("/v1/audio/*", notImplementedHandler())
	app.All("/v1/images/*", notImplementedHandler())
	app.All("/v1/files*", notImplementedHandler())
	app.All("/v1/fine-tuning/*", notImplementedHandler())
	app.All("/v1/assistants/*", notImplementedHandler())
	// Fallback for unknown /v1 endpoints: return 501 rather than 404.
	app.All("/v1/*", notImplementedHandler())

	return app
```

`AuthAPIKey("")` is a no-op pass-through, so when `api_key` is unset the protected group behaves exactly as before. `/health` remains open (health checks).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `go test ./internal/httpserver/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/httpserver/server.go internal/httpserver/server_test.go
git commit -m "feat(httpserver): mount optional bearer auth on inference + status + metrics"
```

---

### Task 11: Update example configs and README for default bind + api_key

**Files:**
- Modify: `configs/example.yaml`
- Modify: `configs/scheduler_example.yaml`
- Modify: `README.md`

- [ ] **Step 1: Update example configs to document the new default and api_key**

In `configs/example.yaml` and `configs/scheduler_example.yaml`, change `host: "0.0.0.0"` to a commented example showing the new default and the opt-in:

```yaml
server:
  # Default bind is 127.0.0.1 (localhost only). Set to 0.0.0.0 to expose on all
  # interfaces — ensure network-level protection or set api_key below.
  host: "0.0.0.0"
  # Optional: require `Authorization: Bearer <key>` on inference + /status + /metrics.
  # Omit to disable auth (backward compatible).
  # api_key: "change-me"
```

- [ ] **Step 2: Update README**

In `README.md`, in the config/examples section, add a note near the existing `host` mentions:

```markdown
> **Note:** The default bind is `127.0.0.1` (localhost). To expose jukebox on the
> network, set `host: "0.0.0.0"` and configure `server.api_key` to require a
> bearer token on inference and operational endpoints.
```

Update the `/status` line that says "bind to localhost / protect in production" to reference the new `api_key` option.

- [ ] **Step 3: Commit**

```bash
git add configs/example.yaml configs/scheduler_example.yaml README.md
git commit -m "docs: document default 127.0.0.1 bind and optional api_key auth"
```

---

### Task 12: Full verification

- [ ] **Step 1: Run the full check**

Run: `make check`
Expected: fmt-check passes, vet passes, all tests pass.

- [ ] **Step 2: Run the smoke tests (no real vLLM needed)**

Run: `make smoke && make smoke-scheduler`
Expected: PASS.

- [ ] **Step 3: Confirm no scope creep**

Run: `git diff main --stat` (or the merge-base). Confirm only the files in the File Structure table are touched; no #6–#15 work leaked in.

- [ ] **Step 4: Final commit if check made any formatting fixes**

```bash
git add -A
git commit -m "chore: gofmt after hardening pass" || echo "nothing to commit"
```
