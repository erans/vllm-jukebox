# vLLM Jukebox MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Build an OpenAI-compatible API server that fronts a single vLLM process and automatically swaps the loaded model based on the incoming request’s `model`.

**Architecture:** A Go HTTP server (Fiber) receives OpenAI-style requests, extracts the requested model, asks a single-goroutine swap coordinator to ensure that model is loaded, then proxies the request to the local vLLM server. During swaps, non-triggering requests get `503` with `Retry-After`.

**Tech Stack:** Go, `github.com/gofiber/fiber/v2`, `gopkg.in/yaml.v3`, Go `log/slog` JSON logging, standard `net/http` client for proxying.

---

## Conventions

- Workspace root: repo root.
- Main binary: `cmd/jukebox`.
- Internal packages: `internal/...`.
- Tests: Go `testing` package (`go test ./...`).
- vLLM base URL: `http://127.0.0.1:<vllm.port>`.

---

### Task 1: Bootstrap Go project + minimal CLI

**Files:**
- Create: `go.mod`
- Create: `cmd/jukebox/main.go`

**Step 1: Create `go.mod`**

Run:
```bash
go mod init vllm-jukebox
go get github.com/gofiber/fiber/v2@latest
go get gopkg.in/yaml.v3@latest
```

Expected:
- `go.mod` and `go.sum` created.

**Step 2: Add minimal `main` that parses `-config` and exits**

Create `cmd/jukebox/main.go` with:
- `-config` flag required
- Read file bytes
- Parse YAML into config struct (stubbed initially)
- Print a single startup log line (JSON via `slog`) and exit `0`

**Step 3: Verify build**

Run:
```bash
go test ./...
go build ./cmd/jukebox
```

Expected:
- `go test` passes (no tests yet).
- `go build` succeeds.

---

### Task 2: Implement config schema + strict validation (TDD)

**Files:**
- Create: `internal/config/config.go`
- Create: `internal/config/config_test.go`

**Step 1: Write failing tests for YAML parsing + KnownFields**

Create `internal/config/config_test.go` with tests:
- `TestLoad_RejectsUnknownTopLevelField`
- `TestLoad_RejectsUnknownModelField`
- `TestValidate_EmptyModelsRejected`
- `TestValidate_AliasToMissingModelRejected`
- `TestValidate_AliasChainRejected`
- `TestValidate_AliasLoopRejected`
- `TestValidate_DefaultModelMustExist`
- `TestValidate_GPUUtilRange`

Use a helper:
```go
func loadFromYAML(t *testing.T, s string) (*config.Config, error) { ... }
```

Expected (RED):
- Fails because `config.Load` / `config.Validate` doesn’t exist yet.

**Step 2: Implement minimal loader**

In `internal/config/config.go`:
- Define `Config` struct mirroring `docs/spec.md`
- Use `yaml.NewDecoder(...).KnownFields(true)` for strictness
- Implement `Load(bytes)` that decodes YAML and calls `Validate()`

**Step 3: Implement `Validate()`**

Must enforce at least:
- non-empty `models`
- aliases reference existing models
- alias cannot reference an alias (no chains)
- detect alias loops
- non-alias models require `path`
- `gpu_memory_utilization` in `[0.0, 1.0]` if set
- `default_model` (if set) must exist
- ports in `[1,65535]`

**Step 4: Run tests**

Run:
```bash
go test ./internal/config -v
```

Expected: PASS.

---

### Task 3: Add model resolution helpers (requested vs canonical)

**Files:**
- Modify: `internal/config/config.go`
- Create: `internal/config/resolve_test.go`

**Step 1: Write failing tests for model resolution**

Create `internal/config/resolve_test.go`:
- `TestResolveModel_AliasResolvesToTarget`
- `TestResolveModel_NonAliasResolvesToSelf`
- `TestResolveModel_UnknownRejected`

Expected (RED):
- Fails because `ResolveModel` doesn’t exist.

**Step 2: Implement `ResolveModel(name string) (resolvedName string, model Model, err error)`**

Rules:
- If `name` is an alias, return referenced model name + model config
- If `name` is non-alias, return `name` + model config
- Error if unknown

**Step 3: Run tests**

Run:
```bash
go test ./internal/config -v
```

Expected: PASS.

---

### Task 4: In-flight tracker (drain coordination)

**Files:**
- Create: `internal/inflight/tracker.go`
- Create: `internal/inflight/tracker_test.go`

Implement a tracker like `docs/spec.md`:
- `Track(ctx)` returns `done()`
- `WaitForDrain(ctx)` blocks until count is `0`
- Ensure `done()` is idempotent-safe (calling twice doesn’t underflow)

Verification:
```bash
go test ./internal/inflight -v
```

---

### Task 5: Swap coordinator state machine (mocked manager)

**Files:**
- Create: `internal/jukebox/coordinator.go`
- Create: `internal/jukebox/coordinator_test.go`

Implement:
- single goroutine Run loop (channels)
- states: `idle|starting|ready|stopping|error`
- cooldown enforcement
- backoff on repeated failures (return 503-style error with retry-after seconds)

Use a `Manager` interface mocked in tests:
```go
type Manager interface {
  Start(ctx context.Context, modelName string) (pid int, err error)
  Stop(ctx context.Context) error
  VerifyReady(ctx context.Context, expectedModel string) error
  CurrentPID() int
}
```

Verification:
```bash
go test ./internal/jukebox -v
```

---

### Task 6: Real vLLM process manager (exec + process groups)

**Files:**
- Create: `internal/vllm/manager.go`
- Create: `internal/vllm/health.go`
- Create: `internal/vllm/manager_test.go` (unit tests for command/env building; no real vLLM)

Implement:
- build args for `vllm serve <path> --host 127.0.0.1 --port <port> ...`
- env merge: inherited + `default_env` + model env (later wins)
- process group (`Setpgid: true`), `SIGTERM` then `SIGKILL` after timeout
- readiness check: poll `GET /health` then verify `GET /v1/models` contains expected id/path

---

### Task 7: HTTP server skeleton + management endpoints

**Files:**
- Create: `internal/httpserver/server.go`
- Create: `internal/httpserver/middleware_request_id.go`
- Create: `internal/httpserver/handlers_health.go`
- Create: `internal/httpserver/handlers_status.go`
- Create: `internal/httpserver/server_test.go`
- Modify: `cmd/jukebox/main.go`

Implement:
- `GET /health` and `GET /status` per `docs/spec.md`
- request ID propagation via `X-Request-ID` (generate if missing)
- JSON logging with request_id

---

### Task 8: Proxy endpoints (model-switching + pass-through)

**Files:**
- Create: `internal/httpserver/handlers_proxy.go`
- Create: `internal/proxy/proxy.go`
- Create: `internal/proxy/rewrite.go`
- Create: `internal/proxy/proxy_test.go`

Implement:
- switching endpoints: `POST /v1/responses`, `/v1/chat/completions`, `/v1/completions`
- pass-through endpoints: `/v1/embeddings`, `/v1/tokenize`, `/v1/detokenize`
- `GET /v1/models` returns configured model list (not vLLM)
- during swap: return `503` + `Retry-After`
- optional `behavior.rewrite_model_name` for JSON and SSE streaming chunks

---

### Task 9: Packaging + docs

**Files:**
- Create: `configs/example.yaml`
- Create: `README.md`

Add:
- how to run (`./jukebox -config configs/example.yaml`)
- operational notes (bind `/status` carefully)
- example curl requests

---

### Task 10: Verification + smoke test script

**Files:**
- Create: `scripts/smoke.sh`

Run:
```bash
go test ./...
go build -o bin/jukebox ./cmd/jukebox
./bin/jukebox -config configs/example.yaml
```

Expected:
- server starts and `/health` returns JSON.

