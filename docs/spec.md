# vLLM Jukebox

**Technical Specification v0.2**

## Overview

vLLM Jukebox is an OpenAI-compatible API server that fronts a single vLLM instance, providing automatic model switching based on incoming requests. When a request arrives for a model that isn't currently loaded, Jukebox gracefully drains in-flight requests, stops the current vLLM process, and starts a new one with the requested model.

## Goals

- **OpenAI API compatibility**: Drop-in replacement for applications using the OpenAI API
- **Transparent model switching**: Clients request models by name; Jukebox handles loading
- **Graceful transitions**: In-flight requests complete before model switches
- **Simple deployment**: Single binary, YAML configuration

## Non-Goals (v1)

- Multiple concurrent vLLM instances
- Request queuing during model switches (beyond the triggering request)
- LoRA adapter hot-swapping
- Load balancing across multiple backends
- API key validation / per-key model allowlists (use an API gateway)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM Jukebox                         │
├─────────────────────────────────────────────────────────────┤
│  HTTP Server (Fiber)                                        │
│    ├── POST /v1/responses                                   │
│    ├── POST /v1/chat/completions                            │
│    ├── POST /v1/completions                                 │
│    ├── GET  /v1/models                                      │
│    ├── GET  /health                                         │
│    ├── GET  /status                                         │
│    └── GET  /metrics (optional)                             │
├─────────────────────────────────────────────────────────────┤
│  Request Handler                                            │
│    ├── Extract model name from request                      │
│    ├── Generate/propagate X-Request-ID                      │
│    ├── Send swap request to coordinator                     │
│    ├── Track in-flight requests                             │
│    └── Return 503 + Retry-After during transitions          │
├─────────────────────────────────────────────────────────────┤
│  Swap Coordinator (single goroutine)                        │
│    ├── Owns all state transitions                           │
│    ├── Serializes swap requests (first-wins)                │
│    ├── Enforces swap cooldown                               │
│    └── Handles startup backoff on failures                  │
├─────────────────────────────────────────────────────────────┤
│  Instance Manager                                           │
│    ├── vLLM process lifecycle (start/stop)                  │
│    ├── Process group management                             │
│    ├── Health + model verification                          │
│    └── Graceful drain coordination                          │
├─────────────────────────────────────────────────────────────┤
│  Proxy                                                      │
│    ├── Forward requests to vLLM                             │
│    ├── Handle SSE streaming                                 │
│    ├── Optional model name rewriting                        │
│    └── Pass through headers (including X-Request-ID)        │
├─────────────────────────────────────────────────────────────┤
│  Config                                                     │
│    ├── Model definitions                                    │
│    ├── Server settings                                      │
│    └── vLLM defaults                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  vLLM Process     │
                    │  (process group)  │
                    └───────────────────┘
```

---

## Configuration

Configuration is provided via a YAML file.

### Example Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: 30s
  write_timeout: 300s  # Long for streaming responses

vllm:
  port: 8000                    # Internal vLLM port
  binary: "uvx"                 # Launcher for vLLM (e.g. "uvx" to run `uvx vllm ...`, or "vllm")
  startup_timeout: 300s         # Max time to wait for model load
  shutdown_timeout: 30s         # Max time to wait for graceful stop
  drain_timeout: 60s            # Max time to wait for in-flight requests
  swap_cooldown: 30s            # Min time between swaps (prevents thrash)
  swap_wait_timeout: 60s        # How long triggering request waits for swap

  # Default arguments applied to all models (can be overridden per-model)
  defaults:
    gpu_memory_utilization: 0.9
    dtype: "auto"
    max_model_len: null         # null = use model default

  # Default environment variables (can be overridden per-model)
  default_env: {}

behavior:
  default_model: null           # Optional: preload on startup
  rewrite_model_name: false     # Rewrite responses to use requested model name

models:
  llama-3-70b:
    path: "/models/Meta-Llama-3-70B-Instruct"
    tensor_parallel_size: 4
    max_model_len: 8192
    env:
      CUDA_VISIBLE_DEVICES: "0,1,2,3"

  llama-3-8b:
    path: "/models/Meta-Llama-3-8B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 8192
    env:
      CUDA_VISIBLE_DEVICES: "0"

  mistral-7b:
    path: "/models/Mistral-7B-Instruct-v0.2"
    tensor_parallel_size: 1
    max_model_len: 32768
    gpu_memory_utilization: 0.85  # Override default
    env:
      CUDA_VISIBLE_DEVICES: "1"

  # Aliases: multiple names can point to same config
  gpt-4: 
    alias: llama-3-70b
  
  gpt-3.5-turbo:
    alias: llama-3-8b
```

### Configuration Schema

#### `server`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Bind address |
| `port` | int | `8080` | Listen port |
| `read_timeout` | duration | `30s` | HTTP read timeout |
| `write_timeout` | duration | `300s` | HTTP write timeout |
| `log_requests` | bool | `true` | Emit per-request logs |

#### `vllm`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | int | `8000` | Port for vLLM to listen on |
| `binary` | string | `"uvx"` | vLLM launcher (use `"uvx"` to run `uvx vllm ...`, or `"vllm"`) |
| `startup_timeout` | duration | `300s` | Max wait for vLLM to become ready |
| `shutdown_timeout` | duration | `30s` | Max wait for vLLM process to exit |
| `drain_timeout` | duration | `60s` | Max wait for in-flight requests before swap |
| `swap_cooldown` | duration | `30s` | Minimum time between model swaps |
| `swap_wait_timeout` | duration | `60s` | Max time triggering request waits for swap |
| `defaults` | object | `{}` | Default vLLM arguments |
| `default_env` | object | `{}` | Default environment variables |

#### `behavior`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_model` | string | `null` | Model to preload on startup (optional) |
| `rewrite_model_name` | bool | `false` | Rewrite response `model` field to match request |

#### `models.<name>`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes* | Path to model weights (local or HuggingFace ID) |
| `alias` | string | No | Reference another model config by name |
| `tensor_parallel_size` | int | No | Number of GPUs for tensor parallelism |
| `pipeline_parallel_size` | int | No | Number of GPUs for pipeline parallelism |
| `max_model_len` | int | No | Maximum sequence length |
| `gpu_memory_utilization` | float | No | GPU memory fraction (0.0-1.0) |
| `dtype` | string | No | Data type: "auto", "float16", "bfloat16" |
| `quantization` | string | No | Quantization method: "awq", "gptq", etc. |
| `extra_args` | []string | No | Additional CLI arguments for vLLM |
| `env` | object | No | Environment variables (merged with default_env) |

*Required unless `alias` is specified.

### Configuration Validation

On startup, Jukebox validates the configuration and fails fast with clear errors:

| Validation | Error |
|------------|-------|
| Alias points to non-existent model | `alias 'gpt-4' references unknown model 'llama-3-70b'` |
| Alias chain (alias → alias) | `alias 'gpt-4' cannot reference another alias 'gpt-3.5-turbo'` |
| Alias loop | `alias loop detected: gpt-4 → llama-3-70b → gpt-4` |
| Missing `path` on non-alias | `model 'llama-3-70b' requires 'path' field` |
| Invalid `gpu_memory_utilization` | `gpu_memory_utilization must be between 0.0 and 1.0` |
| Invalid duration | `invalid duration 'startup_timeout': cannot parse "5 minutes"` |
| Unknown field | `unknown field 'typo_field' in model 'llama-3-70b'` |
| `default_model` not in models | `default_model 'unknown' not found in models` |
| Duplicate model names | `duplicate model name 'llama-3-70b'` |
| Empty models | `at least one model must be configured` |
| Invalid port range | `port must be between 1 and 65535` |

---

## API Compatibility Matrix

### Supported Endpoints (with model switching)

These endpoints trigger model switching logic:

| Endpoint | Description | Model Switching |
|----------|-------------|-----------------|
| `POST /v1/responses` | OpenAI Responses API | ✅ Yes |
| `POST /v1/chat/completions` | Chat completions | ✅ Yes |
| `POST /v1/completions` | Legacy completions | ✅ Yes |

### Supported Endpoints (pass-through)

These endpoints are proxied to vLLM without model switching:

| Endpoint | Description | Notes |
|----------|-------------|-------|
| `GET /v1/models` | List models | Returns Jukebox's configured models |
| `POST /v1/embeddings` | Embeddings | Pass-through if vLLM supports it |
| `POST /v1/tokenize` | Tokenization | Pass-through |
| `POST /v1/detokenize` | Detokenization | Pass-through |

### Not Supported

| Endpoint | Reason |
|----------|--------|
| `POST /v1/audio/*` | Requires specialized models; out of scope |
| `POST /v1/images/*` | Requires specialized models; out of scope |
| `POST /v1/files` | Stateful; requires storage |
| `POST /v1/fine-tuning/*` | Not applicable to inference |
| `GET /v1/assistants/*` | Stateful; requires storage |

Unsupported endpoints return `501 Not Implemented` with a clear error message.

---

## Swap Coordinator

The swap coordinator is a single goroutine that owns all state transitions, eliminating race conditions.

### Design

```go
type SwapRequest struct {
    Model    string
    Response chan SwapResult
}

type SwapResult struct {
    Ready bool
    Error error
}

type Coordinator struct {
    requests     chan SwapRequest
    state        State
    currentModel string
    lastSwapTime time.Time
    failureCount int
}

func (c *Coordinator) Run(ctx context.Context) {
    for {
        select {
        case req := <-c.requests:
            result := c.handleSwapRequest(req.Model)
            req.Response <- result
        case <-ctx.Done():
            return
        }
    }
}
```

### Swap Request Handling

```
Request for model X arrives
        │
        ▼
┌─────────────────────────────────┐
│ currentModel == X && Ready?     │
│                                 │
│  Yes → Return Ready immediately │
│  No  → Continue                 │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ State == Starting/Stopping?     │
│                                 │
│  Yes → Return ErrSwapInProgress │
│  No  → Continue                 │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Time since last swap < cooldown?│
│                                 │
│  Yes → Return ErrCooldown       │
│  No  → Continue                 │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Perform swap:                   │
│  1. Set state = Stopping        │
│  2. Drain in-flight requests    │
│  3. Stop vLLM (process group)   │
│  4. Set state = Starting        │
│  5. Start vLLM with model X     │
│  6. Verify health + model       │
│  7. Set state = Ready           │
│  8. Update lastSwapTime         │
└─────────────────────────────────┘
        │
        ▼
    Return Ready
```

### Policies

| Policy | Behavior |
|--------|----------|
| **Serialization** | One swap at a time; others wait or get 503 |
| **Winner selection** | First request wins; subsequent get `ErrSwapInProgress` |
| **Cooldown** | No swap within `swap_cooldown` of last swap (prevents thrash) |
| **Triggering request** | Waits up to `swap_wait_timeout` for swap to complete |
| **Other requests** | Immediate 503 + Retry-After during swap |

### Failure Backoff

On repeated startup failures, the coordinator applies exponential backoff:

| Consecutive Failures | Backoff Delay |
|---------------------|---------------|
| 1 | 0s (immediate retry) |
| 2 | 5s |
| 3 | 15s |
| 4 | 30s |
| 5+ | 60s (capped) |

Backoff resets on successful startup.

---

## Instance Manager

### State Machine

```
                    ┌──────────────┐
                    │     Idle     │
                    └──────┬───────┘
                           │ start()
                           ▼
                    ┌──────────────┐
        ┌───────────│   Starting   │
        │           └──────┬───────┘
        │                  │ health + model verified
        │ timeout/error    ▼
        │           ┌──────────────┐
        │           │    Ready     │◄─────────────┐
        │           └──────┬───────┘              │
        │                  │ swap triggered       │
        │                  ▼                      │
        │           ┌──────────────┐              │
        │           │   Stopping   │──────────────┘
        │           └──────┬───────┘   new model started
        │                  │
        └──────────────────┴──────────────────────┐
                                                  │
                                           ┌──────▼───────┐
                                           │    Error     │
                                           └──────────────┘
```

### States

| State | Description | Accepts Requests? |
|-------|-------------|-------------------|
| `Idle` | No vLLM process running | No (triggers start) |
| `Starting` | vLLM process launched, waiting for health | No (503) |
| `Ready` | vLLM healthy, serving requests | Yes |
| `Stopping` | Draining in-flight, stopping process | No (503) |
| `Error` | vLLM failed to start or crashed | No (500) |

### In-Flight Request Tracking

```go
type InFlightTracker struct {
    mu      sync.Mutex
    count   int64
    waiters []chan struct{}
}

func (t *InFlightTracker) Track(ctx context.Context) (done func()) {
    t.mu.Lock()
    t.count++
    t.mu.Unlock()
    
    return func() {
        t.mu.Lock()
        defer t.mu.Unlock()
        t.count--
        if t.count == 0 {
            for _, ch := range t.waiters {
                close(ch)
            }
            t.waiters = nil
        }
    }
}

func (t *InFlightTracker) WaitForDrain(ctx context.Context) error {
    t.mu.Lock()
    if t.count == 0 {
        t.mu.Unlock()
        return nil
    }
    ch := make(chan struct{})
    t.waiters = append(t.waiters, ch)
    t.mu.Unlock()
    
    select {
    case <-ch:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}
```

**Critical**: The `done` function must always be called, even on errors or client disconnects. Handlers use:

```go
func (h *Handler) handleRequest(c *fiber.Ctx) error {
    done := h.tracker.Track(c.Context())
    defer done()  // Always cleanup, even on panic
    
    // ... handle request ...
}
```

---

## vLLM Process Management

### Starting vLLM

Command construction:
```bash
vllm serve <model_path> \
  --host 127.0.0.1 \
  --port <vllm.port> \
  --tensor-parallel-size <tp_size> \
  --gpu-memory-utilization <gpu_mem> \
  --max-model-len <max_len> \
  --dtype <dtype> \
  [--quantization <quant>] \
  [<extra_args>...]
```

### Process Group Management

vLLM is started in its own process group to ensure clean termination:

```go
cmd := exec.CommandContext(ctx, vllmBinary, args...)
cmd.SysProcAttr = &syscall.SysProcAttr{
    Setpgid: true,  // Create new process group
}
cmd.Env = buildEnv(modelConfig)  // Merge default_env + model env
```

On termination, kill the entire process group:

```go
func (m *Manager) killProcessGroup(pid int) error {
    // Negative PID = kill process group
    return syscall.Kill(-pid, syscall.SIGKILL)
}
```

This ensures all vLLM worker processes are terminated, preventing orphans.

### Environment Variables

Environment is constructed by merging (later wins):

1. System environment (inherit from Jukebox)
2. `vllm.default_env` from config
3. `models.<name>.env` from config

### Health + Model Verification

After `/health` returns 200, verify the correct model is loaded:

```go
func (m *Manager) verifyModel(ctx context.Context, expectedModel string) error {
    resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/v1/models", m.port))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    var models ModelsResponse
    if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
        return err
    }
    
    for _, model := range models.Data {
        if model.ID == expectedModel || model.ID == m.configs[expectedModel].Path {
            return nil
        }
    }
    
    return fmt.Errorf("expected model %s not found in /v1/models", expectedModel)
}
```

### Stopping vLLM

1. Send `SIGTERM` to process group
2. Wait up to `vllm.shutdown_timeout` for exit
3. If still running, send `SIGKILL` to process group
4. Reap process

### Structured Event Logging

All process lifecycle events are logged with structured fields:

```json
{"time":"2024-01-15T10:30:00Z","level":"info","msg":"vLLM process started","pid":12345,"model":"llama-3-70b","command":"vllm serve /models/..."}
{"time":"2024-01-15T10:30:45Z","level":"info","msg":"vLLM process ready","pid":12345,"model":"llama-3-70b","startup_duration_ms":45000}
{"time":"2024-01-15T11:00:00Z","level":"info","msg":"vLLM process stopped","pid":12345,"model":"llama-3-70b","exit_code":0,"signal":null,"uptime_seconds":1755}
{"time":"2024-01-15T11:00:00Z","level":"error","msg":"vLLM process crashed","pid":12345,"model":"llama-3-70b","exit_code":1,"signal":"SIGSEGV","stderr":"..."}
```

---

## API Endpoints

### OpenAI-Compatible Endpoints

#### `POST /v1/responses`

Responses API endpoint (OpenAI's newer unified API).

**Request**: Standard OpenAI Responses API request
```json
{
  "model": "llama-3-70b",
  "input": "What is the capital of France?",
  "instructions": "You are a helpful assistant."
}
```

**Response**: 
- `200`: Proxied response from vLLM
- `503`: Model switch in progress (Retry-After header included)
- `400`: Unknown model

**Note**: Responses API support depends on vLLM version (v0.8+).

#### `POST /v1/chat/completions`

Chat completion endpoint.

**Request**: Standard OpenAI chat completion request
```json
{
  "model": "llama-3-70b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

**Response**: 
- `200`: Proxied response from vLLM (streaming or non-streaming)
- `503`: Model switch in progress
  ```json
  {
    "error": {
      "message": "Model switch in progress, please retry",
      "type": "service_unavailable",
      "code": "model_switching"
    }
  }
  ```
  Headers: `Retry-After: 5`
- `400`: Unknown model
  ```json
  {
    "error": {
      "message": "Model 'unknown-model' not found in configuration",
      "type": "invalid_request_error",
      "code": "model_not_found"
    }
  }
  ```

#### `POST /v1/completions`

Legacy completion endpoint. Same behavior as chat completions.

#### `GET /v1/models`

List available models (returns Jukebox's configured models, not vLLM's).

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-70b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "jukebox"
    },
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1234567890,
      "owned_by": "jukebox"
    }
  ]
}
```

### Model Name Handling

Model names are **case-sensitive**. Use aliases to handle variations.

When `behavior.rewrite_model_name: true`:
- Response JSON `model` field is rewritten to match the requested model name
- SSE streaming chunks are also rewritten
- Useful when clients expect exact name matching

When `behavior.rewrite_model_name: false` (default):
- Response contains the canonical model name from vLLM
- More transparent; clients see what's actually running

### Jukebox Management Endpoints

#### `GET /health`

Health check for Jukebox itself.

**Response**:
```json
{
  "status": "healthy",
  "accepting_requests": true,
  "vllm": {
    "state": "ready",
    "model": "llama-3-70b",
    "pid": 12345,
    "uptime_seconds": 3600
  }
}
```

| `vllm.state` | `accepting_requests` | HTTP Status |
|--------------|---------------------|-------------|
| `ready` | `true` | 200 |
| `starting` | `false` | 503 |
| `stopping` | `false` | 503 |
| `idle` | `false` | 200 |
| `error` | `false` | 500 |

Load balancers should check `accepting_requests` to determine if traffic should be routed.

#### `GET /status`

Detailed status information.

**Response**:
```json
{
  "state": "ready",
  "accepting_requests": true,
  "current_model": "llama-3-70b",
  "in_flight_requests": 5,
  "uptime_seconds": 3600,
  "swap_cooldown_remaining_seconds": 0,
  "last_swap": {
    "from": "mistral-7b",
    "to": "llama-3-70b", 
    "at": "2024-01-15T10:30:00Z",
    "duration_ms": 45000,
    "triggered_by_request_id": "req_abc123"
  },
  "failure_count": 0,
  "available_models": ["llama-3-70b", "llama-3-8b", "mistral-7b", "gpt-4", "gpt-3.5-turbo"]
}
```

**Security Note**: Bind `/status` to localhost or require authentication in production.

---

## Request ID Tracking

Jukebox generates and propagates request IDs for correlation:

1. Check for incoming `X-Request-ID` header
2. If not present, generate a new UUID
3. Include in all log entries for this request
4. Pass through to vLLM in proxied request
5. Include in response headers

```go
func requestIDMiddleware(c *fiber.Ctx) error {
    requestID := c.Get("X-Request-ID")
    if requestID == "" {
        requestID = uuid.NewString()
    }
    c.Locals("request_id", requestID)
    c.Set("X-Request-ID", requestID)
    return c.Next()
}
```

---

## Request Flow

### Normal Request (Correct Model Loaded)

```
Client                    Jukebox                    vLLM
   │                         │                         │
   │ POST /v1/chat/completions                         │
   │ model: llama-3-70b      │                         │
   │ X-Request-ID: req_123   │                         │
   │────────────────────────►│                         │
   │                         │                         │
   │                    [Coordinator: model matches]   │
   │                    [Track request]                │
   │                         │                         │
   │                         │ POST /v1/chat/completions
   │                         │ X-Request-ID: req_123   │
   │                         │────────────────────────►│
   │                         │                         │
   │                         │◄────────────────────────│
   │                         │      (streaming)        │
   │◄────────────────────────│                         │
   │      (streaming)        │                         │
   │ X-Request-ID: req_123   │                         │
   │                         │                         │
   │                    [Untrack request]              │
   │                         │                         │
```

### Request Triggering Model Swap (Waits)

```
Client                    Jukebox                    vLLM (old)    vLLM (new)
   │                         │                         │               │
   │ POST /v1/chat/completions                         │               │
   │ model: mistral-7b       │                         │               │
   │────────────────────────►│                         │               │
   │                         │                         │               │
   │            [Coordinator: swap needed]             │               │
   │            [Request waits for swap]               │               │
   │                         │                         │               │
   │                    [state = Stopping]             │               │
   │                    [Wait for in-flight]           │               │
   │                         │                         │               │
   │                         │ SIGTERM (process group) │               │
   │                         │────────────────────────►│               │
   │                         │                         X               │
   │                         │                                         │
   │                    [state = Starting]                             │
   │                         │                                         │
   │                         │ vllm serve mistral-7b ─────────────────►│
   │                         │                                         │
   │                         │ GET /health (poll)                      │
   │                         │────────────────────────────────────────►│
   │                         │◄────────────────────────────────────────│
   │                         │         200 OK                          │
   │                         │                                         │
   │                         │ GET /v1/models (verify)                 │
   │                         │────────────────────────────────────────►│
   │                         │◄────────────────────────────────────────│
   │                         │                                         │
   │                    [state = Ready]                                │
   │                    [Request proceeds]                             │
   │                         │                                         │
   │                         │ POST /v1/chat/completions               │
   │                         │────────────────────────────────────────►│
   │                         │◄────────────────────────────────────────│
   │◄────────────────────────│                                         │
   │                         │                                         │
```

### Other Requests During Model Swap

```
Client                    Jukebox                    
   │                         │                         
   │ POST /v1/chat/completions                         
   │ model: any              │                         
   │────────────────────────►│                         
   │                         │                         
   │            [Coordinator: swap in progress]        
   │                         │                         
   │◄────────────────────────│                         
   │  503 Service Unavailable│                         
   │  Retry-After: 5         │                         
   │  X-Request-ID: req_456  │                         
   │                         │                         
```

---

## Error Handling

### Startup Failures

If vLLM fails to start or become healthy within timeout:
- Set `state = Error`
- Increment failure count (for backoff)
- Log error details with request ID
- Subsequent requests return 500 with error details
- Recovery: next request for any model will retry (after backoff)

### Process Crashes

If vLLM process exits unexpectedly while in `Ready` state:
- Detected via process monitoring (wait on PID)
- Set `state = Error`
- Log crash details including exit code and signal
- Next request will attempt restart

### Drain Timeout

If in-flight requests don't complete within `drain_timeout`:
- Log warning with count of remaining requests
- Proceed with SIGKILL to process group
- Those requests will fail with connection errors
- This is a safety valve, not normal operation

### Swap Timeout

If triggering request exceeds `swap_wait_timeout`:
- Return 503 with appropriate error
- Swap continues in background
- Request can retry

---

## Logging

Log format: structured JSON

```json
{
  "time": "2024-01-15T10:30:00Z",
  "level": "info",
  "msg": "Model swap completed",
  "request_id": "req_abc123",
  "from_model": "llama-3-70b",
  "to_model": "mistral-7b",
  "drain_time_ms": 150,
  "startup_time_ms": 45000
}
```

Log levels:
- `debug`: Request details, health poll results, SSE chunks
- `info`: Startup, swap events, state transitions, process lifecycle
- `warn`: Drain timeout, slow health checks, cooldown rejections
- `error`: Process crashes, startup failures, verification failures

---

## Metrics

Prometheus metrics are exposed at `GET /metrics` (when enabled).

Metrics to expose:

```
# Counter: requests by model and status
jukebox_requests_total{model="llama-3-70b", status="200"}

# Counter: model swaps
jukebox_swaps_total{from="llama-3-70b", to="mistral-7b"}

# Counter: swap rejections by reason
jukebox_swap_rejections_total{reason="cooldown"}
jukebox_swap_rejections_total{reason="in_progress"}

# Histogram: swap duration
jukebox_swap_duration_seconds

# Histogram: request duration
jukebox_request_duration_seconds{model="llama-3-70b"}

# Gauge: current state (1 = active)
jukebox_state{state="ready"} 1

# Gauge: in-flight requests
jukebox_in_flight_requests

# Gauge: current model (1 = loaded)
jukebox_current_model{model="llama-3-70b"} 1

# Gauge: consecutive failures
jukebox_consecutive_failures
```

---

## File Structure

```
vllm-jukebox/
├── main.go                 # Entry point, CLI flags
├── config/
│   ├── config.go           # Config structs and loading
│   ├── validate.go         # Config validation
│   └── config_test.go
├── server/
│   ├── server.go           # Fiber app setup
│   └── middleware.go       # Logging, request ID, recovery
├── handlers/
│   ├── openai.go           # /v1/* handlers (responses, chat, completions)
│   ├── health.go           # /health, /status
│   └── handlers_test.go
├── proxy/
│   ├── proxy.go            # HTTP/SSE proxying to vLLM
│   ├── rewrite.go          # Model name rewriting
│   └── proxy_test.go
├── coordinator/
│   ├── coordinator.go      # Swap coordinator loop
│   ├── backoff.go          # Failure backoff logic
│   └── coordinator_test.go
├── instance/
│   ├── manager.go          # vLLM lifecycle management
│   ├── process.go          # Process group handling
│   ├── tracker.go          # In-flight request tracking
│   └── manager_test.go
├── types/
│   └── openai.go           # OpenAI request/response types
├── go.mod
├── go.sum
├── Dockerfile
└── config.example.yaml
```

---

## Design Decisions

These questions were resolved during design review:

| Question | Decision | Rationale |
|----------|----------|-----------|
| Model name normalization | Case-sensitive | Predictable; use aliases for variations |
| Concurrent swap requests | First-wins via coordinator | Simplest; prevents thrash |
| Default model on startup | Optional via `default_model` config | Flexibility for different use cases |
| API key handling | Pass-through to vLLM | Jukebox is not a security boundary; use gateway |
| Response model name | Configurable via `rewrite_model_name` | Support both transparency and client compatibility |
| Triggering request behavior | Waits (up to timeout) | Better UX; not true queuing |

---

## Security Considerations

Jukebox is designed to sit behind an API gateway in production:

1. **No built-in auth**: API keys are passed through to vLLM
2. **DoS via model switching**: Mitigated by `swap_cooldown`, but rate limiting should happen at gateway
3. **Management endpoints**: `/status` should be localhost-only or authenticated
4. **Process isolation**: vLLM runs as same user; consider containerization

Recommended deployment:
```
[Clients] → [API Gateway (auth, rate limit)] → [Jukebox] → [vLLM]
```

---

## Future Considerations

These are explicitly out of scope for v1 but worth noting:

1. **Request queuing**: Full queue during swap (beyond triggering request)
2. **Predictive loading**: Analyze request patterns to preload likely-needed models
3. **Multi-instance**: Run multiple vLLM instances for zero-downtime swaps
4. **LoRA support**: Hot-swap LoRA adapters without full model reload
5. **Model eviction policies**: LRU/LFU for multi-instance setups
6. **Admin API**: Force model loads, clear state, etc.
7. **Websocket support**: For vLLM's native websocket interface
8. **Per-key model allowlists**: Restrict which API keys can request which models
9. **Dynamic Retry-After**: Calculate based on remaining timeout
