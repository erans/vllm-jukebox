# llama.cpp Runtime Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-class `llama-server` (llama.cpp) runtime support to vllm-jukebox, swap mode only. Existing vLLM configs keep working unchanged.

**Architecture:** Introduce a `Runtime` interface (`internal/runtime/`) that abstracts arg-building, binary resolution, and readiness verification. Two implementations: `vllm` (wraps the existing `BuildServeArgsForPort` / `VerifyModelLoaded`) and `llama_cpp` (new). `Manager.Start` and `Manager.VerifyReady` pick the runtime based on a new per-model `runtime` field; default is `vllm`. New top-level `llama_cpp:` config block carries the `binary` path and optional `default_args`.

**Tech Stack:** Go 1.24+, `gopkg.in/yaml.v3`, `net/http/httptest` for verifier tests.

**Design source:** `docs/plans/2026-05-19-llamacpp-runtime-design.md`. Read that first for the rationale.

**Out of scope (do not implement):** MTP, scheduler-mode coverage for llama.cpp, building llama.cpp from source, renaming the `internal/vllm` package or YAML `vllm:` block.

---

## Conventions for executor

- **Workdir:** `/home/eran/work/vllm-jukebox/.worktrees/llamacpp-runtime`. Run all commands from there.
- **TDD:** for every task that adds behavior, write the failing test before the implementation, run it, see it fail, then implement, then see it pass.
- **Commits:** one commit per task, message format `<area>: <what>` (no Conventional Commits scope rules required, just be descriptive). Always end with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- **Verification command** after each task: `go build ./... && go test ./...` from the worktree root. The full suite must stay green.

---

## Task 1 — Config: add `runtime` per-model field and `llama_cpp` top-level block (no validation yet)

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`

**Step 1 — Write failing test.** Append to `internal/config/config_test.go`:

```go
func TestLoad_ModelRuntimeDefaultsToVLLM(t *testing.T) {
    cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    path: "/models/m"
`))
    if err != nil {
        t.Fatalf("load: %v", err)
    }
    if got := cfg.Models["m"].Runtime; got != "" && got != "vllm" {
        t.Fatalf("expected runtime to default to \"\" or \"vllm\", got %q", got)
    }
}

func TestLoad_LlamaCppBlockParses(t *testing.T) {
    cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "/opt/llama.cpp/llama-server"
  default_args: ["--jinja"]
models:
  m:
    runtime: llama_cpp
    path: "unsloth/foo-GGUF:Q4_K_M"
`))
    if err != nil {
        t.Fatalf("load: %v", err)
    }
    if cfg.LlamaCpp == nil {
        t.Fatalf("expected llama_cpp block to parse, got nil")
    }
    if cfg.LlamaCpp.Binary != "/opt/llama.cpp/llama-server" {
        t.Fatalf("binary: %q", cfg.LlamaCpp.Binary)
    }
    if got := cfg.Models["m"].Runtime; got != "llama_cpp" {
        t.Fatalf("expected runtime=llama_cpp, got %q", got)
    }
    if len(cfg.LlamaCpp.DefaultArgs) != 1 || cfg.LlamaCpp.DefaultArgs[0] != "--jinja" {
        t.Fatalf("default_args: %v", cfg.LlamaCpp.DefaultArgs)
    }
}
```

**Step 2 — Run, see failures.**

```
go test ./internal/config/ -run 'TestLoad_(Model|LlamaCpp)' -v
```

Expected: compilation error (field `LlamaCpp` doesn't exist) or unknown-field error (`field llama_cpp not found in type config.Config`).

**Step 3 — Implement.** In `internal/config/config.go`:

a) Add to `Config` struct (after the `Scheduler` line):

```go
LlamaCpp           *LlamaCppConfig        `yaml:"llama_cpp"`
```

b) Add new type next to `SchedulerConfig`:

```go
type LlamaCppConfig struct {
    Binary      string   `yaml:"binary"`
    DefaultArgs []string `yaml:"default_args"`
}
```

c) Add to `ModelConfig` struct (between `Path` and `Alias`):

```go
Runtime              string            `yaml:"runtime"`
```

**Step 4 — Run, see pass.**

```
go test ./internal/config/ -run 'TestLoad_(Model|LlamaCpp)' -v
go build ./...
go test ./...
```

All pass.

**Step 5 — Commit.**

```
config: add runtime field and llama_cpp block (schema only)
```

---

## Task 2 — Config: validation for `runtime` and `llama_cpp.binary`

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`

**Step 1 — Write failing tests.** Append to `internal/config/config_test.go`:

```go
func TestValidate_UnknownRuntimeRejected(t *testing.T) {
    _, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    runtime: "tgi"
    path: "/models/m"
`))
    if err == nil {
        t.Fatalf("expected unknown runtime to be rejected")
    }
    if !strings.Contains(err.Error(), "runtime") {
        t.Fatalf("expected error to mention runtime, got: %v", err)
    }
}

func TestValidate_LlamaCppRequiresBinary(t *testing.T) {
    _, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
`))
    if err == nil {
        t.Fatalf("expected missing llama_cpp.binary to be rejected")
    }
    if !strings.Contains(err.Error(), "llama_cpp.binary") {
        t.Fatalf("expected error to mention llama_cpp.binary, got: %v", err)
    }
}

func TestValidate_LlamaCppOK(t *testing.T) {
    if _, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
`)); err != nil {
        t.Fatalf("expected llama_cpp config with binary to load, got: %v", err)
    }
}

func TestValidate_LlamaCppRejectedInSchedulerMode(t *testing.T) {
    _, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
scheduler:
  port_range_start: 9000
  port_range_end: 9009
models:
  m:
    runtime: llama_cpp
    path: "/models/m.gguf"
    gpus: [0]
    min_free_mem_mb_per_gpu: 1000
`))
    if err == nil {
        t.Fatalf("expected llama_cpp under scheduler mode to be rejected")
    }
    if !strings.Contains(err.Error(), "scheduler") {
        t.Fatalf("expected error to mention scheduler, got: %v", err)
    }
}
```

If `strings` isn't already imported in this test file, add it (`grep ^import internal/config/config_test.go` first; if it's a `import (...)` block just add `"strings"` inside).

**Step 2 — Run, see failures.**

```
go test ./internal/config/ -run 'TestValidate_(Unknown|LlamaCpp)' -v
```

Expected: tests fail because the config currently accepts these without complaint.

**Step 3 — Implement.** In `internal/config/config.go`:

a) Add a constant block near the top (after `Duration` type):

```go
const (
    RuntimeVLLM     = "vllm"
    RuntimeLlamaCpp = "llama_cpp"
)
```

b) In `Validate()`, after the existing model loop (after `validateAliases`), add:

```go
if err := c.validateRuntimes(); err != nil {
    return err
}
```

c) Add the validator method:

```go
func (c *Config) validateRuntimes() error {
    anyLlama := false
    for name, model := range c.Models {
        switch model.Runtime {
        case "", RuntimeVLLM, RuntimeLlamaCpp:
            // ok
        default:
            return fmt.Errorf("model %q: unknown runtime %q (must be one of: vllm, llama_cpp)", name, model.Runtime)
        }
        if model.Runtime == RuntimeLlamaCpp {
            anyLlama = true
            if c.Scheduler != nil {
                return fmt.Errorf("model %q: runtime llama_cpp is not supported in scheduler mode (use swap mode for llama.cpp)", name)
            }
        }
    }
    if anyLlama {
        if c.LlamaCpp == nil || c.LlamaCpp.Binary == "" {
            return fmt.Errorf("at least one model uses runtime: llama_cpp but llama_cpp.binary is not set")
        }
    }
    return nil
}
```

**Step 4 — Run, see pass.**

```
go test ./internal/config/ -v
go build ./...
go test ./...
```

All pass.

**Step 5 — Commit.**

```
config: validate runtime values and require llama_cpp.binary
```

---

## Task 3 — Runtime interface + vLLM adapter (no behavior change)

**Files:**
- Create: `internal/runtime/runtime.go`
- Create: `internal/runtime/vllm.go`
- Create: `internal/runtime/vllm_test.go`

**Step 1 — Write failing test.** Create `internal/runtime/vllm_test.go`:

```go
package runtime_test

import (
    "reflect"
    "testing"

    "vllm-jukebox/internal/config"
    "vllm-jukebox/internal/runtime"
    "vllm-jukebox/internal/vllm"
)

func TestVLLMRuntime_BuildArgsMatchesLegacy(t *testing.T) {
    cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  binary: "vllm"
  defaults:
    gpu_memory_utilization: 0.9
    dtype: auto
models:
  m:
    path: "/models/m"
    tensor_parallel_size: 4
    max_model_len: 8192
    extra_args: ["--enforce-eager"]
`))
    if err != nil {
        t.Fatalf("load: %v", err)
    }

    rt, err := runtime.For(cfg.Models["m"].Runtime)
    if err != nil {
        t.Fatalf("For: %v", err)
    }
    if rt.Name() != "vllm" {
        t.Fatalf("expected vllm, got %q", rt.Name())
    }
    if got := rt.Binary(cfg); got != "vllm" {
        t.Fatalf("Binary: %q", got)
    }

    legacy, err := vllm.BuildServeArgsForPort(cfg, "m", 8000)
    if err != nil {
        t.Fatalf("legacy: %v", err)
    }
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    if !reflect.DeepEqual(legacy, got) {
        t.Fatalf("vllm runtime BuildArgs diverges from legacy.\n  legacy: %v\n     got: %v", legacy, got)
    }
}

func TestRuntimeFor_UnknownName(t *testing.T) {
    if _, err := runtime.For("tgi"); err == nil {
        t.Fatalf("expected unknown runtime to error")
    }
}
```

**Step 2 — Run, see failure.**

```
go test ./internal/runtime/ -v
```

Expected: package doesn't exist; compile error.

**Step 3 — Implement.** Create `internal/runtime/runtime.go`:

```go
package runtime

import (
    "context"
    "fmt"

    "vllm-jukebox/internal/config"
)

// Runtime abstracts the per-backend bits of process startup and readiness checks
// so the Manager can drive either vLLM or llama.cpp behind a common interface.
type Runtime interface {
    Name() string
    Binary(cfg *config.Config) string
    BuildArgs(cfg *config.Config, model config.ModelConfig, resolvedName string, port int) ([]string, error)
    VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error
}

// For returns the Runtime for the given name. Empty string maps to vllm so
// every existing config continues to work unchanged.
func For(name string) (Runtime, error) {
    switch name {
    case "", config.RuntimeVLLM:
        return vllmRuntime{}, nil
    case config.RuntimeLlamaCpp:
        return llamaCppRuntime{}, nil
    default:
        return nil, fmt.Errorf("unknown runtime %q", name)
    }
}
```

Create `internal/runtime/vllm.go`:

```go
package runtime

import (
    "context"

    "vllm-jukebox/internal/config"
    "vllm-jukebox/internal/vllm"
)

type vllmRuntime struct{}

func (vllmRuntime) Name() string { return config.RuntimeVLLM }

func (vllmRuntime) Binary(cfg *config.Config) string { return cfg.VLLM.Binary }

func (vllmRuntime) BuildArgs(cfg *config.Config, _ config.ModelConfig, resolvedName string, port int) ([]string, error) {
    return vllm.BuildServeArgsForPort(cfg, resolvedName, port)
}

func (vllmRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
    return vllm.VerifyModelLoaded(ctx, baseURL, expectedID, expectedPath)
}
```

Create a stub `internal/runtime/llamacpp.go` so `For` compiles. It'll get filled in next task:

```go
package runtime

import (
    "context"
    "errors"

    "vllm-jukebox/internal/config"
)

type llamaCppRuntime struct{}

func (llamaCppRuntime) Name() string { return config.RuntimeLlamaCpp }

func (llamaCppRuntime) Binary(cfg *config.Config) string {
    if cfg.LlamaCpp == nil {
        return ""
    }
    return cfg.LlamaCpp.Binary
}

func (llamaCppRuntime) BuildArgs(cfg *config.Config, model config.ModelConfig, resolvedName string, port int) ([]string, error) {
    return nil, errors.New("llama_cpp runtime: BuildArgs not implemented yet")
}

func (llamaCppRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
    return errors.New("llama_cpp runtime: VerifyModelLoaded not implemented yet")
}
```

**Step 4 — Run, see pass.**

```
go test ./internal/runtime/ -v
go build ./...
go test ./...
```

**Step 5 — Commit.**

```
runtime: introduce Runtime interface and vLLM adapter
```

---

## Task 4 — llama.cpp adapter: arg building

**Files:**
- Modify: `internal/runtime/llamacpp.go`
- Create: `internal/runtime/llamacpp_test.go`

**Step 1 — Write failing tests.** Create `internal/runtime/llamacpp_test.go`:

```go
package runtime_test

import (
    "os"
    "path/filepath"
    "reflect"
    "testing"

    "vllm-jukebox/internal/config"
    "vllm-jukebox/internal/runtime"
)

func loadLlamaCfg(t *testing.T, body string) *config.Config {
    t.Helper()
    cfg, err := config.Load([]byte(body))
    if err != nil {
        t.Fatalf("load: %v", err)
    }
    return cfg
}

func TestLlamaCppRuntime_BuildArgs_HFRepoPath(t *testing.T) {
    cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
    max_model_len: 65536
    extra_args: ["--jinja", "--n-gpu-layers", "all"]
`)
    rt, _ := runtime.For("llama_cpp")
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    want := []string{
        "--host", "127.0.0.1",
        "--port", "8000",
        "--hf-repo", "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
        "--ctx-size", "65536",
        "--jinja", "--n-gpu-layers", "all",
    }
    if !reflect.DeepEqual(got, want) {
        t.Fatalf("args mismatch\n  got: %v\n want: %v", got, want)
    }
}

func TestLlamaCppRuntime_BuildArgs_LocalGGUFPath(t *testing.T) {
    dir := t.TempDir()
    gguf := filepath.Join(dir, "model.gguf")
    if err := os.WriteFile(gguf, []byte("x"), 0o644); err != nil {
        t.Fatalf("write: %v", err)
    }
    cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "`+gguf+`"
`)
    rt, _ := runtime.For("llama_cpp")
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    want := []string{"--host", "127.0.0.1", "--port", "8000", "-m", gguf}
    if !reflect.DeepEqual(got, want) {
        t.Fatalf("args mismatch\n  got: %v\n want: %v", got, want)
    }
}

func TestLlamaCppRuntime_BuildArgs_DotGGUFSuffixUsesLocal(t *testing.T) {
    // File doesn't exist on disk, but `.gguf` suffix should still pick `-m`.
    cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "model-q4.gguf"
`)
    rt, _ := runtime.For("llama_cpp")
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    if got[len(got)-2] != "-m" || got[len(got)-1] != "model-q4.gguf" {
        t.Fatalf("expected `-m model-q4.gguf` tail, got: %v", got)
    }
}

func TestLlamaCppRuntime_BuildArgs_DefaultArgsPrependExtraArgs(t *testing.T) {
    cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
  default_args: ["--metrics", "--no-webui"]
models:
  m:
    runtime: llama_cpp
    path: "org/repo-GGUF"
    extra_args: ["--jinja"]
`)
    rt, _ := runtime.For("llama_cpp")
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    // default_args must appear before extra_args, both after path flags.
    wantTail := []string{"--metrics", "--no-webui", "--jinja"}
    if !reflect.DeepEqual(got[len(got)-len(wantTail):], wantTail) {
        t.Fatalf("expected tail %v, got %v (full: %v)", wantTail, got[len(got)-len(wantTail):], got)
    }
}

func TestLlamaCppRuntime_BuildArgs_IgnoresVLLMOnlyFields(t *testing.T) {
    tp := 4
    gmu := 0.9
    cfg := loadLlamaCfg(t, `
vllm:
  port: 8000
  binary: "vllm"
llama_cpp:
  binary: "llama-server"
models:
  m:
    runtime: llama_cpp
    path: "org/repo-GGUF"
    dtype: "auto"
    quantization: "awq"
`)
    // Mutate in-memory to set the pointer fields too (yaml doesn't carry them above).
    mc := cfg.Models["m"]
    mc.TensorParallelSize = &tp
    mc.GPUMemoryUtilization = &gmu
    cfg.Models["m"] = mc

    rt, _ := runtime.For("llama_cpp")
    got, err := rt.BuildArgs(cfg, cfg.Models["m"], "m", 8000)
    if err != nil {
        t.Fatalf("BuildArgs: %v", err)
    }
    for _, banned := range []string{"--tensor-parallel-size", "--gpu-memory-utilization", "--dtype", "--quantization"} {
        for _, a := range got {
            if a == banned {
                t.Fatalf("vLLM-only flag %q leaked into llama.cpp args: %v", banned, got)
            }
        }
    }
}
```

**Step 2 — Run, see failures.**

```
go test ./internal/runtime/ -run TestLlamaCppRuntime_BuildArgs -v
```

Expected: all four fail (current stub returns the "not implemented" error).

**Step 3 — Implement.** Rewrite `internal/runtime/llamacpp.go`:

```go
package runtime

import (
    "context"
    "fmt"
    "log/slog"
    "os"
    "strconv"
    "strings"

    "vllm-jukebox/internal/config"
)

type llamaCppRuntime struct{}

func (llamaCppRuntime) Name() string { return config.RuntimeLlamaCpp }

func (llamaCppRuntime) Binary(cfg *config.Config) string {
    if cfg.LlamaCpp == nil {
        return ""
    }
    return cfg.LlamaCpp.Binary
}

func (llamaCppRuntime) BuildArgs(cfg *config.Config, model config.ModelConfig, resolvedName string, port int) ([]string, error) {
    if model.Path == "" {
        return nil, fmt.Errorf("model %q: path is required for llama_cpp runtime", resolvedName)
    }
    warnIgnoredVLLMFields(resolvedName, model)

    args := []string{
        "--host", "127.0.0.1",
        "--port", strconv.Itoa(port),
    }

    if looksLikeLocalGGUF(model.Path) {
        args = append(args, "-m", model.Path)
    } else {
        args = append(args, "--hf-repo", model.Path)
    }

    if model.MaxModelLen != nil {
        args = append(args, "--ctx-size", strconv.Itoa(*model.MaxModelLen))
    }

    if cfg.LlamaCpp != nil {
        args = append(args, cfg.LlamaCpp.DefaultArgs...)
    }
    args = append(args, model.ExtraArgs...)

    return args, nil
}

func (llamaCppRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
    return fmt.Errorf("llama_cpp runtime: VerifyModelLoaded not implemented yet")
}

// looksLikeLocalGGUF matches the heuristic from scripts/llama-server-wrapper.sh:
// a value is treated as a local file if it exists on disk, is an absolute or
// explicitly-relative path, or carries a .gguf suffix. Anything else is
// assumed to be a Hugging Face repo coordinate like "org/repo[:quant]".
func looksLikeLocalGGUF(p string) bool {
    if strings.HasPrefix(p, "/") || strings.HasPrefix(p, "./") || strings.HasPrefix(p, "../") {
        return true
    }
    if strings.HasSuffix(strings.ToLower(p), ".gguf") {
        return true
    }
    if _, err := os.Stat(p); err == nil {
        return true
    }
    return false
}

func warnIgnoredVLLMFields(name string, m config.ModelConfig) {
    ignored := []string{}
    if m.TensorParallelSize != nil {
        ignored = append(ignored, "tensor_parallel_size")
    }
    if m.PipelineParallelSize != nil {
        ignored = append(ignored, "pipeline_parallel_size")
    }
    if m.GPUMemoryUtilization != nil {
        ignored = append(ignored, "gpu_memory_utilization")
    }
    if m.DType != "" {
        ignored = append(ignored, "dtype")
    }
    if m.Quantization != "" {
        ignored = append(ignored, "quantization")
    }
    if len(ignored) > 0 {
        slog.Warn("llama_cpp runtime: ignoring vLLM-only model fields",
            "model", name,
            "fields", ignored,
            "hint", "translate equivalents into extra_args (e.g. --n-gpu-layers, --split-mode)")
    }
}
```

**Step 4 — Run, see pass.**

```
go test ./internal/runtime/ -v
go build ./...
go test ./...
```

**Step 5 — Commit.**

```
runtime: implement llama_cpp arg building
```

---

## Task 5 — llama.cpp adapter: model-loaded verification

**Files:**
- Modify: `internal/runtime/llamacpp.go`
- Modify: `internal/runtime/llamacpp_test.go`

llama-server's `/v1/models` typically returns the GGUF basename or the repo coordinate as `id`. We accept any of: `expectedID`, `expectedPath`, or `filepath.Base(expectedPath)`.

**Step 1 — Write failing tests.** Append to `internal/runtime/llamacpp_test.go`:

```go
import (
    "context"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "time"
)

func newModelsServer(t *testing.T, ids ...string) *httptest.Server {
    t.Helper()
    return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path != "/v1/models" {
            http.NotFound(w, r)
            return
        }
        type m struct{ ID string `json:"id"` }
        data := []m{}
        for _, id := range ids {
            data = append(data, m{ID: id})
        }
        _ = json.NewEncoder(w).Encode(map[string]any{"data": data})
    }))
}

func TestLlamaCppRuntime_VerifyModelLoaded_MatchesExpectedID(t *testing.T) {
    s := newModelsServer(t, "qwen36-35b-a3b")
    defer s.Close()
    rt, _ := runtime.For("llama_cpp")
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    if err := rt.VerifyModelLoaded(ctx, s.URL, "qwen36-35b-a3b", "unsloth/qwen-GGUF"); err != nil {
        t.Fatalf("VerifyModelLoaded: %v", err)
    }
}

func TestLlamaCppRuntime_VerifyModelLoaded_MatchesPathBasename(t *testing.T) {
    s := newModelsServer(t, "Qwen3.6-35B-A3B-Q4_K_M.gguf")
    defer s.Close()
    rt, _ := runtime.For("llama_cpp")
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    if err := rt.VerifyModelLoaded(ctx, s.URL, "qwen36", "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"); err != nil {
        t.Fatalf("VerifyModelLoaded: %v", err)
    }
}

func TestLlamaCppRuntime_VerifyModelLoaded_RejectsMismatch(t *testing.T) {
    s := newModelsServer(t, "wrong-model")
    defer s.Close()
    rt, _ := runtime.For("llama_cpp")
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()
    if err := rt.VerifyModelLoaded(ctx, s.URL, "qwen", "/models/qwen.gguf"); err == nil {
        t.Fatalf("expected mismatch to error")
    }
}
```

If the test file already has an `import (...)` block, merge into that one instead of adding a second one.

**Step 2 — Run, see failures.**

```
go test ./internal/runtime/ -run TestLlamaCppRuntime_VerifyModelLoaded -v
```

Expected: stub error.

**Step 3 — Implement.** Replace the stub `VerifyModelLoaded` in `internal/runtime/llamacpp.go`:

```go
import (  // merge into existing import block
    "encoding/json"
    "net/http"
    "path/filepath"
)

func (llamaCppRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
    url := strings.TrimRight(baseURL, "/") + "/v1/models"
    client := &http.Client{}

    req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
    if err != nil {
        return err
    }
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    if resp.StatusCode < 200 || resp.StatusCode >= 300 {
        return fmt.Errorf("GET /v1/models returned %s", resp.Status)
    }

    var decoded struct {
        Data []struct {
            ID string `json:"id"`
        } `json:"data"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
        return err
    }

    base := ""
    if expectedPath != "" {
        base = filepath.Base(expectedPath)
    }
    for _, m := range decoded.Data {
        if m.ID == expectedID {
            return nil
        }
        if expectedPath != "" && m.ID == expectedPath {
            return nil
        }
        if base != "" && m.ID == base {
            return nil
        }
    }
    return fmt.Errorf("expected model %q (or path %q) not found in /v1/models", expectedID, expectedPath)
}
```

**Step 4 — Run, see pass.**

```
go test ./internal/runtime/ -v
go build ./...
go test ./...
```

**Step 5 — Commit.**

```
runtime: implement llama_cpp /v1/models verification
```

---

## Task 6 — Wire `Runtime` into `Manager.Start` and `Manager.VerifyReady`

**Files:**
- Modify: `internal/vllm/runtime.go`

This swaps in the Runtime interface at the two existing call sites. All existing tests should keep passing because the empty/`vllm` runtime delegates to the same functions.

**Step 1 — Implement.** In `internal/vllm/runtime.go`:

a) Add import: `rt "vllm-jukebox/internal/runtime"`. (Aliased to avoid shadowing the existing `Runtime` filename and to make call sites read `rt.For(...)`.)

b) Replace this block inside `Manager.Start` (currently around line 108–119):

```go
    args, err := BuildServeArgsForPort(m.cfg, modelName, m.port)
    if err != nil {
        return 0, err
    }

    _, modelCfg, err := m.cfg.ResolveModel(modelName)
    if err != nil {
        return 0, err
    }

    bin, binArgs := wrapBinaryArgs(m.cfg.VLLM.Binary, args)
```

with:

```go
    resolvedName, modelCfg, err := m.cfg.ResolveModel(modelName)
    if err != nil {
        return 0, err
    }

    runtimeImpl, err := rt.For(modelCfg.Runtime)
    if err != nil {
        return 0, err
    }

    args, err := runtimeImpl.BuildArgs(m.cfg, modelCfg, resolvedName, m.port)
    if err != nil {
        return 0, err
    }

    bin, binArgs := wrapBinaryArgs(runtimeImpl.Binary(m.cfg), args)
```

c) In `Manager.VerifyReady`, replace the final return (currently around line 351–356):

```go
    _, modelCfg, err := m.cfg.ResolveModel(expectedModel)
    if err != nil {
        return err
    }

    return VerifyModelLoaded(ctx, base, expectedModel, modelCfg.Path)
```

with:

```go
    _, modelCfg, err := m.cfg.ResolveModel(expectedModel)
    if err != nil {
        return err
    }

    runtimeImpl, err := rt.For(modelCfg.Runtime)
    if err != nil {
        return err
    }
    return runtimeImpl.VerifyModelLoaded(ctx, base, expectedModel, modelCfg.Path)
```

**Step 2 — Verify build + tests.**

```
go build ./...
go test ./...
```

All packages must stay green. In particular:
- `internal/vllm` tests must still pass (vllm runtime path is the default and produces identical args).
- `internal/runtime` tests must still pass.

If `go vet` complains about an import cycle, that's a sign: `internal/vllm` now imports `internal/runtime`, and `internal/runtime/vllm.go` imports `internal/vllm`. That's a cycle. Resolution: keep `internal/vllm` importing `internal/runtime` only at the `Manager` level, but the `runtime/vllm.go` adapter still imports `internal/vllm`. Since both packages refer to each other this WILL cycle. Detection step is below.

**Step 3 — Cycle check + fix if needed.**

```
go build ./internal/...
```

If you see `import cycle not allowed`, the minimal fix is to move the two pure functions the vllm adapter calls (`BuildServeArgsForPort` and `VerifyModelLoaded`) into a new leaf package, e.g. `internal/vllm/cli`, and have both `internal/vllm` and `internal/runtime/vllm.go` import that. Concretely:

- Create `internal/vllmcli/` containing the contents of `manager.go`'s arg builders and `health.go`'s `VerifyModelLoaded`.
- Re-export from `internal/vllm/manager.go` and `internal/vllm/health.go` as thin shims (`func BuildServeArgs(...) = vllmcli.BuildServeArgs(...)`) so existing tests keep compiling.
- Update `internal/runtime/vllm.go` to import `vllmcli` instead.

Only do this if the cycle actually fires. Many Go layouts skirt it because the consumer (`Manager.Start`) and the adapter (`runtime/vllm.go`) sit in different packages; the only cycle risk is `internal/vllm` ↔ `internal/runtime`. **Try the simpler wiring first; only break the cycle if needed.**

**Step 4 — Commit.**

```
vllm: route Manager.Start and VerifyReady through Runtime interface
```

---

## Task 7 — Example config

**Files:**
- Create: `configs/qwen36-35b-a3b-llamacpp.yaml`

**Step 1 — Write the file:**

```yaml
server:
  host: "0.0.0.0"
  port: 8080

vllm:
  port: 8000
  startup_timeout: 1800s
  shutdown_timeout: 30s
  drain_timeout: 60s
  swap_cooldown: 30s
  swap_wait_timeout: 60s
  log_dir: "/tmp/vllm-logs"
  log_max_size_mb: 100
  log_max_files: 3

llama_cpp:
  binary: "llama-server"

behavior:
  rewrite_model_name: true

models:
  qwen36-35b-a3b:
    runtime: llama_cpp
    # Hugging Face repo coordinate; llama-server downloads on first start.
    # Pick a quant tag that fits your VRAM budget.
    path: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
    max_model_len: 65536
    extra_args:
      - "--jinja"
      - "--flash-attn"
      - "on"
      - "--n-gpu-layers"
      - "all"
      - "--split-mode"
      - "layer"
      - "--cache-type-k"
      - "q8_0"
      - "--cache-type-v"
      - "q8_0"
      - "--parallel"
      - "4"
      - "--batch-size"
      - "2048"
      - "--ubatch-size"
      - "512"
```

**Step 2 — Verify it parses.** Add a quick check via the existing `cmd/jukebox` entrypoint:

```
go build -o /tmp/jukebox-validate ./cmd/jukebox
/tmp/jukebox-validate -config configs/qwen36-35b-a3b-llamacpp.yaml -validate 2>&1 | head -20 || true
```

If `cmd/jukebox` lacks a `-validate` flag (likely — check `cmd/jukebox/main.go` first), instead run a small inline test from the worktree root:

```
go run ./cmd/jukebox -config configs/qwen36-35b-a3b-llamacpp.yaml -dry-run 2>&1 | head -5 || true
```

If neither flag exists, fall back to a tiny Go script: don't add one; just trust the unit-test coverage from Tasks 1–6 plus a manual `go test ./internal/config/... -run TestLoad` against an inline fixture. If you want belt-and-suspenders, write an extra test in `internal/config/config_test.go` that loads this file:

```go
func TestLoad_QwenLlamaCppExampleParses(t *testing.T) {
    data, err := os.ReadFile("../../configs/qwen36-35b-a3b-llamacpp.yaml")
    if err != nil {
        t.Skipf("example config not readable: %v", err)
    }
    if _, err := config.Load(data); err != nil {
        t.Fatalf("expected example to parse, got: %v", err)
    }
}
```

(`os` is likely already imported; if not, add it.)

**Step 3 — Run.**

```
go test ./internal/config/ -v
```

**Step 4 — Commit.**

```
configs: add Qwen3.6 llama_cpp example
```

---

## Task 8 — Docs: update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Step 1 — Update the Project Overview paragraph.** Change:

> vLLM Jukebox is an OpenAI-compatible HTTP server written in Go that orchestrates vLLM (Large Language Model inference server) instances.

to:

> vLLM Jukebox is an OpenAI-compatible HTTP server written in Go that orchestrates LLM inference backends. The default backend is vLLM; `llama-server` (llama.cpp) is supported in swap mode as an alternative per-model runtime.

**Step 2 — Add a new "Runtimes" section** after "Configuration" and before "GPU Power Limits":

```markdown
## Runtimes

Each model picks an inference runtime via the `runtime` field (default: `vllm`).
Two runtimes are supported today:

- `vllm` — the default. Uses the `vllm:` block's `binary` and the existing
  vLLM CLI shape (`serve <path> --host ... --tensor-parallel-size ...`).
- `llama_cpp` — runs llama.cpp's `llama-server`. Requires the top-level
  `llama_cpp:` block.

```yaml
llama_cpp:
  binary: "llama-server"   # required if any model uses runtime: llama_cpp
  default_args: []         # optional; prepended to every llama_cpp launch

models:
  my-gguf:
    runtime: llama_cpp
    path: "org/repo-GGUF:Q4_K_M"   # treated as --hf-repo
    # or: path: "/models/model.gguf" # treated as -m
    max_model_len: 32768            # mapped to --ctx-size
    extra_args:
      - "--n-gpu-layers"
      - "all"
      - "--jinja"
```

Field mapping for `runtime: llama_cpp`:

| YAML field            | llama-server flag                                        |
|-----------------------|----------------------------------------------------------|
| `path`                | `--hf-repo <p>` if not a local path, else `-m <p>`       |
| `max_model_len`       | `--ctx-size <n>`                                         |
| `extra_args`          | passthrough (after `llama_cpp.default_args`)             |
| `tensor_parallel_size`, `pipeline_parallel_size`, `gpu_memory_utilization`, `dtype`, `quantization` | ignored (with a startup warning) — translate to the equivalent llama.cpp flags via `extra_args` |

The `--hf-repo` vs `-m` choice mirrors the heuristic in
`scripts/llama-server-wrapper.sh`: a value is treated as a local file when it
exists on disk, is an absolute or `./`/`../` path, or ends in `.gguf`.

Scheduler mode does not yet support `runtime: llama_cpp`; the config
validator rejects that combination.
```

**Step 3 — Commit.**

```
docs: document llama_cpp runtime in CLAUDE.md
```

---

## Final verification

After Task 8:

```
go build ./...
go test ./...
make check          # if your dev env has golangci-lint installed
```

All green. Then surface the branch back to the user (no auto-push, no PR; the
user has explicit policies on those).

## Notes for the executor

- If any test in the existing suite (`internal/vllm/`, `internal/jukebox/`,
  `internal/httpserver/`) breaks after Task 6, **stop and report** rather than
  patching tests to fit. The wiring is supposed to be transparent for the
  vLLM path; a failure means the adapter diverges from the legacy behavior.
- If the example config in Task 7 fails to parse for a reason unrelated to
  the schema (e.g. a typo), fix the typo. If it fails because validation is
  too strict for valid llama.cpp use cases, **stop and report** — that's a
  design issue, not an executor decision.
- Do not implement MTP or scheduler-mode hooks even if it seems "almost
  free" — they were explicitly cut from v1.
