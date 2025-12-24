# vLLM Log File Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add rotating log file support for vLLM process output with per-model overrides.

**Architecture:** Add config fields for log directory and rotation settings. Create a RotatingFileWriter that handles size-based rotation. Integrate with runtime.go using io.MultiWriter to tee output to both tail buffer and log file.

**Tech Stack:** Go standard library (os, io, sync, filepath)

---

### Task 1: Add Log Config Fields to VLLMConfig

**Files:**
- Modify: `internal/config/config.go:50-61`

**Step 1: Add log fields to VLLMConfig struct**

Add these fields after `DefaultEnv`:

```go
type VLLMConfig struct {
	Port            int      `yaml:"port"`
	Binary          string   `yaml:"binary"`
	StartupTimeout  Duration `yaml:"startup_timeout"`
	ShutdownTimeout Duration `yaml:"shutdown_timeout"`
	DrainTimeout    Duration `yaml:"drain_timeout"`
	SwapCooldown    Duration `yaml:"swap_cooldown"`
	SwapWaitTimeout Duration `yaml:"swap_wait_timeout"`

	Defaults   VLLMDefaults      `yaml:"defaults"`
	DefaultEnv map[string]string `yaml:"default_env"`

	LogDir       string `yaml:"log_dir"`
	LogMaxSizeMB int    `yaml:"log_max_size_mb"`
	LogMaxFiles  int    `yaml:"log_max_files"`
}
```

**Step 2: Add defaults in applyDefaults()**

Add after the `DefaultEnv` default (around line 157):

```go
	if c.VLLM.LogMaxSizeMB == 0 {
		c.VLLM.LogMaxSizeMB = 50
	}
	if c.VLLM.LogMaxFiles == 0 {
		c.VLLM.LogMaxFiles = 5
	}
```

**Step 3: Run tests**

Run: `go test ./internal/config/...`
Expected: PASS

**Step 4: Commit**

```bash
git add internal/config/config.go
git commit -m "feat(config): add vLLM log file configuration fields"
```

---

### Task 2: Add LogFile Field to ModelConfig

**Files:**
- Modify: `internal/config/config.go:82-98`

**Step 1: Add log_file field to ModelConfig struct**

Add after `PowerLimits`:

```go
type ModelConfig struct {
	Path                 string            `yaml:"path"`
	Alias                string            `yaml:"alias"`
	GPUs                 []int             `yaml:"gpus"`
	MinFreeMemMBPerGPU   *int              `yaml:"min_free_mem_mb_per_gpu"`
	Pinned               *bool             `yaml:"pinned"`
	TensorParallelSize   *int              `yaml:"tensor_parallel_size"`
	PipelineParallelSize *int              `yaml:"pipeline_parallel_size"`
	MaxModelLen          *int              `yaml:"max_model_len"`
	GPUMemoryUtilization *float64          `yaml:"gpu_memory_utilization"`
	DType                string            `yaml:"dtype"`
	Quantization         string            `yaml:"quantization"`
	ExtraArgs            []string          `yaml:"extra_args"`
	Env                  map[string]string `yaml:"env"`
	PowerLimit           *int              `yaml:"power_limit"`
	PowerLimits          map[int]int       `yaml:"power_limits"`
	LogFile              string            `yaml:"log_file"`
}
```

**Step 2: Run tests**

Run: `go test ./internal/config/...`
Expected: PASS

**Step 3: Commit**

```bash
git add internal/config/config.go
git commit -m "feat(config): add per-model log_file override"
```

---

### Task 3: Add Validation for Log Config

**Files:**
- Modify: `internal/config/config.go` (Validate function)
- Test: `internal/config/config_test.go`

**Step 1: Add validation in Validate() function**

Add after power limit validation (around line 193):

```go
	// Validate log settings
	if err := c.validateLogSettings(); err != nil {
		return err
	}
```

**Step 2: Add validateLogSettings function**

Add after validateModelPowerLimits:

```go
func (c *Config) validateLogSettings() error {
	if c.VLLM.LogMaxSizeMB < 0 {
		return fmt.Errorf("vllm.log_max_size_mb must be >= 0, got %d", c.VLLM.LogMaxSizeMB)
	}
	if c.VLLM.LogMaxFiles < 0 {
		return fmt.Errorf("vllm.log_max_files must be >= 0, got %d", c.VLLM.LogMaxFiles)
	}
	return nil
}
```

**Step 3: Add test for validation**

Add to `internal/config/config_test.go`:

```go
func TestValidate_LogSettings(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "valid log settings",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_dir: /var/log/vllm
  log_max_size_mb: 100
  log_max_files: 10
`,
			wantErr: "",
		},
		{
			name: "negative log_max_size_mb",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_max_size_mb: -1
`,
			wantErr: "log_max_size_mb must be >= 0",
		},
		{
			name: "negative log_max_files",
			yaml: `
models:
  test:
    path: /models/test
vllm:
  log_max_files: -1
`,
			wantErr: "log_max_files must be >= 0",
		},
		{
			name: "per-model log_file",
			yaml: `
models:
  test:
    path: /models/test
    log_file: /var/log/test.log
`,
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Load([]byte(tt.yaml))
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected error containing %q", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
			}
		})
	}
}
```

**Step 4: Run tests**

Run: `go test ./internal/config/... -v`
Expected: PASS

**Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): add validation for log settings"
```

---

### Task 4: Create RotatingFileWriter

**Files:**
- Create: `internal/vllm/logwriter.go`
- Test: `internal/vllm/logwriter_test.go`

**Step 1: Create logwriter.go with RotatingFileWriter**

```go
package vllm

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RotatingFileWriter writes to a file with size-based rotation.
// When the file exceeds maxBytes, it rotates: file.log -> file.log.1, etc.
// Old files beyond maxFiles are deleted.
type RotatingFileWriter struct {
	mu          sync.Mutex
	path        string
	maxBytes    int64
	maxFiles    int
	currentSize int64
	file        *os.File
}

// NewRotatingFileWriter creates a new rotating file writer.
// maxSizeMB is the maximum size in megabytes before rotation.
// maxFiles is the number of rotated files to keep.
func NewRotatingFileWriter(path string, maxSizeMB, maxFiles int) (*RotatingFileWriter, error) {
	// Create parent directories if needed
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory %s: %w", dir, err)
	}

	w := &RotatingFileWriter{
		path:     path,
		maxBytes: int64(maxSizeMB) * 1024 * 1024,
		maxFiles: maxFiles,
	}

	if err := w.openFile(); err != nil {
		return nil, err
	}

	return w, nil
}

func (w *RotatingFileWriter) openFile() error {
	f, err := os.OpenFile(w.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file %s: %w", w.path, err)
	}

	// Get current file size
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return fmt.Errorf("failed to stat log file %s: %w", w.path, err)
	}

	w.file = f
	w.currentSize = info.Size()
	return nil
}

func (w *RotatingFileWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return 0, fmt.Errorf("log file is closed")
	}

	// Check if we need to rotate before writing
	if w.maxBytes > 0 && w.currentSize+int64(len(p)) > w.maxBytes {
		if err := w.rotate(); err != nil {
			// Log rotation failed, but continue writing to current file
			// This is graceful degradation
		}
	}

	n, err := w.file.Write(p)
	w.currentSize += int64(n)
	return n, err
}

func (w *RotatingFileWriter) rotate() error {
	// Close current file
	if w.file != nil {
		w.file.Close()
		w.file = nil
	}

	// Delete oldest file if it exists
	oldest := fmt.Sprintf("%s.%d", w.path, w.maxFiles)
	os.Remove(oldest)

	// Rotate existing files: .4 -> .5, .3 -> .4, etc.
	for i := w.maxFiles - 1; i >= 1; i-- {
		oldPath := fmt.Sprintf("%s.%d", w.path, i)
		newPath := fmt.Sprintf("%s.%d", w.path, i+1)
		os.Rename(oldPath, newPath)
	}

	// Rotate current file to .1
	if err := os.Rename(w.path, w.path+".1"); err != nil && !os.IsNotExist(err) {
		// If rename fails, try to reopen the original file
		return w.openFile()
	}

	// Open new file
	w.currentSize = 0
	return w.openFile()
}

func (w *RotatingFileWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return nil
	}

	err := w.file.Close()
	w.file = nil
	return err
}
```

**Step 2: Create logwriter_test.go**

```go
package vllm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRotatingFileWriter_Write(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	w, err := NewRotatingFileWriter(path, 1, 3) // 1MB max, keep 3 files
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer w.Close()

	// Write some data
	msg := "hello world\n"
	n, err := w.Write([]byte(msg))
	if err != nil {
		t.Fatalf("write failed: %v", err)
	}
	if n != len(msg) {
		t.Fatalf("expected %d bytes written, got %d", len(msg), n)
	}

	// Verify file exists and contains data
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if string(data) != msg {
		t.Fatalf("expected %q, got %q", msg, string(data))
	}
}

func TestRotatingFileWriter_Rotation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	// Use very small max size to trigger rotation
	w, err := NewRotatingFileWriter(path, 0, 3) // 0 MB = will rotate on any write > 0
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	// Manually set maxBytes to something small for testing
	w.maxBytes = 100

	// Write enough to trigger rotation
	for i := 0; i < 5; i++ {
		msg := strings.Repeat("x", 50) + "\n"
		_, err := w.Write([]byte(msg))
		if err != nil {
			t.Fatalf("write %d failed: %v", i, err)
		}
	}
	w.Close()

	// Check that rotated files exist
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("main log file should exist")
	}
	if _, err := os.Stat(path + ".1"); os.IsNotExist(err) {
		t.Fatal("rotated file .1 should exist")
	}
}

func TestRotatingFileWriter_CreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "subdir", "nested", "test.log")

	w, err := NewRotatingFileWriter(path, 50, 5)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer w.Close()

	// Verify directory was created
	if _, err := os.Stat(filepath.Dir(path)); os.IsNotExist(err) {
		t.Fatal("directory should have been created")
	}
}

func TestRotatingFileWriter_MaxFilesRespected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	w, err := NewRotatingFileWriter(path, 0, 2) // keep only 2 rotated files
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	w.maxBytes = 50 // small size for testing

	// Write enough to trigger multiple rotations
	for i := 0; i < 10; i++ {
		msg := strings.Repeat("y", 60) + "\n"
		w.Write([]byte(msg))
	}
	w.Close()

	// .1 and .2 should exist, .3 should not
	if _, err := os.Stat(path + ".1"); os.IsNotExist(err) {
		t.Fatal(".1 should exist")
	}
	if _, err := os.Stat(path + ".2"); os.IsNotExist(err) {
		t.Fatal(".2 should exist")
	}
	if _, err := os.Stat(path + ".3"); !os.IsNotExist(err) {
		t.Fatal(".3 should NOT exist")
	}
}
```

**Step 3: Run tests**

Run: `go test ./internal/vllm/... -v -run TestRotating`
Expected: PASS

**Step 4: Commit**

```bash
git add internal/vllm/logwriter.go internal/vllm/logwriter_test.go
git commit -m "feat(vllm): add RotatingFileWriter for log files"
```

---

### Task 5: Add Helper to Resolve Log Path

**Files:**
- Modify: `internal/config/config.go`

**Step 1: Add ResolveLogPath method to Config**

Add after ResolveModel function:

```go
// ResolveLogPath returns the log file path for a model.
// Returns empty string if logging is not configured.
func (c *Config) ResolveLogPath(modelName string) string {
	model, ok := c.Models[modelName]
	if !ok {
		return ""
	}

	// Per-model override takes precedence
	if model.LogFile != "" {
		return model.LogFile
	}

	// Fall back to log_dir/<model>.log
	if c.VLLM.LogDir != "" {
		return filepath.Join(c.VLLM.LogDir, modelName+".log")
	}

	return ""
}
```

**Step 2: Add import for filepath**

Add `"path/filepath"` to the imports.

**Step 3: Run tests**

Run: `go test ./internal/config/...`
Expected: PASS

**Step 4: Commit**

```bash
git add internal/config/config.go
git commit -m "feat(config): add ResolveLogPath helper method"
```

---

### Task 6: Integrate Log Writer into Runtime

**Files:**
- Modify: `internal/vllm/runtime.go`

**Step 1: Add logWriter field to Manager struct**

Add after `stdoutTail`:

```go
type Manager struct {
	cfg *config.Config
	port int

	// extraEnv is applied after default and model env, overriding on conflict.
	extraEnv map[string]string

	mu            sync.Mutex
	cmd           *exec.Cmd
	pid           int
	waitCh        chan error
	exitCh        chan struct{}
	exitInfo      *processExitInfo
	stopRequested bool
	currentModel  string
	startedAt     time.Time
	stderrTail    *tailBuffer
	stdoutTail    *tailBuffer
	logWriter     *RotatingFileWriter
}
```

**Step 2: Add import for io**

Add `"io"` to the imports at the top of the file.

**Step 3: Modify Start() to set up log writer**

Replace the stdout/stderr setup section (around lines 120-123) with:

```go
	stdoutTail := newTailBuffer(16 * 1024)
	stderrTail := newTailBuffer(16 * 1024)

	// Set up log file if configured
	var logWriter *RotatingFileWriter
	logPath := m.cfg.ResolveLogPath(modelName)
	if logPath != "" {
		var err error
		logWriter, err = NewRotatingFileWriter(logPath, m.cfg.VLLM.LogMaxSizeMB, m.cfg.VLLM.LogMaxFiles)
		if err != nil {
			slog.Warn("failed to create log file, continuing without file logging",
				"path", logPath,
				"error", err)
		}
	}

	// Tee output to both tail buffer and log file (if configured)
	if logWriter != nil {
		cmd.Stdout = io.MultiWriter(stdoutTail, logWriter)
		cmd.Stderr = io.MultiWriter(stderrTail, logWriter)
	} else {
		cmd.Stdout = stdoutTail
		cmd.Stderr = stderrTail
	}
```

**Step 4: Store logWriter in manager state**

After storing stdoutTail (around line 141), add:

```go
	m.logWriter = logWriter
```

**Step 5: Close log writer on process exit**

In the goroutine that handles process exit (around line 186), before clearing fields, add:

```go
		if m.logWriter != nil {
			m.logWriter.Close()
			m.logWriter = nil
		}
```

**Step 6: Run tests**

Run: `go test ./internal/vllm/...`
Expected: PASS

**Step 7: Commit**

```bash
git add internal/vllm/runtime.go
git commit -m "feat(vllm): integrate log file writer into runtime"
```

---

### Task 7: Update Example Configs

**Files:**
- Modify: `configs/example.yaml`
- Modify: `configs/scheduler_example.yaml`

**Step 1: Add log config to example.yaml**

Add after `default_env` section:

```yaml
  # Log vLLM output to files (optional)
  # log_dir: "/var/log/vllm"
  # log_max_size_mb: 50   # default: 50
  # log_max_files: 5      # default: 5
```

**Step 2: Add per-model log_file example**

In the models section, add a comment:

```yaml
    # Per-model log file override (optional)
    # log_file: "/var/log/vllm/custom-model.log"
```

**Step 3: Add same comments to scheduler_example.yaml**

**Step 4: Commit**

```bash
git add configs/example.yaml configs/scheduler_example.yaml
git commit -m "docs(config): add log file examples to config files"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add log file documentation**

Add a new section after GPU Power Limits:

```markdown
## vLLM Log Files

vLLM process output can be redirected to rotating log files:

```yaml
vllm:
  log_dir: "/var/log/vllm"    # Directory for log files
  log_max_size_mb: 50         # Max size before rotation (default: 50)
  log_max_files: 5            # Rotated files to keep (default: 5)

models:
  mymodel:
    path: "..."
    log_file: "/custom/path.log"  # Per-model override (optional)
```

- If `log_dir` is set, each model logs to `<log_dir>/<model-name>.log`
- Per-model `log_file` overrides the auto-generated path
- Logs are appended with size-based rotation
- Tail buffer still available via `/status` endpoint
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add vLLM log file documentation"
```

---

### Task 9: Add Integration Test

**Files:**
- Create: `internal/vllm/logwriter_integration_test.go`

**Step 1: Create integration test**

```go
package vllm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"vllm-jukebox/internal/config"
)

func TestResolveLogPath(t *testing.T) {
	tests := []struct {
		name      string
		yaml      string
		modelName string
		wantPath  string
	}{
		{
			name: "no log config",
			yaml: `
models:
  test:
    path: /models/test
`,
			modelName: "test",
			wantPath:  "",
		},
		{
			name: "log_dir set",
			yaml: `
vllm:
  log_dir: /var/log/vllm
models:
  test:
    path: /models/test
`,
			modelName: "test",
			wantPath:  "/var/log/vllm/test.log",
		},
		{
			name: "per-model override",
			yaml: `
vllm:
  log_dir: /var/log/vllm
models:
  test:
    path: /models/test
    log_file: /custom/path.log
`,
			modelName: "test",
			wantPath:  "/custom/path.log",
		},
		{
			name: "per-model without global log_dir",
			yaml: `
models:
  test:
    path: /models/test
    log_file: /custom/path.log
`,
			modelName: "test",
			wantPath:  "/custom/path.log",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := config.Load([]byte(tt.yaml))
			if err != nil {
				t.Fatalf("failed to load config: %v", err)
			}

			got := cfg.ResolveLogPath(tt.modelName)
			if got != tt.wantPath {
				t.Errorf("ResolveLogPath() = %q, want %q", got, tt.wantPath)
			}
		})
	}
}

func TestLogWriterEndToEnd(t *testing.T) {
	dir := t.TempDir()
	logDir := filepath.Join(dir, "logs")

	yaml := `
vllm:
  log_dir: ` + logDir + `
  log_max_size_mb: 1
  log_max_files: 3
models:
  test:
    path: /models/test
`

	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	logPath := cfg.ResolveLogPath("test")
	if logPath == "" {
		t.Fatal("expected log path to be set")
	}

	w, err := NewRotatingFileWriter(logPath, cfg.VLLM.LogMaxSizeMB, cfg.VLLM.LogMaxFiles)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	// Write some data
	w.Write([]byte("test log output\n"))
	w.Close()

	// Verify log file exists
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}
	if !strings.Contains(string(data), "test log output") {
		t.Fatalf("log file should contain test output")
	}
}
```

**Step 2: Run all tests**

Run: `go test ./... -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add internal/vllm/logwriter_integration_test.go
git commit -m "test(vllm): add log file integration tests"
```

---

### Task 10: Final Verification

**Step 1: Run full test suite**

Run: `make check`
Expected: All checks pass

**Step 2: Build and verify startup**

Run: `make build && timeout 2 ./bin/jukebox -config configs/scheduler_example.yaml; true`
Expected: Server starts without errors

**Step 3: Push all commits**

```bash
git push
```
