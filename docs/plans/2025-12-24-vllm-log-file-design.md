# vLLM Log File Design

## Overview

Add support for redirecting vLLM process output to log files with built-in rotation. Each running vLLM instance writes to its own log file, with configurable size-based rotation.

## Configuration Schema

```yaml
vllm:
  binary: "uvx"              # or "vllm" or "/path/to/vllm"
  log_dir: "/var/log/vllm"   # optional - if not set, no file logging
  log_max_size_mb: 50        # default: 50
  log_max_files: 5           # default: 5

models:
  devstral:
    path: "cyankiwi/Devstral-..."
    gpus: [0,1,2,3]
    log_file: "/custom/path/devstral.log"  # optional override

  qwen-small:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
    gpus: [4]
    # uses log_dir/qwen-small.log if log_dir is set
```

### Behavior

- If `log_dir` is set, each model logs to `<log_dir>/<model-name>.log`
- Per-model `log_file` overrides the auto-generated path
- If neither is set, no file logging (current behavior - tail buffer only)
- Tail buffer still kept for `/status` endpoint diagnostics

### Validation Rules

- `log_max_size_mb` must be > 0 if set
- `log_max_files` must be >= 1 if set
- Per-model `log_file` and global `log_dir` can coexist (per-model takes precedence)

## Log Rotation

### Rotation Behavior

- Check file size before each write
- When current log exceeds `log_max_size_mb`, rotate:
  - Delete `model.log.5` (oldest)
  - Rename `model.log.4` → `model.log.5`
  - Rename `model.log.3` → `model.log.4`
  - ... and so on
  - Rename `model.log` → `model.log.1`
  - Create new `model.log`

### File Naming

- Current log: `<model-name>.log`
- Rotated logs: `<model-name>.log.1`, `<model-name>.log.2`, etc.
- Higher numbers = older files

## Implementation

### New File: `internal/vllm/logwriter.go`

```go
type RotatingFileWriter struct {
    mu          sync.Mutex
    path        string
    maxBytes    int64
    maxFiles    int
    currentSize int64
    file        *os.File
}

func NewRotatingFileWriter(path string, maxSizeMB, maxFiles int) (*RotatingFileWriter, error)
func (w *RotatingFileWriter) Write(p []byte) (n int, err error)
func (w *RotatingFileWriter) Close() error
func (w *RotatingFileWriter) rotate() error
```

### Output Tee

vLLM output goes to both:
1. Rotating file writer (for persistence)
2. Tail buffer (for `/status` endpoint)

Uses `io.MultiWriter` to combine them.

### Integration Points

1. **Config parsing** (`internal/config/config.go`): Add new fields
2. **Runtime start** (`internal/vllm/runtime.go`): Create file writer, set up MultiWriter
3. **Runtime stop**: Close file writer

## Error Handling

### Directory Creation

- Auto-create `log_dir` if it doesn't exist (0755 permissions)
- Auto-create parent directories for per-model `log_file` paths

### Graceful Degradation

- If log file can't be opened/written: log warning, continue without file logging
- Rotation errors: log warning, continue writing to current file

### Model Lifecycle

- File opened when vLLM process starts
- File closed when vLLM process stops
- In swap mode: close old model's file, open new model's file
- In scheduler mode: each instance manages its own file independently

## Files to Modify/Create

- `internal/config/config.go` - Add config fields
- `internal/config/config_test.go` - Validation tests
- `internal/vllm/logwriter.go` - New RotatingFileWriter
- `internal/vllm/logwriter_test.go` - Unit tests
- `internal/vllm/runtime.go` - Integration with file writer
- `configs/example.yaml` - Document new options
- `configs/scheduler_example.yaml` - Document new options
