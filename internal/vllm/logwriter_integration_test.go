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
