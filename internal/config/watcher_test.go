package config_test

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

const validYAML = `
server:
  host: "127.0.0.1"
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpu_memory_utilization: 0.8
`

const validYAMLWithBeta = `
server:
  host: "127.0.0.1"
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
behavior:
  default_model: "alpha"
models:
  alpha:
    path: "/models/alpha"
    gpu_memory_utilization: 0.8
  beta:
    path: "/models/beta"
    gpu_memory_utilization: 0.7
`

// invalidValidationYAML — parses as YAML, fails Validate (default_model
// references a nonexistent model name).
const invalidValidationYAML = `
server:
  host: "127.0.0.1"
  port: 8080
vllm:
  binary: "/usr/bin/vllm"
behavior:
  default_model: "ghost"
models:
  alpha:
    path: "/models/alpha"
    gpu_memory_utilization: 0.8
`

// brokenYAML — YAML parser failure (unclosed string, mis-indent).
const brokenYAML = `
server:
  host: "127.0.0.1
  port: 8080
`

// resultRecorder buffers reload outcomes so tests can wait for them
// without sleeping arbitrary amounts.
type resultRecorder struct {
	mu      sync.Mutex
	results []config.ReloadResult
	errs    []error
	count   atomic.Int32
	ch      chan config.ReloadResult
}

func newResultRecorder() *resultRecorder {
	return &resultRecorder{ch: make(chan config.ReloadResult, 32)}
}

func (r *resultRecorder) observe(res config.ReloadResult, err error) {
	r.mu.Lock()
	r.results = append(r.results, res)
	r.errs = append(r.errs, err)
	r.mu.Unlock()
	r.count.Add(1)
	select {
	case r.ch <- res:
	default:
	}
}

func (r *resultRecorder) waitFor(t *testing.T, timeout time.Duration) config.ReloadResult {
	t.Helper()
	select {
	case res := <-r.ch:
		return res
	case <-time.After(timeout):
		t.Fatalf("timed out waiting for reload result after %s", timeout)
		return ""
	}
}

func writeFileAtomic(t *testing.T, path, contents string) {
	t.Helper()
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".active-*.tmp")
	if err != nil {
		t.Fatalf("CreateTemp: %v", err)
	}
	if _, err := tmp.WriteString(contents); err != nil {
		_ = tmp.Close()
		t.Fatalf("write: %v", err)
	}
	if err := tmp.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		t.Fatalf("rename: %v", err)
	}
}

func TestWatcher_ReloadOnWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "active.yaml")
	if err := os.WriteFile(path, []byte(validYAML), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load([]byte(validYAML))
	if err != nil {
		t.Fatalf("initial Load: %v", err)
	}
	config.SetCurrent(cfg)
	if got := len(config.Current().Models); got != 1 {
		t.Fatalf("initial model count: got %d, want 1", got)
	}

	rec := newResultRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := config.Watch(ctx, path, rec.observe); err != nil {
		t.Fatalf("Watch: %v", err)
	}

	writeFileAtomic(t, path, validYAMLWithBeta)
	res := rec.waitFor(t, 5*time.Second)
	if res != config.ReloadSuccess {
		t.Fatalf("reload result: got %q, want success", res)
	}
	got := config.Current()
	if _, ok := got.Models["beta"]; !ok {
		t.Fatalf("expected beta model after reload, got models=%v", keys(got.Models))
	}
}

func TestWatcher_InvalidYAML_PreservesPreviousConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "active.yaml")
	if err := os.WriteFile(path, []byte(validYAML), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load([]byte(validYAML))
	if err != nil {
		t.Fatalf("initial Load: %v", err)
	}
	config.SetCurrent(cfg)
	prev := config.Current()

	rec := newResultRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := config.Watch(ctx, path, rec.observe); err != nil {
		t.Fatalf("Watch: %v", err)
	}

	// Validation failure (parses as YAML, fails Validate)
	writeFileAtomic(t, path, invalidValidationYAML)
	res := rec.waitFor(t, 5*time.Second)
	if res != config.ReloadValidationFailed {
		t.Fatalf("reload result: got %q, want validation_failed", res)
	}
	if config.Current() != prev {
		t.Fatalf("Current() changed despite validation failure")
	}

	// YAML syntax failure
	writeFileAtomic(t, path, brokenYAML)
	res = rec.waitFor(t, 5*time.Second)
	if res != config.ReloadParseFailed {
		t.Fatalf("reload result: got %q, want parse_failed", res)
	}
	if config.Current() != prev {
		t.Fatalf("Current() changed despite parse failure")
	}
}

func TestWatcher_DebouncesRapidWrites(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "active.yaml")
	if err := os.WriteFile(path, []byte(validYAML), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load([]byte(validYAML))
	if err != nil {
		t.Fatalf("initial Load: %v", err)
	}
	config.SetCurrent(cfg)

	rec := newResultRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := config.Watch(ctx, path, rec.observe); err != nil {
		t.Fatalf("Watch: %v", err)
	}

	// Write three times in rapid succession; debounce window is 500ms.
	writeFileAtomic(t, path, validYAMLWithBeta)
	time.Sleep(50 * time.Millisecond)
	writeFileAtomic(t, path, validYAML)
	time.Sleep(50 * time.Millisecond)
	writeFileAtomic(t, path, validYAMLWithBeta)

	// Wait for the (one) reload to settle.
	res := rec.waitFor(t, 5*time.Second)
	if res != config.ReloadSuccess {
		t.Fatalf("first reload: got %q, want success", res)
	}

	// Quiet period — assert NO additional reloads fire.
	time.Sleep(800 * time.Millisecond)
	if got := rec.count.Load(); got != 1 {
		t.Fatalf("expected exactly 1 reload, got %d", got)
	}

	// Final config should reflect the LAST written file (validYAMLWithBeta).
	if _, ok := config.Current().Models["beta"]; !ok {
		t.Fatalf("expected beta model after debounced reload, got %v", keys(config.Current().Models))
	}
}

// TestWatcher_ConcurrentReadersAndWriter is intended to be run with -race.
// N reader goroutines call Current() in a tight loop while a single
// writer goroutine triggers reloads. The race detector flags torn reads
// or unsynchronized accesses.
func TestWatcher_ConcurrentReadersAndWriter(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "active.yaml")
	if err := os.WriteFile(path, []byte(validYAML), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load([]byte(validYAML))
	if err != nil {
		t.Fatalf("initial Load: %v", err)
	}
	config.SetCurrent(cfg)

	rec := newResultRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := config.Watch(ctx, path, rec.observe); err != nil {
		t.Fatalf("Watch: %v", err)
	}

	stopReaders := make(chan struct{})
	var wg sync.WaitGroup
	const readers = 8
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopReaders:
					return
				default:
				}
				c := config.Current()
				if c == nil {
					t.Errorf("Current() returned nil")
					return
				}
				// Touch a few fields to provoke any torn-read.
				_ = c.Server.Host
				_ = len(c.Models)
				_ = c.Behavior.DefaultModel
			}
		}()
	}

	// Trigger ~5 reloads, alternating between two valid configs. Wait
	// for the debounce window between writes so each lands distinctly.
	for i := 0; i < 5; i++ {
		body := validYAML
		if i%2 == 0 {
			body = validYAMLWithBeta
		}
		writeFileAtomic(t, path, body)
		// Wait for this reload to settle before launching the next.
		_ = rec.waitFor(t, 5*time.Second)
	}

	close(stopReaders)
	wg.Wait()
}

func keys(m map[string]config.ModelConfig) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
