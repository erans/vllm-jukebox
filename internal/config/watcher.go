package config

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
)

// current holds the latest validated config. Reads via Current() are
// lock-free; writes via setCurrent() atomically swap the pointer.
//
// Hot-reload semantics:
//   - The pointer starts nil; SetCurrent must be called once after the
//     initial Load() before Current() is meaningful.
//   - On a successful hot-reload, the pointer is swapped. Any goroutine
//     that calls Current() after the swap sees the new view; goroutines
//     mid-operation continue to use whatever pointer they captured.
//   - On a failed hot-reload (parse or validation error), the pointer is
//     NOT swapped — last-known-good config keeps serving.
var current atomic.Pointer[Config]

// Current returns the latest validated config, or nil if SetCurrent has
// never been called. Callers that need a stable view across an operation
// should capture the return value once.
func Current() *Config {
	return current.Load()
}

// SetCurrent installs cfg as the active configuration. Intended to be
// called exactly once at startup (after the initial Load) before any
// subsystem reads via Current(). The watcher uses the same pointer for
// subsequent atomic swaps.
func SetCurrent(cfg *Config) {
	current.Store(cfg)
}

// ReloadResult is the outcome label emitted on the
// jukebox_config_reloads_total counter and surfaced to test hooks.
type ReloadResult string

const (
	ReloadSuccess          ReloadResult = "success"
	ReloadValidationFailed ReloadResult = "validation_failed"
	ReloadParseFailed      ReloadResult = "parse_failed"
)

// ReloadObserver is invoked after every reload attempt (success or
// failure). Used by callers to hook a Prometheus counter without
// creating a config→metrics package dependency, and by tests to wait
// for reload settling.
type ReloadObserver func(result ReloadResult, err error)

// reloadDebounce is the quiet window the watcher waits after the last
// fsnotify event before re-reading the file. Sized to coalesce both
// the "atomic editor writes temp + renames" double-fire and the
// "operator vim :w" multi-write storm in well under a second.
const reloadDebounce = 500 * time.Millisecond

// Watch installs an fsnotify watcher on the directory containing path
// and re-loads the file on WRITE/CREATE/RENAME events targeting that
// basename. Successful reloads atomic-swap config.Current(). Failed
// reloads leave Current() untouched and log slog.Error.
//
// The watcher goroutine exits when ctx is cancelled. Returns an error
// only if the watcher cannot be initialized (fsnotify creation failure
// or directory not readable). The caller is expected to start this in
// a goroutine after a successful initial Load() + SetCurrent().
//
// observer is optional (may be nil). When non-nil it fires on every
// reload outcome — used by main to bump Prometheus counters and by
// tests to await settling.
func Watch(ctx context.Context, path string, observer ReloadObserver) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("config watcher: resolve path: %w", err)
	}
	dir := filepath.Dir(abs)
	base := filepath.Base(abs)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("config watcher: create: %w", err)
	}
	// Watch the directory, not the file. Atomic editors (vim, helm
	// update, k8s configmap projection) RENAME a temp file over the
	// target; most fsnotify backends drop a per-file watch when the
	// inode is replaced.
	if err := watcher.Add(dir); err != nil {
		_ = watcher.Close()
		return fmt.Errorf("config watcher: add %q: %w", dir, err)
	}

	go runWatcher(ctx, watcher, abs, base, observer)
	return nil
}

func runWatcher(ctx context.Context, watcher *fsnotify.Watcher, absPath, base string, observer ReloadObserver) {
	defer watcher.Close()

	// Debounce timer is created stopped; we (re)Reset on each event.
	var debounce *time.Timer
	debounceC := func() <-chan time.Time {
		if debounce == nil {
			return nil
		}
		return debounce.C
	}

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-watcher.Events:
			if !ok {
				return
			}
			if filepath.Base(ev.Name) != base {
				continue
			}
			// Only react to events that meaningfully change the file's
			// contents or replace its inode.
			if ev.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename) == 0 {
				continue
			}
			if debounce == nil {
				debounce = time.NewTimer(reloadDebounce)
			} else {
				if !debounce.Stop() {
					select {
					case <-debounce.C:
					default:
					}
				}
				debounce.Reset(reloadDebounce)
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			slog.Warn("active.yaml watcher error", "err", err)
		case <-debounceC():
			debounce = nil
			doReload(absPath, observer)
		}
	}
}

func doReload(path string, observer ReloadObserver) {
	data, err := os.ReadFile(path)
	if err != nil {
		// File momentarily missing during atomic-replace is common; log
		// once and let the next event re-trigger.
		if errors.Is(err, os.ErrNotExist) {
			slog.Warn("active.yaml hot-reload: file missing (atomic replace in flight?)", "path", path)
			if observer != nil {
				observer(ReloadParseFailed, err)
			}
			return
		}
		slog.Error("active.yaml hot-reload: read failed", "path", path, "err", err, "keeping_previous_config", true)
		if observer != nil {
			observer(ReloadParseFailed, err)
		}
		return
	}
	if len(data) == 0 {
		slog.Error("active.yaml hot-reload: file empty", "path", path, "keeping_previous_config", true)
		if observer != nil {
			observer(ReloadParseFailed, errors.New("file empty"))
		}
		return
	}
	cfg, err := Load(data)
	if err != nil {
		// Load() runs both YAML parse and Validate(). We can't easily
		// distinguish parse-vs-validate from the returned error without
		// re-doing the work, so categorize by error string prefix —
		// good enough for the operator dashboard.
		result := ReloadValidationFailed
		if isYAMLSyntaxError(err) {
			result = ReloadParseFailed
		}
		slog.Error("active.yaml hot-reload validation failed",
			"err", err,
			"path", path,
			"keeping_previous_config", true,
		)
		if observer != nil {
			observer(result, err)
		}
		return
	}
	SetCurrent(cfg)
	slog.Info("active.yaml hot-reloaded",
		"path", path,
		"tracked_models", len(cfg.Models),
	)
	if observer != nil {
		observer(ReloadSuccess, nil)
	}
}

// isYAMLSyntaxError approximates whether the wrapped error came from
// yaml.Decode (parse) vs Validate(). Used only for the metric label.
func isYAMLSyntaxError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	// gopkg.in/yaml.v3 errors all carry a "yaml:" prefix; Validate's
	// errors are plain wrapped strings produced by Validate().
	return len(msg) >= 5 && msg[:5] == "yaml:"
}
