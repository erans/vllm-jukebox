package vllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"vllm-jukebox/internal/config"
)

type Manager struct {
	cfg  *config.Config
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

type processExitInfo struct {
	pid        int
	model      string
	err        error
	stdoutTail string
	stderrTail string
}

type ProcessExitedError struct {
	PID        int
	Model      string
	Err        error
	StdoutTail string
	StderrTail string
}

func (e *ProcessExitedError) Error() string {
	msg := fmt.Sprintf("vLLM process exited (pid=%d model=%q)", e.PID, e.Model)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *ProcessExitedError) Unwrap() error { return e.Err }

func NewManager(cfg *config.Config) *Manager {
	return &Manager{cfg: cfg, port: cfg.VLLM.Port}
}

func NewInstanceManager(cfg *config.Config, port int, extraEnv map[string]string) *Manager {
	copied := map[string]string{}
	for k, v := range extraEnv {
		copied[k] = v
	}
	return &Manager{cfg: cfg, port: port, extraEnv: copied}
}

func (m *Manager) CurrentPID() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.pid
}

func (m *Manager) BaseURL() string {
	u := url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", m.port),
	}
	return u.String()
}

func (m *Manager) Start(ctx context.Context, modelName string) (int, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
	}

	m.mu.Lock()
	if m.cmd != nil {
		m.mu.Unlock()
		return 0, fmt.Errorf("vLLM process already running (pid=%d)", m.pid)
	}
	m.mu.Unlock()

	args, err := BuildServeArgsForPort(m.cfg, modelName, m.port)
	if err != nil {
		return 0, err
	}

	_, modelCfg, err := m.cfg.ResolveModel(modelName)
	if err != nil {
		return 0, err
	}

	bin, binArgs := wrapBinaryArgs(m.cfg.VLLM.Binary, args)
	cmd := exec.Command(bin, binArgs...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = BuildEnv(os.Environ(), m.cfg.VLLM.DefaultEnv, mergeEnv(modelCfg.Env, m.extraEnv))
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

	if err := cmd.Start(); err != nil {
		return 0, err
	}

	waitCh := make(chan error, 1)
	exitCh := make(chan struct{})
	m.mu.Lock()
	m.cmd = cmd
	m.pid = cmd.Process.Pid
	m.waitCh = waitCh
	m.exitCh = exitCh
	m.exitInfo = nil
	m.stopRequested = false
	m.currentModel = modelName
	m.startedAt = time.Now()
	m.stderrTail = stderrTail
	m.stdoutTail = stdoutTail
	m.logWriter = logWriter
	m.mu.Unlock()

	slog.Info(
		"vllm_process_started",
		"pid", cmd.Process.Pid,
		"model", modelName,
		"command", strings.Join(append([]string{bin}, binArgs...), " "),
	)

	go func() {
		err := cmd.Wait()
		waitCh <- err

		m.mu.Lock()
		stopRequested := m.stopRequested
		pid := m.pid
		model := m.currentModel
		stderr := ""
		stdout := ""
		if m.stderrTail != nil {
			stderr = m.stderrTail.String()
		}
		if m.stdoutTail != nil {
			stdout = m.stdoutTail.String()
		}

		// Only clear if this is still the current process.
		if m.cmd == cmd {
			m.exitInfo = &processExitInfo{
				pid:        pid,
				model:      model,
				err:        err,
				stdoutTail: stdout,
				stderrTail: stderr,
			}
			if m.logWriter != nil {
				m.logWriter.Close()
				m.logWriter = nil
			}
			m.cmd = nil
			m.pid = 0
			m.waitCh = nil
			m.exitCh = nil
			m.stopRequested = false
			m.currentModel = ""
			m.startedAt = time.Time{}
			m.stderrTail = nil
			m.stdoutTail = nil
		}
		m.mu.Unlock()

		close(exitCh)

		if err != nil && !stopRequested {
			slog.Error(
				"vllm_process_exited",
				"pid", pid,
				"model", model,
				"err", err,
				"stdout_tail", stdout,
				"stderr_tail", stderr,
			)
		} else {
			slog.Info(
				"vllm_process_stopped",
				"pid", pid,
				"model", model,
			)
		}
	}()

	return cmd.Process.Pid, nil
}

func mergeEnv(base map[string]string, overlay map[string]string) map[string]string {
	if base == nil && overlay == nil {
		return nil
	}
	out := map[string]string{}
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		out[k] = v
	}
	return out
}

func (m *Manager) Stop(ctx context.Context) error {
	m.mu.Lock()
	cmd := m.cmd
	pid := m.pid
	waitCh := m.waitCh
	m.mu.Unlock()

	if cmd == nil || pid == 0 {
		return nil
	}

	m.mu.Lock()
	// Mark this process as an intentional stop (so wait() doesn't log it as a crash).
	if m.cmd == cmd {
		m.stopRequested = true
	}
	m.mu.Unlock()

	slog.Info("vllm_process_stopping", "pid", pid)
	_ = syscall.Kill(-pid, syscall.SIGTERM)

	select {
	case err := <-waitCh:
		return normalizeStopWaitErr(err)
	case <-ctx.Done():
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		select {
		case err := <-waitCh:
			_ = normalizeStopWaitErr(err)
			return nil
		case <-time.After(2 * time.Second):
		}
		return ctx.Err()
	}
}

func wrapBinaryArgs(binary string, args []string) (string, []string) {
	switch filepath.Base(binary) {
	case "uvx":
		return binary, append([]string{"vllm"}, args...)
	}
	return binary, args
}

func normalizeStopWaitErr(err error) error {
	if err == nil {
		return nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		// During an intentional shutdown, any non-zero exit is acceptable.
		return nil
	}
	return err
}

func (m *Manager) VerifyReady(ctx context.Context, expectedModel string) error {
	base := m.BaseURL()

	m.mu.Lock()
	exitCh := m.exitCh
	exitInfo := m.exitInfo
	cmd := m.cmd
	pid := m.pid
	m.mu.Unlock()

	if cmd == nil || pid == 0 {
		if exitInfo != nil {
			return &ProcessExitedError{
				PID:        exitInfo.pid,
				Model:      exitInfo.model,
				Err:        exitInfo.err,
				StdoutTail: exitInfo.stdoutTail,
				StderrTail: exitInfo.stderrTail,
			}
		}
		return fmt.Errorf("vLLM process is not running")
	}

	if err := WaitForHealthOrExit(ctx, base, exitCh); err != nil {
		if errors.Is(err, ErrProcessExited) {
			m.mu.Lock()
			exitInfo := m.exitInfo
			m.mu.Unlock()
			if exitInfo != nil {
				return &ProcessExitedError{
					PID:        exitInfo.pid,
					Model:      exitInfo.model,
					Err:        exitInfo.err,
					StdoutTail: exitInfo.stdoutTail,
					StderrTail: exitInfo.stderrTail,
				}
			}
			return fmt.Errorf("vLLM process exited while waiting for health")
		}
		return err
	}

	_, modelCfg, err := m.cfg.ResolveModel(expectedModel)
	if err != nil {
		return err
	}

	return VerifyModelLoaded(ctx, base, expectedModel, modelCfg.Path)
}

type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func newTailBuffer(max int) *tailBuffer {
	return &tailBuffer{max: max}
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.buf = append(t.buf, p...)
	if len(t.buf) > t.max {
		t.buf = append([]byte(nil), t.buf[len(t.buf)-t.max:]...)
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}
