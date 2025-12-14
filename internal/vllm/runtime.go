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
	cfg *config.Config

	mu            sync.Mutex
	cmd           *exec.Cmd
	pid           int
	waitCh        chan error
	stopRequested bool
	currentModel  string
	startedAt     time.Time
	stderrTail    *tailBuffer
}

func NewManager(cfg *config.Config) *Manager {
	return &Manager{cfg: cfg}
}

func (m *Manager) CurrentPID() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.pid
}

func (m *Manager) BaseURL() string {
	u := url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("127.0.0.1:%d", m.cfg.VLLM.Port),
	}
	return u.String()
}

func (m *Manager) Start(ctx context.Context, modelName string) (int, error) {
	m.mu.Lock()
	if m.cmd != nil {
		m.mu.Unlock()
		return 0, fmt.Errorf("vLLM process already running (pid=%d)", m.pid)
	}
	m.mu.Unlock()

	args, err := BuildServeArgs(m.cfg, modelName)
	if err != nil {
		return 0, err
	}

	_, modelCfg, err := m.cfg.ResolveModel(modelName)
	if err != nil {
		return 0, err
	}

	bin, binArgs := wrapBinaryArgs(m.cfg.VLLM.Binary, args)
	cmd := exec.CommandContext(ctx, bin, binArgs...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = BuildEnv(os.Environ(), m.cfg.VLLM.DefaultEnv, modelCfg.Env)
	cmd.Stdout = io.Discard
	stderrTail := newTailBuffer(16 * 1024)
	cmd.Stderr = stderrTail

	if err := cmd.Start(); err != nil {
		return 0, err
	}

	waitCh := make(chan error, 1)
	m.mu.Lock()
	m.cmd = cmd
	m.pid = cmd.Process.Pid
	m.waitCh = waitCh
	m.stopRequested = false
	m.currentModel = modelName
	m.startedAt = time.Now()
	m.stderrTail = stderrTail
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
		if m.stderrTail != nil {
			stderr = m.stderrTail.String()
		}

		// Only clear if this is still the current process.
		if m.cmd == cmd {
			m.cmd = nil
			m.pid = 0
			m.waitCh = nil
			m.stopRequested = false
			m.currentModel = ""
			m.startedAt = time.Time{}
			m.stderrTail = nil
		}
		m.mu.Unlock()

		if err != nil && !stopRequested {
			slog.Error(
				"vllm_process_exited",
				"pid", pid,
				"model", model,
				"err", err,
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
	if filepath.Base(binary) == "uvx" {
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
	if err := WaitForHealth(ctx, base); err != nil {
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
