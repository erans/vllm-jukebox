package vllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"vllm-jukebox/internal/config"
)

type Manager struct {
	cfg *config.Config

	mu  sync.Mutex
	cmd *exec.Cmd
	pid int
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
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return 0, err
	}

	m.mu.Lock()
	m.cmd = cmd
	m.pid = cmd.Process.Pid
	m.mu.Unlock()

	return cmd.Process.Pid, nil
}

func (m *Manager) Stop(ctx context.Context) error {
	m.mu.Lock()
	cmd := m.cmd
	pid := m.pid
	m.mu.Unlock()

	if cmd == nil || pid == 0 {
		return nil
	}

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	_ = syscall.Kill(-pid, syscall.SIGTERM)

	select {
	case err := <-waitCh:
		m.clear()
		return normalizeStopWaitErr(err)
	case <-ctx.Done():
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		select {
		case err := <-waitCh:
			m.clear()
			_ = normalizeStopWaitErr(err)
		case <-time.After(2 * time.Second):
			// Best effort; avoid blocking forever.
		}
		m.clear()
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

func (m *Manager) clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cmd = nil
	m.pid = 0
}
