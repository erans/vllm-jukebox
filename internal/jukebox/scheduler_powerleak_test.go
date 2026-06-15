package jukebox_test

// Fix 2 (power-limit leak on failed cold-load) regression test.
//
// AcquireRoute applies a per-GPU watt cap via powerMgr.ApplyModelLimits
// before mgr.Start; on start-fail (or verify-fail) the FIX reverts that cap
// via powerMgr.RevertModelLimits. Pre-fix only the apply was issued and the
// override stayed pinned after a doomed cold-load.
//
// We drive a real gpu.PowerManager whose binary is a fake nvidia-smi shell
// script that appends every `-pl` invocation to a log file. A factory whose
// Start always errors forces the start-fail branch. PASS = the log shows the
// apply -pl followed by the revert -pl; pre-fix only the apply appears.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/ports"
)

// startFailInstance is an InstanceManager whose Start always fails, forcing
// the start-fail branch in AcquireRoute.
type startFailInstance struct{ port int }

func (s *startFailInstance) Start(_ context.Context, _ string) (int, error) {
	return 0, errStartFail
}
func (s *startFailInstance) Stop(_ context.Context) error                  { return nil }
func (s *startFailInstance) VerifyReady(_ context.Context, _ string) error { return nil }
func (s *startFailInstance) CurrentPID() int                               { return 0 }
func (s *startFailInstance) BaseURL() string {
	return ""
}

var errStartFail = &startFailErr{}

type startFailErr struct{}

func (*startFailErr) Error() string { return "forced start failure" }

// writeFakeNvidiaSmi writes a shell script that logs every -pl invocation
// (the watt-cap apply/revert calls) to logPath, one line per call:
// "pl <gpu> <watts>". Returns the script path.
func writeFakeNvidiaSmi(t *testing.T, dir, logPath string) string {
	t.Helper()
	script := "#!/bin/sh\n" +
		"gpu=\"\"\n" +
		"pl=\"\"\n" +
		"while [ $# -gt 0 ]; do\n" +
		"  case \"$1\" in\n" +
		"    -i) gpu=\"$2\"; shift 2;;\n" +
		"    -pl) pl=\"$2\"; shift 2;;\n" +
		"    *) shift;;\n" +
		"  esac\n" +
		"done\n" +
		"if [ -n \"$pl\" ]; then echo \"pl $gpu $pl\" >> \"" + logPath + "\"; fi\n" +
		"exit 0\n"
	path := filepath.Join(dir, "fake-nvidia-smi.sh")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake nvidia-smi: %v", err)
	}
	return path
}

func TestAcquireRoute_RevertsPowerLimitsOnStartFailure(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pl.log")
	smi := writeFakeNvidiaSmi(t, dir, logPath)

	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8400
  port_range_end: 8409
vllm:
  port: 8000
  startup_timeout: 2s
  shutdown_timeout: 2s
models:
  pw:
    path: "/models/pw"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    power_limit: 250
`)

	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8400, 8409)

	// Default 300W on GPU 0 so RevertToDefault has somewhere to revert to
	// AND the model's 250W override differs from it (the revert is a no-op
	// when current == default).
	powerMgr := gpu.NewPowerManager(smi, map[int]int{0: 300}, false)

	factory := func(port int, _ string) jukebox.InstanceManager {
		return &startFailInstance{port: port}
	}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, powerMgr)

	_, err := s.AcquireRoute(context.Background(), "pw", "req-pw")
	if err == nil {
		t.Fatalf("expected AcquireRoute to fail (forced start failure), got nil")
	}

	data, readErr := os.ReadFile(logPath)
	if readErr != nil {
		t.Fatalf("read pl log: %v (no -pl calls were logged at all?)", readErr)
	}
	lines := splitNonEmpty(string(data))

	// Expect exactly: the apply (250W) then the revert (back to default 300W).
	if len(lines) < 2 {
		t.Fatalf("expected an apply -pl AND a revert -pl; got %d call(s): %v\n"+
			"PRE-FIX this fails: only the apply is logged, the revert is missing — "+
			"the 250W override stays pinned after the failed cold-load.", len(lines), lines)
	}
	if lines[0] != "pl 0 250" {
		t.Fatalf("expected first -pl call to be the apply 'pl 0 250', got %q (all: %v)", lines[0], lines)
	}
	if lines[len(lines)-1] != "pl 0 300" {
		t.Fatalf("expected last -pl call to be the revert-to-default 'pl 0 300', got %q (all: %v)", lines[len(lines)-1], lines)
	}
}

func splitNonEmpty(s string) []string {
	var out []string
	for _, ln := range strings.Split(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			out = append(out, strings.TrimSpace(ln))
		}
	}
	return out
}

// verifyFailInstance is an InstanceManager whose Start succeeds but
// VerifyReady always fails, forcing the verify-fail branch in AcquireRoute
// (start OK → verify error → Stop → power revert).
type verifyFailInstance struct{ port int }

func (v *verifyFailInstance) Start(_ context.Context, _ string) (int, error) { return 1234, nil }
func (v *verifyFailInstance) Stop(_ context.Context) error                   { return nil }
func (v *verifyFailInstance) VerifyReady(_ context.Context, _ string) error {
	return errors.New("forced verify failure")
}
func (v *verifyFailInstance) CurrentPID() int { return 1234 }
func (v *verifyFailInstance) BaseURL() string { return "" }

// TestAcquireRoute_RevertsPowerLimitsOnVerifyFailure mirrors the start-fail
// test for the OTHER doomed-cold-load branch: Start succeeds but VerifyReady
// fails. The FIX reverts the watt cap in the verify-fail branch too; pre-fix
// only the apply was logged and the 250W override stayed pinned.
func TestAcquireRoute_RevertsPowerLimitsOnVerifyFailure(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pl.log")
	smi := writeFakeNvidiaSmi(t, dir, logPath)

	cfg := mustLoadSchedulerCfg(t, `
scheduler:
  port_range_start: 8420
  port_range_end: 8429
vllm:
  port: 8000
  startup_timeout: 2s
  shutdown_timeout: 2s
models:
  pw:
    path: "/models/pw"
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    power_limit: 250
`)

	inv := &fakeInventory{gpus: []gpu.GPU{
		{Index: 0, TotalMB: 100, FreeMB: 100},
	}}
	pool := ports.New(8420, 8429)
	powerMgr := gpu.NewPowerManager(smi, map[int]int{0: 300}, false)

	factory := func(port int, _ string) jukebox.InstanceManager {
		return &verifyFailInstance{port: port}
	}
	s := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, factory, powerMgr)

	_, err := s.AcquireRoute(context.Background(), "pw", "req-pw-verify")
	if err == nil {
		t.Fatalf("expected AcquireRoute to fail (forced verify failure), got nil")
	}

	data, readErr := os.ReadFile(logPath)
	if readErr != nil {
		t.Fatalf("read pl log: %v (no -pl calls were logged at all?)", readErr)
	}
	lines := splitNonEmpty(string(data))
	if len(lines) < 2 {
		t.Fatalf("expected an apply -pl AND a revert -pl on verify-fail; got %d call(s): %v\n"+
			"PRE-FIX this fails: only the apply is logged — the 250W override stays pinned "+
			"after the failed verify.", len(lines), lines)
	}
	if lines[0] != "pl 0 250" {
		t.Fatalf("expected first -pl call to be the apply 'pl 0 250', got %q (all: %v)", lines[0], lines)
	}
	if lines[len(lines)-1] != "pl 0 300" {
		t.Fatalf("expected last -pl call to be the revert-to-default 'pl 0 300', got %q (all: %v)", lines[len(lines)-1], lines)
	}
}

// coordinatorPowerCfg builds a single-model config wired for the power-revert
// coordinator tests: GPU 0, 250W override, default 300W. Shared by the
// start-fail and verify-fail coordinator cases below.
func coordinatorPowerCfg(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
  startup_timeout: 2s
  shutdown_timeout: 2s
  drain_timeout: 1s
models:
  pw:
    path: "/models/pw"
    gpus: [0]
    power_limit: 250
`))
	if err != nil {
		t.Fatalf("load cfg: %v", err)
	}
	return cfg
}

// TestCoordinator_RevertsPowerLimitsOnStartFailure covers the coordinator
// (swap-mode) doSwap start-fail branch. doSwap applies the watt cap before
// mgr.Start; on start-fail the FIX reverts it. Pre-fix only the apply was
// issued and the 250W override stayed pinned after the doomed swap.
func TestCoordinator_RevertsPowerLimitsOnStartFailure(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pl.log")
	smi := writeFakeNvidiaSmi(t, dir, logPath)

	cfg := coordinatorPowerCfg(t)
	powerMgr := gpu.NewPowerManager(smi, map[int]int{0: 300}, false)

	var tr inflight.Tracker
	mgr := &fakeManager{startErr: errors.New("forced start failure")}
	clock := func() time.Time { return time.Unix(0, 0) }

	c := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, clock, powerMgr)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "pw", "req_start_fail"); err == nil {
		t.Fatalf("expected EnsureModel to fail (forced start failure), got nil")
	}

	data, readErr := os.ReadFile(logPath)
	if readErr != nil {
		t.Fatalf("read pl log: %v (no -pl calls were logged at all?)", readErr)
	}
	lines := splitNonEmpty(string(data))
	if len(lines) < 2 {
		t.Fatalf("expected an apply -pl AND a revert -pl on coordinator start-fail; got %d call(s): %v\n"+
			"PRE-FIX this fails: only the apply is logged — the 250W override stays pinned "+
			"after the failed swap start.", len(lines), lines)
	}
	if lines[0] != "pl 0 250" {
		t.Fatalf("expected first -pl call to be the apply 'pl 0 250', got %q (all: %v)", lines[0], lines)
	}
	if lines[len(lines)-1] != "pl 0 300" {
		t.Fatalf("expected last -pl call to be the revert-to-default 'pl 0 300', got %q (all: %v)", lines[len(lines)-1], lines)
	}
}

// TestCoordinator_RevertsPowerLimitsOnVerifyFailure covers the coordinator
// doSwap verify-fail branch: Start succeeds, VerifyReady fails → Stop →
// power revert. Pre-fix the revert was missing and the override stayed pinned.
func TestCoordinator_RevertsPowerLimitsOnVerifyFailure(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "pl.log")
	smi := writeFakeNvidiaSmi(t, dir, logPath)

	cfg := coordinatorPowerCfg(t)
	powerMgr := gpu.NewPowerManager(smi, map[int]int{0: 300}, false)

	var tr inflight.Tracker
	mgr := &fakeManager{verifyErr: errors.New("forced verify failure")}
	clock := func() time.Time { return time.Unix(0, 0) }

	c := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, clock, powerMgr)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.Run(ctx)

	if err := c.EnsureModel(context.Background(), "pw", "req_verify_fail"); err == nil {
		t.Fatalf("expected EnsureModel to fail (forced verify failure), got nil")
	}

	data, readErr := os.ReadFile(logPath)
	if readErr != nil {
		t.Fatalf("read pl log: %v (no -pl calls were logged at all?)", readErr)
	}
	lines := splitNonEmpty(string(data))
	if len(lines) < 2 {
		t.Fatalf("expected an apply -pl AND a revert -pl on coordinator verify-fail; got %d call(s): %v\n"+
			"PRE-FIX this fails: only the apply is logged — the 250W override stays pinned "+
			"after the failed swap verify.", len(lines), lines)
	}
	if lines[0] != "pl 0 250" {
		t.Fatalf("expected first -pl call to be the apply 'pl 0 250', got %q (all: %v)", lines[0], lines)
	}
	if lines[len(lines)-1] != "pl 0 300" {
		t.Fatalf("expected last -pl call to be the revert-to-default 'pl 0 300', got %q (all: %v)", lines[len(lines)-1], lines)
	}
}
