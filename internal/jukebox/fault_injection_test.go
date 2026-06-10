package jukebox

// Fault-injection tests. Every external dependency (docker CLI, vLLM HTTP
// endpoints, request contexts, operator-supplied input) can fail. Each
// test below injects ONE failure mode and asserts that jukebox degrades
// gracefully — books are not corrupted, locks are not leaked, goroutines
// terminate, and operator-meaningful error messages surface.
//
// Conventions:
//   - Real-clock waits are short (≤2s) so the suite stays fast.
//   - Docker exec failures are injected via SetSleepDockerCmdForTest /
//     SetDockerCmdForTest.
//   - vLLM HTTP failures are injected by either (a) substituting a fake
//     SleepCapable manager whose Sleep/Wake/IsSleeping return canned
//     errors, or (b) pointing a vllmHTTPMgr at an httptest server that
//     emits the failure mode under test.
//   - Each test name starts with TestFault_ so the suite can be filtered
//     via `-run TestFault`.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
	"vllm-jukebox/internal/vllmcli"
)

// ---------------------------------------------------------------------------
// Fault helpers
// ---------------------------------------------------------------------------

// failingDockerCmd returns a docker exec hook that simulates a named
// failure scenario. Recognized scenarios:
//
//	"nonzero":          returns an error wrapping a non-zero exit
//	"timeout":          blocks until ctx is cancelled, then returns ctx.Err
//	"panic":            panics on first call
//	"not-installed":    returns the canonical "executable file not found in $PATH"
//	"no-such-container": returns a "docker stop: No such container" style error
//
// All scenarios accept any docker verb (start/stop/etc.) — the scenario
// determines the failure shape, not the verb.
func failingDockerCmd(scenario string) func(ctx context.Context, name string, args ...string) ([]byte, error) {
	return func(ctx context.Context, _ string, _ ...string) ([]byte, error) {
		switch scenario {
		case "nonzero":
			return []byte("Error response from daemon: container is not running"), errors.New("exit status 1")
		case "timeout":
			<-ctx.Done()
			return nil, ctx.Err()
		case "panic":
			panic("simulated docker exec panic")
		case "not-installed":
			return []byte(""), errors.New(`exec: "docker": executable file not found in $PATH`)
		case "no-such-container":
			return []byte("Error response from daemon: No such container: vllm-moe"), errors.New("exit status 1")
		default:
			return []byte("ok"), nil
		}
	}
}

// vllmHTTPMgr is a SleepCapable wrapper around vllmcli that lets tests
// inject HTTP-level failures by pointing it at an httptest server. Used
// by the vLLM-HTTP fault tests so we exercise the real wire-level
// decoder/transport instead of bypassing it via a fake.
//
// Implements the InstanceManager + SleepCapable interfaces required by
// schedInstance.mgr.
type vllmHTTPMgr struct {
	baseURL string
	port    int
}

func (m *vllmHTTPMgr) Start(_ context.Context, _ string) (int, error) { return 1 + m.port, nil }
func (m *vllmHTTPMgr) Stop(_ context.Context) error                   { return nil }
func (m *vllmHTTPMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *vllmHTTPMgr) CurrentPID() int                                { return 1 + m.port }
func (m *vllmHTTPMgr) BaseURL() string                                { return m.baseURL }
func (m *vllmHTTPMgr) Sleep(ctx context.Context, level int) error {
	return vllmcli.Sleep(ctx, m.baseURL, level)
}
func (m *vllmHTTPMgr) Wake(ctx context.Context, timeout time.Duration) error {
	return vllmcli.Wake(ctx, m.baseURL, timeout)
}
func (m *vllmHTTPMgr) IsSleeping(ctx context.Context) (bool, error) {
	return vllmcli.IsSleeping(ctx, m.baseURL)
}

// failingVLLMServer returns an httptest server that fails per scenario:
//
//	"sleep-500":              POST /sleep returns 500; /is_sleeping returns true
//	                          (the wedge-recovery case — vLLM accepted the
//	                          /sleep before the connection dropped)
//	"sleep-500-vllm-awake":   POST /sleep returns 500; /is_sleeping returns
//	                          false (the legitimate-rollback case — vLLM is
//	                          honestly still awake)
//	"wake-hang":              POST /wake_up never responds (until ctx cancel)
//	"is_sleeping-refused":    server is pre-closed; every endpoint refuses
//	"is_sleeping-flap":       /is_sleeping returns false, then true, then true
//	"is_sleeping-garbage":    /is_sleeping returns 200 with non-JSON body
//
// Caller is responsible for srv.Close() — except "is_sleeping-refused"
// which is pre-closed (the test typically just uses srv.URL).
func failingVLLMServer(scenario string) *httptest.Server {
	var flapCount atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/sleep", func(w http.ResponseWriter, r *http.Request) {
		if scenario == "sleep-500" || scenario == "sleep-500-vllm-awake" {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte("internal: cuda OOM during sleep"))
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/wake_up", func(w http.ResponseWriter, r *http.Request) {
		if scenario == "wake-hang" {
			// Block until the client gives up.
			<-r.Context().Done()
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/is_sleeping", func(w http.ResponseWriter, r *http.Request) {
		switch scenario {
		case "is_sleeping-flap":
			n := flapCount.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			if n == 1 {
				_, _ = w.Write([]byte(`{"is_sleeping": false}`))
			} else {
				_, _ = w.Write([]byte(`{"is_sleeping": true}`))
			}
			return
		case "is_sleeping-garbage":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("<<NOT JSON AT ALL>>"))
			return
		case "sleep-500-vllm-awake":
			// /sleep failed AND vLLM is honestly still awake — sleepInstance
			// must roll back state to its pre-call value (Ready). Used by
			// TestFault_VLLM_SleepEndpoint_Returns500.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"is_sleeping": false}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"is_sleeping": true}`))
	})
	srv := httptest.NewServer(mux)
	if scenario == "is_sleeping-refused" {
		// Pre-close — clients will see connection refused on the bound
		// port for the lifetime of the test (no other process can grab
		// the ephemeral port that fast in practice).
		srv.Close()
	}
	return srv
}

// ---------------------------------------------------------------------------
// Docker exec failures (1-5)
// ---------------------------------------------------------------------------

// TestFault_DockerStop_ReturnsNonZero asserts that when `docker stop`
// returns a non-zero exit, StopForEviction propagates the error and
// leaves the instance state restored to its pre-stop value (so a future
// retry can run). admission's books are owned by the caller (RequestWake)
// and rolled back via the documented evictor contract.
func TestFault_DockerStop_ReturnsNonZero(t *testing.T) {
	s, _, _ := makeStopOnEvictScheduler(t)

	SetSleepDockerCmdForTest(failingDockerCmd("nonzero"))
	defer SetSleepDockerCmdForTest(nil)

	// Pre-state: moe is Stopped (per setup). Manually transition it to
	// Sleeping so StopForEviction has a reachable victim to operate on.
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()

	ev := &SchedulerEvictor{S: s}
	err := ev.StopForEviction(context.Background(), "moe", "test")
	if err == nil {
		t.Fatalf("expected docker stop nonzero exit to surface as error, got nil")
	}
	if !strings.Contains(err.Error(), "docker stop") {
		t.Errorf("expected error to mention docker stop, got: %v", err)
	}

	// Instance state must be restored to Sleeping (the pre-stop state),
	// not left wedged in StateStopping or flipped to StateStopped.
	s.mu.RLock()
	got := s.instances["moe"].state
	draining := s.instances["moe"].draining
	s.mu.RUnlock()
	if got != StateSleeping {
		t.Errorf("expected state restored to Sleeping after docker stop failure, got %v", got)
	}
	if draining {
		t.Errorf("expected draining=false after error rollback, got true")
	}
}

// TestFault_DockerStop_TimesOut asserts that when docker stop blocks
// past the deadline, ctx.Err propagates and the rollback fires.
func TestFault_DockerStop_TimesOut(t *testing.T) {
	s, _, _ := makeStopOnEvictScheduler(t)

	SetSleepDockerCmdForTest(failingDockerCmd("timeout"))
	defer SetSleepDockerCmdForTest(nil)

	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()

	// Use a short outer ctx so the in-helper 90s timeout doesn't gate us.
	// failingDockerCmd("timeout") respects ctx.Done(), so this returns
	// quickly.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	ev := &SchedulerEvictor{S: s}
	err := ev.StopForEviction(ctx, "moe", "test")
	if err == nil {
		t.Fatalf("expected timeout to surface as error, got nil")
	}

	// State rolled back, books not flipped.
	s.mu.RLock()
	got := s.instances["moe"].state
	s.mu.RUnlock()
	if got != StateSleeping {
		t.Errorf("expected state restored to Sleeping after timeout, got %v", got)
	}
}

// TestFault_DockerStart_ReturnsNonZero asserts that when docker start
// fails inside coldLoadStoppedMember, the goroutine calls
// NotifyStartFailed (NOT NotifySleep), state stays admissionStopped,
// and no goroutine leak (coldLoadKicks cleared via defer).
func TestFault_DockerStart_ReturnsNonZero(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	startErr := errors.New("simulated docker start failure")
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			return []byte("daemon error"), startErr
		}
		// stop / cleanup calls succeed (so the cleanup path itself works).
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// moe starts admissionStopped; verify cold-load failure leaves it Stopped.
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe Stopped pre-test")
	}
	a.mu.Lock()
	preAwake := a.awakeByGPU[0]
	preResidual := a.l1ResidualByGPU[0]
	a.mu.Unlock()

	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false; expected true")
	}

	// Cold-load should fail and the goroutine should clean up the kick map.
	if ok := waitForCondition(3*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Fatalf("coldLoadKicks[moe] still set after start failure (goroutine leaked)")
	}

	// admission books unchanged: moe still Stopped, no awake/residual delta.
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still admissionStopped after docker start failure, got state %d", a.models["moe"].State)
	}
	a.mu.Lock()
	postAwake := a.awakeByGPU[0]
	postResidual := a.l1ResidualByGPU[0]
	a.mu.Unlock()
	if postAwake != preAwake {
		t.Errorf("awakeByGPU[0] changed: pre=%d post=%d", preAwake, postAwake)
	}
	if postResidual != preResidual {
		t.Errorf("l1ResidualByGPU[0] changed: pre=%d post=%d", preResidual, postResidual)
	}
}

// TestFault_DockerExec_Panics asserts the recover middleware in
// StopForEviction restores state and re-panics. The lock must NOT be
// leaked (we test that by acquiring s.mu after the panic).
func TestFault_DockerExec_Panics(t *testing.T) {
	s, _, _ := makeStopOnEvictScheduler(t)

	SetSleepDockerCmdForTest(failingDockerCmd("panic"))
	defer SetSleepDockerCmdForTest(nil)

	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()

	ev := &SchedulerEvictor{S: s}

	// The deferred recover in StopForEviction restores state then re-panics.
	// Catch the re-panic here.
	var panicVal any
	func() {
		defer func() { panicVal = recover() }()
		_ = ev.StopForEviction(context.Background(), "moe", "test")
	}()
	if panicVal == nil {
		t.Fatalf("expected the panic to propagate after state rollback, got nil")
	}

	// Lock must not be held; we should be able to acquire it.
	acquired := make(chan struct{})
	go func() {
		s.mu.Lock()
		s.mu.Unlock()
		close(acquired)
	}()
	select {
	case <-acquired:
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("scheduler mu still held after panic — lock leak")
	}

	// State restored to pre-panic value.
	s.mu.RLock()
	got := s.instances["moe"].state
	draining := s.instances["moe"].draining
	s.mu.RUnlock()
	if got != StateSleeping {
		t.Errorf("expected state restored to Sleeping after panic recover, got %v", got)
	}
	if draining {
		t.Errorf("expected draining=false after panic recover")
	}
}

// TestFault_DockerNotInstalled distinguishes "executable not found"
// (operator misconfig: no docker binary in image) from "no such container"
// (operator state drift: container name typo or already removed). Both
// paths surface as errors but the operator-meaningful messages differ.
func TestFault_DockerNotInstalled(t *testing.T) {
	s, _, _ := makeStopOnEvictScheduler(t)

	// Scenario 1: docker binary missing.
	SetSleepDockerCmdForTest(failingDockerCmd("not-installed"))
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()
	ev := &SchedulerEvictor{S: s}
	err := ev.StopForEviction(context.Background(), "moe", "test")
	SetSleepDockerCmdForTest(nil)
	if err == nil {
		t.Fatalf("expected error when docker binary missing")
	}
	if !strings.Contains(err.Error(), "executable file not found") {
		t.Errorf("expected error mentioning missing binary, got: %v", err)
	}

	// Scenario 2: docker present but container missing.
	SetSleepDockerCmdForTest(failingDockerCmd("no-such-container"))
	defer SetSleepDockerCmdForTest(nil)
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()
	err2 := ev.StopForEviction(context.Background(), "moe", "test")
	if err2 == nil {
		t.Fatalf("expected error when container missing")
	}
	if !strings.Contains(err2.Error(), "No such container") {
		t.Errorf("expected error mentioning missing container, got: %v", err2)
	}

	// The two error strings MUST be distinguishable to a parser/operator.
	if err.Error() == err2.Error() {
		t.Errorf("not-installed and no-such-container produce identical errors — operator can't differentiate")
	}
}

// ---------------------------------------------------------------------------
// vLLM HTTP failures (6-10)
// ---------------------------------------------------------------------------

// makeVLLMFaultScheduler builds a Scheduler + admission with one model
// (`moe`) whose manager talks to the supplied baseURL via vllmcli. Used
// by the HTTP-level fault tests so failures travel through the real
// wire decoder.
func makeVLLMFaultScheduler(t *testing.T, baseURL string, initState State, admState admissionModelState) (*Scheduler, *AdmissionController, *vllmHTTPMgr) {
	t.Helper()
	yaml := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  moe:
    lifecycle: external
    host: vllm-moe
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 500ms
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgr := &vllmHTTPMgr{baseURL: baseURL, port: 8002}
	s.SeedInstanceForTest("moe", 8002, []int{0}, false, initState, mgr)

	a.mu.Lock()
	switch admState {
	case admissionAwake:
		// Snapshot BookedExpectedVRAMMB to match the awake booking so
		// later markSleepingLocked / markStoppedLocked correctly reverse
		// it. Mirrors the snapshot discipline that the production
		// RequestWake awake-commit path uses.
		m := a.models["moe"]
		m.State = admissionAwake
		booked := make(map[int]int, len(m.GPUs))
		for _, g := range m.GPUs {
			v := m.ExpectedOn(g)
			a.awakeByGPU[g] += v
			booked[g] = v
		}
		m.BookedExpectedVRAMMB = booked
	case admissionSleeping:
		m := a.models["moe"]
		m.State = admissionSleeping
		if m.L1ResidualMB > 0 {
			for _, g := range m.GPUs {
				a.l1ResidualByGPU[g] += m.L1ResidualMB
			}
			m.BookedL1ResidualMB = m.L1ResidualMB
			m.BookedL1ResidualGPUs = append([]int(nil), m.GPUs...)
		}
	}
	a.mu.Unlock()

	return s, a, mgr
}

// TestFault_VLLM_SleepEndpoint_Returns500 asserts that a 5xx response
// from /sleep where vLLM is HONESTLY still awake (verified via the
// post-error /is_sleeping wedge-recovery probe returning false)
// surfaces as an error from sleepInstance and the instance state is
// restored to its pre-call value (admission books not flipped).
//
// The wedge-recovery counterpart (where /is_sleeping returns true after
// /sleep errored) is exercised by TestFault_VLLM_SleepEndpoint_Returns500_VLLMActuallyAsleep.
func TestFault_VLLM_SleepEndpoint_Returns500(t *testing.T) {
	srv := failingVLLMServer("sleep-500-vllm-awake")
	defer srv.Close()

	s, a, _ := makeVLLMFaultScheduler(t, srv.URL, StateReady, admissionAwake)

	preAwake := a.SnapshotBudgets()[0].AwakeMB

	// SleepForEviction should fail (vLLM /sleep returned 500).
	ev := &SchedulerEvictor{S: s}
	err := ev.SleepForEviction(context.Background(), "moe", "test")
	if err == nil {
		t.Fatalf("expected SleepForEviction error on /sleep 500, got nil")
	}
	if !strings.Contains(err.Error(), "500") && !strings.Contains(err.Error(), "Internal") {
		t.Errorf("expected error mentioning 500 status, got: %v", err)
	}

	// Instance state restored — did NOT flip to Sleeping.
	s.mu.RLock()
	got := s.instances["moe"].state
	s.mu.RUnlock()
	if got != StateReady {
		t.Errorf("expected state restored to Ready after /sleep 500, got %v", got)
	}

	// Admission books: this is the SchedulerEvictor → sleepInstance path,
	// which on failure restores state. admission books are owned by
	// the caller of SleepForEviction (RequestWake) per the evictor
	// contract — those books should not have been touched.
	postAwake := a.SnapshotBudgets()[0].AwakeMB
	if postAwake != preAwake {
		t.Errorf("admission awake books changed across failed /sleep: pre=%d post=%d", preAwake, postAwake)
	}
}

// TestFault_VLLM_SleepEndpoint_Returns500_VLLMActuallyAsleep asserts the
// wedge-recovery counterpart: /sleep returns 500 BUT the post-error
// /is_sleeping probe returns true (vLLM accepted the /sleep call before
// the connection dropped — connection error / ctx cancellation between
// vLLM acknowledging and the response reaching the client). In that
// case sleepInstance MUST NOT roll back to StateReady — that would
// leave the proxy forwarding requests to a sleeping backend. Instead
// the instance state flips to StateSleeping, NotifySleep fires with
// the dedicated "sleep-api-error-but-vllm-asleep" reason, and an error
// still surfaces to the caller (so they know the API call failed even
// though the underlying state is consistent).
func TestFault_VLLM_SleepEndpoint_Returns500_VLLMActuallyAsleep(t *testing.T) {
	srv := failingVLLMServer("sleep-500")
	defer srv.Close()

	s, a, _ := makeVLLMFaultScheduler(t, srv.URL, StateReady, admissionAwake)

	preAwake := a.SnapshotBudgets()[0].AwakeMB
	if preAwake == 0 {
		t.Fatalf("test setup: expected non-zero pre-call awake budget, got 0")
	}

	ev := &SchedulerEvictor{S: s}
	err := ev.SleepForEviction(context.Background(), "moe", "test")
	if err == nil {
		t.Fatalf("expected SleepForEviction error on /sleep 500 even with wedge-recovery, got nil")
	}
	// Error must surface the underlying 500 AND mention the recovery so
	// operators can grep for the wedge-recovery path in logs.
	if !strings.Contains(err.Error(), "500") && !strings.Contains(err.Error(), "Internal") {
		t.Errorf("expected error mentioning 500 status, got: %v", err)
	}
	if !strings.Contains(err.Error(), "actually sleeping") && !strings.Contains(err.Error(), "recovered") {
		t.Errorf("expected error mentioning wedge-recovery, got: %v", err)
	}

	// Instance state MUST flip to Sleeping — NOT roll back to Ready.
	// Rolling back here is the wedge bug Fix 12 closes: the proxy would
	// forward to a sleeping backend and requests hang.
	s.mu.RLock()
	got := s.instances["moe"].state
	draining := s.instances["moe"].draining
	s.mu.RUnlock()
	if got != StateSleeping {
		t.Errorf("expected state advanced to Sleeping after wedge-recovery, got %v", got)
	}
	if draining {
		t.Errorf("expected draining=false after wedge-recovery, got true")
	}

	// Admission books: NotifySleep should have fired (reason="test" is
	// NOT admissionReason, so the !admissionReason branch in sleepInstance
	// triggers). Awake budget for the model (1000MB) should have moved
	// out of awakeByGPU; with sleep_l1_residual_mb=0 it goes nowhere
	// (full reclaim), leaving awakeByGPU at preAwake-1000.
	postAwake := a.SnapshotBudgets()[0].AwakeMB
	if postAwake >= preAwake {
		t.Errorf("expected admission awake budget to drop after wedge-recovery NotifySleep: pre=%d post=%d", preAwake, postAwake)
	}
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected admission model state admissionSleeping after wedge-recovery, got %d", a.models["moe"].State)
	}
}

// TestFault_VLLM_WakeEndpoint_TimesOut asserts that when /wake_up never
// returns, performWake hits its bounded timeout and rolls back via
// NotifySleep("wake-rollback"). The model returns to admissionSleeping,
// not admissionAwake.
func TestFault_VLLM_WakeEndpoint_TimesOut(t *testing.T) {
	srv := failingVLLMServer("wake-hang")
	defer srv.Close()

	s, a, _ := makeVLLMFaultScheduler(t, srv.URL, StateSleeping, admissionSleeping)

	modelCfg := s.cfg.Models["moe"]
	s.mu.RLock()
	inst := s.instances["moe"]
	s.mu.RUnlock()

	// Use a tight outer context so even if vllmcli's deadline somehow
	// races us, the test still terminates fast.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := s.performWake(ctx, inst, modelCfg, "test")
	if err == nil {
		t.Fatalf("expected wake to fail when /wake_up hangs, got nil")
	}

	// State must be back to Sleeping (NOT Awake or any partial state).
	if a.models["moe"].State != admissionSleeping {
		t.Errorf("expected admissionSleeping after wake rollback, got state %d", a.models["moe"].State)
	}
	// Scheduler instance state was never flipped to Ready.
	s.mu.RLock()
	got := s.instances["moe"].state
	s.mu.RUnlock()
	if got == StateReady {
		t.Errorf("expected instance NOT in StateReady after wake timeout, got %v", got)
	}
}

// TestFault_VLLM_IsSleeping_FlipsMidPoll asserts that doColdLoad's poll
// loop tolerates a transient flap (false then true). The endpoint
// returning false is not the success condition — the success condition
// per current implementation is "no error" on the response. This test
// confirms the loop exits on the FIRST non-error response (whatever its
// boolean value), which is the documented contract.
func TestFault_VLLM_IsSleeping_FlipsMidPoll(t *testing.T) {
	srv := failingVLLMServer("is_sleeping-flap")
	defer srv.Close()

	s, _, _ := makeStopOnEvictScheduler(t)

	// Replace moe's manager with a real-HTTP one pointing at our flapping server.
	httpMgr := &vllmHTTPMgr{baseURL: srv.URL, port: 8002}
	s.mu.Lock()
	s.instances["moe"].mgr = httpMgr
	s.mu.Unlock()

	// docker start should succeed so doColdLoad reaches the poll loop.
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Tight ctx; a single successful poll suffices to exit the loop.
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	if err := s.coldLoadStoppedMember(ctx, s.instances["moe"], s.cfg.Models["moe"]); err != nil {
		t.Fatalf("expected cold-load to succeed despite flap, got: %v", err)
	}
}

// TestFault_VLLM_IsSleeping_AlwaysReturnsConnectionRefused asserts that
// when /is_sleeping is unreachable for the entire poll budget, the
// timeout fires and the best-effort cleanup `docker stop` runs.
func TestFault_VLLM_IsSleeping_AlwaysReturnsConnectionRefused(t *testing.T) {
	// Pre-closed server — every connection refused.
	srv := failingVLLMServer("is_sleeping-refused")
	// (already Close()'d inside the helper)

	s, a, _ := makeStopOnEvictScheduler(t)

	httpMgr := &vllmHTTPMgr{baseURL: srv.URL, port: 8002}
	s.mu.Lock()
	s.instances["moe"].mgr = httpMgr
	s.mu.Unlock()

	var dockerMu sync.Mutex
	startCount := 0
	stopCount := 0
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		defer dockerMu.Unlock()
		if len(args) > 0 && args[0] == "start" {
			startCount++
		}
		if len(args) > 0 && args[0] == "stop" {
			stopCount++
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Force a tiny startup timeout so the test stays under 5s. The helper
	// bumps to 600s default if <60s, so we use a tight outer ctx as the
	// real upper bound (doColdLoad's poll respects ctx.Done).
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()

	err := s.coldLoadStoppedMember(ctx, s.instances["moe"], s.cfg.Models["moe"])
	if err == nil {
		t.Fatalf("expected cold-load to fail when /is_sleeping unreachable")
	}

	// Cleanup stop must have fired.
	dockerMu.Lock()
	if startCount != 1 {
		t.Errorf("expected exactly 1 docker start, got %d", startCount)
	}
	if stopCount < 1 {
		t.Errorf("expected at least 1 best-effort docker stop cleanup, got %d", stopCount)
	}
	dockerMu.Unlock()

	// Admission books must reflect Stopped — not wedged Awake/Sleeping.
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still admissionStopped after cold-load failure, got state %d", a.models["moe"].State)
	}
}

// TestFault_VLLM_ResponseInvalidJSON asserts that a 200 OK with non-JSON
// body to /is_sleeping surfaces as an error rather than crashing the
// caller.
func TestFault_VLLM_ResponseInvalidJSON(t *testing.T) {
	srv := failingVLLMServer("is_sleeping-garbage")
	defer srv.Close()

	got, err := vllmcli.IsSleeping(context.Background(), srv.URL)
	if err == nil {
		t.Fatalf("expected error decoding garbage JSON, got is_sleeping=%v", got)
	}
	// Error must surface as a decode error, not a panic / nil deref.
	if !strings.Contains(err.Error(), "decode") && !strings.Contains(err.Error(), "invalid") && !strings.Contains(err.Error(), "JSON") {
		t.Errorf("expected decode-style error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Context propagation (11-12)
// ---------------------------------------------------------------------------

// TestFault_ContextCancelled_Mid_RequestWake asserts that when the
// caller's ctx is cancelled mid-wake, admission does not leave
// half-committed state. Specifically: if the evictor is invoked and ctx
// cancels DURING the evictor call, the controller must not have
// awake/residual books reflecting a partial transition.
func TestFault_ContextCancelled_Mid_RequestWake(t *testing.T) {
	s, a, mgrs := makeStopOnEvictScheduler(t)

	// main is awake, moe is sleeping (not stopped) so a wake of moe will
	// trigger main as victim — putting us in admission's eviction loop
	// when we cancel.
	a.mu.Lock()
	a.markStartedLocked(a.models["moe"]) // Stopped → Sleeping
	a.mu.Unlock()
	s.mu.Lock()
	s.instances["moe"].state = StateSleeping
	s.mu.Unlock()
	mgrs["moe"].pid.Store(int64(7000 + 8002))
	mgrs["moe"].isSleeping.Store(true)

	preAwake := a.SnapshotBudgets()[0].AwakeMB
	preResidual := a.SnapshotBudgets()[0].L1ResidualMB

	// Cancel before calling so the very first ctx-aware op observes the cancel.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := a.RequestWake(ctx, "moe")
	// RequestWake might or might not reject for ctx cancel — depends on
	// where the cancel surfaces. The load-bearing assertion is the
	// invariant: books are coherent, not half-committed.
	_ = err

	// Books must be self-consistent. Either the wake fully succeeded
	// (moe Awake, main Sleeping) or fully failed (no change). No
	// "evicted A but didn't book B" state is permitted.
	a.mu.Lock()
	moeState := a.models["moe"].State
	mainState := a.models["main"].State
	awake := a.awakeByGPU[0]
	residual := a.l1ResidualByGPU[0]
	a.mu.Unlock()

	if moeState == admissionAwake {
		// Success path: main must be Sleeping, books reflect both transitions.
		if mainState != admissionSleeping {
			t.Errorf("inconsistent: moe=Awake but main not Sleeping (got %d)", mainState)
		}
	} else {
		// Failure/no-change path: state and books must match pre-test.
		if awake != preAwake {
			t.Errorf("inconsistent: failed wake but awakeByGPU changed (pre=%d post=%d)", preAwake, awake)
		}
		if residual != preResidual {
			t.Errorf("inconsistent: failed wake but residual changed (pre=%d post=%d)", preResidual, residual)
		}
	}
}

// TestFault_ContextCancelled_Mid_ColdLoad asserts that the cold-load
// goroutine kicked by KickColdLoad has its OWN context (not the inbound
// request's), so cancelling the inbound request does not abort the
// cold-load. A subsequent retry sees the cold-load already in flight
// (deduped via coldLoadKicks) and rides it.
func TestFault_ContextCancelled_Mid_ColdLoad(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	// Slow docker start so the cold-load is in flight when we cancel.
	startEntered := make(chan struct{})
	releaseStart := make(chan struct{})
	var startFired atomic.Bool
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			if startFired.CompareAndSwap(false, true) {
				close(startEntered)
				<-releaseStart
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Inbound request ctx (will be cancelled).
	inboundCtx, inboundCancel := context.WithCancel(context.Background())

	// Kick — this spawns a goroutine that does NOT consume inboundCtx.
	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false")
	}

	// Wait for docker start to be in flight, then cancel inbound.
	// Budget 5s — the BUG 1 fix evicts pinned main (sleepInstance) before
	// docker start, and sleepInstance pays a hardcoded 2s settleAfterSleep.
	select {
	case <-startEntered:
	case <-time.After(5 * time.Second):
		releaseStart <- struct{}{}
		t.Fatalf("docker start did not fire within 5s")
	}
	inboundCancel() // cancel the original inbound

	// A second KickColdLoad while the goroutine is still running must
	// dedupe (return false — kicks map says one is in flight).
	if s.KickColdLoad("moe") {
		t.Errorf("expected dedupe: second KickColdLoad while cold-load in flight should return false")
	}

	// Release docker start; the goroutine should still complete despite
	// the inbound ctx being cancelled.
	close(releaseStart)

	if ok := waitForCondition(8*time.Second, func() bool {
		return !a.IsStopped("moe")
	}); !ok {
		t.Fatalf("cold-load did not complete after inbound ctx cancel — goroutine inherited cancelled ctx (BUG)")
	}

	// Avoid the "ctx never cancelled" lint.
	_ = inboundCtx
}

// ---------------------------------------------------------------------------
// Edge inputs (13-14)
// ---------------------------------------------------------------------------

// TestFault_RedeployMember_ModelNameWithSpecialChars asserts that
// RedeployMember rejects malicious model names cleanly (via
// ErrRedeployUnknownModel), without ever passing the name as a docker
// container argument. Defense in depth — docker stop/start would NOT
// shell-interpret args (Go's exec.Command does not invoke a shell), but
// rejecting at the lookup layer avoids surprising edge cases.
func TestFault_RedeployMember_ModelNameWithSpecialChars(t *testing.T) {
	s, _, _ := makeStopOnEvictScheduler(t)

	// Track every docker invocation — none should fire for a bad name.
	var dockerCalls atomic.Int32
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		dockerCalls.Add(1)
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)
	SetDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		dockerCalls.Add(1)
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)

	malicious := []string{
		"../../../etc/passwd",
		"x; rm -rf /",
		"$(touch /tmp/owned)",
		"`whoami`",
		"\x00null",
		"vllm-moe\nINJECTED",
	}
	for _, name := range malicious {
		t.Run(fmt.Sprintf("%q", name), func(t *testing.T) {
			_, err := s.RedeployMember(context.Background(), name)
			if err == nil {
				t.Errorf("expected error for malicious name %q, got nil", name)
				return
			}
			if !errors.Is(err, ErrRedeployUnknownModel) && !errors.Is(err, ErrRedeployIneligible) {
				t.Errorf("expected ErrRedeployUnknownModel or ErrRedeployIneligible, got: %v", err)
			}
		})
	}

	// No docker exec should have fired for any malicious name.
	if got := dockerCalls.Load(); got != 0 {
		t.Errorf("docker was invoked %d times for malicious model names — should be zero", got)
	}
}

// TestFault_RedeployMember_OversizedBody asserts that handling an
// oversized JSON body does not crash the JSON decoder. The HTTP layer
// (Fiber) has its own body-size limit which produces 413 when exceeded;
// the assertion here is that the in-process decode of a 10MB payload
// returns cleanly (success or error), not a panic.
//
// Lives in the jukebox package (not httpserver_test) so it can share
// the deferred-tool helpers; the assertion target is the parser
// robustness rather than the precise HTTP status, which depends on
// fiber.Config (BodyLimit) wired in main.go and not in the test harness.
func TestFault_RedeployMember_OversizedBody(t *testing.T) {
	const size = 10 * 1024 * 1024
	var sb strings.Builder
	sb.Grow(size + 64)
	sb.WriteString(`{"model":"`)
	for sb.Len() < size {
		sb.WriteString("aaaaaaaaaa")
	}
	sb.WriteString(`"}`)
	body := sb.String()

	var req redeployMemberRequestShape
	if err := json.Unmarshal([]byte(body), &req); err == nil {
		// 10MB of "a"s as a string field IS valid JSON. The real
		// rejection is at the body-size limit (Fiber default 4MB returns
		// 413). Confirm the parser captured the bulk of the payload
		// without panicking.
		if len(req.Model) < 1024*1024 {
			t.Errorf("expected the model field to capture the bulk of the payload, got len=%d", len(req.Model))
		}
	}

	// A garbage payload of equal size must surface a clean decode error.
	garbage := strings.Repeat("not-json-at-all", size/15)
	var req2 redeployMemberRequestShape
	if err := json.Unmarshal([]byte(garbage), &req2); err == nil {
		t.Errorf("expected garbage JSON to error, got nil")
	}
}

// redeployMemberRequestShape mirrors the handler's body shape so the
// oversized-body test can decode without importing httpserver (which
// would create a package cycle from internal/jukebox).
type redeployMemberRequestShape struct {
	Model string `json:"model"`
}

// ---------------------------------------------------------------------------
// Architect-fixes regression tests (peer-stop ctx-cancel, cleanup-stop drift,
// is_sleeping bool propagation)
// ---------------------------------------------------------------------------

// TestFault_RedeployMember_PeerStopLoop_BailsOnCtxCancel asserts Fix 1:
// when an operator cancels /admin/redeploy-member mid peer-stop loop, the
// loop must check ctx.Err() at the top of each iteration and bail
// without mutating admission state for peers it never reached. The
// pre-fix behavior unconditionally ran NotifyStopped + state flip + an
// instantly-cancelled docker stop, marking peers Stopped on the books
// while their containers continued running and holding tens of GiB of
// VRAM.
//
// This needs ≥3 evict_action: stop peers in the same swap_group so that
// when we redeploy one, there are ≥2 stopPeers to iterate — we cancel
// after the first peer's docker stop fires, asserting the second peer
// is NOT touched.
func TestFault_RedeployMember_PeerStopLoop_BailsOnCtxCancel(t *testing.T) {
	const cfgYAML = `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  target:
    lifecycle: external
    host: vllm-target
    port: 8001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
    cold_load_timeout_seconds: 30
  peerA:
    lifecycle: external
    host: vllm-peerA
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
  peerB:
    lifecycle: external
    host: vllm-peerB
    port: 8003
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    swap_group: G
    evict_action: stop
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 1s
`
	s, a, mgrs := makeRedeployScheduler(t, cfgYAML,
		map[string]State{
			"target": StateSleeping,
			"peerA":  StateSleeping,
			"peerB":  StateSleeping,
		},
		map[int]int{0: 24000},
	)

	// Block on the first peer's docker stop, signal the test, then
	// continue when released. After the first peer is fully stopped,
	// the loop will iterate to the second peer — at which point we
	// want the ctx to already be cancelled.
	firstStopEntered := make(chan struct{})
	releaseFirstStop := make(chan struct{})
	var stopFired atomic.Bool
	var dockerCallNames []string
	var dockerMu sync.Mutex
	SetDockerCmdForTest(func(_ context.Context, name string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		dockerCallNames = append(dockerCallNames, name+" "+strings.Join(args, " "))
		dockerMu.Unlock()
		// First peer stop: block until released so we can cancel mid-flight.
		if len(args) >= 2 && args[0] == "stop" && stopFired.CompareAndSwap(false, true) {
			close(firstStopEntered)
			<-releaseFirstStop
		}
		// docker start: flip the target's isSleeping=true so the post-
		// cold-load poll succeeds (defensive — we expect ctx cancel to
		// short-circuit before reaching docker start, but be safe).
		if len(args) >= 2 && args[0] == "start" {
			for _, m := range mgrs {
				m.isSleeping.Store(true)
			}
		}
		return []byte("ok"), nil
	})
	defer SetDockerCmdForTest(nil)
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Pre-state: neither peer admissionStopped.
	if a.IsStopped("peerA") || a.IsStopped("peerB") {
		t.Fatalf("setup: expected neither peer to start admissionStopped")
	}

	ctx, cancel := context.WithCancel(context.Background())
	resultCh := make(chan error, 1)
	go func() {
		_, err := s.RedeployMember(ctx, "target")
		resultCh <- err
	}()

	// Wait for the first peer-stop to be in flight.
	select {
	case <-firstStopEntered:
	case <-time.After(5 * time.Second):
		cancel()
		close(releaseFirstStop)
		t.Fatalf("first docker stop did not fire within 5s")
	}

	// Cancel ctx — the redeploy must observe ctx.Err() at the TOP of
	// the next loop iteration and bail without mutating admission for
	// the unreached peer.
	cancel()
	close(releaseFirstStop)

	// Wait for redeploy to return.
	select {
	case err := <-resultCh:
		if err == nil {
			t.Fatalf("expected redeploy to return ctx.Err(), got nil")
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("expected context.Canceled, got: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("redeploy did not return after ctx cancel")
	}

	// At most ONE peer should have been admission-stopped. The other
	// must NOT be flipped — that would be the silent VRAM drift the fix
	// closes.
	flippedCount := 0
	if a.IsStopped("peerA") {
		flippedCount++
	}
	if a.IsStopped("peerB") {
		flippedCount++
	}
	if flippedCount > 1 {
		t.Errorf("expected at most 1 peer flipped to admissionStopped after ctx-cancel, got %d", flippedCount)
	}

	// At most one peer docker stop landed.
	dockerMu.Lock()
	defer dockerMu.Unlock()
	peerStopCount := 0
	for _, call := range dockerCallNames {
		if strings.Contains(call, "stop") &&
			(strings.Contains(call, "vllm-peerA") || strings.Contains(call, "vllm-peerB")) {
			peerStopCount++
		}
	}
	if peerStopCount > 1 {
		t.Errorf("expected at most 1 peer docker stop after ctx-cancel, got %d (calls=%v)",
			peerStopCount, dockerCallNames)
	}
}

// TestFault_ColdLoad_CleanupStopFails_DoesNotMarkAdmissionStopped asserts
// Fix 2: when doColdLoad fails AND the best-effort cleanup `docker stop`
// ALSO fails, admission must NOT call NotifyStartFailed (which would
// mark the model admissionStopped → zero VRAM on the books). The
// container may still be running and holding tens of GiB; marking it
// Stopped would hide the leak from every future scheduling decision.
// The drift-risk metric must bump and the wrapped error must surface.
func TestFault_ColdLoad_CleanupStopFails_DoesNotMarkAdmissionStopped(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	// vLLM /is_sleeping always returns connection-refused → poll exhausts
	// the budget → doColdLoad returns timeout error → cleanup `docker stop`
	// runs → cleanup ALSO fails (we wire it to error).
	srv := failingVLLMServer("is_sleeping-refused")
	httpMgr := &vllmHTTPMgr{baseURL: srv.URL, port: 8002}
	s.mu.Lock()
	s.instances["moe"].mgr = httpMgr
	s.mu.Unlock()

	var startCount, stopCount atomic.Int32
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			startCount.Add(1)
			return []byte("ok"), nil // start succeeds, but vLLM is refusing connections
		}
		if len(args) > 0 && args[0] == "stop" {
			stopCount.Add(1)
			// Cleanup stop fails — daemon hiccup simulation.
			return []byte("Error response from daemon: connection refused"),
				errors.New("exit status 1")
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Capture pre-state metric value (counters are package-global and
	// can have nonzero pre-state from earlier tests in the same run).
	preMetric := readDriftRiskCounter(t, "moe")

	// Tight ctx so the test stays under 5s.
	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()

	err := s.coldLoadStoppedMember(ctx, s.instances["moe"], s.cfg.Models["moe"])
	if err == nil {
		t.Fatalf("expected coldLoadStoppedMember to error when both cold-load and cleanup fail")
	}
	// The error message must surface the drift-risk warning.
	if !strings.Contains(err.Error(), "vram-drift-risk") {
		t.Errorf("expected error to mention vram-drift-risk, got: %v", err)
	}

	// admission books MUST still consider moe Stopped (its starting state
	// in makeStopOnEvictScheduler) — but the load-bearing assertion is
	// that the metric bumped, indicating the drift-risk path executed
	// rather than the "all clear, NotifyStartFailed" path.
	if !a.IsStopped("moe") {
		t.Errorf("expected moe to remain admissionStopped (in-flight state preserved on drift-risk)")
	}

	// docker start ran exactly once, docker stop (cleanup) ran at least once.
	if startCount.Load() != 1 {
		t.Errorf("expected 1 docker start, got %d", startCount.Load())
	}
	if stopCount.Load() < 1 {
		t.Errorf("expected at least 1 cleanup docker stop, got %d", stopCount.Load())
	}

	// Drift-risk metric bumped.
	postMetric := readDriftRiskCounter(t, "moe")
	if postMetric <= preMetric {
		t.Errorf("expected AdmissionVRAMDriftRiskTotal to bump for moe (pre=%v post=%v)",
			preMetric, postMetric)
	}
}

// readDriftRiskCounter returns the float64 value of
// metrics.AdmissionVRAMDriftRiskTotal{model=name}. Used by Fix 2 tests
// to assert the metric bumped — uses the prometheus dto directly to
// avoid pulling in testutil (whose transitive deps the offline-proxy
// rejects).
func readDriftRiskCounter(t *testing.T, model string) float64 {
	t.Helper()
	c, err := metrics.AdmissionVRAMDriftRiskTotal.GetMetricWithLabelValues(model)
	if err != nil {
		t.Fatalf("get drift-risk counter: %v", err)
	}
	d := &dto.Metric{}
	if err := c.Write(d); err != nil {
		t.Fatalf("write counter dto: %v", err)
	}
	if d.Counter == nil {
		return 0
	}
	return d.Counter.GetValue()
}

// TestFault_DoColdLoad_DoesNotReturnEarlyOnIsSleepingFalse asserts Fix 3:
// doColdLoad must wait for is_sleeping=true, NOT return on the first
// 200 response that says is_sleeping=false. Sister fn pollUntilSleeping
// (redeploy.go) was already correct; the fix propagates to doColdLoad.
//
// vLLM's HTTP server can come up before the model is loaded into VRAM,
// answering /is_sleeping with 200 + {"is_sleeping": false}. Returning
// early on err==nil flips admission to Sleeping prematurely; a consumer
// /wake_up arrives before the model is resident. The test wires a vLLM
// fake whose first poll returns false, second returns true, and asserts
// doColdLoad keeps polling until the transition.
func TestFault_DoColdLoad_DoesNotReturnEarlyOnIsSleepingFalse(t *testing.T) {
	srv := failingVLLMServer("is_sleeping-flap")
	defer srv.Close()

	s, _, _ := makeStopOnEvictScheduler(t)

	httpMgr := &vllmHTTPMgr{baseURL: srv.URL, port: 8002}
	s.mu.Lock()
	s.instances["moe"].mgr = httpMgr
	s.mu.Unlock()

	SetSleepDockerCmdForTest(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Speed up the test: shrink the poll interval so we don't wait 5s
	// for the second poll. coldLoadPollInterval is a package-level
	// atomic.Int64 (LOW #9) so tests can override; restore on exit.
	prevInterval := SetColdLoadPollIntervalForTest(50 * time.Millisecond)
	defer SetColdLoadPollIntervalForTest(prevInterval)

	t0 := time.Now()
	err := s.coldLoadStoppedMember(context.Background(), s.instances["moe"], s.cfg.Models["moe"])
	elapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("expected cold-load to succeed at the is_sleeping=true transition, got: %v", err)
	}

	// Must have waited AT LEAST one poll interval — proving we did NOT
	// return on the first poll's `is_sleeping: false`. The flap server
	// returns false on n==1 and true on n>=2, so success requires at
	// least 2 polls (≥ 2x interval, conservatively 1x).
	curInterval := coldLoadPollIntervalDuration()
	if elapsed < curInterval {
		t.Errorf("doColdLoad returned faster (%s) than one poll interval (%s) — looks like it returned on is_sleeping=false (BUG)",
			elapsed, curInterval)
	}
}

// flakyProbeMgr is a SleepCapable fake whose IsSleeping returns errors
// on the first N calls then returns (true, nil). Used by the MEDIUM #5
// wedge-probe-retry test to verify that a transient probe flake doesn't
// cause sleepInstance to misclassify a wedged-asleep vLLM as wedged-awake.
type flakyProbeMgr struct {
	probeFailsBeforeSuccess int
	probeCalls              atomic.Int32
	sleepErr                error
}

func (m *flakyProbeMgr) Start(_ context.Context, _ string) (int, error) { return 1, nil }
func (m *flakyProbeMgr) Stop(_ context.Context) error                   { return nil }
func (m *flakyProbeMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *flakyProbeMgr) CurrentPID() int                                { return 1 }
func (m *flakyProbeMgr) BaseURL() string                                { return "" }
func (m *flakyProbeMgr) Sleep(_ context.Context, _ int) error           { return m.sleepErr }
func (m *flakyProbeMgr) Wake(_ context.Context, _ time.Duration) error  { return nil }
func (m *flakyProbeMgr) IsSleeping(_ context.Context) (bool, error) {
	n := m.probeCalls.Add(1)
	if int(n) <= m.probeFailsBeforeSuccess {
		return false, errors.New("transient probe network hiccup")
	}
	return true, nil
}

// TestFault_WedgeProbe_RetriesOnFlake_RecoversOnThird asserts MEDIUM #5:
// the wedge-recovery probe retries up to 3 times with bounded backoff.
// First 2 IsSleeping calls error (network hiccup, busy box, etc.); the
// 3rd returns true. sleepInstance must take the wedge-recovery path
// (NOT optimistically rollback as the prior single-attempt code did).
func TestFault_WedgeProbe_RetriesOnFlake_RecoversOnThird(t *testing.T) {
	yaml := `
scheduler:
  port_range_start: 8100
  port_range_end: 8199
vllm:
  port: 8000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  moe:
    lifecycle: external
    host: vllm-moe
    port: 8002
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    sleep_mode: true
    expected_vram_mb_per_gpu: 1000
    sleep_l1_residual_mb: 0
    wake_timeout: 500ms
`
	cfg, err := config.Load([]byte(yaml))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgr := &flakyProbeMgr{
		probeFailsBeforeSuccess: 2, // first 2 probes error, 3rd succeeds
		sleepErr:                errors.New("simulated /sleep 500"),
	}
	s.SeedInstanceForTest("moe", 8002, []int{0}, false, StateReady, mgr)

	a.mu.Lock()
	a.awakeByGPU[0] += a.models["moe"].ExpectedOn(0)
	a.models["moe"].State = admissionAwake
	a.mu.Unlock()

	t0 := time.Now()
	err = s.sleepInstance(context.Background(), s.instances["moe"], 1, "test")
	elapsed := time.Since(t0)

	// Sleep API errored, so an error must surface.
	if err == nil {
		t.Fatalf("expected sleep-API error to surface even on wedge-recovery, got nil")
	}
	if !strings.Contains(err.Error(), "actually sleeping") && !strings.Contains(err.Error(), "recovered") {
		t.Errorf("expected error to mention wedge-recovery, got: %v", err)
	}

	// Probes called exactly 3 times — 2 failures + 1 success.
	if got := mgr.probeCalls.Load(); got != 3 {
		t.Errorf("expected 3 probe attempts (retry on flake), got %d", got)
	}

	// State MUST be StateSleeping (wedge-recovered), NOT StateReady (rollback).
	s.mu.RLock()
	gotState := s.instances["moe"].state
	s.mu.RUnlock()
	if gotState != StateSleeping {
		t.Errorf("expected wedge-recovery to flip state to StateSleeping, got %v (probe-flake regressed to optimistic rollback)", gotState)
	}

	// Must have waited at least 500ms + 1s = 1.5s of backoff between attempts.
	// (Actually 500ms before attempt 2, 1s before attempt 3.) Allow some slop
	// for fast machines but expect ≥ 1.4s.
	if elapsed < 1400*time.Millisecond {
		t.Errorf("expected wedge-probe-retry to take ≥ 1.4s of bounded backoff, got %v (retries may not be sleeping correctly)", elapsed)
	}
}

// TestFault_WedgeRecovery_PreservesOriginalReason asserts LOW #8: when
// wedge-recovery fires from an idle-suspend (reason="idle"), NotifySleep
// must receive the ORIGINAL "idle" reason so admission's swap-group
// auto-restore gate (`reason == "idle"`) fires. Previously the wedge
// path passed a hardcoded "sleep-api-error-but-vllm-asleep" string,
// silently skipping auto-restore on a swap-group member.
func TestFault_WedgeRecovery_PreservesOriginalReason(t *testing.T) {
	srv := failingVLLMServer("sleep-500")
	defer srv.Close()

	s, a, _ := makeVLLMFaultScheduler(t, srv.URL, StateReady, admissionAwake)

	// Install a swap-group with a sleeping peer and arm auto-restore.
	// Add a 2nd model with the same swap_group so admission has a peer
	// to wake.
	a.mu.Lock()
	a.models["moe"].SwapGroup = "g"
	a.mu.Unlock()
	// Add a fake peer in same swap_group, sleeping, so auto-restore picks it.
	peerCfg := config.ModelConfig{
		Lifecycle:            "external",
		Host:                 "vllm-peer",
		Port:                 8003,
		GPUs:                 []int{0},
		SleepMode:            true,
		ExpectedVRAMMBPerGPU: 100,
		SwapGroup:            "g",
	}
	s.cfg.Models["peer"] = peerCfg
	a.mu.Lock()
	a.models["peer"] = &modelAdmissionState{
		Name:           "peer",
		GPUs:           []int{0},
		ExpectedVRAMMB: map[int]int{0: 100},
		L1ResidualMB:   0,
		Priority:       config.PriorityNormal,
		SwapGroup:      "g",
		State:          admissionSleeping,
	}
	a.mu.Unlock()

	// Hook auto-restore so we can observe whether it fires (and with what
	// reason). Captured via channel since the callback runs in a goroutine.
	autoRestoreCalled := make(chan string, 1)
	a.SetAutoRestoreWake(func(name, reason string) {
		select {
		case autoRestoreCalled <- name + ":" + reason:
		default:
		}
	})

	// Trigger sleepInstance with reason="idle" — same shape as auto-suspend.
	err := s.sleepInstance(context.Background(), s.instances["moe"], 1, "idle")
	if err == nil {
		t.Fatalf("expected sleep-API error to surface, got nil")
	}

	// Auto-restore must have fired (this is the LOW #8 check). Wait briefly
	// for the goroutine.
	select {
	case got := <-autoRestoreCalled:
		// Must reference the peer (not moe). Reason is admission's choice
		// ("swap-group-auto-restore").
		if !strings.Contains(got, "peer") {
			t.Errorf("expected auto-restore to wake 'peer', got %q", got)
		}
	case <-time.After(2 * time.Second):
		t.Errorf("expected swap-group auto-restore to fire on wedge-recovery with idle reason (LOW #8 regressed: original reason was lost)")
	}
}
