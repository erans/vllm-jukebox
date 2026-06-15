package jukebox

// Forensics A2 regression test.
//
// BUG (confirmed in live logs): when an operator runs
// `docker compose up --force-recreate vllm-main`, docker removes-then-
// recreates the container. During the rm-gap the container name does not
// resolve. If jukebox's cold-load path fires `docker start vllm-main` in
// that window, docker returns "No such container: vllm-main".
//
// PRE-FIX: that start error fell through to the generic cold-load failure
// handling. The cleanup `docker stop` ALSO hit "No such container", which
// latched the hard ErrAdmissionVRAMDriftRisk drift-risk audit + metric AND
// recorded the 30s cold-load failure cooldown. The cooldown then made the
// ExternalStartMonitor's KickColdLoad refuse (cooldown_active) for the full
// window — actively DELAYING the external-start reconcile that adopts the
// recreated container once it shows running.
//
// POST-FIX: doColdLoad detects the missing-container start error and returns
// ErrColdLoadContainerMissing; coldLoadStoppedMemberLocked short-circuits
// before the cleanup-stop / drift-risk block, and the KickColdLoad goroutine
// skips the cooldown record — deferring to external-start reconcile.
//
// This test FAILS pre-fix (cooldown latched + drift-risk audit fires) and
// PASSES post-fix (no cooldown, no drift-risk audit, container-missing audit).

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestFault_ColdLoad_ContainerMissing_NoCooldownLatch_DefersToExternalStart(t *testing.T) {
	s, a, _ := makeStopOnEvictScheduler(t)

	// Keep the production-size cooldown so a latch would be unambiguous,
	// but make poll/timeout fast so the (short-circuited) path is quick.
	restoreCooldown := SetColdLoadFailureCooldownForTest(30 * time.Second)
	defer SetColdLoadFailureCooldownForTest(restoreCooldown)

	// Mock docker: `start moe` returns the docker rm-gap error.
	// Track verbs so we can assert the cleanup `stop` of the missing
	// container is NOT issued on the container-missing short-circuit path.
	var dockerMu sync.Mutex
	var startCalls, stopCalls int
	noSuchContainerErr := errors.New("Error response from daemon: No such container: moe")
	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		dockerMu.Lock()
		defer dockerMu.Unlock()
		if len(args) > 0 {
			switch args[0] {
			case "start":
				startCalls++
				// docker CLI prints to stderr + nonzero exit.
				return []byte("Error: No such container: moe"), noSuchContainerErr
			case "stop":
				stopCalls++
				// Even if a stop WERE issued against the missing container,
				// it would fail the same way — but the fix must not issue it.
				return []byte("Error: No such container: moe"), noSuchContainerErr
			}
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	// Capture the lifecycle audit stream so we can assert NO drift-risk
	// audit fires and the container-missing audit DOES.
	var auditMu sync.Mutex
	var reasons []string
	prev := SetLifecycleAuditSinkForTest(func(ev LifecycleEvent) {
		auditMu.Lock()
		reasons = append(reasons, ev.Reason)
		auditMu.Unlock()
	})
	defer SetLifecycleAuditSinkForTest(prev)

	// Setup sanity: moe is Stopped so the cold-load path engages.
	if !a.IsStopped("moe") {
		t.Fatalf("setup: expected moe admissionStopped")
	}

	// Kick the cold-load. It will hit the rm-gap and short-circuit.
	if !s.KickColdLoad("moe") {
		t.Fatalf("KickColdLoad(moe) returned false; expected true (moe is Stopped)")
	}

	// Wait for the goroutine to finish (kick map cleared via defer).
	if ok := waitForCondition(3*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["moe"]
	}); !ok {
		t.Fatalf("coldLoadKicks[moe] still set after container-missing cold-load (goroutine leaked)")
	}

	// docker start must have actually been attempted (we exercised the
	// real path, not a no-op).
	dockerMu.Lock()
	gotStart, gotStop := startCalls, stopCalls
	dockerMu.Unlock()
	if gotStart < 1 {
		t.Fatalf("expected at least one `docker start moe`, got %d", gotStart)
	}

	// CORE ASSERTION (FAIL pre-fix): the container-missing cold-load must
	// NOT latch the 30s failure cooldown. Pre-fix the cleanup-stop drift
	// path recorded coldLoadFailures[moe]; post-fix the goroutine skips it.
	if s.HasColdLoadCooldownForTest("moe") {
		t.Fatalf("container-missing cold-load latched the failure cooldown; " +
			"this BLOCKS the external-start reconcile for 30s (Forensics A2 bug)")
	}

	// The fix must NOT issue a cleanup `docker stop` against the missing
	// container — there is nothing to stop, and that stop is exactly what
	// latched the vram-drift-risk audit pre-fix.
	if gotStop != 0 {
		t.Errorf("expected NO cleanup `docker stop` on container-missing path, got %d", gotStop)
	}

	// No vram-drift-risk audit may fire for this transient rm-gap.
	auditMu.Lock()
	gotReasons := append([]string(nil), reasons...)
	auditMu.Unlock()
	for _, r := range gotReasons {
		if strings.Contains(r, "vram-drift-risk") {
			t.Errorf("vram-drift-risk audit fired on container-missing path (reason=%q); "+
				"this is the false drift latch the fix removes", r)
		}
	}
	// And the dedicated defer-to-external-start audit MUST fire so the
	// signal is greppable in postmortems.
	foundDefer := false
	for _, r := range gotReasons {
		if r == "container-missing-defer-external-start" {
			foundDefer = true
			break
		}
	}
	if !foundDefer {
		t.Errorf("expected a 'container-missing-defer-external-start' lifecycle audit, got reasons=%v", gotReasons)
	}

	// admission books unchanged: moe still Stopped (the external-start
	// monitor will reconcile the recreated container once it shows running).
	if !a.IsStopped("moe") {
		t.Errorf("expected moe still admissionStopped after container-missing cold-load")
	}

	// FOLLOW-ON ASSERTION (the actual recovery-delay symptom): because no
	// cooldown was latched, a subsequent KickColdLoad must NOT be refused
	// with cooldown_active. (It may refuse for other benign reasons, but
	// NEVER cooldown_active on this path.)
	if _, reason := s.KickColdLoadWithReason("moe"); reason == KickColdLoadRefusalCooldownActive {
		t.Fatalf("follow-on KickColdLoad refused with cooldown_active; the rm-gap " +
			"must not gate the external-start reconcile retry (Forensics A2 bug)")
	}
	// Drain any goroutine spawned by the follow-on kick so the suite is clean.
	waitForCondition(3*time.Second, func() bool {
		s.mu.RLock()
		defer s.mu.RUnlock()
		return !s.coldLoadKicks["moe"]
	})
}

// TestUnit_dockerNoSuchContainer covers the substring detector across the
// CLI / daemon-API phrasings docker emits.
func TestUnit_dockerNoSuchContainer(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"Error response from daemon: No such container: vllm-main", true},
		{"Error: No such container: moe", true},
		{"no such container", true}, // lower-case daemon path
		{"NO SUCH CONTAINER: X", true},
		{"docker start vllm-main: exit status 1 (output: Error: No such container: vllm-main)", true},
		{"Cannot connect to the Docker daemon", false},
		{"container is restarting, wait until the container is running", false},
		{"", false},
	}
	for _, c := range cases {
		if got := dockerNoSuchContainer(c.in); got != c.want {
			t.Errorf("dockerNoSuchContainer(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestUnit_doColdLoad_ContainerMissing_ReturnsSentinel asserts doColdLoad
// classifies a "No such container" start failure as ErrColdLoadContainerMissing
// (so callers can short-circuit) rather than a generic error.
func TestUnit_doColdLoad_ContainerMissing_ReturnsSentinel(t *testing.T) {
	s, _, mgrs := makeStopOnEvictScheduler(t)

	SetSleepDockerCmdForTest(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "start" {
			return []byte("Error: No such container: moe"),
				errors.New("Error response from daemon: No such container: moe")
		}
		return []byte("ok"), nil
	})
	defer SetSleepDockerCmdForTest(nil)

	s.mu.RLock()
	inst := s.instances["moe"]
	s.mu.RUnlock()
	if inst == nil {
		t.Fatalf("setup: moe instance missing")
	}
	_ = mgrs

	err := s.doColdLoad(context.Background(), inst, "moe", 100*time.Millisecond)
	if err == nil {
		t.Fatalf("expected error from doColdLoad on missing container, got nil")
	}
	if !errors.Is(err, ErrColdLoadContainerMissing) {
		t.Fatalf("expected ErrColdLoadContainerMissing, got %v", err)
	}
	// Must NOT be misclassified as a container-exited (post-poll) failure.
	if errors.Is(err, ErrColdLoadContainerExited) {
		t.Errorf("container-missing must not be classified as container-exited")
	}
}

// TestUnit_mapWakeError_ContainerMissing_IsRetryable asserts the wake-path
// error mapper routes the rm-gap sentinel to a retryable RejectColdLoading
// (NOT a terminal RejectAdminIntervention as the vram-drift path would).
func TestUnit_mapWakeError_ContainerMissing_IsRetryable(t *testing.T) {
	err := mapWakeError(ErrColdLoadContainerMissing)
	re, ok := err.(*RejectError)
	if !ok {
		t.Fatalf("expected *RejectError, got %T", err)
	}
	if re.Reason != RejectColdLoading {
		t.Errorf("expected RejectColdLoading (retryable), got %v", re.Reason)
	}
	if re.Reason == RejectAdminIntervention {
		t.Errorf("rm-gap must NOT surface as terminal admin-intervention")
	}
	if re.RetryAfter <= 0 {
		t.Errorf("expected a positive Retry-After for the transient rm-gap, got %v", re.RetryAfter)
	}
}
