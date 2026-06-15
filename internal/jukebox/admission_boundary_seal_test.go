package jukebox

// Fix 1 (drain-barrier tracker-seal) regression tests — defends
// vllm-project/vllm#45520. sleepInstance seals inst.inflight in the same
// critical section that flips StateStopping, so no request can Track() into
// the window between WaitForDrain observing zero and sc.Sleep() firing.
//
// PRE-FIX (Track gated only on the concurrency cap): the racing
// routeForInstance Track()s successfully and fakeSeal Sleep() observes
// inflight==1 at entry — a /sleep mid-decode → cudaErrorIllegalAddress.
// POST-FIX: Track returns errSealedRetryViaWake, the count stays 0, and the
// request re-routes via the wake-coalesced path with no *RejectError.

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"vllm-jukebox/internal/ports"
)

// sealRaceMgr is a SleepCapable stub whose Sleep() fires the racing
// routeForInstance at the exact instant the drain barrier has returned —
// i.e. inside the post-drain / pre-sleep window the seal must close. It
// records the in-flight count observed at Sleep entry and the error the
// racing route returned.
type sealRaceMgr struct {
	s    *Scheduler
	inst *schedInstance

	observedInflight int64
	raceErr          error
	raceRoute        Route
	raceOnce         sync.Once
	raceDone         chan struct{}

	mu       sync.Mutex
	sleeping bool
}

func (m *sealRaceMgr) Start(_ context.Context, _ string) (int, error) { return 1, nil }
func (m *sealRaceMgr) Stop(_ context.Context) error                   { return nil }
func (m *sealRaceMgr) VerifyReady(_ context.Context, _ string) error  { return nil }
func (m *sealRaceMgr) CurrentPID() int                                { return 1 }
func (m *sealRaceMgr) BaseURL() string                                { return "" }

func (m *sealRaceMgr) Sleep(_ context.Context, _ int) error {
	// Snapshot the in-flight count the instant /sleep would land. With the
	// seal in place this MUST be 0; pre-fix the racing route below would
	// have Track()'d it to 1 before we got here.
	m.observedInflight = m.inst.inflight.Count()
	// Fire the racing request from a second goroutine — it must hit the
	// sealed tracker and be refused. Run synchronously-joined so the
	// assertion sees a stable result.
	m.raceOnce.Do(func() {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			m.raceRoute, m.raceErr = m.s.routeForInstance(context.Background(), m.inst, "upstream")
		}()
		wg.Wait()
		close(m.raceDone)
	})
	m.mu.Lock()
	m.sleeping = true
	m.mu.Unlock()
	return nil
}

func (m *sealRaceMgr) Wake(_ context.Context, _ time.Duration) error {
	m.mu.Lock()
	m.sleeping = false
	m.mu.Unlock()
	return nil
}

func (m *sealRaceMgr) IsSleeping(_ context.Context) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sleeping, nil
}

// TestSleepInstance_NoAdmitAfterDrainObservation proves the drain-barrier
// seal closes the TOCTOU window: a request that races routeForInstance in
// the post-drain / pre-sleep window is refused (errSealedRetryViaWake, no
// Route, no *RejectError) and the slept backend observes zero in-flight.
//
// Pre-fix this fails because routeForInstance Track()s successfully (gated
// only on the concurrency cap), so fakeSleep observes inflight==1.
func TestSleepInstance_NoAdmitAfterDrainObservation(t *testing.T) {
	const model = "seal-race-target"
	cfg := guardTestCfg(t, model, 1 /* 1s drain */, 0 /* settle off */)

	totalsByGPU := map[int]int{0: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(8200, 8299)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)

	mgr := &sealRaceMgr{s: s, raceDone: make(chan struct{})}
	s.SeedInstanceForTest(model, 8210, []int{0}, false, StateReady, mgr)
	inst := s.instances[model]
	mgr.inst = inst

	// One legitimately-admitted in-flight request before the sleep starts.
	preDone, ok := inst.inflight.Track(context.Background())
	if !ok || preDone == nil {
		t.Fatalf("setup: expected initial Track to succeed")
	}

	// Drive sleepInstance from a goroutine. It seals, waits for the drain,
	// then calls mgr.Sleep — which fires the racing routeForInstance.
	sleepErrCh := make(chan error, 1)
	go func() {
		sleepErrCh <- s.sleepInstance(context.Background(), inst, 1, "test-seal-race")
	}()

	// Let sleepInstance reach the strict drain wait, then release the
	// original in-flight so the drain returns 0 and Sleep proceeds.
	time.Sleep(50 * time.Millisecond)
	preDone()

	select {
	case err := <-sleepErrCh:
		if err != nil {
			t.Fatalf("sleepInstance unexpectedly errored: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("sleepInstance timed out")
	}

	// The racing route must have run inside Sleep().
	select {
	case <-mgr.raceDone:
	case <-time.After(2 * time.Second):
		t.Fatalf("racing routeForInstance never ran (Sleep not invoked?)")
	}

	if mgr.observedInflight != 0 {
		t.Fatalf("Sleep observed inflight=%d at entry; expected 0 (a request admitted into the drain/sleep window → cumem race)", mgr.observedInflight)
	}
	if !errors.Is(mgr.raceErr, errSealedRetryViaWake) {
		t.Fatalf("racing route err = %v; expected errSealedRetryViaWake", mgr.raceErr)
	}
	if mgr.raceRoute.BaseURL != "" {
		t.Fatalf("racing route returned a Route (BaseURL=%q); expected empty Route on seal-refuse", mgr.raceRoute.BaseURL)
	}
	var rej *RejectError
	if errors.As(mgr.raceErr, &rej) {
		t.Fatalf("racing route returned a *RejectError (%v); seal-refuse must NOT 503 — it re-routes via wake", rej)
	}

	// Instance is slept and re-opened for admission (Unseal on success).
	if st, _ := s.InstanceStateForTest(model); st != StateSleeping {
		t.Fatalf("expected StateSleeping after sleep, got %s", st)
	}

	// Wake-path leg: a request driven through AcquireRoute on the slept
	// instance must succeed via the wake-coalesced path with NO *RejectError
	// (the seal-refuse must never surface as a 503).
	route, err := s.AcquireRoute(context.Background(), model, "req-after-sleep")
	if err != nil {
		var rej2 *RejectError
		if errors.As(err, &rej2) {
			t.Fatalf("AcquireRoute after sleep returned *RejectError %v; expected wake-path success", rej2)
		}
		t.Fatalf("AcquireRoute after sleep errored: %v", err)
	}
	if route.Done != nil {
		route.Done()
	}
	if st, _ := s.InstanceStateForTest(model); st != StateReady {
		t.Fatalf("expected StateReady after wake, got %s", st)
	}
}
