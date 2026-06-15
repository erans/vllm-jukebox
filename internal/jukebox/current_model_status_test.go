package jukebox

import (
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/ports"
)

// statusConfig: two scheduler-mode external models on disjoint GPUs so
// both can be Ready simultaneously. Used by the Bug #4 current_model
// tests.
const statusConfig = `
scheduler:
  port_range_start: 9100
  port_range_end: 9199
vllm:
  port: 9000
  startup_timeout: 1s
  drain_timeout: 50ms
  shutdown_timeout: 50ms
models:
  alpha:
    lifecycle: external
    host: vllm-alpha
    port: 9001
    gpus: [0]
    min_free_mem_mb_per_gpu: 10
    expected_vram_mb_per_gpu: 8000
    wake_timeout: 1s
  beta:
    lifecycle: external
    host: vllm-beta
    port: 9002
    gpus: [1]
    min_free_mem_mb_per_gpu: 10
    expected_vram_mb_per_gpu: 8000
    wake_timeout: 1s
`

func makeStatusScheduler(t *testing.T) (*Scheduler, map[string]*fakeRedeployMgr) {
	t.Helper()
	cfg, err := config.Load([]byte(statusConfig))
	if err != nil {
		t.Fatalf("config load: %v", err)
	}
	totalsByGPU := map[int]int{0: 24000, 1: 24000}
	inv := newRedeployInventory(totalsByGPU)
	pool := ports.New(9100, 9199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totalsByGPU, &SchedulerEvictor{S: s})
	s.SetAdmission(a)

	mgrs := map[string]*fakeRedeployMgr{
		"alpha": {port: 9001},
		"beta":  {port: 9002},
	}
	for name, m := range mgrs {
		modelCfg := cfg.Models[name]
		m.pid.Store(int64(7000 + m.port)) // Ready needs a non-zero PID
		m.isSleeping.Store(false)
		s.SeedInstanceForTest(name, m.port, modelCfg.GPUs, false, StateReady, m)
	}
	return s, mgrs
}

// TestStatus_CurrentModelReflectsMostRecentlyUsedReady is the Bug #4
// regression: /status.current_model historically reported "" in
// scheduler mode even while instances served 200s, because
// Scheduler.Status() never populated CurrentModel (only the swap-mode
// coordinator did). Post-fix it reports the most-recently-USED Ready
// instance — the model a consumer would observe as "currently serving".
func TestStatus_CurrentModelReflectsMostRecentlyUsedReady(t *testing.T) {
	s, _ := makeStatusScheduler(t)

	// beta was used more recently than alpha.
	base := time.Now()
	s.SetInstanceLastUsedForTest("alpha", base.Add(-1*time.Minute))
	s.SetInstanceLastUsedForTest("beta", base)

	st := s.Status()
	if st.CurrentModel != "beta" {
		t.Fatalf("Bug #4: expected current_model=beta (most-recently-used Ready), got %q", st.CurrentModel)
	}

	// Now alpha becomes the most-recently-used; current_model must follow.
	s.SetInstanceLastUsedForTest("alpha", base.Add(1*time.Minute))
	if got := s.Status().CurrentModel; got != "alpha" {
		t.Fatalf("Bug #4: expected current_model=alpha after it served most recently, got %q", got)
	}
}

// TestStatus_CurrentModelNonEmptyWhileServing pins the core symptom: at
// least one Ready instance ⇒ current_model is never blank.
func TestStatus_CurrentModelNonEmptyWhileServing(t *testing.T) {
	s, _ := makeStatusScheduler(t)
	st := s.Status()
	if st.State != StateReady {
		t.Fatalf("expected scheduler StateReady with two Ready instances, got %v", st.State)
	}
	if st.CurrentModel == "" {
		t.Fatalf("Bug #4: current_model is empty while instances are Ready and serving (this is the exact reported symptom)")
	}
}

// TestStatus_CurrentModelPrefersReadyOverSleeping proves a Ready
// instance is reported even when a Sleeping instance was used more
// recently — current_model should name what's actually serving, not a
// slept peer.
func TestStatus_CurrentModelPrefersReadyOverSleeping(t *testing.T) {
	s, mgrs := makeStatusScheduler(t)

	// Make beta Sleeping but more-recently-used; alpha stays Ready.
	mgrs["beta"].isSleeping.Store(true)
	s.SetInstanceStateForTest("beta", StateSleeping)
	base := time.Now()
	s.SetInstanceLastUsedForTest("alpha", base.Add(-1*time.Minute))
	s.SetInstanceLastUsedForTest("beta", base)

	if got := s.Status().CurrentModel; got != "alpha" {
		t.Fatalf("Bug #4: expected current_model=alpha (the Ready instance), got %q (must prefer Ready over more-recently-used Sleeping)", got)
	}
}

// TestStatus_CurrentModelBlankWhenAllStopped is the MED#4 regression: when
// NO instance is Ready or Sleeping (all Stopped), current_model must be ""
// — honest blank. The pre-rework MRU fallback considered ALL instances
// (including Stopped) and would name a down model, making /status lie
// "serving vllm-main" while it was actually stopped.
func TestStatus_CurrentModelBlankWhenAllStopped(t *testing.T) {
	s, mgrs := makeStatusScheduler(t)

	// Drive both instances to Stopped (container down: PID 0).
	for name, m := range mgrs {
		m.pid.Store(0)
		m.isSleeping.Store(true)
		s.SetInstanceStateForTest(name, StateStopped)
	}
	// Give them recent last-used so the old MRU fallback would have picked
	// one of them — proving the fix excludes Stopped regardless of recency.
	base := time.Now()
	s.SetInstanceLastUsedForTest("alpha", base)
	s.SetInstanceLastUsedForTest("beta", base.Add(1*time.Second))

	st := s.Status()
	if st.CurrentModel != "" {
		t.Fatalf("MED#4: expected current_model=\"\" when all instances Stopped, got %q (must never name a down model)", st.CurrentModel)
	}
}

// TestStatus_CurrentModelSleepingWhenOnlySleeping confirms a Sleeping
// instance IS eligible as current_model when nothing is Ready (it is
// wakeable-on-demand), distinguishing it from a Stopped instance which is
// not eligible.
func TestStatus_CurrentModelSleepingWhenOnlySleeping(t *testing.T) {
	s, mgrs := makeStatusScheduler(t)

	// alpha Sleeping (eligible), beta Stopped (not eligible).
	mgrs["alpha"].isSleeping.Store(true)
	s.SetInstanceStateForTest("alpha", StateSleeping)
	mgrs["beta"].pid.Store(0)
	mgrs["beta"].isSleeping.Store(true)
	s.SetInstanceStateForTest("beta", StateStopped)

	if got := s.Status().CurrentModel; got != "alpha" {
		t.Fatalf("MED#4: expected current_model=alpha (only Sleeping is eligible; Stopped beta excluded), got %q", got)
	}
}
