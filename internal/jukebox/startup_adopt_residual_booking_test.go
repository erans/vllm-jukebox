package jukebox

import (
	"context"
	"errors"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/ports"
)

// TestAdoptPreExistingPeer_StoppedBooksFullAwakeFootprint is the P1-B
// regression guard (#314 residual-booking gap → OOM).
//
// The boot-adopt path's StateStopped branch reconciles a peer whose
// container is ALIVE (transient boot-probe failure) directly to
// StateReady — which makes tryRouteReady route 100% of traffic to it
// with NO RequestWake gate. The admission books MUST therefore reflect
// the FULL awake VRAM footprint.
//
// Pre-fix, the branch called NotifyStarted → markStartedLocked, which
// books only the small L1 residual and transitions the model to
// admissionSleeping. With only the residual booked, the per-GPU budget
// reports the GPU as almost-entirely-free even though a fully-resident
// engine is actually mapping the whole GPU. A concurrent
// GPU-overlapping wake of a sibling then sees the (phantom) free VRAM,
// admits WITHOUT eviction, and both engines map memory on the same GPU
// → CUDA OOM.
//
// Post-fix, the branch calls NotifyStartedAwake → markStartedAwakeLocked,
// booking the full expected awake VRAM (admissionAwake). The budget then
// matches the routing decision and the GPU-overlapping sibling wake is
// correctly rejected as infeasible (the resident peer is critical /
// non-evictable).
//
// Assertions:
//  1. SnapshotBudgets AwakeMB == C (full GPU). Pre-fix: 0 awake +
//     residual-only (the model is admissionSleeping, AwakeMB==0).
//  2. A GPU-overlapping wake for sibling B is rejected with
//     ErrAdmissionInfeasible. Pre-fix: it WRONGLY succeeds (admits onto
//     the phantom-free GPU).
func TestAdoptPreExistingPeer_StoppedBooksFullAwakeFootprint(t *testing.T) {
	const capacityC = 24000

	cfg := makeCfg(map[string]config.ModelConfig{
		// A: the adopted peer. expected == full GPU capacity; tiny L1
		// residual. Critical + NO swap_group => once awake it is
		// untouchable by a lower-priority cross-group requester, so B's
		// infeasibility is a clean signal of "the GPU is genuinely full",
		// not "B happened to be evictable".
		"A": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: capacityC,
			SleepL1ResidualMB:    500,
			Priority:             config.PriorityCritical,
		},
		// B: a sibling on the SAME GPU. Normal priority, small footprint.
		// Post-fix it cannot fit (GPU fully booked by A, A non-evictable).
		"B": {
			GPUs:                 []int{0},
			ExpectedVRAMMBPerGPU: 4000,
			SleepL1ResidualMB:    300,
			Priority:             config.PriorityNormal,
		},
	})

	totals := map[int]int{0: capacityC}
	inv := &redeployInventory{gpus: []gpu.GPU{{Index: 0, TotalMB: capacityC, FreeMB: capacityC}}}
	pool := ports.New(8100, 8199)
	s := NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, nil)
	a := NewAdmissionController(cfg, totals, &stubAdmissionEvictor{})
	s.SetAdmission(a)

	// Seed A as a StateStopped instance (the boot path's seed for a peer
	// whose /health was unreachable at boot probe + evict_action stop)
	// and drive admission to admissionStopped to match — this is exactly
	// the post-boot precondition for the adopt-Stopped branch.
	mgrA := &fakeRedeployMgr{port: 8101}
	mgrA.pid.Store(int64(7000 + 8101))
	s.SeedInstanceForTest("A", 8101, []int{0}, false, StateStopped, mgrA)
	a.NotifyStopped("A")

	// Sanity: before adopt, A contributes nothing (admissionStopped).
	if pre := a.SnapshotBudgets(); len(pre) != 1 || pre[0].AwakeMB != 0 {
		t.Fatalf("precondition: expected A admissionStopped (AwakeMB==0); got %+v", pre)
	}

	// The container is actually ALIVE — boot probe had a transient
	// failure. Adopt it. Post-fix this must book the FULL awake footprint.
	settled := s.adoptPreExistingPeer("A", StateStopped, cfg.Models["A"])
	if settled != StateReady {
		t.Fatalf("expected adopt-Stopped to settle StateReady; got %s", settled)
	}

	// (1) Budget must reflect the full resident footprint, not the residual.
	snap := a.SnapshotBudgets()
	if len(snap) != 1 {
		t.Fatalf("expected 1 GPU snapshot; got %d", len(snap))
	}
	if snap[0].AwakeMB != capacityC {
		t.Errorf("P1-B: expected AwakeMB==%d (full awake footprint booked on adopt); got %d (pre-fix books only residual → phantom-free GPU → OOM)",
			capacityC, snap[0].AwakeMB)
	}
	if snap[0].AvailableMB != 0 {
		t.Errorf("expected AvailableMB==0 after booking full GPU; got %d", snap[0].AvailableMB)
	}

	// (2) A concurrent GPU-overlapping wake for B must be rejected as
	// infeasible — the GPU is full and A (critical, non-swap-group) is
	// untouchable. Pre-fix B would mis-admit onto the phantom-free GPU.
	_, err := a.RequestWake(context.Background(), "B")
	if err == nil {
		t.Fatalf("P1-B: expected RequestWake(B) to be rejected (GPU fully booked by A); got nil (pre-fix over-admit → OOM)")
	}
	if !errors.Is(err, ErrAdmissionInfeasible) {
		t.Errorf("expected ErrAdmissionInfeasible for B; got %v", err)
	}
}
