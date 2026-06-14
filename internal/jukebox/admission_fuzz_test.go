package jukebox

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"testing"
	"time"

	"vllm-jukebox/internal/config"
)

// admission_fuzz_test.go: property-based / fuzz-style tests for the
// admission state machine. Each test generates a deterministic random
// sequence of admission transitions and asserts invariants hold at every
// step. Goal: catch arithmetic / state-machine bugs that the targeted
// admission_test.go suite would miss.
//
// All RNG seeds are derived from defaultFuzzSeed unless the FUZZ_SEED env
// var is set — failures reproduce verbatim. Each test runs ~1000 ops in
// well under 1 second on a laptop.

const defaultFuzzSeed int64 = 0xC0FFEE1234

// seedFromEnv returns FUZZ_SEED if set+parseable, otherwise the provided
// fallback. Lets a maintainer rerun a specific failing seed via
// `FUZZ_SEED=<n> go test ...`.
func seedFromEnv(fallback int64) int64 {
	if s := os.Getenv("FUZZ_SEED"); s != "" {
		if n, err := strconv.ParseInt(s, 0, 64); err == nil {
			return n
		}
	}
	return fallback
}

// fuzzModel is a per-test description of a synthetic model.
type fuzzModel struct {
	name string
	mc   config.ModelConfig
}

// newRandomAdmission builds a synthetic 4-GPU controller with five model
// "shapes" matching the real fleet topology: pinned-with-swap-group
// (main-shape), 4-GPU swap-group with evict_action stop (moe-shape),
// 4-GPU swap-group with evict_action sleep (longctx-shape), single-GPU
// evict-group member (vision-shape), single-GPU non-group plain
// (task-shape).
func newRandomAdmission(_ *testing.T, _ int64) (*AdmissionController, []fuzzModel) {
	mods := []fuzzModel{
		{
			name: "main",
			mc: config.ModelConfig{
				GPUs:                 []int{0, 1, 2, 3},
				ExpectedVRAMMBPerGPU: 4000,
				SleepL1ResidualMB:    500,
				Pinned:               boolPtr(true),
				SwapGroup:            "g-main",
				EvictAction:          config.EvictActionSleep,
			},
		},
		{
			name: "moe",
			mc: config.ModelConfig{
				GPUs:                 []int{0, 1, 2, 3},
				ExpectedVRAMMBPerGPU: 5000,
				SleepL1ResidualMB:    1000,
				Priority:             config.PriorityNormal,
				SwapGroup:            "g-main",
				EvictAction:          config.EvictActionStop,
				Lifecycle:            config.LifecycleExternal,
				Host:                 "vllm-moe",
			},
		},
		{
			name: "longctx",
			mc: config.ModelConfig{
				GPUs:                 []int{0, 1, 2, 3},
				ExpectedVRAMMBPerGPU: 4500,
				SleepL1ResidualMB:    800,
				Priority:             config.PriorityNormal,
				SwapGroup:            "g-main",
				EvictAction:          config.EvictActionSleep,
			},
		},
		{
			name: "vision",
			mc: config.ModelConfig{
				GPUs:                 []int{3},
				ExpectedVRAMMBPerGPU: 3000,
				SleepL1ResidualMB:    400,
				Priority:             config.PriorityBestEffort,
				SwapGroup:            "g-vision",
				EvictAction:          config.EvictActionStop,
				Lifecycle:            config.LifecycleExternal,
				Host:                 "vllm-vision",
			},
		},
		{
			name: "task",
			mc: config.ModelConfig{
				GPUs:                 []int{2},
				ExpectedVRAMMBPerGPU: 2500,
				SleepL1ResidualMB:    300,
				Priority:             config.PriorityNormal,
			},
		},
	}
	cfgModels := map[string]config.ModelConfig{}
	for _, m := range mods {
		cfgModels[m.name] = m.mc
	}
	cfg := makeCfg(cfgModels)
	ev := &stubEvictor{}
	totals := map[int]int{0: 24000, 1: 24000, 2: 24000, 3: 24000}
	a := NewAdmissionController(cfg, totals, ev)
	return a, mods
}

// fuzzOpKind enumerates the random ops we apply. Restricted to legal
// transitions; RequestWake on Stopped is gated in production by IsStopped
// + cold-load, mirrored here.
type fuzzOpKind int

const (
	opRequestWake fuzzOpKind = iota
	opNotifySleep
	opNotifyStopped
	opNotifyStarted
	opNotifyStartFailed
	opNotifyWakeComplete
	opTouchActivity
)

func (k fuzzOpKind) String() string {
	switch k {
	case opRequestWake:
		return "RequestWake"
	case opNotifySleep:
		return "NotifySleep"
	case opNotifyStopped:
		return "NotifyStopped"
	case opNotifyStarted:
		return "NotifyStarted"
	case opNotifyStartFailed:
		return "NotifyStartFailed"
	case opNotifyWakeComplete:
		return "NotifyWakeComplete"
	case opTouchActivity:
		return "TouchActivity"
	default:
		return "?"
	}
}

// randomOp picks a random op + target model.
func randomOp(rng *rand.Rand, models []fuzzModel) (fuzzOpKind, string) {
	r := rng.Intn(100)
	var kind fuzzOpKind
	switch {
	case r < 30:
		kind = opRequestWake
	case r < 50:
		kind = opNotifySleep
	case r < 60:
		kind = opNotifyStopped
	case r < 70:
		kind = opNotifyStarted
	case r < 75:
		kind = opNotifyStartFailed
	case r < 85:
		kind = opNotifyWakeComplete
	default:
		kind = opTouchActivity
	}
	name := models[rng.Intn(len(models))].name
	return kind, name
}

// applyOp dispatches `kind` against `name`. RequestWake errors are
// expected — the controller doing its job — and intentionally swallowed.
func applyOp(t *testing.T, a *AdmissionController, kind fuzzOpKind, name string) {
	t.Helper()
	switch kind {
	case opRequestWake:
		// Production guard: if Stopped, the cold-load path runs
		// NotifyStarted first. Mirror that.
		if a.IsStopped(name) {
			a.NotifyStarted(name)
		}
		_, _ = a.RequestWake(context.Background(), name)
	case opNotifySleep:
		a.NotifySleep(name, "manual")
	case opNotifyStopped:
		a.NotifyStopped(name)
	case opNotifyStarted:
		a.NotifyStarted(name)
	case opNotifyStartFailed:
		a.NotifyStartFailed(name)
	case opNotifyWakeComplete:
		a.NotifyWakeComplete(name)
	case opTouchActivity:
		a.TouchActivity(name)
	}
}

// assertBooksNonNegative is invariant 1.
func assertBooksNonNegative(t *testing.T, a *AdmissionController, step int, kind fuzzOpKind, target string) {
	t.Helper()
	a.mu.Lock()
	defer a.mu.Unlock()
	for g := range a.totalsByGPU {
		if a.awakeByGPU[g] < 0 {
			t.Fatalf("step %d (%s %s): GPU %d awakeByGPU went negative: %d", step, kind, target, g, a.awakeByGPU[g])
		}
		if a.l1ResidualByGPU[g] < 0 {
			t.Fatalf("step %d (%s %s): GPU %d l1ResidualByGPU went negative: %d", step, kind, target, g, a.l1ResidualByGPU[g])
		}
		if a.pinnedByGPU[g] < 0 {
			t.Fatalf("step %d (%s %s): GPU %d pinnedByGPU went negative: %d", step, kind, target, g, a.pinnedByGPU[g])
		}
	}
}

// assertBooksBounded is invariant 2.
func assertBooksBounded(t *testing.T, a *AdmissionController, step int, kind fuzzOpKind, target string) {
	t.Helper()
	a.mu.Lock()
	defer a.mu.Unlock()
	for g, total := range a.totalsByGPU {
		used := a.awakeByGPU[g] + a.l1ResidualByGPU[g] + a.pinnedByGPU[g]
		if used > total {
			t.Fatalf("step %d (%s %s): GPU %d overcommitted: awake=%d + l1=%d + pinned=%d = %d > total=%d",
				step, kind, target, g, a.awakeByGPU[g], a.l1ResidualByGPU[g], a.pinnedByGPU[g], used, total)
		}
	}
}

func modelStateLocked(a *AdmissionController, name string) (admissionModelState, bool) {
	m, ok := a.models[name]
	if !ok {
		return admissionUnknown, false
	}
	return m.State, true
}

func mustState(a *AdmissionController, name string) (admissionModelState, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return modelStateLocked(a, name)
}

// bookSnapshot is a deep-comparable snapshot of all three per-GPU maps
// plus every tracked model's state.
type bookSnapshot struct {
	awake     map[int]int
	l1        map[int]int
	pinned    map[int]int
	stateByMo map[string]admissionModelState
}

func snapshotBooks(a *AdmissionController) bookSnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	bs := bookSnapshot{
		awake:     map[int]int{},
		l1:        map[int]int{},
		pinned:    map[int]int{},
		stateByMo: map[string]admissionModelState{},
	}
	for g, v := range a.awakeByGPU {
		bs.awake[g] = v
	}
	for g, v := range a.l1ResidualByGPU {
		bs.l1[g] = v
	}
	for g, v := range a.pinnedByGPU {
		bs.pinned[g] = v
	}
	for n, m := range a.models {
		bs.stateByMo[n] = m.State
	}
	return bs
}

func bookEqual(a, b bookSnapshot) bool {
	if !mapIntEqual(a.awake, b.awake) || !mapIntEqual(a.l1, b.l1) || !mapIntEqual(a.pinned, b.pinned) {
		return false
	}
	if len(a.stateByMo) != len(b.stateByMo) {
		return false
	}
	for k, v := range a.stateByMo {
		if b.stateByMo[k] != v {
			return false
		}
	}
	return true
}

func mapIntEqual(a, b map[int]int) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func statesEqual(a, b map[string]admissionModelState) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func perGPUMap(a *AdmissionController, which string) map[int]int {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := map[int]int{}
	var src map[int]int
	switch which {
	case "awake":
		src = a.awakeByGPU
	case "l1":
		src = a.l1ResidualByGPU
	case "pinned":
		src = a.pinnedByGPU
	}
	for g, v := range src {
		out[g] = v
	}
	return out
}

// ============================================================
// Tests
// ============================================================

// TestAdmissionFuzz_BooksAlwaysNonNegative drives a random walk of 1000
// ops and asserts invariants 1 and 2 (non-negative + bounded above) hold
// at every step. Covers all transition methods.
func TestAdmissionFuzz_BooksAlwaysNonNegative(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed)
	rng := rand.New(rand.NewSource(seed))
	a, models := newRandomAdmission(t, seed)

	const N = 1000
	for i := 0; i < N; i++ {
		kind, name := randomOp(rng, models)
		applyOp(t, a, kind, name)
		assertBooksNonNegative(t, a, i, kind, name)
		assertBooksBounded(t, a, i, kind, name)
	}
	t.Logf("fuzz seed=0x%x: %d ops, no invariant violations", seed, N)
}

// TestAdmissionFuzz_NotifyIdempotency is invariant 3. Catches
// double-decrement / double-add bugs in markSleepingLocked,
// markStoppedLocked, markStartedLocked.
func TestAdmissionFuzz_NotifyIdempotency(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 1)
	rng := rand.New(rand.NewSource(seed))

	type notify struct {
		name string
		fn   func(a *AdmissionController, name string)
	}
	notifies := []notify{
		{"NotifySleep", func(a *AdmissionController, n string) { a.NotifySleep(n, "manual") }},
		{"NotifyStopped", func(a *AdmissionController, n string) { a.NotifyStopped(n) }},
		{"NotifyStarted", func(a *AdmissionController, n string) { a.NotifyStarted(n) }},
		{"NotifyStartFailed", func(a *AdmissionController, n string) { a.NotifyStartFailed(n) }},
		{"NotifyWakeComplete", func(a *AdmissionController, n string) { a.NotifyWakeComplete(n) }},
	}

	for _, nf := range notifies {
		a, models := newRandomAdmission(t, seed)
		for _, m := range models {
			driveToRandomState(t, a, m.name, rng)
		}
		for _, m := range models {
			nf.fn(a, m.name)
			snap1 := snapshotBooks(a)
			nf.fn(a, m.name)
			snap2 := snapshotBooks(a)
			if !bookEqual(snap1, snap2) {
				t.Errorf("idempotency violation: %s(%s) changed books on second call\n  after-1st: %v\n  after-2nd: %v",
					nf.name, m.name, snap1, snap2)
			}
		}
	}
}

func driveToRandomState(t *testing.T, a *AdmissionController, name string, rng *rand.Rand) {
	t.Helper()
	switch rng.Intn(4) {
	case 0:
		// leave as-is
	case 1:
		_, _ = a.RequestWake(context.Background(), name)
	case 2:
		_, _ = a.RequestWake(context.Background(), name)
		a.NotifySleep(name, "manual")
	case 3:
		a.NotifyStopped(name)
	}
}

// TestAdmissionFuzz_RoundTripCleanliness is invariant 4. After getting
// every non-pinned model to Sleeping, a wake/sleep round-trip on a model
// (that doesn't trigger an eviction) must restore books bit-for-bit.
func TestAdmissionFuzz_RoundTripCleanliness(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 2)
	a, models := newRandomAdmission(t, seed)

	for _, m := range models {
		st, _ := mustState(a, m.name)
		if st == admissionAwake {
			a.NotifySleep(m.name, "manual")
			continue
		}
		if _, err := a.RequestWake(context.Background(), m.name); err != nil {
			continue
		}
		a.NotifySleep(m.name, "manual")
	}

	for _, m := range models {
		st, _ := mustState(a, m.name)
		if st != admissionSleeping {
			continue
		}
		before := snapshotBooks(a)
		if _, err := a.RequestWake(context.Background(), m.name); err != nil {
			continue
		}
		a.NotifySleep(m.name, "manual")
		after := snapshotBooks(a)
		// If RequestWake had to evict a peer, the world changed; skip
		// — the invariant is about an isolated round trip.
		if !statesEqual(before.stateByMo, after.stateByMo) {
			continue
		}
		if !bookEqual(before, after) {
			t.Errorf("round-trip non-clean for %s\n  before: %v\n  after:  %v",
				m.name, before, after)
		}
	}
}

// TestAdmissionFuzz_StoppedToStartedToSleeping is invariant 5.
func TestAdmissionFuzz_StoppedToStartedToSleeping(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 3)
	a, models := newRandomAdmission(t, seed)

	for _, m := range models {
		a.NotifyStopped(m.name)
		st, _ := mustState(a, m.name)
		if st != admissionStopped {
			t.Errorf("%s: expected Stopped after NotifyStopped, got %d", m.name, st)
		}
	}

	beforeL1 := perGPUMap(a, "l1")
	for _, m := range models {
		a.NotifyStarted(m.name)
	}
	afterL1 := perGPUMap(a, "l1")

	expectedDelta := map[int]int{}
	for _, m := range models {
		for _, g := range m.mc.GPUs {
			expectedDelta[g] += m.mc.SleepL1ResidualMB
		}
	}
	for g, want := range expectedDelta {
		got := afterL1[g] - beforeL1[g]
		if got != want {
			t.Errorf("GPU %d: expected residual delta %d, got %d (before=%d after=%d)",
				g, want, got, beforeL1[g], afterL1[g])
		}
	}
	for _, m := range models {
		st, _ := mustState(a, m.name)
		if st != admissionSleeping {
			t.Errorf("%s: expected Sleeping after NotifyStarted, got %d", m.name, st)
		}
	}
}

// TestAdmissionFuzz_PinnedSwapGroupFootprintInAwake is invariant 6.
// Regression guard for the pinned-in-pinnedByGPU branch being taken when
// swap_group is non-empty.
func TestAdmissionFuzz_PinnedSwapGroupFootprintInAwake(t *testing.T) {
	a, models := newRandomAdmission(t, seedFromEnv(defaultFuzzSeed+4))
	var main *fuzzModel
	for i := range models {
		if models[i].mc.Pinned != nil && *models[i].mc.Pinned && models[i].mc.SwapGroup != "" {
			main = &models[i]
			break
		}
	}
	if main == nil {
		t.Fatal("fuzz fixture missing the pinned+swap_group model")
	}

	a.mu.Lock()
	for _, g := range main.mc.GPUs {
		if a.pinnedByGPU[g] != 0 {
			t.Errorf("invariant: pinned+swap_group must NOT use pinnedByGPU; GPU %d has pinnedByGPU=%d",
				g, a.pinnedByGPU[g])
		}
		if a.awakeByGPU[g] < main.mc.ExpectedVRAMMBPerGPU {
			t.Errorf("invariant: pinned+swap_group footprint missing from awakeByGPU on GPU %d (have %d, want >=%d)",
				g, a.awakeByGPU[g], main.mc.ExpectedVRAMMBPerGPU)
		}
	}
	a.mu.Unlock()

	beforeAwake := perGPUMap(a, "awake")
	beforeL1 := perGPUMap(a, "l1")
	a.NotifySleep(main.name, "manual")
	afterAwake := perGPUMap(a, "awake")
	afterL1 := perGPUMap(a, "l1")

	for _, g := range main.mc.GPUs {
		awakeDelta := afterAwake[g] - beforeAwake[g]
		l1Delta := afterL1[g] - beforeL1[g]
		if awakeDelta != -main.mc.ExpectedVRAMMBPerGPU {
			t.Errorf("GPU %d: expected awake to drop by %d after sleep, got delta %d",
				g, main.mc.ExpectedVRAMMBPerGPU, awakeDelta)
		}
		if l1Delta != main.mc.SleepL1ResidualMB {
			t.Errorf("GPU %d: expected residual to grow by %d after sleep, got delta %d",
				g, main.mc.SleepL1ResidualMB, l1Delta)
		}
	}
}

// TestAdmissionFuzz_StoppedContributesZero is invariant 7 (strong form).
// After each NotifyStopped, books on every GPU must equal the SUM of the
// remaining (non-stopped) models' per-state contributions, computed
// independently.
func TestAdmissionFuzz_StoppedContributesZero(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 5)
	rng := rand.New(rand.NewSource(seed))

	for trial := 0; trial < 20; trial++ {
		a, models := newRandomAdmission(t, seed)
		for _, m := range models {
			if rng.Intn(2) == 0 {
				_, _ = a.RequestWake(context.Background(), m.name)
			}
		}

		shuffled := append([]fuzzModel(nil), models...)
		rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })

		for _, m := range shuffled {
			a.NotifyStopped(m.name)
			expAwake := map[int]int{}
			expL1 := map[int]int{}
			expPinned := map[int]int{}
			a.mu.Lock()
			for _, other := range models {
				st := a.models[other.name].State
				for _, g := range other.mc.GPUs {
					switch st {
					case admissionAwake:
						if other.mc.Pinned != nil && *other.mc.Pinned && other.mc.SwapGroup == "" {
							expPinned[g] += other.mc.ExpectedVRAMMBPerGPU
						} else {
							expAwake[g] += other.mc.ExpectedVRAMMBPerGPU
						}
					case admissionSleeping:
						expL1[g] += other.mc.SleepL1ResidualMB
					case admissionStopped, admissionUnknown:
						// contributes 0
					}
				}
			}
			for g := range a.totalsByGPU {
				if a.awakeByGPU[g] != expAwake[g] {
					t.Errorf("trial %d after stopping %s: GPU %d awakeByGPU=%d, expected=%d",
						trial, m.name, g, a.awakeByGPU[g], expAwake[g])
				}
				if a.l1ResidualByGPU[g] != expL1[g] {
					t.Errorf("trial %d after stopping %s: GPU %d l1ResidualByGPU=%d, expected=%d",
						trial, m.name, g, a.l1ResidualByGPU[g], expL1[g])
				}
				if a.pinnedByGPU[g] != expPinned[g] {
					t.Errorf("trial %d after stopping %s: GPU %d pinnedByGPU=%d, expected=%d",
						trial, m.name, g, a.pinnedByGPU[g], expPinned[g])
				}
			}
			a.mu.Unlock()
		}
	}
}

// TestAdmissionFuzz_NoGoroutineLeaks. goleak isn't in go.mod; we use
// runtime.NumGoroutine() with a settle deadline. Auto-restore dispatches
// goroutines on idle-sleeps — a hang there would leak.
func TestAdmissionFuzz_NoGoroutineLeaks(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 6)
	rng := rand.New(rand.NewSource(seed))
	a, models := newRandomAdmission(t, seed)

	a.SetAutoRestoreWake(func(name, reason string) {
		_ = name
		_ = reason
	})

	time.Sleep(20 * time.Millisecond)
	runtime.GC()
	baseline := runtime.NumGoroutine()

	const N = 500
	for i := 0; i < N; i++ {
		kind, name := randomOp(rng, models)
		if kind == opNotifySleep && rng.Intn(3) == 0 {
			a.NotifySleep(name, "idle")
			continue
		}
		applyOp(t, a, kind, name)
	}

	deadline := time.Now().Add(2 * time.Second)
	var final int
	for time.Now().Before(deadline) {
		runtime.GC()
		final = runtime.NumGoroutine()
		if final <= baseline+2 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if final > baseline+2 {
		t.Errorf("goroutine leak suspected: baseline=%d, final=%d (allowed slack: +2)", baseline, final)
	}
}

// TestAdmissionFuzz_SnapshotBudgetsMatchInternal: SnapshotBudgets stays in
// lock-step with the internal books at every step. Regression here means
// /status would lie.
func TestAdmissionFuzz_SnapshotBudgetsMatchInternal(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 7)
	rng := rand.New(rand.NewSource(seed))
	a, models := newRandomAdmission(t, seed)

	const N = 500
	for i := 0; i < N; i++ {
		kind, name := randomOp(rng, models)
		applyOp(t, a, kind, name)

		snap := a.SnapshotBudgets()
		byGPU := map[int]GPUBudgetSnapshot{}
		for _, s := range snap {
			byGPU[s.GPUID] = s
		}
		a.mu.Lock()
		for g, total := range a.totalsByGPU {
			s, ok := byGPU[g]
			if !ok {
				a.mu.Unlock()
				t.Fatalf("step %d (%s %s): SnapshotBudgets missing GPU %d", i, kind, name, g)
			}
			if s.TotalMB != total {
				a.mu.Unlock()
				t.Fatalf("step %d: GPU %d total mismatch: snap=%d internal=%d", i, g, s.TotalMB, total)
			}
			if s.AwakeMB != a.awakeByGPU[g] {
				a.mu.Unlock()
				t.Fatalf("step %d (%s %s): GPU %d awake mismatch: snap=%d internal=%d",
					i, kind, name, g, s.AwakeMB, a.awakeByGPU[g])
			}
			if s.L1ResidualMB != a.l1ResidualByGPU[g] {
				a.mu.Unlock()
				t.Fatalf("step %d (%s %s): GPU %d residual mismatch: snap=%d internal=%d",
					i, kind, name, g, s.L1ResidualMB, a.l1ResidualByGPU[g])
			}
			if s.PinnedMB != a.pinnedByGPU[g] {
				a.mu.Unlock()
				t.Fatalf("step %d (%s %s): GPU %d pinned mismatch: snap=%d internal=%d",
					i, kind, name, g, s.PinnedMB, a.pinnedByGPU[g])
			}
			expectedAvail := total - a.pinnedByGPU[g] - a.awakeByGPU[g] - a.l1ResidualByGPU[g]
			if s.AvailableMB != expectedAvail {
				a.mu.Unlock()
				t.Fatalf("step %d: GPU %d available mismatch: snap=%d expected=%d",
					i, g, s.AvailableMB, expectedAvail)
			}
		}
		a.mu.Unlock()
	}
}

// TestAdmissionFuzz_StateTransitionsLegal asserts every observed state
// transition is in the legal set documented in admission.go.
//
// Legal transitions (per code + comments):
//   Unknown   → {Unknown, Awake, Sleeping, Stopped}
//   Awake     → {Awake, Sleeping, Stopped}
//   Sleeping  → {Awake, Sleeping, Stopped}
//   Stopped   → {Sleeping (via NotifyStarted), Stopped}
//
// Stopped → Awake is FORBIDDEN at the single-API-call level — the
// contract requires NotifyStarted (→ Sleeping) before any RequestWake.
// However, applyOp's RequestWake branch internally runs that
// NotifyStarted first when it sees IsStopped, so the observed
// before→after composite is Stopped → Awake. We treat that one case as
// legal because the harness, not the controller, performed the
// double-step.
func TestAdmissionFuzz_StateTransitionsLegal(t *testing.T) {
	seed := seedFromEnv(defaultFuzzSeed + 8)
	rng := rand.New(rand.NewSource(seed))
	a, models := newRandomAdmission(t, seed)

	legal := map[admissionModelState]map[admissionModelState]bool{
		admissionUnknown:  {admissionUnknown: true, admissionAwake: true, admissionSleeping: true, admissionStopped: true},
		admissionAwake:    {admissionAwake: true, admissionSleeping: true, admissionStopped: true},
		admissionSleeping: {admissionAwake: true, admissionSleeping: true, admissionStopped: true},
		admissionStopped:  {admissionSleeping: true, admissionStopped: true},
	}

	const N = 1000
	for i := 0; i < N; i++ {
		kind, name := randomOp(rng, models)
		a.mu.Lock()
		before, ok := modelStateLocked(a, name)
		a.mu.Unlock()
		if !ok {
			continue
		}
		applyOp(t, a, kind, name)
		a.mu.Lock()
		after, _ := modelStateLocked(a, name)
		a.mu.Unlock()
		if before == admissionStopped && after == admissionAwake && kind == opRequestWake {
			continue
		}
		if !legal[before][after] {
			t.Fatalf("step %d (%s %s): illegal transition %d → %d", i, kind, name, before, after)
		}
	}
}

// TestAdmissionFuzz_FixedSeedReproducibility: same seed → same op
// sequence → same final book state. Smoke test for the harness itself.
func TestAdmissionFuzz_FixedSeedReproducibility(t *testing.T) {
	seed := int64(0xDEADBEEF)
	run := func() bookSnapshot {
		rng := rand.New(rand.NewSource(seed))
		a, models := newRandomAdmission(t, seed)
		for i := 0; i < 200; i++ {
			kind, name := randomOp(rng, models)
			applyOp(t, a, kind, name)
		}
		return snapshotBooks(a)
	}
	s1 := run()
	s2 := run()
	if !bookEqual(s1, s2) {
		t.Errorf("non-deterministic fuzz: same seed produced different books\n  run1: %v\n  run2: %v", s1, s2)
	}
}

// --- helpers for compact diagnostics ---

func (bs bookSnapshot) String() string {
	gpus := make([]int, 0, len(bs.awake))
	for g := range bs.awake {
		gpus = append(gpus, g)
	}
	sort.Ints(gpus)
	out := "{"
	for _, g := range gpus {
		out += fmt.Sprintf("gpu%d:[a=%d l1=%d p=%d] ", g, bs.awake[g], bs.l1[g], bs.pinned[g])
	}
	names := make([]string, 0, len(bs.stateByMo))
	for n := range bs.stateByMo {
		names = append(names, n)
	}
	sort.Strings(names)
	out += "states:["
	for _, n := range names {
		out += fmt.Sprintf("%s=%d ", n, bs.stateByMo[n])
	}
	out += "]}"
	return out
}
