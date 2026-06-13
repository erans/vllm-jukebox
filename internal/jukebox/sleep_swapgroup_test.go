package jukebox

import (
	"reflect"
	"testing"

	"vllm-jukebox/internal/config"
)

// TestSwapGroupMembersForColdLoad_NilConfigDoesNotPanic regresses a
// nil-deref at sleep.go:420 reachable when config.Current() returns nil
// (documented at watcher.go: pointer is nil until SetCurrent is called,
// and tests/early-startup may exercise this path before any SetCurrent
// has fired). The cold-load goroutine spawned by KickColdLoad must
// degrade gracefully — it cannot crash the daemon — so the function
// must return [target] when there's no live config to consult.
//
// Repro before the fix: TestStress_ConcurrentKickColdLoad_SameModel and
// every TestIntegration_*ColdLoad* in this package panic on a fresh
// `go test ./internal/jukebox/...` run if they execute before any test
// that calls config.SetCurrent.
func TestSwapGroupMembersForColdLoad_NilConfigDoesNotPanic(t *testing.T) {
	prev := config.Current()
	config.SetCurrent(nil)
	t.Cleanup(func() { config.SetCurrent(prev) })

	if got := config.Current(); got != nil {
		t.Fatalf("test setup: config.Current() expected nil, got %v", got)
	}

	s := &Scheduler{}

	// swapGroup empty: the early return at the top of the function
	// short-circuits before the deref. Sanity check that path still
	// returns just [target].
	if got := s.swapGroupMembersForColdLoad("moe", ""); !reflect.DeepEqual(got, []string{"moe"}) {
		t.Errorf("empty swapGroup: got %v, want [moe]", got)
	}

	// swapGroup non-empty + nil config: pre-fix, this panicked at
	// `range config.Current().Models`. Post-fix, must return just
	// [target] (no peers discoverable without a live config).
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("nil config panicked (regression): %v", r)
		}
	}()
	if got := s.swapGroupMembersForColdLoad("moe", "core"); !reflect.DeepEqual(got, []string{"moe"}) {
		t.Errorf("nil config: got %v, want [moe]", got)
	}
}

// TestSwapGroupMembersForColdLoad_LiveConfigReturnsPeers covers the
// happy path: a live config with two same-group peers + one cross-group
// model — the function must return target + same-group peers, never the
// cross-group model. Belt-and-suspenders alongside the nil-deref test.
func TestSwapGroupMembersForColdLoad_LiveConfigReturnsPeers(t *testing.T) {
	cfg := &config.Config{
		Models: map[string]config.ModelConfig{
			"moe":      {SwapGroup: "core"},
			"main":     {SwapGroup: "core"},
			"long":     {SwapGroup: "core"},
			"vision":   {SwapGroup: "vision"}, // different group — must be excluded
			"isolated": {SwapGroup: ""},       // no group — must be excluded
		},
	}
	prev := config.Current()
	config.SetCurrent(cfg)
	t.Cleanup(func() { config.SetCurrent(prev) })

	s := &Scheduler{}
	got := s.swapGroupMembersForColdLoad("moe", "core")

	// Sort-insensitive comparison (map iteration order is random).
	gotSet := make(map[string]bool, len(got))
	for _, n := range got {
		gotSet[n] = true
	}
	want := map[string]bool{"moe": true, "main": true, "long": true}
	if !reflect.DeepEqual(gotSet, want) {
		t.Errorf("got %v (set %v), want set %v", got, gotSet, want)
	}
}
