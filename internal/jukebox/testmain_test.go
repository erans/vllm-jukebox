package jukebox

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestMain runs once for the entire `internal/jukebox` test package and
// shrinks the cold-load /is_sleeping poll interval from its production
// 5s default down to 10ms. Every test that exercises doColdLoad,
// pollUntilSleeping (used by RedeployMember), or KickColdLoad's full
// goroutine path was paying a 5s wall-clock minimum per cold-load —
// the production tuning makes sense (5min cold loads dominate the wait,
// so a 5s tick is invisible) but in tests with mocked docker the poll
// is the ENTIRE wait. Shrinking it brings the per-test wall-clock from
// ~5-6s to ~50ms.
//
// Test-overridable design: coldLoadPollInterval is a package-level
// atomic.Int64 (nanoseconds) declared in sleep.go specifically so this
// kind of bulk override is possible. Individual tests that want to
// assert poll-tick behavior can use SetColdLoadPollIntervalForTest to
// swap to a different value and defer-restore.
//
// Also installs a package-default no-op wake-verify probe. The
// production default (vllmcli.VerifyWakeWithProbe POSTing
// /v1/completions) would fire on every wake in every test, but most
// jukebox unit tests either use fake instance managers (BaseURL=="",
// probe self-skips) or use httptest servers that mock only the
// /wake_up + /health endpoints (and would 404 the probe, causing
// every wake test to false-positive as a phantom-wake). The no-op
// keeps existing tests unaffected; phantom-wake tests opt INTO the
// production behavior (or inject a deterministic probe) via
// SetWakeVerifyProbeForTest.
func TestMain(m *testing.M) {
	old := coldLoadPollInterval.Load()
	coldLoadPollInterval.Store(int64(10 * time.Millisecond))
	restoreProbe := SetWakeVerifyProbeForTest(func(_ context.Context, _, _ string, _ time.Duration) error {
		return nil
	})
	code := m.Run()
	restoreProbe()
	coldLoadPollInterval.Store(old)
	os.Exit(code)
}
