package jukebox

import (
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
func TestMain(m *testing.M) {
	old := coldLoadPollInterval.Load()
	coldLoadPollInterval.Store(int64(10 * time.Millisecond))
	code := m.Run()
	coldLoadPollInterval.Store(old)
	os.Exit(code)
}
