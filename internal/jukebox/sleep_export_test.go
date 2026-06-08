package jukebox

import "context"

// CheckIdleForTest is a test-only export that triggers a single idle-check
// pass synchronously. Lets tests validate auto-suspend without waiting for
// the 30s ticker. The _test.go suffix means this symbol is only present
// during `go test` builds.
func (s *Scheduler) CheckIdleForTest(ctx context.Context) { s.checkIdle(ctx) }
