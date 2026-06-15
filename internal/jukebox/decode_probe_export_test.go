package jukebox

import (
	"context"
	"time"
)

// SetDecodeProbeFnForTest overrides the package-level /health/decode
// probe hook so decode-stall tests can drive the probe without a real
// vLLM server. Returns a restore func the test should defer. See
// decode_probe.go for the production default (defaultDecodeProbe).
func SetDecodeProbeFnForTest(fn func(ctx context.Context, baseURL string) (int, error)) func() {
	var f decodeProbeFn = fn
	decodeProbeHook.Store(&f)
	return func() { decodeProbeHook.Store(nil) }
}

// CheckDecodeStallsForTest runs one decode-probe tick synchronously,
// bypassing the ticker. Lets tests assert the sustained-stall window,
// the no-inflight guard, and the endpoint-absent graceful degrade
// deterministically.
func (s *Scheduler) CheckDecodeStallsForTest(ctx context.Context) {
	s.checkDecodeStalls(ctx)
}

// SetInstanceDrainingForTest reads the draining flag of a seeded
// instance under the scheduler lock — lets the decode-probe tests assert
// the instance was taken out of routing on a trip. Returns (draining, ok).
func (s *Scheduler) InstanceDrainingForTest(model string) (bool, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	inst, ok := s.instances[model]
	if !ok {
		return false, false
	}
	return inst.draining, true
}

// SetInstanceInflightForTest bumps the in-flight tracker of a seeded
// instance to `n` (best-effort, by calling Track n times and discarding
// the done funcs) so decode-probe tests can express "this engine has
// running requests" without driving real routing. Returns false if the
// instance isn't seeded. The leaked done-funcs are harmless in a test
// process and keep the count pinned at n for the duration of the test.
func (s *Scheduler) SetInstanceInflightForTest(model string, n int) bool {
	s.mu.RLock()
	inst, ok := s.instances[model]
	s.mu.RUnlock()
	if !ok {
		return false
	}
	for i := 0; i < n; i++ {
		_, _ = inst.inflight.Track(context.Background())
	}
	return true
}

// DecodeStallCountForTest returns the current consecutive-stall counter
// for a model (0 if none). Lets tests assert the sustained window
// advances and resets correctly.
func (s *Scheduler) DecodeStallCountForTest(model string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.decodeStallCounts == nil {
		return 0
	}
	return s.decodeStallCounts[model]
}

// nowFuncForTest is unused outside tests but kept here so the export
// file imports time (the seed helpers in redeploy_test.go set up the
// clock). Avoids an unused-import churn if a future test needs it.
var _ = time.Now
