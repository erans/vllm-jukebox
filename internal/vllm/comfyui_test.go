package vllm_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/vllm"
)

// comfyAt builds a ComfyUIManager pointing at an httptest server's URL.
// Mirrors externalAt in external_test.go.
func comfyAt(t *testing.T, rawURL string) *vllm.ComfyUIManager {
	t.Helper()
	hostPort := strings.TrimPrefix(rawURL, "http://")
	parts := strings.SplitN(hostPort, ":", 2)
	if len(parts) != 2 {
		t.Fatalf("unexpected httptest URL shape: %s", rawURL)
	}
	var port int
	for _, c := range parts[1] {
		if c < '0' || c > '9' {
			t.Fatalf("non-numeric port in URL: %s", rawURL)
		}
		port = port*10 + int(c-'0')
	}
	return vllm.NewComfyUIManager(parts[0], port)
}

func TestComfyUIManager_BaseURL(t *testing.T) {
	m := vllm.NewComfyUIManager("comfyui-svc", 8188)
	got := m.BaseURL()
	want := "http://comfyui-svc:8188"
	if got != want {
		t.Fatalf("BaseURL: got %q want %q", got, want)
	}
}

func TestComfyUIManager_CurrentPIDNonZero(t *testing.T) {
	m := vllm.NewComfyUIManager("x", 1234)
	if m.CurrentPID() == 0 {
		t.Fatalf("CurrentPID must be non-zero so scheduler guards don't trip")
	}
}

func TestComfyUIManager_StartIsNoOpAndReturnsSentinelPID(t *testing.T) {
	m := vllm.NewComfyUIManager("x", 1234)
	pid, err := m.Start(context.Background(), "any-model")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if pid != m.CurrentPID() {
		t.Fatalf("Start pid (%d) != CurrentPID (%d)", pid, m.CurrentPID())
	}
}

func TestComfyUIManager_StopIsNoOp(t *testing.T) {
	m := vllm.NewComfyUIManager("x", 1234)
	if err := m.Stop(context.Background()); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if m.CurrentPID() == 0 {
		t.Fatalf("Stop must not zero CurrentPID")
	}
}

func TestComfyUIManager_SleepPostsFreeWithRightBody(t *testing.T) {
	var sawMethod, sawPath, sawCT string
	var sawBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawMethod = r.Method
		sawPath = r.URL.Path
		sawCT = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		sawBody = b
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	if err := m.Sleep(context.Background(), 1); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if sawMethod != http.MethodPost {
		t.Fatalf("expected POST, got %s", sawMethod)
	}
	if sawPath != "/free" {
		t.Fatalf("expected /free, got %s", sawPath)
	}
	if sawCT != "application/json" {
		t.Fatalf("expected Content-Type application/json, got %q", sawCT)
	}
	var got map[string]any
	if err := json.Unmarshal(sawBody, &got); err != nil {
		t.Fatalf("body not JSON: %v (body=%q)", err, string(sawBody))
	}
	if v, ok := got["unload_models"].(bool); !ok || !v {
		t.Fatalf("expected unload_models: true, got %v (body=%q)", got["unload_models"], string(sawBody))
	}
	if v, ok := got["free_memory"].(bool); !ok || !v {
		t.Fatalf("expected free_memory: true, got %v (body=%q)", got["free_memory"], string(sawBody))
	}
}

func TestComfyUIManager_SleepIgnoresLevelArgument(t *testing.T) {
	// ComfyUI has only one freeing mode. Level 1, 2, or anything else
	// should all produce the same body — verified by the unconditional
	// {"unload_models": true, "free_memory": true} the manager sends.
	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, b)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	for _, lvl := range []int{0, 1, 2, 99} {
		if err := m.Sleep(context.Background(), lvl); err != nil {
			t.Fatalf("Sleep(level=%d): %v", lvl, err)
		}
	}
	for i, b := range bodies {
		if i == 0 {
			continue
		}
		if string(b) != string(bodies[0]) {
			t.Fatalf("Sleep body varies by level (call %d differs): %q vs %q", i, string(b), string(bodies[0]))
		}
	}
}

func TestComfyUIManager_VerifyReadyChecksSystemStats(t *testing.T) {
	var hit int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/system_stats" {
			t.Errorf("VerifyReady made unexpected request to %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		hit++
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"system":{},"devices":[]}`)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := m.VerifyReady(ctx, "any-model"); err != nil {
		t.Fatalf("VerifyReady: %v", err)
	}
	if hit == 0 {
		t.Fatalf("expected /system_stats to be hit at least once")
	}
}

func TestComfyUIManager_VerifyReadyFailsOnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	if err := m.VerifyReady(ctx, "any-model"); err == nil {
		t.Fatalf("expected VerifyReady to fail when /system_stats returns 500")
	}
}

func TestComfyUIManager_SleepFailsOn5xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	if err := m.Sleep(context.Background(), 1); err == nil {
		t.Fatalf("expected Sleep to fail on 502, got nil")
	}
	// Sleeping state must not flip on failure.
	asleep, _ := m.IsSleeping(context.Background())
	if asleep {
		t.Fatalf("Sleep error must not set sleeping=true")
	}
}

func TestComfyUIManager_IsSleepingReturnsInternalState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	m := comfyAt(t, srv.URL)
	if asleep, err := m.IsSleeping(context.Background()); err != nil || asleep {
		t.Fatalf("initial IsSleeping: asleep=%v err=%v (want false,nil)", asleep, err)
	}
	if err := m.Sleep(context.Background(), 1); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if asleep, err := m.IsSleeping(context.Background()); err != nil || !asleep {
		t.Fatalf("after Sleep, IsSleeping: asleep=%v err=%v (want true,nil)", asleep, err)
	}
	// Wake against the same reachable server flips state back to false.
	if err := m.Wake(context.Background(), 2*time.Second); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	if asleep, err := m.IsSleeping(context.Background()); err != nil || asleep {
		t.Fatalf("after Wake, IsSleeping: asleep=%v err=%v (want false,nil)", asleep, err)
	}
}

func TestComfyUIManager_WakeReachabilityCheck(t *testing.T) {
	// Point at a port we know is unreachable. Wake should fail within
	// its timeout (we use a short one so the test doesn't hang).
	// 127.0.0.1:1 is a privileged port that won't accept connections in
	// any sane test environment.
	m := vllm.NewComfyUIManager("127.0.0.1", 1)

	// Mark it asleep via the Sleep API against an unreachable server
	// would also fail, so just call Wake against an unreachable URL
	// directly — we're testing the wake-time reachability gate.
	start := time.Now()
	err := m.Wake(context.Background(), 500*time.Millisecond)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("expected Wake against unreachable URL to fail")
	}
	// Should fail roughly within the timeout (allow some slack for CI).
	if elapsed > 2*time.Second {
		t.Fatalf("Wake took %v, expected ≤ ~500ms+slack", elapsed)
	}
}
