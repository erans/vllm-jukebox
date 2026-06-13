package vllm_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"vllm-jukebox/internal/vllm"
)

func TestExternalManager_BaseURL(t *testing.T) {
	m := vllm.NewExternalManager("vllm-embeddings", 8003)
	got := m.BaseURL()
	want := "http://vllm-embeddings:8003"
	if got != want {
		t.Fatalf("BaseURL: got %q want %q", got, want)
	}
}

func TestExternalManager_CurrentPIDNonZero(t *testing.T) {
	m := vllm.NewExternalManager("x", 9000)
	if m.CurrentPID() == 0 {
		t.Fatalf("CurrentPID must be non-zero so state-machine guards don't trip")
	}
}

func TestExternalManager_StartIsNoOpAndReturnsSentinelPID(t *testing.T) {
	m := vllm.NewExternalManager("x", 9000)
	pid, err := m.Start(context.Background(), "any-model")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if pid != m.CurrentPID() {
		t.Fatalf("Start pid (%d) must match CurrentPID (%d)", pid, m.CurrentPID())
	}
}

func TestExternalManager_StopIsNoOp(t *testing.T) {
	m := vllm.NewExternalManager("x", 9000)
	if err := m.Stop(context.Background()); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	// Stop must not change CurrentPID (we never owned the process).
	if m.CurrentPID() == 0 {
		t.Fatalf("Stop must not zero CurrentPID")
	}
}

func TestExternalManager_VerifyReadyChecksHealth(t *testing.T) {
	var hit int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			hit++
			w.WriteHeader(http.StatusOK)
			return
		}
		// VerifyReady must NOT hit /v1/models (unlike Manager.VerifyReady)
		t.Errorf("ExternalManager.VerifyReady made an unexpected request to %s", r.URL.Path)
	}))
	defer srv.Close()

	m := externalAt(srv.URL)
	if err := m.VerifyReady(context.Background(), "any-model"); err != nil {
		t.Fatalf("VerifyReady: %v", err)
	}
	if hit == 0 {
		t.Fatalf("expected /health to be hit at least once")
	}
}

func TestExternalManager_SleepDelegatesToVLLMCLI(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := externalAt(srv.URL).Sleep(context.Background(), 2); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if gotPath != "/sleep?level=2" {
		t.Fatalf("unexpected request path: %q", gotPath)
	}
}

func TestExternalManager_WakeDelegatesAndPollsHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := externalAt(srv.URL).Wake(context.Background(), 3*time.Second); err != nil {
		t.Fatalf("Wake: %v", err)
	}
}

func TestExternalManager_IsSleepingDelegates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"is_sleeping": true}`)
	}))
	defer srv.Close()

	got, err := externalAt(srv.URL).IsSleeping(context.Background())
	if err != nil {
		t.Fatalf("IsSleeping: %v", err)
	}
	if !got {
		t.Fatalf("expected is_sleeping=true")
	}
}

// externalAt constructs an ExternalManager pointing at the host:port
// portion of an arbitrary URL (typically from httptest.NewServer.URL).
func externalAt(rawURL string) *vllm.ExternalManager {
	// httptest URLs look like http://127.0.0.1:42137
	hostPort := strings.TrimPrefix(rawURL, "http://")
	// SplitN to handle IPv6 in the future
	parts := strings.SplitN(hostPort, ":", 2)
	if len(parts) != 2 {
		panic("unexpected httptest URL shape: " + rawURL)
	}
	var port int
	if _, err := fmt.Sscanf(parts[1], "%d", &port); err != nil {
		panic(err)
	}
	return vllm.NewExternalManager(parts[0], port)
}
