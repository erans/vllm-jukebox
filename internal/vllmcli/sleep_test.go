package vllmcli_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"vllm-jukebox/internal/vllmcli"
)

func TestSleep_SuccessReturnsNil(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := vllmcli.Sleep(context.Background(), srv.URL, 1); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if gotPath != "/sleep?level=1" {
		t.Fatalf("unexpected request path/query: %q", gotPath)
	}
}

func TestSleep_PassesLevelInQuery(t *testing.T) {
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := vllmcli.Sleep(context.Background(), srv.URL, 2); err != nil {
		t.Fatalf("Sleep: %v", err)
	}
	if gotQuery != "level=2" {
		t.Fatalf("expected level=2 in query, got: %s", gotQuery)
	}
}

func TestSleep_ErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	err := vllmcli.Sleep(context.Background(), srv.URL, 1)
	if err == nil {
		t.Fatalf("expected error from 500 response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Fatalf("expected error to include status, got: %v", err)
	}
}

func TestSleep_ErrorOnConnectionRefused(t *testing.T) {
	// Use an unroutable URL — port 1 is reserved and won't accept.
	err := vllmcli.Sleep(context.Background(), "http://127.0.0.1:1", 1)
	if err == nil {
		t.Fatalf("expected connection error")
	}
}

func TestWake_SuccessWhenHealthyImmediately(t *testing.T) {
	var wakePosted int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/wake_up":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST /wake_up, got %s", r.Method)
			}
			atomic.StoreInt32(&wakePosted, 1)
			w.WriteHeader(http.StatusOK)
		case "/health":
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	if err := vllmcli.Wake(context.Background(), srv.URL, 5*time.Second); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	if atomic.LoadInt32(&wakePosted) != 1 {
		t.Fatalf("expected /wake_up to have been POSTed")
	}
}

func TestWake_PollsHealthUntilReady(t *testing.T) {
	// Health is 503 for the first 2 polls, then 200.
	var healthHits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/wake_up":
			w.WriteHeader(http.StatusOK)
		case "/health":
			n := atomic.AddInt32(&healthHits, 1)
			if n < 3 {
				w.WriteHeader(http.StatusServiceUnavailable)
			} else {
				w.WriteHeader(http.StatusOK)
			}
		}
	}))
	defer srv.Close()

	if err := vllmcli.Wake(context.Background(), srv.URL, 10*time.Second); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	if got := atomic.LoadInt32(&healthHits); got < 3 {
		t.Fatalf("expected at least 3 health hits, got %d", got)
	}
}

func TestWake_TimesOutWhenHealthNeverReady(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/wake_up":
			w.WriteHeader(http.StatusOK)
		case "/health":
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()

	start := time.Now()
	err := vllmcli.Wake(context.Background(), srv.URL, 600*time.Millisecond)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected timeout error, got: %v", err)
	}
	if elapsed < 500*time.Millisecond {
		t.Fatalf("returned too fast (%s) — did the loop short-circuit?", elapsed)
	}
}

func TestWake_ErrorOnWakeNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/wake_up" {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	err := vllmcli.Wake(context.Background(), srv.URL, 2*time.Second)
	if err == nil || !strings.Contains(err.Error(), "wake_up") {
		t.Fatalf("expected wake_up error, got: %v", err)
	}
}

func TestWake_ZeroTimeoutRejected(t *testing.T) {
	err := vllmcli.Wake(context.Background(), "http://127.0.0.1:1", 0)
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("expected zero-timeout error, got: %v", err)
	}
}

func TestIsSleeping_ReturnsTrue(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/is_sleeping" {
			t.Errorf("expected /is_sleeping, got %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"is_sleeping": true}`)
	}))
	defer srv.Close()

	got, err := vllmcli.IsSleeping(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("IsSleeping: %v", err)
	}
	if !got {
		t.Fatalf("expected is_sleeping=true")
	}
}

func TestIsSleeping_ReturnsFalse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"is_sleeping": false}`)
	}))
	defer srv.Close()

	got, err := vllmcli.IsSleeping(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("IsSleeping: %v", err)
	}
	if got {
		t.Fatalf("expected is_sleeping=false")
	}
}

func TestIsSleeping_ErrorOnMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"is_sleeping": notabool}`)
	}))
	defer srv.Close()

	_, err := vllmcli.IsSleeping(context.Background(), srv.URL)
	if err == nil {
		t.Fatalf("expected decode error")
	}
}

func TestIsSleeping_ErrorOnNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := vllmcli.IsSleeping(context.Background(), srv.URL)
	if err == nil {
		t.Fatalf("expected error from 500")
	}
}

func TestIsSleeping_ErrorOnConnectionRefused(t *testing.T) {
	_, err := vllmcli.IsSleeping(context.Background(), "http://127.0.0.1:1")
	if err == nil {
		t.Fatalf("expected connection error")
	}
}
