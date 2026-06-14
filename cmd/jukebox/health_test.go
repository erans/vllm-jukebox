package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// startTestServer starts a httptest.Server on 127.0.0.1, returns its port
// and a teardown closer. We need the actual port number because
// runHealthCheck hits 127.0.0.1:<port>/health.
func startTestServer(t *testing.T, status int) (int, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(status)
		_, _ = fmt.Fprintf(w, `{"status":%d}`, status)
	}))
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatalf("parse port: %v", err)
	}
	return p, srv.Close
}

func TestRunHealthCheck_OK(t *testing.T) {
	port, stop := startTestServer(t, 200)
	defer stop()

	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", strconv.Itoa(port))
	t.Setenv("JUKEBOX_CONFIG", "")

	if code := runHealthCheck(nil); code != 0 {
		t.Fatalf("expected exit 0, got %d", code)
	}
}

func TestRunHealthCheck_5xx(t *testing.T) {
	port, stop := startTestServer(t, 500)
	defer stop()

	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", strconv.Itoa(port))
	t.Setenv("JUKEBOX_CONFIG", "")

	if code := runHealthCheck(nil); code != 1 {
		t.Fatalf("expected exit 1 on 500, got %d", code)
	}
}

func TestRunHealthCheck_Unreachable(t *testing.T) {
	// Bind-and-immediately-close to get a port that's guaranteed not
	// listening. This is more reliable than picking a "probably unused"
	// number.
	port, stop := startTestServer(t, 200)
	stop() // tear down NOW — port is now silent

	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", strconv.Itoa(port))
	t.Setenv("JUKEBOX_CONFIG", "")

	if code := runHealthCheck(nil); code != 1 {
		t.Fatalf("expected exit 1 on unreachable, got %d", code)
	}
}

func TestResolveHealthCheckPort_EnvWins(t *testing.T) {
	// Even if a config file is present, env takes precedence.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "active.yaml")
	if err := os.WriteFile(cfgPath, []byte("server:\n  port: 7777\nmodels:\n  m1: {path: /tmp/x}\nbehavior:\n  default_model: m1\n"), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", "9999")
	t.Setenv("JUKEBOX_CONFIG", cfgPath)

	port, source := resolveHealthCheckPort(nil)
	if port != 9999 {
		t.Fatalf("expected port 9999 (env), got %d (source=%s)", port, source)
	}
	if source != "env:JUKEBOX_HEALTHCHECK_PORT" {
		t.Fatalf("expected env source, got %q", source)
	}
}

func TestResolveHealthCheckPort_FlagBeatsEnv(t *testing.T) {
	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", "9999")
	t.Setenv("JUKEBOX_CONFIG", "")

	port, source := resolveHealthCheckPort([]string{"-port", "1234"})
	if port != 1234 {
		t.Fatalf("expected port 1234 (flag), got %d (source=%s)", port, source)
	}
	if source != "flag" {
		t.Fatalf("expected flag source, got %q", source)
	}
}

func TestResolveHealthCheckPort_ConfigSecond(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "active.yaml")
	// Minimum config that survives validation: needs at least one model
	// + behavior.default_model. server.port is the value we care about.
	yaml := `
server:
  port: 7777
models:
  m1:
    path: /tmp/x
behavior:
  default_model: m1
`
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", "")
	t.Setenv("JUKEBOX_CONFIG", cfgPath)

	port, source := resolveHealthCheckPort(nil)
	if port != 7777 {
		t.Fatalf("expected port 7777 (config), got %d (source=%s)", port, source)
	}
	if source != "config:"+cfgPath {
		t.Fatalf("expected config source, got %q", source)
	}
}

func TestResolveHealthCheckPort_DefaultThird(t *testing.T) {
	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", "")
	t.Setenv("JUKEBOX_CONFIG", "")

	port, source := resolveHealthCheckPort(nil)
	if port != defaultHealthCheckPort {
		t.Fatalf("expected default port %d, got %d (source=%s)", defaultHealthCheckPort, port, source)
	}
	if source != "default" {
		t.Fatalf("expected default source, got %q", source)
	}
}

func TestResolveHealthCheckPort_InvalidEnvFallsThrough(t *testing.T) {
	// Bogus env value should be ignored — not crash, not used. Resolution
	// continues to config → default.
	t.Setenv("JUKEBOX_HEALTHCHECK_PORT", "not-a-number")
	t.Setenv("JUKEBOX_CONFIG", "")

	port, source := resolveHealthCheckPort(nil)
	if port != defaultHealthCheckPort {
		t.Fatalf("expected default port %d (env was bogus), got %d (source=%s)", defaultHealthCheckPort, port, source)
	}
	if source != "default" {
		t.Fatalf("expected default source, got %q", source)
	}
}
