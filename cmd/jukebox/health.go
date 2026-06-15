package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"vllm-jukebox/internal/config"
)

// defaultHealthCheckPort is the fallback when no env var, no -port flag, and
// no config file value can be found. Matches config.ServerConfig.Port's
// default (see internal/config/config.go ApplyDefaults).
const defaultHealthCheckPort = 8080

// healthCheckTimeout bounds the probe end-to-end. Docker's HEALTHCHECK
// --timeout SIGKILLs after its own ceiling; we want to return non-zero
// with a useful stderr line well before that.
const healthCheckTimeout = 3 * time.Second

// runHealthCheck is the entrypoint for `jukebox health`. It opens a short
// HTTP GET against the local /health endpoint and returns a process exit
// code: 0 on HTTP 200, 1 on any other response, network error, or timeout.
//
// Port resolution order (first non-empty wins):
//  1. JUKEBOX_HEALTHCHECK_PORT env var (operator override)
//  2. server.port from the YAML config pointed to by -config / JUKEBOX_CONFIG
//  3. defaultHealthCheckPort (8080, matches config defaults)
//
// Designed for distroless containers — no shell, no curl. Image-level
// HEALTHCHECK directive: HEALTHCHECK CMD ["/jukebox", "health"].
func runHealthCheck(args []string) int {
	port, source := resolveHealthCheckPort(args)

	url := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	client := &http.Client{Timeout: healthCheckTimeout}

	resp, err := client.Get(url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "health: GET %s failed (port_source=%s): %v\n", url, source, err)
		return 1
	}
	defer func() {
		// Drain so the connection can be reused (harmless for a one-shot
		// process, but it's the polite thing to do).
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "health: %s returned status %d (port_source=%s)\n", url, resp.StatusCode, source)
		return 1
	}
	return 0
}

// resolveHealthCheckPort walks the resolution order and returns the
// chosen port plus a short string describing where it came from (for
// observability in stderr on failure).
//
// `args` is the slice AFTER the "health" subcommand token — e.g. for
// `jukebox health -port 9000 -config foo.yaml`, args == ["-port", "9000",
// "-config", "foo.yaml"].
func resolveHealthCheckPort(args []string) (port int, source string) {
	// Optional flags on the subcommand: -port and -config. Both override
	// nothing if absent. Errors here are non-fatal — we fall through to
	// env/config/default.
	var (
		flagPort       int
		flagConfigPath string
	)
	parseSubArgs(args, &flagPort, &flagConfigPath)

	if flagPort > 0 {
		return flagPort, "flag"
	}

	if envPort := os.Getenv("JUKEBOX_HEALTHCHECK_PORT"); envPort != "" {
		if p, err := strconv.Atoi(envPort); err == nil && p > 0 && p < 65536 {
			return p, "env:JUKEBOX_HEALTHCHECK_PORT"
		}
		fmt.Fprintf(os.Stderr, "health: ignoring invalid JUKEBOX_HEALTHCHECK_PORT=%q\n", envPort)
	}

	configPath := flagConfigPath
	if configPath == "" {
		configPath = os.Getenv("JUKEBOX_CONFIG")
	}
	if configPath != "" {
		if p, err := portFromConfigFile(configPath); err == nil && p > 0 {
			return p, "config:" + configPath
		}
		// Don't spam stderr on the common path (no config provided is
		// fine); only emit when the operator explicitly pointed us at
		// a file that didn't yield a usable port.
	}

	return defaultHealthCheckPort, "default"
}

// parseSubArgs is a tiny hand-rolled flag parser. The std flag package
// would also work, but using it here would emit usage text on parse
// errors which is wrong for a health probe (we want silent fallback).
func parseSubArgs(args []string, port *int, configPath *string) {
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-port", "--port":
			if i+1 < len(args) {
				if p, err := strconv.Atoi(args[i+1]); err == nil {
					*port = p
				}
				i++
			}
		case "-config", "--config":
			if i+1 < len(args) {
				*configPath = args[i+1]
				i++
			}
		}
	}
}

// portFromConfigFile reads a YAML config file and returns server.port.
// Uses the full config.Load path so any defaults (port=8080 when unset)
// also apply — operator running with the canonical example.yaml gets
// the right answer even if the file omits server.port entirely.
func portFromConfigFile(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	if len(data) == 0 {
		return 0, fmt.Errorf("empty config")
	}
	cfg, err := config.Load(data)
	if err != nil {
		return 0, err
	}
	return cfg.Server.Port, nil
}
