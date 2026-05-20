package vllm

import (
	"fmt"
	"strings"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/vllmcli"
)

// BuildServeArgs is a thin re-export of vllmcli.BuildServeArgs so existing
// callers (and tests) inside this package keep compiling. The real
// implementation lives in internal/vllmcli to avoid an import cycle with
// internal/runtime.
func BuildServeArgs(cfg *config.Config, requestedModel string) ([]string, error) {
	return vllmcli.BuildServeArgs(cfg, requestedModel)
}

// BuildServeArgsForPort is a thin re-export of vllmcli.BuildServeArgsForPort.
func BuildServeArgsForPort(cfg *config.Config, requestedModel string, port int) ([]string, error) {
	return vllmcli.BuildServeArgsForPort(cfg, requestedModel, port)
}

func BuildEnv(base []string, defaultEnv map[string]string, modelEnv map[string]string) []string {
	envMap := map[string]string{}
	for _, kv := range base {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		envMap[k] = v
	}

	for k, v := range defaultEnv {
		envMap[k] = v
	}
	for k, v := range modelEnv {
		envMap[k] = v
	}

	out := make([]string, 0, len(envMap))
	for k, v := range envMap {
		out = append(out, fmt.Sprintf("%s=%s", k, v))
	}
	return out
}
