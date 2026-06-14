package vllm

import (
	"vllm-jukebox/internal/config"
)

// liveCfg returns the latest hot-reloaded *config.Config installed by
// the fsnotify watcher (config.SetCurrent), falling back to the
// construction-time pointer when Current() is nil — primarily a test
// path that bypasses SetCurrent.
//
// Per the hot-reload schema matrix at the top of config.go, per-model
// fields read at vllm.Manager.Start (Path, Runtime, SleepMode, Env,
// PowerLimit*, ExtraArgs, ...) must reflect the operator's most recent
// active.yaml edit. Top-level fields (vllm.Port, vllm.Binary, the
// log_dir / log_max_*) do NOT hot-reload — they are captured at
// construction.
func liveCfg(fallback *config.Config) *config.Config {
	if c := config.Current(); c != nil {
		return c
	}
	return fallback
}
