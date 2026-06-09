package jukebox

import (
	"vllm-jukebox/internal/config"
)

// liveCfg returns the latest validated *config.Config installed by the
// fsnotify watcher (via config.SetCurrent), or the construction-time
// fallback when the watcher hasn't run yet — primarily a test path that
// constructs a Scheduler / Coordinator without calling config.SetCurrent.
//
// Use this everywhere a subsystem needs a HOT-RELOADABLE per-model
// field (Pinned, SwapGroup, EvictAction, Priority,
// ExpectedVRAMMBPerGPU, SleepL1ResidualMB, ColdLoadTimeoutSeconds,
// IdleTimeout, SleepLevel, WakeTimeout, GPUs). Structural / boot-time
// fields (server.port, vllm.binary, scheduler block topology,
// log_dir paths) are intentionally NOT hot-reloadable and may safely
// continue to read from the captured s.cfg pointer.
//
// See README of the hot-reload-matrix below — keep in sync with the
// schema comment block at the top of internal/config/config.go.
//
// Hot-reload matrix (per-model fields)
//
//	Field                          Hot-reloadable  Notes
//	---------------------------    --------------  ---------------------------
//	expected_vram_mb_per_gpu       yes             Grandfathered: existing
//	                                               awake bookings keep their
//	                                               original value; NEW
//	                                               admission decisions use
//	                                               the new value.
//	cold_load_timeout_seconds      yes             Picked up on next
//	                                               cold-load.
//	evict_action                   yes             Picked up on next
//	                                               eviction.
//	pinned                         yes             Picked up on next
//	                                               admission decision /
//	                                               idle scan.
//	swap_group                     yes             Picked up on next
//	                                               eviction / auto-restore
//	                                               cycle.
//	sleep_l1_residual_mb           yes             Grandfathered like
//	                                               expected_vram_mb_per_gpu.
//	priority                       yes             Picked up on next
//	                                               eviction.
//	gpus                           yes             Picked up on next
//	                                               eviction / admission
//	                                               decision; existing
//	                                               instance keeps its
//	                                               originally-allocated
//	                                               GPU set.
//	idle_timeout                   yes             Picked up on next idle
//	                                               scan tick.
//	sleep_level / wake_timeout     yes             Picked up on next
//	                                               sleep / wake.
//	power_limit / power_limits     yes             Picked up on next
//	                                               model start (managed
//	                                               lifecycle).
//	---------------------------    --------------  ---------------------------
//	Top-level fields NOT hot-reloadable:
//	  - server.port, server.host
//	  - vllm.port, vllm.binary
//	  - vllm.startup_timeout, vllm.drain_timeout, vllm.shutdown_timeout
//	  - vllm.swap_cooldown, vllm.swap_wait_timeout
//	  - vllm.log_dir, vllm.log_max_size_mb, vllm.log_max_files
//	  - scheduler.* (port range, max_instances, nvlink_pairs, ...)
//	  - llama_cpp.binary
//	  - gpu_power_limits, default_power_limit, power_limit_required
//	  - watcher target path itself
//
// fallback is the construction-time *Config; never nil in production.
func liveCfg(fallback *config.Config) *config.Config {
	if c := config.Current(); c != nil {
		return c
	}
	return fallback
}

// liveModelCfg returns the latest ModelConfig for `name` from
// config.Current() when available, else from the fallback. Returns
// (zero, false) if the model is absent from both. Aliases are NOT
// resolved — callers that want alias-resolution should use
// liveResolveModel.
func liveModelCfg(fallback *config.Config, name string) (config.ModelConfig, bool) {
	cfg := liveCfg(fallback)
	if cfg == nil {
		return config.ModelConfig{}, false
	}
	m, ok := cfg.Models[name]
	return m, ok
}

// liveResolveModel is the hot-reload-aware equivalent of
// (*config.Config).ResolveModel. It resolves alias chains against the
// latest Current() config (falling back to the captured config when
// Current() is nil — primarily a test path).
func liveResolveModel(fallback *config.Config, name string) (string, config.ModelConfig, error) {
	return liveCfg(fallback).ResolveModel(name)
}
