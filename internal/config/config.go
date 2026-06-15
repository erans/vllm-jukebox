package config

import (
	"bytes"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type Duration struct {
	time.Duration
}

const (
	RuntimeVLLM     = "vllm"
	RuntimeLlamaCpp = "llama_cpp"
)

const (
	RunnerGenerate = "generate"
	RunnerPooling  = "pooling"
)

const (
	// LifecycleManaged: jukebox owns the vLLM process (default). Existing
	// behavior — jukebox spawns, monitors, and stops the vllm subprocess.
	LifecycleManaged = "managed"
	// LifecycleExternal: an external service (e.g. a separate compose
	// container) owns the vLLM process. Jukebox only proxies requests and,
	// when sleep_mode is enabled, sends sleep/wake HTTP calls. The external
	// service must be started with `--enable-sleep-mode` and
	// `VLLM_SERVER_DEV_MODE=1` for sleep-mode to function.
	LifecycleExternal = "external"
	// LifecycleComfyUI: a sibling for ComfyUI (Stable Diffusion / image-gen).
	// Jukebox does not own the process; it proxies requests and uses
	// ComfyUI's own POST /free verb for sleep (frees VRAM by unloading
	// models). There is no explicit wake call — the next POST /prompt
	// auto-reloads models from disk. There is also no /is_sleeping
	// endpoint; jukebox tracks sleeping state internally rather than
	// probing the engine. Like LifecycleExternal, sleep is implicitly
	// always-on; the lifecycle exists specifically so ComfyUI can
	// participate in the scheduler's sleep/eviction story.
	LifecycleComfyUI = "comfyui"
)

const (
	// DefaultSleepLevel is the L1 sleep level (weights→CPU RAM, KV cache
	// discarded, GPU memory released via CUDA managed memory).
	DefaultSleepLevel = 1
	// DefaultWakeTimeout is the max time to wait for /health to return 200
	// after POSTing /wake_up.
	DefaultWakeTimeout = 120 * time.Second
	// DefaultColdLoadTimeout is the max time to wait for /is_sleeping=true
	// after `docker start` on a Stopped member during a cold load. Sized
	// for big-context FP8-KV models with cudagraph capture (e.g. Qwen3
	// 27B @ 512K observed taking ~6-7 min between docker start and
	// is_sleeping=true). Smaller models cold-load in seconds and
	// tolerate the headroom. Override per-model via
	// `cold_load_timeout_seconds`.
	DefaultColdLoadTimeout = 10 * time.Minute
	// MinColdLoadTimeout is the validator floor for
	// cold_load_timeout_seconds. Anything tighter than 30s is almost
	// certainly a config bug — even cached page-cache reloads of small
	// weights take longer than that once you include CUDA context init.
	MinColdLoadTimeout = 30 * time.Second
	// MaxColdLoadTimeout is the validator ceiling. 30 minutes is far
	// beyond the worst observed cold load and exceeds it intentionally
	// gives operators a hard upper bound — anything longer is a sign
	// that something else is broken (disk thrash, NCCL hang, etc.) and
	// should fail loud rather than wait silently.
	MaxColdLoadTimeout = 30 * time.Minute
	// DefaultCumemDrainTimeout is the cumem PP-broadcast drain barrier
	// before /sleep — defense against vllm-project/vllm#45520, which
	// observes that calling /sleep with in-flight decode corrupts the
	// cumem allocator state via cudaErrorIllegalAddress. We refuse to
	// /sleep while requests are still mid-flight, waiting up to this
	// budget for them to drain. 5s comfortably covers in-flight
	// generations from interactive callers (typical p99 ~1-3s) without
	// stalling idle-evict for streaming workloads.
	DefaultCumemDrainTimeout = 5 * time.Second
	// DefaultWakeSettleSeconds is the post-wake settle barrier — defense
	// against vllm-project/vllm#45519, which observes that /wake_up
	// returns 200 before all PP workers have actually settled. A /sleep
	// that lands in that gap re-enters cumem mid-settle and wedges the
	// engine. 2s is short enough that legitimate idle-evicts of a
	// just-woken model still trigger normally, long enough that the
	// PP-broadcast settle has provably completed on observed hardware.
	DefaultWakeSettleSeconds = 2 * time.Second
	// DefaultMinTimeSinceWake is the broader rapid-cycle anti-race gate
	// applied AFTER the post-wake settle barrier. Where WakeSettle
	// (2s) defends the narrow PP-broadcast window upstream calls out
	// in vllm#45519, MinTimeSinceWake defends the wider observed race
	// in 100-cycle stress where a 2-3s sleep-after-wake on cycle ~23
	// triggers cudaErrorIllegalAddress in cumem.py:202; /wake_up returns
	// 200 but the engine is silently wedged and only `docker rm -f` +
	// recreate recovers. The 5s default brackets the longest observed
	// safe interval in our fleet measurement; operators can tune via
	// per-model min_time_since_wake_seconds (0 = disable, only if you
	// have a very strong reason — disabling re-opens the wedge race).
	DefaultMinTimeSinceWake = 5 * time.Second
	// DefaultWakeVerifyTimeout is the budget for the post-/wake_up
	// verification probe — a single-token /v1/completions request issued
	// AFTER /wake_up returns 200 and BEFORE jukebox routes the user
	// request. Defends against the "phantom wake" failure mode where
	// vLLM's cumem_tag wake returns 200 from the API server but one or
	// more PP/TP workers have wedged on "CUDA Error: invalid argument
	// at cumem_allocator.cpp:169"; /health continues to return 200
	// (the FastAPI process is alive) but every subsequent inference
	// request hangs forever waiting for the shm_broadcast block. The
	// probe takes ~30-80ms on a healthy wake (one decode step at
	// max_tokens:1, prefix-cache-defeating prompt) and hard-fails on
	// a wedged engine inside this budget. 5s is short enough that the
	// added wake latency is negligible relative to a hot wake (1-2s)
	// and far below cold-load (5min); long enough to forgive a slow
	// post-wake cudagraph re-replay on big-context models.
	// Set per-model via wake_verify_timeout_ms; 0 = use default; -1 =
	// explicitly disable the probe (caller takes the phantom-wake risk;
	// recommended only for non-cumem-sleep models).
	DefaultWakeVerifyTimeout = 5 * time.Second
	// DefaultDecodeProbeInterval is the cadence applied when a decode
	// probe is constructed without an explicit interval. Callers treat 0
	// as "disabled" and only fall back to this when they have a reason to
	// run the probe. 30s is deliberate: PR vllm#45097's /health/decode
	// has a known long-prefill false-positive caveat, and the probe also
	// requires the stall to persist across >=2 intervals before tripping,
	// so 30s × 2 = ~60s detection latency clears honest long prefills
	// while still catching a real #45094 wedge promptly. See
	// BehaviorConfig.DecodeProbeInterval for the full rationale.
	DefaultDecodeProbeInterval = 30 * time.Second
)

// Priority controls how the admission controller picks eviction victims.
// "best-effort" is evicted first regardless of recency; "normal" is evicted
// by LRU among non-critical peers; "critical" is never evicted. Pinned
// models are always treated as critical regardless of this field.
const (
	PriorityCritical   = "critical"
	PriorityNormal     = "normal"
	PriorityBestEffort = "best-effort"
	DefaultPriority    = PriorityNormal
)

// EvictAction selects how admission frees a model's VRAM when an in-group
// peer wakes. Sleep is the default (vLLM /sleep, fast wake); Stop tears
// the container down for full reclaim of CUDA context + NCCL buffers,
// at the cost of a cold-load (~5 min from page cache) on next demand.
const (
	EvictActionSleep   = "sleep"
	EvictActionStop    = "stop"
	DefaultEvictAction = EvictActionSleep
)

func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode && value.Tag == "!!null" {
		d.Duration = 0
		return nil
	}
	if value.Kind != yaml.ScalarNode {
		return fmt.Errorf("duration must be a scalar")
	}
	parsed, err := time.ParseDuration(value.Value)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", value.Value, err)
	}
	d.Duration = parsed
	return nil
}

// Hot-reload schema matrix.
//
// fsnotify-driven reload (config.Watch + atomic.Pointer swap in
// watcher.go) is partial — only fields read at request-time via
// config.Current() pick up changes within ~500ms of an active.yaml
// write. Fields captured at construction time still require a process
// restart.
//
// Per-model fields that DO hot-reload (read via jukebox.liveModelCfg
// at decision time):
//   - expected_vram_mb_per_gpu  (next admission decision; existing
//     bookings grandfathered)
//   - sleep_l1_residual_mb      (next admission decision; existing
//     bookings grandfathered)
//   - cold_load_timeout_seconds (next cold-load)
//   - evict_action              (next eviction)
//   - pinned                    (next admission decision / idle scan)
//   - swap_group                (next eviction / auto-restore)
//   - priority                  (next eviction)
//   - gpus                      (next eviction / admission decision;
//     existing instance keeps its allocated GPU set)
//   - idle_timeout              (next idle scan tick)
//   - sleep_level / wake_timeout (next sleep / wake call)
//   - power_limit / power_limits (next model start)
//
// Top-level fields that DO NOT hot-reload (restart required):
//   - server.* (host, port, timeouts)
//   - vllm.* (port, binary, all timeouts, log_dir/log_*, default_env)
//   - scheduler.* (port range, max_instances, nvlink_pairs,
//     min_instance_uptime)
//   - llama_cpp.* (binary, default_args)
//   - gpu_power_limits, default_power_limit, power_limit_required
//   - the active.yaml watcher target path itself
//   - behavior.default_model, behavior.rewrite_model_name
//
// See internal/jukebox/livecfg.go for the production read-site helper
// and the per-field grandfathering notes. AdmissionController.RefreshConfig
// reconciles the per-model state (Pinned/SwapGroup/EvictAction/Priority/
// VRAM expectations) on the watcher callback.
type Config struct {
	Server             ServerConfig           `yaml:"server"`
	VLLM               VLLMConfig             `yaml:"vllm"`
	Behavior           BehaviorConfig         `yaml:"behavior"`
	Scheduler          *SchedulerConfig       `yaml:"scheduler"`
	LlamaCpp           *LlamaCppConfig        `yaml:"llama_cpp"`
	Models             map[string]ModelConfig `yaml:"models"`
	GPUPowerLimits     map[int]int            `yaml:"gpu_power_limits"`
	DefaultPowerLimit  *int                   `yaml:"default_power_limit"`
	PowerLimitRequired bool                   `yaml:"power_limit_required"`
}

type ServerConfig struct {
	Host         string   `yaml:"host"`
	Port         int      `yaml:"port"`
	ReadTimeout  Duration `yaml:"read_timeout"`
	WriteTimeout Duration `yaml:"write_timeout"`
	LogRequests  *bool    `yaml:"log_requests"`
}

type VLLMConfig struct {
	Port            int      `yaml:"port"`
	Binary          string   `yaml:"binary"`
	StartupTimeout  Duration `yaml:"startup_timeout"`
	ShutdownTimeout Duration `yaml:"shutdown_timeout"`
	DrainTimeout    Duration `yaml:"drain_timeout"`
	SwapCooldown    Duration `yaml:"swap_cooldown"`
	SwapWaitTimeout Duration `yaml:"swap_wait_timeout"`

	Defaults   VLLMDefaults      `yaml:"defaults"`
	DefaultEnv map[string]string `yaml:"default_env"`

	LogDir       string `yaml:"log_dir"`
	LogMaxSizeMB int    `yaml:"log_max_size_mb"`
	LogMaxFiles  int    `yaml:"log_max_files"`

	// TCPLiveness controls the per-upstream TCP dial probe that demotes
	// a peer when its listening socket stops accepting connections. The
	// HTTP /health endpoint cannot catch this case because a wedged
	// inference loop can leave the socket half-listening while every
	// request hangs to timeout. Defaults to enabled — opt-out by
	// setting `enabled: false`.
	TCPLiveness TCPLivenessConfig `yaml:"tcp_liveness"`
}

// TCPLivenessConfig configures the TCP-level liveness probe in
// internal/health. Zero values get sensible defaults applied at load.
type TCPLivenessConfig struct {
	// Enabled gates the probe entirely. Defaults to true.
	Enabled *bool `yaml:"enabled"`

	// Interval between probe attempts.
	Interval Duration `yaml:"interval"`

	// Timeout per dial attempt.
	Timeout Duration `yaml:"timeout"`

	// FailThreshold is consecutive failures before demote.
	FailThreshold int `yaml:"fail_threshold"`

	// RecoverThreshold is consecutive successes required to recover.
	RecoverThreshold int `yaml:"recover_threshold"`

	// RetryAfterSeconds is the Retry-After header value emitted on the
	// 503 returned while the upstream is demoted. Defaults to 10.
	RetryAfterSeconds int `yaml:"retry_after_seconds"`
}

type SchedulerConfig struct {
	NvidiaSMIBinary   string    `yaml:"nvidia_smi_binary"`
	PortRangeStart    int       `yaml:"port_range_start"`
	PortRangeEnd      int       `yaml:"port_range_end"`
	MaxInstances      *int      `yaml:"max_instances"`
	MinInstanceUptime *Duration `yaml:"min_instance_uptime"`
	NVLinkPairs       [][]int   `yaml:"nvlink_pairs"`
}

type LlamaCppConfig struct {
	Binary      string   `yaml:"binary"`
	DefaultArgs []string `yaml:"default_args"`
}

type VLLMDefaults struct {
	GPUMemoryUtilization *float64 `yaml:"gpu_memory_utilization"`
	DType                string   `yaml:"dtype"`
	MaxModelLen          *int     `yaml:"max_model_len"`
}

type BehaviorConfig struct {
	DefaultModel     string `yaml:"default_model"`
	RewriteModelName bool   `yaml:"rewrite_model_name"`

	// --- content-aware routing (Stage J overnight build, 2026-06-10) ---
	//
	// All fields below are gated on ContentRouting (kill-switch, default
	// false). When false, the proxy handler skips the classifier entirely
	// and behaves byte-for-byte as before. Hot-reloadable via active.yaml
	// fsnotify watcher.

	// ContentRouting enables the content classifier in the chat-completions
	// proxy path. When true, inbound POST /v1/chat/completions bodies are
	// inspected for (a) image_url content blocks (image-reroute rule P1)
	// and (b) byte-length-derived token estimate (length-overflow rule P2).
	// Other endpoints (/v1/embeddings, /v1/rerank, /v1/tokenize, etc.) are
	// untouched regardless of this flag.
	//
	// Default false — must be explicitly enabled per stack.
	ContentRouting bool `yaml:"content_routing"`

	// MultimodalCapableModels lists served-names that natively accept
	// image_url content blocks. Requests with images targeting models in
	// this list are forwarded as-is. Requests with images targeting models
	// NOT in this list are rerouted to VisionModel (or rejected if no
	// VisionModel is configured — see DecisionRejectNoVision).
	MultimodalCapableModels []string `yaml:"multimodal_capable_models"`

	// VisionModel is the served-name to which image-bearing requests are
	// rerouted when the requested model is not multimodal-capable. When
	// empty, image requests to text-only models return 503 with a clear
	// pointer rather than silently 200 with a hallucinated description.
	VisionModel string `yaml:"vision_model"`

	// OverflowModel maps {requested-served-name → long-context-served-name}.
	// When a request's estimated token count exceeds
	// LengthOverflowThresholdTokens AND there's an entry here for the
	// requested model, the proxy rewrites the target to the overflow model.
	// Models not in this map are forwarded as-is (vLLM will return its own
	// "context too long" error).
	OverflowModel map[string]string `yaml:"overflow_model"`

	// LengthOverflowThresholdTokens is the estimated-token boundary above
	// which OverflowModel is consulted. Set ~6K under the requested model's
	// max_model_len to leave room for response tokens (e.g. 256000 for a
	// 262144-cap model with 6K reserved for output).
	//
	// Default 0 = length-overflow rule disabled.
	LengthOverflowThresholdTokens int `yaml:"length_overflow_threshold_tokens"`

	// LengthHardCapTokens is the absolute ceiling above which NO model can
	// serve the request. Even the overflow model has a max — once an
	// estimate exceeds this, the proxy returns 503 "exceeds long-context
	// capacity" rather than wasting a swap to a model that will also
	// reject.
	//
	// Default 0 = hard-cap disabled (only forwarder/overflow rules apply).
	LengthHardCapTokens int `yaml:"length_hard_cap_tokens"`

	// EstTokensCharsPerToken calibrates the byte-length token estimate.
	// JSON-encoded chat (with role markers, escapes, tool definitions)
	// runs ~3 chars/tok in English vs the naive 4 chars/tok. Defaulting
	// to 3.3 gives ±25% accuracy at the 256K threshold, which is fine for
	// the coarse reroute-or-not decision.
	//
	// Default 0 = use 3.3.
	EstTokensCharsPerToken float64 `yaml:"est_tokens_chars_per_token"`

	// CircuitBreaker is the per-model 5xx auto-restart policy. Replaces
	// the external sidecar that polls proxy access logs. Default
	// disabled; opt in via behavior.circuit_breaker.enabled = true.
	CircuitBreaker CircuitBreakerConfig `yaml:"circuit_breaker"`

	// DecodeProbeInterval is the cadence of the background /health/decode
	// decode-stall probe (DETECTION + RECOVERY half of the
	// vllm-project/vllm#45094 fix). When > 0, jukebox periodically GETs
	// <baseURL>/health/decode on each external-lifecycle instance that is
	// StateReady with a live PID; a sustained non-200 response indicates
	// the engine is wedged (the #45094 PP-split-brain manifests as
	// Running>0 with 0 decode progress, which plain /health misses because
	// the FastAPI process stays alive and keeps answering 200). On a
	// sustained stall the probe marks the instance draining and fires the
	// circuit breaker's docker-restart path.
	//
	// 0 (the default) DISABLES the probe entirely — zero behavior change,
	// so deploying an image carrying this field is a no-op until an
	// operator opts in.
	//
	// Default-when-enabled is 30s (DefaultDecodeProbeInterval), NOT 15s:
	// PR vllm#45097 (/health/decode) has a documented long-prefill
	// false-positive caveat (a legitimately long prefill can show as
	// no-decode-progress for several seconds), and the probe additionally
	// requires the stall to persist across >=2 intervals before tripping.
	// 30s × 2 intervals comfortably clears the longest observed honest
	// prefill while still catching a real wedge within ~1 minute.
	//
	// If the upstream lacks /health/decode (older vLLM returns 404, or the
	// socket connection-refuses), the probe graceful-degrades: it
	// debug-logs and SKIPS — it never trips on a missing endpoint.
	DecodeProbeInterval Duration `yaml:"decode_probe_interval"`
}

// CircuitBreakerConfig configures the jukebox-native circuit breaker.
// When a model's recent response stream shows THRESHOLD consecutive 5xx
// AND the scheduler reports the model as Ready (not sleeping/stopped),
// the breaker `docker restart`s the model's container (per
// ModelConfig.Container) to recover from a hung engine. Sleep/Stopped
// states are intentional jukebox-managed states and never trigger a
// trip — the breaker only catches /v1/models-says-healthy-but-
// EngineCore-is-dead crashes.
//
// All fields are hot-reloadable via active.yaml fsnotify watcher.
type CircuitBreakerConfig struct {
	// Enabled is the master kill-switch. Default false — when false the
	// breaker observes nothing and trips nothing.
	Enabled bool `yaml:"enabled"`

	// Window is the number of most-recent responses kept per model in
	// the rolling decision buffer. Default 5.
	Window int `yaml:"window"`

	// Threshold is the count of consecutive 5xx from the head of Window
	// required to fire a trip. Default 3.
	Threshold int `yaml:"threshold"`

	// CooldownSeconds suppresses re-trips on the same container within
	// this many seconds of the most recent trip. Default 300 (5min).
	CooldownSeconds int `yaml:"cooldown_seconds"`

	// CrashLoopMaxTrips is the cap on trips per container within
	// CrashLoopWindowSeconds before the breaker freezes that container
	// (no further trips, operator intervention required). Default 3.
	CrashLoopMaxTrips int `yaml:"crash_loop_max_trips"`

	// CrashLoopWindowSeconds is the sliding window for the crash-loop
	// guard. Default 1800 (30min).
	CrashLoopWindowSeconds int `yaml:"crash_loop_window_seconds"`

	// RestartTimeoutSeconds bounds the docker restart wall-clock.
	// Default 60.
	RestartTimeoutSeconds int `yaml:"restart_timeout_seconds"`

	// ColdLoadGraceMaxSeconds bounds the cold-load grace gate so a
	// container that NEVER returns a 2xx doesn't silently suppress every
	// trip forever. After this many seconds since the first response was
	// observed for a container, the grace gate exits and trips are
	// allowed even without a prior 2xx. Default 900 (15min). Set 0 to
	// disable the timeout (latch-only mode).
	ColdLoadGraceMaxSeconds int `yaml:"cold_load_grace_max_seconds"`

	// DryRun logs decisions without executing docker restart. Useful
	// during initial rollout. Default false.
	DryRun bool `yaml:"dry_run"`
}

// PredictiveColdLoadConfig per-model knobs for the histogram-driven
// pre-warm predictor. Only meaningful when EvictAction = "stop" (so the
// model is the cold-load target). Validation: see config.Validate.
//
// All fields are optional; zero values mean "use defaults":
//   - Enabled: false → predictor does not even tick this model.
//   - PThreshold: 0 → 0.4 (predictor's package default).
//   - MaxPerDay: 0 → 24 (one warm per hour budget).
//   - CooldownMinutes: 0 → 30 (mute after a warm fires).
//   - WindowMinutes: 0 → 10 (forecast horizon).
type PredictiveColdLoadConfig struct {
	Enabled         bool    `yaml:"enabled"`
	PThreshold      float64 `yaml:"p_threshold,omitempty"`
	MaxPerDay       int     `yaml:"max_per_day,omitempty"`
	CooldownMinutes int     `yaml:"cooldown_minutes,omitempty"`
	WindowMinutes   int     `yaml:"window_minutes,omitempty"`
}

type ModelConfig struct {
	Path                 string            `yaml:"path"`
	Runtime              string            `yaml:"runtime"`
	Runner               string            `yaml:"runner"`
	Alias                string            `yaml:"alias"`
	GPUs                 []int             `yaml:"gpus"`
	MinFreeMemMBPerGPU   *int              `yaml:"min_free_mem_mb_per_gpu"`
	Pinned               *bool             `yaml:"pinned"`
	TensorParallelSize   *int              `yaml:"tensor_parallel_size"`
	PipelineParallelSize *int              `yaml:"pipeline_parallel_size"`
	MaxModelLen          *int              `yaml:"max_model_len"`
	GPUMemoryUtilization *float64          `yaml:"gpu_memory_utilization"`
	DType                string            `yaml:"dtype"`
	Quantization         string            `yaml:"quantization"`
	ExtraArgs            []string          `yaml:"extra_args"`
	Env                  map[string]string `yaml:"env"`
	PowerLimit           *int              `yaml:"power_limit"`
	PowerLimits          map[int]int       `yaml:"power_limits"`
	LogFile              string            `yaml:"log_file"`

	// --- sleep mode / lifecycle / runner ---

	// SleepMode enables vLLM's sleep/wake API in place of full process
	// stop/start when this model is evicted or auto-suspended. Requires
	// runtime: vllm. Default false (zero behavior change).
	SleepMode bool `yaml:"sleep_mode"`
	// SleepLevel selects the vLLM sleep depth. 1 = L1 (weights→CPU RAM,
	// fastest wake). 2 = L2 (discard weights, requires disk reload on wake).
	// 0 = use DefaultSleepLevel. Only meaningful when SleepMode is true.
	SleepLevel int `yaml:"sleep_level"`
	// WakeTimeout caps how long jukebox waits for /health to return 200
	// after POSTing /wake_up. 0 = use DefaultWakeTimeout.
	WakeTimeout Duration `yaml:"wake_timeout"`
	// ColdLoadTimeoutSeconds caps how long jukebox waits for the model
	// to reach is_sleeping=true after `docker start` on a Stopped
	// evict_action: stop member. 0 = use DefaultColdLoadTimeout (10 min).
	// Bump this for big-context FP8-KV models — Qwen3 27B @ 512K with
	// cudagraph capture has been observed taking ~6-7 min cleanly, and
	// the historical 5-min default SIGKILLed it mid-capture. Validator
	// requires 30s..30m when set.
	ColdLoadTimeoutSeconds int `yaml:"cold_load_timeout_seconds,omitempty"`
	// CumemDrainTimeoutSeconds is the budget for the in-flight drain
	// barrier executed before /sleep. When the in-flight tracker is
	// non-empty, sleepInstance waits up to this duration for requests
	// to drain; if any remain, /sleep is rejected with
	// ErrSleepRejectedInflight. Defends against vllm-project/vllm#45520
	// (calling /sleep with in-flight decode corrupts cumem state).
	// 0 = use DefaultCumemDrainTimeout (5s). Validator floor: >= 0;
	// values > 30s emit a warning (very large drain budgets stall
	// admission-driven evicts on streaming workloads).
	CumemDrainTimeoutSeconds int `yaml:"cumem_drain_timeout_seconds,omitempty"`
	// WakeSettleSeconds is the post-wake settle barrier window. After
	// a successful /wake_up, /sleep on the same model is refused for
	// this duration. Defends against vllm-project/vllm#45519 (/wake_up
	// returns 200 before PP workers settle; a /sleep in that gap wedges
	// the engine). 0 = use DefaultWakeSettleSeconds (2s). Per-model
	// because vision pipelines need a longer settle than text-only
	// (more PP ranks + image-prefill cumem touch).
	WakeSettleSeconds int `yaml:"wake_settle_seconds,omitempty"`
	// MinTimeSinceWakeSeconds is the broader rapid-cycle anti-race gate.
	// After a successful /wake_up, /sleep on the same model is refused
	// for this duration regardless of the WakeSettle window. Defends
	// against the observed cycle-23 stress race (jukebox-side memory
	// project_jukebox_rapid_sleep_wake_race): /wake_up returns 200 but
	// the engine is silently wedged by cudaErrorIllegalAddress in
	// cumem.py:202 when /sleep lands too close behind the wake; only
	// `docker rm -f` + recreate recovers. 0 = use DefaultMinTimeSinceWake
	// (5s, default-on). -1 = explicitly disable. Positive = explicit
	// window. The asymmetry from WakeSettle (opt-in) is deliberate.
	MinTimeSinceWakeSeconds int `yaml:"min_time_since_wake_seconds,omitempty"`
	// WakeVerifyTimeoutMs is the budget (milliseconds) for the post-wake
	// verification probe — a single-token /v1/completions issued after
	// /wake_up returns 200 to confirm the engine isn't wedged in a
	// "phantom wake" (cumem-allocator wedge where /health lies 200).
	// 0 = use DefaultWakeVerifyTimeout (5s); -1 = disable the probe
	// entirely (NOT recommended for cumem-sleep models — re-opens the
	// phantom-wake hole). See DefaultWakeVerifyTimeout for rationale.
	WakeVerifyTimeoutMs int `yaml:"wake_verify_timeout_ms,omitempty"`
	// ServedModelName is the name vLLM is actually serving (i.e. what
	// --served-model-name was set to via extra_args, or what vLLM
	// defaults to — typically the path basename). The post-wake verify
	// probe (VerifyWakeWithProbe) sends this in the `model` field of
	// /v1/completions; if it doesn't match what vLLM is serving, vLLM
	// returns 404/400 NotFoundError and the probe (pre-fix) treated
	// that as a phantom wake → infinite redeploy loop.
	//
	// When unset, the probe falls back to the jukebox config key
	// (map key in models{}), which matches when --served-model-name is
	// not overridden. Set this explicitly any time extra_args contains
	// --served-model-name <X> where X differs from the config key.
	//
	// The validator warns loudly when sleep_mode=true + extra_args has
	// --served-model-name + this field is unset, since that's the
	// exact footgun that triggered HIGH-3 in the wake-verify review.
	ServedModelName string `yaml:"served_model_name,omitempty"`
	// IdleTimeout: if > 0 and SleepMode is true and the model is not pinned,
	// jukebox auto-sleeps the instance after this much idle time. 0 = never
	// auto-suspend.
	IdleTimeout Duration `yaml:"idle_timeout"`
	// Lifecycle selects who owns the vLLM process. "managed" (default) =
	// jukebox spawns it. "external" = jukebox only proxies + sleeps/wakes
	// an already-running vLLM at Host:Port.
	Lifecycle string `yaml:"lifecycle"`
	// Host is the hostname/IP of the external vLLM service. Required when
	// Lifecycle == "external"; defaults to "127.0.0.1" if unset.
	Host string `yaml:"host"`
	// Port is the listen port of the external vLLM service. Required when
	// Lifecycle == "external".
	Port int `yaml:"port"`

	// --- admission control ---

	// ExpectedVRAMMBPerGPU is the model's awake VRAM footprint, per GPU it
	// occupies. When > 0, the admission controller uses this to track per-GPU
	// budgets and may evict lower-priority peers to make room for a wake.
	// Default 0 = no admission control for this model (legacy behavior:
	// jukebox calls /wake_up directly and any OOM surfaces as 503).
	ExpectedVRAMMBPerGPU int `yaml:"expected_vram_mb_per_gpu"`
	// ExpectedVRAMMBByGPU is the per-GPU override of ExpectedVRAMMBPerGPU
	// — needed when a model's footprint is asymmetric across its GPUs
	// (e.g. vLLM pipeline parallelism: rank 1 carries the embedding +
	// lm_head + sampler, +1-2 GiB heavier than rank 0). Using a flat
	// scalar in that case forces the operator to pick the heavier rank
	// for the whole set, stranding the delta on the lighter ranks.
	//
	// When set, every GPU in `gpus` MUST have an entry, each value MUST
	// be > 0, and ExpectedVRAMMBPerGPU MUST be unset (mutually exclusive
	// — picking one source of truth avoids drift). When unset, admission
	// falls back to ExpectedVRAMMBPerGPU on every GPU.
	//
	// Default nil = admission uses the flat ExpectedVRAMMBPerGPU value
	// (legacy behavior).
	ExpectedVRAMMBByGPU map[int]int `yaml:"expected_vram_mb_by_gpu,omitempty" json:"expected_vram_mb_by_gpu,omitempty"`
	// SleepL1ResidualMB is the VRAM that vLLM's cumem allocator keeps
	// reserved after an L1 sleep (weights freed, but the pool is not
	// returned to the driver until full process exit). Measured empirically
	// via `nvidia-smi` before and after a sleep cycle. Used by admission to
	// understand how much VRAM a "sleeping" model still costs on each GPU.
	// Default 0 = assume sleep frees the full expected footprint.
	SleepL1ResidualMB int `yaml:"sleep_l1_residual_mb"`
	// Priority controls eviction order — one of "critical", "normal",
	// "best-effort". Empty defaults to "normal". Pinned models are always
	// treated as critical regardless of this field.
	Priority string `yaml:"priority"`
	// SwapGroup names a mutual-exclusion group. Members of the same group
	// can evict each other through the admission controller regardless of
	// priority — overriding the pinned-is-critical-never-evicted rule, but
	// ONLY within the group. Outside the group, priority still applies
	// normally. When a group member auto-sleeps via idle_timeout, jukebox
	// auto-wakes the highest-priority sleeping member of the same group
	// (so the pinned default can auto-restore when a transient peer idles).
	// Default empty = model is not in any swap group (legacy behavior).
	SwapGroup string `yaml:"swap_group"`
	// EvictAction selects how admission frees this model's VRAM when a
	// peer wakes and needs the room: "sleep" (default — vLLM /sleep,
	// fast wake via /wake_up) or "stop" (docker compose stop the
	// container, full reclaim of CUDA context + NCCL buffers, restored
	// on next demand via docker compose up + health-wait).
	//
	// Stop is the right choice for rarely-woken evict-group members
	// where the L1 residual that survives sleep (~2 GB/GPU per peer
	// stack) accumulates to meaningful headroom across multiple slept
	// peers, AND the cold-load wall (~5 min from page cache) is
	// acceptable because consumer demand is sparse.
	//
	// Default empty = "sleep" (backward-compat).
	EvictAction string `yaml:"evict_action"`

	// Container is the docker container name backing this model — used
	// by the jukebox-native circuit breaker (behavior.circuit_breaker)
	// to issue `docker restart <container>` when N consecutive 5xx
	// indicate a hung engine. Empty = breaker skips this model.
	Container string `yaml:"container,omitempty"`

	// PredictiveColdLoad opts this model into traffic-histogram-driven
	// pre-warming. Default: disabled (zero behavior change).
	PredictiveColdLoad PredictiveColdLoadConfig `yaml:"predictive_cold_load,omitempty"`

	// MaxConcurrentRequests caps how many requests jukebox will allow
	// in-flight against this model's instance at once. When > 0, the
	// proxy-dispatch admission point refuses a request that would push
	// the instance's in-flight count above the cap, returning a 503 +
	// Retry-After (RejectConcurrencyLimit). When 0 (the default) the cap
	// is disabled and concurrency is unbounded — byte-for-byte legacy
	// behavior, so deploying an image carrying this field is a no-op
	// until an operator sets a positive value.
	//
	// PREVENTION half of the vllm-project/vllm#45094 fix. That issue is a
	// pipeline-parallel cudagraph-vs-eager dispatch split-brain: when too
	// many long-decode requests are admitted into a single engine step,
	// the admitted batch shape crosses a cudagraph-bucket boundary that
	// PP stage 0 and PP stage 1 disagree on (one replays a captured graph
	// with baked-in P2P send/recv shapes, the other runs eager), and the
	// cross-stage pipeline P2P never rendezvous — all GPUs pin at 100%
	// util @ 0 tok/s forever while /health keeps lying 200. py-spy-proven
	// 2026-06-14: 7 concurrent @ 512 tok drained cleanly; 12 concurrent @
	// 1500 tok wedged on the first wave, deterministically. Bounding the
	// admitted concurrency keeps the engine below the batch-shape
	// threshold that triggers the divergence.
	//
	// The cap is per-MODEL (matched against the live in-flight count on
	// the routed instance), hot-reloadable via the active.yaml fsnotify
	// watcher (read at dispatch time through liveModelCfg), and
	// independent of admission-control VRAM tracking — a model with no
	// expected_vram_mb_per_gpu can still carry a concurrency cap.
	//
	// Operator guidance (measured on the TP2×PP2 Qwen3.6-27B stack that
	// reproduced #45094): 7 concurrent @ 512-tok decode is safe; 12 @
	// 1500-tok wedges. A cap of ~8 leaves headroom under the observed
	// wedge threshold. The right value is workload-specific (it scales
	// with max_tokens / decode length, not just request count) — leave
	// this 0 and tune empirically per stack.
	MaxConcurrentRequests int `yaml:"max_concurrent_requests,omitempty"`
}

func Load(data []byte) (*Config, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)

	var cfg Config
	if err := dec.Decode(&cfg); err != nil {
		return nil, err
	}
	cfg.applyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *Config) applyDefaults() {
	if c.Server.Host == "" {
		c.Server.Host = "0.0.0.0"
	}
	if c.Server.Port == 0 {
		c.Server.Port = 8080
	}
	if c.Server.ReadTimeout.Duration == 0 {
		c.Server.ReadTimeout = Duration{Duration: 30 * time.Second}
	}
	if c.Server.WriteTimeout.Duration == 0 {
		c.Server.WriteTimeout = Duration{Duration: 300 * time.Second}
	}
	if c.Server.LogRequests == nil {
		// Default to logging requests (can be noisy in prod; make it configurable).
		v := true
		c.Server.LogRequests = &v
	}

	if c.VLLM.Port == 0 {
		c.VLLM.Port = 8000
	}
	if c.VLLM.Binary == "" {
		c.VLLM.Binary = "uvx"
	}
	if c.VLLM.StartupTimeout.Duration == 0 {
		c.VLLM.StartupTimeout = Duration{Duration: 300 * time.Second}
	}
	if c.VLLM.ShutdownTimeout.Duration == 0 {
		c.VLLM.ShutdownTimeout = Duration{Duration: 30 * time.Second}
	}
	if c.VLLM.DrainTimeout.Duration == 0 {
		c.VLLM.DrainTimeout = Duration{Duration: 60 * time.Second}
	}
	if c.VLLM.SwapCooldown.Duration == 0 {
		c.VLLM.SwapCooldown = Duration{Duration: 30 * time.Second}
	}
	if c.VLLM.SwapWaitTimeout.Duration == 0 {
		c.VLLM.SwapWaitTimeout = Duration{Duration: 60 * time.Second}
	}
	if c.VLLM.DefaultEnv == nil {
		c.VLLM.DefaultEnv = map[string]string{}
	}
	if c.VLLM.LogMaxSizeMB == 0 {
		c.VLLM.LogMaxSizeMB = 50
	}
	if c.VLLM.LogMaxFiles == 0 {
		c.VLLM.LogMaxFiles = 5
	}

	// TCP liveness: enabled by default, with conservative timing that
	// catches a wedged upstream within ~15s without flapping on a single
	// slow GC pause.
	if c.VLLM.TCPLiveness.Enabled == nil {
		t := true
		c.VLLM.TCPLiveness.Enabled = &t
	}
	if c.VLLM.TCPLiveness.Interval.Duration == 0 {
		c.VLLM.TCPLiveness.Interval = Duration{Duration: 5 * time.Second}
	}
	if c.VLLM.TCPLiveness.Timeout.Duration == 0 {
		c.VLLM.TCPLiveness.Timeout = Duration{Duration: 2 * time.Second}
	}
	if c.VLLM.TCPLiveness.FailThreshold == 0 {
		c.VLLM.TCPLiveness.FailThreshold = 3
	}
	if c.VLLM.TCPLiveness.RecoverThreshold == 0 {
		c.VLLM.TCPLiveness.RecoverThreshold = 1
	}
	if c.VLLM.TCPLiveness.RetryAfterSeconds == 0 {
		c.VLLM.TCPLiveness.RetryAfterSeconds = 10
	}

	if c.Scheduler != nil {
		if c.Scheduler.NvidiaSMIBinary == "" {
			c.Scheduler.NvidiaSMIBinary = "nvidia-smi"
		}
		if c.Scheduler.MinInstanceUptime == nil {
			c.Scheduler.MinInstanceUptime = &Duration{Duration: 30 * time.Second}
		}
	}

	// Per-model defaults (sleep mode / lifecycle).
	for name, model := range c.Models {
		mutated := false
		// LifecycleExternal historically defaults Host to 127.0.0.1.
		// LifecycleComfyUI intentionally does NOT — ComfyUI is almost
		// always a sibling container reachable by service-name, and
		// silently defaulting to localhost has bitten operators in
		// production. Require explicit host.
		if model.Lifecycle == LifecycleExternal && model.Host == "" {
			model.Host = "127.0.0.1"
			mutated = true
		}
		if mutated {
			c.Models[name] = model
		}
	}
}

func (c *Config) Validate() error {
	if len(c.Models) == 0 {
		return fmt.Errorf("at least one model must be configured")
	}
	if err := validatePort("server.port", c.Server.Port); err != nil {
		return err
	}
	if err := validatePort("vllm.port", c.VLLM.Port); err != nil {
		return err
	}

	if err := c.validateRuntimes(); err != nil {
		return err
	}

	if err := c.validateScheduler(); err != nil {
		return err
	}

	if c.Behavior.DefaultModel != "" {
		if _, ok := c.Models[c.Behavior.DefaultModel]; !ok {
			return fmt.Errorf("default_model %q not found in models", c.Behavior.DefaultModel)
		}
	}

	// Content-aware routing validation. Only enforced when ContentRouting
	// is enabled — when disabled, fields can be partially-set (operator
	// staging the rollout) without erroring out.
	if c.Behavior.ContentRouting {
		if c.Behavior.VisionModel != "" {
			if _, ok := c.Models[c.Behavior.VisionModel]; !ok {
				return fmt.Errorf("behavior.vision_model %q not found in models", c.Behavior.VisionModel)
			}
		}
		for _, mm := range c.Behavior.MultimodalCapableModels {
			if _, ok := c.Models[mm]; !ok {
				return fmt.Errorf("behavior.multimodal_capable_models entry %q not found in models", mm)
			}
		}
		for from, to := range c.Behavior.OverflowModel {
			if _, ok := c.Models[from]; !ok {
				return fmt.Errorf("behavior.overflow_model: source %q not found in models", from)
			}
			if _, ok := c.Models[to]; !ok {
				return fmt.Errorf("behavior.overflow_model: target %q (for %q) not found in models", to, from)
			}
		}
		if c.Behavior.LengthOverflowThresholdTokens < 0 {
			return fmt.Errorf("behavior.length_overflow_threshold_tokens must be >= 0 (got %d)", c.Behavior.LengthOverflowThresholdTokens)
		}
		if c.Behavior.LengthHardCapTokens < 0 {
			return fmt.Errorf("behavior.length_hard_cap_tokens must be >= 0 (got %d)", c.Behavior.LengthHardCapTokens)
		}
		if c.Behavior.LengthHardCapTokens > 0 && c.Behavior.LengthOverflowThresholdTokens > 0 &&
			c.Behavior.LengthHardCapTokens < c.Behavior.LengthOverflowThresholdTokens {
			return fmt.Errorf("behavior.length_hard_cap_tokens (%d) must be >= length_overflow_threshold_tokens (%d)",
				c.Behavior.LengthHardCapTokens, c.Behavior.LengthOverflowThresholdTokens)
		}
		if c.Behavior.EstTokensCharsPerToken < 0 {
			return fmt.Errorf("behavior.est_tokens_chars_per_token must be >= 0 (got %f)", c.Behavior.EstTokensCharsPerToken)
		}
	}

	// Validate global power limit settings
	if err := c.validatePowerLimits(); err != nil {
		return err
	}

	// Validate log settings
	if err := c.validateLogSettings(); err != nil {
		return err
	}

	for name, model := range c.Models {
		if model.Alias == "" && model.Path == "" && model.Lifecycle != LifecycleExternal && model.Lifecycle != LifecycleComfyUI {
			return fmt.Errorf("model %q requires 'path' field", name)
		}
		if model.GPUMemoryUtilization != nil {
			if *model.GPUMemoryUtilization < 0.0 || *model.GPUMemoryUtilization > 1.0 {
				return fmt.Errorf("model %q gpu_memory_utilization must be between 0.0 and 1.0", name)
			}
		}

		// Validate per-model power limit settings
		if err := c.validateModelPowerLimits(name, model); err != nil {
			return err
		}
	}

	if err := c.validateAliases(); err != nil {
		return err
	}

	if err := c.validateAdmission(); err != nil {
		return err
	}

	return nil
}

// validateAdmission checks the per-model admission control fields. Any
// model with admission enabled (ExpectedVRAMMBPerGPU > 0) must:
//   - run in scheduler mode (the legacy coordinator owns at most one
//     instance, so admission has nothing to coordinate)
//   - declare at least one GPU (admission tracks per-GPU budget) UNLESS
//     it's an alias (aliases inherit their target's admission state)
//   - have a Priority value within the allowed set (or empty for default)
//   - have a non-negative SleepL1ResidualMB that doesn't exceed the
//     awake footprint (a sleeping model can't reserve more than it took
//     while awake)
func (c *Config) validateAdmission() error {
	for name, model := range c.Models {
		// max_concurrent_requests is independent of admission VRAM
		// tracking (a model with no expected_vram_mb_per_gpu can still
		// carry a cap), so validate it here for every model regardless of
		// AdmissionEnabled(). 0 = disabled (default); negatives are an
		// obvious config bug.
		if model.MaxConcurrentRequests < 0 {
			return fmt.Errorf("model %q: max_concurrent_requests must be >= 0 (0 = unlimited); got %d", name, model.MaxConcurrentRequests)
		}

		switch model.Priority {
		case "", PriorityCritical, PriorityNormal, PriorityBestEffort:
			// ok
		default:
			return fmt.Errorf("model %q: unknown priority %q (must be one of: critical, normal, best-effort)", name, model.Priority)
		}

		switch model.EvictAction {
		case "", EvictActionSleep, EvictActionStop:
			// ok
		default:
			return fmt.Errorf("model %q: unknown evict_action %q (must be one of: sleep, stop)", name, model.EvictAction)
		}
		// evict_action: stop is only meaningful for swap_group members
		// (admission only triggers eviction within a group). And it's a
		// noisy operational mode for a pinned daily-driver, so reject
		// stop on pinned models — operators who really want it can drop
		// `pinned: true` first.
		if model.EvictAction == EvictActionStop {
			if model.SwapGroup == "" {
				return fmt.Errorf("model %q: evict_action: stop requires swap_group membership (admission only evicts within a group)", name)
			}
			if model.Pinned != nil && *model.Pinned {
				return fmt.Errorf("model %q: evict_action: stop cannot be combined with pinned: true (operators must drop pinned first to opt the daily driver into stop-on-evict)", name)
			}
			if model.Lifecycle != LifecycleExternal {
				return fmt.Errorf("model %q: evict_action: stop currently requires lifecycle: external (jukebox stops a docker-compose-managed container; managed-lifecycle members would be re-spawned by jukebox itself, defeating the purpose)", name)
			}
		}

		if !model.AdmissionEnabled() {
			continue
		}

		if model.ExpectedVRAMMBPerGPU < 0 {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu must be >= 0", name)
		}
		// Mutually exclusive: pick one source of truth so drift between the
		// scalar and the per-GPU map can't silently desync admission books.
		if model.ExpectedVRAMMBPerGPU > 0 && len(model.ExpectedVRAMMBByGPU) > 0 {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu and expected_vram_mb_by_gpu are mutually exclusive — pick one", name)
		}
		// Per-GPU map validation: every value > 0, every model GPU has an
		// entry, no stray entries for GPUs the model doesn't touch.
		if len(model.ExpectedVRAMMBByGPU) > 0 {
			for gpu, mb := range model.ExpectedVRAMMBByGPU {
				if mb <= 0 {
					return fmt.Errorf("model %q: expected_vram_mb_by_gpu[%d] must be > 0, got %d", name, gpu, mb)
				}
			}
			gpuSet := map[int]bool{}
			for _, g := range model.GPUs {
				gpuSet[g] = true
			}
			for gpu := range model.ExpectedVRAMMBByGPU {
				if !gpuSet[gpu] {
					return fmt.Errorf("model %q: expected_vram_mb_by_gpu has entry for GPU %d which is not in gpus list %v", name, gpu, model.GPUs)
				}
			}
			for _, g := range model.GPUs {
				if _, ok := model.ExpectedVRAMMBByGPU[g]; !ok {
					return fmt.Errorf("model %q: expected_vram_mb_by_gpu must have an entry for every GPU in gpus list; missing GPU %d", name, g)
				}
			}
		}
		if model.SleepL1ResidualMB < 0 {
			return fmt.Errorf("model %q: sleep_l1_residual_mb must be >= 0", name)
		}
		// L1 residual must not exceed the model's per-GPU expected footprint
		// on ANY GPU. With per-GPU calibration, the smallest per-GPU expected
		// value is the binding constraint.
		minExpected := model.ExpectedVRAMMBPerGPU
		if len(model.ExpectedVRAMMBByGPU) > 0 {
			first := true
			for _, mb := range model.ExpectedVRAMMBByGPU {
				if first || mb < minExpected {
					minExpected = mb
					first = false
				}
			}
		}
		if model.SleepL1ResidualMB > minExpected {
			return fmt.Errorf("model %q: sleep_l1_residual_mb (%d) must not exceed expected_vram_mb_per_gpu / expected_vram_mb_by_gpu (min %d) — a sleeping model cannot hold more VRAM than it took while awake",
				name, model.SleepL1ResidualMB, minExpected)
		}
		if model.Alias != "" {
			// Aliases must NOT declare their own admission footprint —
			// they share the target's state, and double-counting would
			// break budget accounting.
			return fmt.Errorf("model %q: aliases must not set expected_vram_mb_per_gpu or expected_vram_mb_by_gpu (admission state is owned by the alias target)", name)
		}
		if c.Scheduler == nil {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu/expected_vram_mb_by_gpu requires scheduler mode (admission control is scheduler-only)", name)
		}
		if len(model.GPUs) == 0 {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu/expected_vram_mb_by_gpu requires 'gpus' to be set", name)
		}
	}
	return nil
}

func (c *Config) validateRuntimes() error {
	anyLlama := false
	for name, model := range c.Models {
		switch model.Runtime {
		case "", RuntimeVLLM, RuntimeLlamaCpp:
			// ok
		default:
			return fmt.Errorf("model %q: unknown runtime %q (must be one of: vllm, llama_cpp)", name, model.Runtime)
		}
		switch model.Runner {
		case "", RunnerGenerate, RunnerPooling:
			// ok
		default:
			return fmt.Errorf("model %q: unknown runner %q (must be one of: generate, pooling)", name, model.Runner)
		}
		if model.Runner == RunnerPooling && model.Runtime == RuntimeLlamaCpp {
			return fmt.Errorf("model %q: runner: pooling is not supported with runtime: llama_cpp (vLLM-only)", name)
		}

		// Lifecycle: validate enum
		switch model.Lifecycle {
		case "", LifecycleManaged, LifecycleExternal, LifecycleComfyUI:
			// ok
		default:
			return fmt.Errorf("model %q: unknown lifecycle %q (must be one of: managed, external, comfyui)", name, model.Lifecycle)
		}

		// SleepMode incompatibility
		if model.SleepMode && model.Runtime == RuntimeLlamaCpp {
			return fmt.Errorf("model %q: sleep_mode is not supported with runtime: llama_cpp (vLLM-only)", name)
		}
		if model.SleepLevel != 0 && model.SleepLevel != 1 && model.SleepLevel != 2 {
			return fmt.Errorf("model %q: sleep_level must be 0 (default), 1, or 2; got %d", name, model.SleepLevel)
		}

		// cold_load_timeout_seconds bounds: 30s..30min when set.
		// 0/unset means "use DefaultColdLoadTimeout" via
		// EffectiveColdLoadTimeout — that's a permitted sentinel, not an
		// out-of-range error. Negative values are obvious config bugs.
		if model.ColdLoadTimeoutSeconds < 0 {
			return fmt.Errorf("model %q: cold_load_timeout_seconds must be >= 0 (0 = use default %s); got %d",
				name, DefaultColdLoadTimeout, model.ColdLoadTimeoutSeconds)
		}
		if model.ColdLoadTimeoutSeconds > 0 {
			d := time.Duration(model.ColdLoadTimeoutSeconds) * time.Second
			if d < MinColdLoadTimeout {
				return fmt.Errorf("model %q: cold_load_timeout_seconds (%ds) is below minimum %s — values that small are almost certainly a config bug",
					name, model.ColdLoadTimeoutSeconds, MinColdLoadTimeout)
			}
			if d > MaxColdLoadTimeout {
				return fmt.Errorf("model %q: cold_load_timeout_seconds (%ds) exceeds maximum %s — anything longer suggests a deeper problem (disk thrash, NCCL hang) that should fail loud",
					name, model.ColdLoadTimeoutSeconds, MaxColdLoadTimeout)
			}
		}

		// cumem_drain_timeout_seconds + wake_settle_seconds: defenses
		// against vllm#45520 / #45519 respectively. Both default-on via
		// Effective accessors when unset; the validator rejects only
		// negative values (obvious config bugs) and warns on suspiciously
		// large drain budgets. No upper bound is fatal — operator may
		// have measured a workload that legitimately needs a long drain.
		if model.CumemDrainTimeoutSeconds < 0 {
			return fmt.Errorf("model %q: cumem_drain_timeout_seconds must be >= 0 (0 = use default %s); got %d",
				name, DefaultCumemDrainTimeout, model.CumemDrainTimeoutSeconds)
		}
		if model.CumemDrainTimeoutSeconds > 30 {
			slog.Warn("config_cumem_drain_timeout_unusually_large",
				"model", name,
				"value_seconds", model.CumemDrainTimeoutSeconds,
				"note", "values >30s may stall admission-driven evicts on streaming workloads; verify intentional",
			)
		}
		if model.WakeSettleSeconds < 0 {
			return fmt.Errorf("model %q: wake_settle_seconds must be >= 0 (0 = use default %s); got %d",
				name, DefaultWakeSettleSeconds, model.WakeSettleSeconds)
		}
		if model.MinTimeSinceWakeSeconds < -1 {
			return fmt.Errorf("model %q: min_time_since_wake_seconds must be >= -1 (-1 = disable; 0 = use default %s; positive = explicit window); got %d",
				name, DefaultMinTimeSinceWake, model.MinTimeSinceWakeSeconds)
		}
		// WakeVerifyTimeoutMs: -1 = disable (operator opt-out of the
		// phantom-wake probe), 0 = use default, positive = explicit
		// budget. Reject < -1 as an obvious config bug; warn on
		// suspiciously tiny budgets (< 500ms) — a healthy probe round
		// trip on a cold-graph model can exceed 200ms and the entire
		// budget covers both round-trip latency AND the actual decode
		// step, so anything <500ms risks false-positive phantom
		// detection on legitimately slow wakes.
		if model.WakeVerifyTimeoutMs < -1 {
			return fmt.Errorf("model %q: wake_verify_timeout_ms must be >= -1 (-1 = disable; 0 = use default %s; positive = explicit budget ms); got %d",
				name, DefaultWakeVerifyTimeout, model.WakeVerifyTimeoutMs)
		}
		if model.WakeVerifyTimeoutMs > 0 && model.WakeVerifyTimeoutMs < 500 {
			slog.Warn("config_wake_verify_timeout_unusually_small",
				"model", name,
				"value_ms", model.WakeVerifyTimeoutMs,
				"note", "values <500ms risk false-positive phantom-wake detection on legitimately slow wakes; verify intentional",
			)
		}

		// HIGH-3 defense: if sleep_mode is on AND extra_args overrides
		// --served-model-name AND the operator didn't set served_model_name
		// in the jukebox config, the post-wake verify probe will send the
		// jukebox config key (e.g. "vllm-main") to vLLM's /v1/completions,
		// vLLM will return 404 because it's actually serving the
		// extra_args-supplied name, the probe will classify that as a
		// config-error and skip redeploy — which is correct but the
		// peer will never come up at all. Warn loud so the operator
		// notices BEFORE the next wake instead of staring at a stuck
		// model wondering why every wake reports
		// ErrWakeVerifyConfigError.
		if model.SleepMode && extraArgsHasServedModelName(model.ExtraArgs) && strings.TrimSpace(model.ServedModelName) == "" {
			slog.Warn("config_served_model_name_missing_with_extra_args_override",
				"model", name,
				"note", "extra_args sets --served-model-name but served_model_name field is unset; post-wake verify probe will 404 → ErrWakeVerifyConfigError on every wake; set served_model_name to match the --served-model-name value",
			)
		}

		// LifecycleExternal constraints
		if model.Lifecycle == LifecycleExternal {
			if model.Alias != "" {
				return fmt.Errorf("model %q: lifecycle: external is incompatible with alias", name)
			}
			if model.Runtime == RuntimeLlamaCpp {
				return fmt.Errorf("model %q: lifecycle: external is not supported with runtime: llama_cpp (vLLM-only)", name)
			}
			if model.Port == 0 {
				return fmt.Errorf("model %q: lifecycle: external requires 'port' to be set", name)
			}
			if c.Scheduler == nil {
				return fmt.Errorf("model %q: lifecycle: external requires scheduler mode (the legacy single-instance coordinator cannot manage external instances)", name)
			}
		}

		// LifecycleComfyUI constraints — sibling of external, but for a
		// different engine. ComfyUI sleep semantics are baked into the
		// ComfyUIManager (POST /free; no /is_sleeping; no explicit wake);
		// the SleepMode flag is implicit, so we don't require the user to
		// set it. LlamaCpp is meaningless here. Aliases are rejected for
		// the same reason as external (the manager is bound to a single
		// host:port and aliasing breaks shared-state assumptions).
		if model.Lifecycle == LifecycleComfyUI {
			if model.Alias != "" {
				return fmt.Errorf("model %q: lifecycle: comfyui is incompatible with alias", name)
			}
			if model.Runtime == RuntimeLlamaCpp {
				return fmt.Errorf("model %q: lifecycle: comfyui is not supported with runtime: llama_cpp", name)
			}
			if model.Host == "" {
				return fmt.Errorf("model %q: lifecycle: comfyui requires 'host' to be set", name)
			}
			if model.Port == 0 {
				return fmt.Errorf("model %q: lifecycle: comfyui requires 'port' to be set", name)
			}
			if c.Scheduler == nil {
				return fmt.Errorf("model %q: lifecycle: comfyui requires scheduler mode (the legacy single-instance coordinator cannot manage comfyui instances)", name)
			}
		}

		if model.Runtime == RuntimeLlamaCpp {
			anyLlama = true
			if c.Scheduler != nil {
				return fmt.Errorf("model %q: runtime llama_cpp is not supported in scheduler mode (use swap mode for llama.cpp)", name)
			}
		}
	}
	if anyLlama {
		if c.LlamaCpp == nil || c.LlamaCpp.Binary == "" {
			return fmt.Errorf("at least one model uses runtime: llama_cpp but llama_cpp.binary is not set")
		}
	}
	return nil
}

func (c *Config) validateScheduler() error {
	if c.Scheduler == nil {
		return nil
	}

	if err := validatePort("scheduler.port_range_start", c.Scheduler.PortRangeStart); err != nil {
		return err
	}
	if err := validatePort("scheduler.port_range_end", c.Scheduler.PortRangeEnd); err != nil {
		return err
	}
	if c.Scheduler.PortRangeStart > c.Scheduler.PortRangeEnd {
		return fmt.Errorf("scheduler.port_range_start must be <= scheduler.port_range_end")
	}

	if c.Scheduler.MaxInstances != nil {
		if *c.Scheduler.MaxInstances <= 0 {
			return fmt.Errorf("scheduler.max_instances must be > 0")
		}
		portCount := c.Scheduler.PortRangeEnd - c.Scheduler.PortRangeStart + 1
		if *c.Scheduler.MaxInstances > portCount {
			return fmt.Errorf("scheduler.max_instances (%d) must be <= number of ports in range (%d)", *c.Scheduler.MaxInstances, portCount)
		}
	}

	if err := c.validateNVLinkPairs(); err != nil {
		return err
	}

	if _, ok := c.VLLM.DefaultEnv["CUDA_VISIBLE_DEVICES"]; ok {
		return fmt.Errorf("vllm.default_env must not set CUDA_VISIBLE_DEVICES when scheduler is enabled")
	}
	for name, model := range c.Models {
		if model.Env != nil {
			if _, ok := model.Env["CUDA_VISIBLE_DEVICES"]; ok {
				return fmt.Errorf("model %q env must not set CUDA_VISIBLE_DEVICES when scheduler is enabled", name)
			}
		}
	}

	for name, model := range c.Models {
		if model.Alias != "" {
			continue
		}

		if len(model.GPUs) == 0 {
			return fmt.Errorf("model %q requires 'gpus' when scheduler is enabled", name)
		}
		seen := map[int]bool{}
		for _, gpu := range model.GPUs {
			if gpu < 0 {
				return fmt.Errorf("model %q gpus must be >= 0", name)
			}
			if seen[gpu] {
				return fmt.Errorf("model %q gpus must not contain duplicates", name)
			}
			seen[gpu] = true
		}

		if model.MinFreeMemMBPerGPU == nil || *model.MinFreeMemMBPerGPU <= 0 {
			return fmt.Errorf("model %q requires 'min_free_mem_mb_per_gpu' > 0 when scheduler is enabled", name)
		}

		warnNVLinkTopology(name, model.GPUs, c.Scheduler.NVLinkPairs)
	}

	return nil
}

// validateNVLinkPairs ensures the optional scheduler.nvlink_pairs config has a
// sane shape: each pair is exactly two distinct non-negative GPU IDs, and no
// GPU appears in more than one pair. When unset, validation is a no-op.
func (c *Config) validateNVLinkPairs() error {
	if c.Scheduler == nil || len(c.Scheduler.NVLinkPairs) == 0 {
		return nil
	}
	seen := map[int]int{} // gpu → pair index
	for i, pair := range c.Scheduler.NVLinkPairs {
		if len(pair) != 2 {
			return fmt.Errorf("scheduler.nvlink_pairs[%d]: each pair must have exactly 2 GPUs, got %d", i, len(pair))
		}
		if pair[0] == pair[1] {
			return fmt.Errorf("scheduler.nvlink_pairs[%d]: pair must contain two distinct GPUs, got [%d,%d]", i, pair[0], pair[1])
		}
		for _, g := range pair {
			if g < 0 {
				return fmt.Errorf("scheduler.nvlink_pairs[%d]: gpu ids must be >= 0, got %d", i, g)
			}
			if prev, dup := seen[g]; dup {
				return fmt.Errorf("scheduler.nvlink_pairs: gpu %d appears in pair %d and pair %d (each GPU may appear in at most one pair)", g, prev, i)
			}
			seen[g] = i
		}
	}
	return nil
}

// warnNVLinkTopology emits a slog.Warn when a model's GPU layout looks
// inefficient against the configured NVLink pairs. It never returns an error
// — topology is operator advice, not a hard constraint.
//
// Heuristic:
//   - 2-GPU models: warn if the pair isn't one of the configured nvlink_pairs
//   - 4-GPU models: warn if the four GPUs aren't the union of exactly two
//     configured pairs
//   - Other model sizes (1, 3, 5+): no warning (NVLink topology doesn't apply
//     to single-GPU models, and uncommon sizes have no canonical layout)
func warnNVLinkTopology(modelName string, gpus []int, pairs [][]int) {
	if len(pairs) == 0 {
		return
	}
	switch len(gpus) {
	case 2:
		if !pairMatches(gpus, pairs) {
			slog.Warn("model gpus do not match a configured NVLink pair — collective ops may cross slower interconnect",
				"model", modelName,
				"gpus", gpus,
				"nvlink_pairs", pairs,
			)
		}
	case 4:
		if !fourGPUsMatchTwoPairs(gpus, pairs) {
			slog.Warn("model gpus do not cleanly span two configured NVLink pairs — collective ops may cross slower interconnect",
				"model", modelName,
				"gpus", gpus,
				"nvlink_pairs", pairs,
				"hint", "for TP=2 PP=2 prefer arranging gpus as the union of two NVLink-bonded pairs",
			)
		}
	}
}

// pairMatches returns true when {gpus[0], gpus[1]} equals one of the configured
// pairs (order-independent within the pair).
func pairMatches(gpus []int, pairs [][]int) bool {
	if len(gpus) != 2 {
		return false
	}
	for _, p := range pairs {
		if (p[0] == gpus[0] && p[1] == gpus[1]) || (p[0] == gpus[1] && p[1] == gpus[0]) {
			return true
		}
	}
	return false
}

// fourGPUsMatchTwoPairs returns true when the 4 GPUs are exactly the union of
// two distinct configured pairs (order-independent).
func fourGPUsMatchTwoPairs(gpus []int, pairs [][]int) bool {
	if len(gpus) != 4 {
		return false
	}
	gpuSet := map[int]bool{}
	for _, g := range gpus {
		gpuSet[g] = true
	}
	if len(gpuSet) != 4 {
		return false
	}
	// Find pairs whose BOTH members are in the model's gpu set.
	matched := 0
	for _, p := range pairs {
		if gpuSet[p[0]] && gpuSet[p[1]] {
			matched++
		}
	}
	return matched == 2
}

func (c *Config) validateAliases() error {
	for name, model := range c.Models {
		if model.Alias == "" {
			continue
		}

		seen := map[string]bool{name: true}
		current := name
		steps := 0
		for {
			next := c.Models[current].Alias
			if next == "" {
				break
			}
			steps++
			if seen[next] {
				return fmt.Errorf("alias loop detected: %s → %s", name, next)
			}
			seen[next] = true

			nextModel, ok := c.Models[next]
			if !ok {
				return fmt.Errorf("alias %q references unknown model %q", name, next)
			}

			current = next
			if nextModel.Alias == "" {
				break
			}
		}

		if steps > 1 {
			return fmt.Errorf("alias %q cannot reference another alias %q", name, c.Models[name].Alias)
		}
	}
	return nil
}

func validatePort(field string, port int) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("%s must be between 1 and 65535", field)
	}
	return nil
}

// EffectiveLifecycle returns LifecycleManaged if Lifecycle is unset, otherwise
// the configured value. Use this everywhere instead of comparing the raw field
// so the default doesn't have to be repeated.
func (m ModelConfig) EffectiveLifecycle() string {
	if m.Lifecycle == "" {
		return LifecycleManaged
	}
	return m.Lifecycle
}

// EffectiveSleepLevel returns the configured SleepLevel, or DefaultSleepLevel
// if unset. Only meaningful when SleepMode is true.
func (m ModelConfig) EffectiveSleepLevel() int {
	if m.SleepLevel == 0 {
		return DefaultSleepLevel
	}
	return m.SleepLevel
}

// EffectiveWakeTimeout returns the configured WakeTimeout, or
// DefaultWakeTimeout if unset.
func (m ModelConfig) EffectiveWakeTimeout() time.Duration {
	if m.WakeTimeout.Duration <= 0 {
		return DefaultWakeTimeout
	}
	return m.WakeTimeout.Duration
}

// EffectiveWakeVerifyTimeout returns the configured WakeVerifyTimeoutMs
// as a Duration when explicitly set (>0), DefaultWakeVerifyTimeout when
// unset (0), or 0 when explicitly disabled (-1). A return of 0 tells the
// caller to SKIP the verification probe; any positive value is the
// per-probe deadline.
func (m ModelConfig) EffectiveWakeVerifyTimeout() time.Duration {
	if m.WakeVerifyTimeoutMs < 0 {
		return 0
	}
	if m.WakeVerifyTimeoutMs == 0 {
		return DefaultWakeVerifyTimeout
	}
	return time.Duration(m.WakeVerifyTimeoutMs) * time.Millisecond
}

// EffectiveServedModelName returns the served model name to use when
// addressing the post-wake verification probe. If ServedModelName is
// set in config, returns it verbatim. Otherwise returns the fallback
// (which the caller passes as the jukebox config key). The probe will
// 404 if the fallback doesn't match what vLLM is actually serving —
// see HIGH-3 in the PR #27 review.
func (m ModelConfig) EffectiveServedModelName(fallback string) string {
	if s := strings.TrimSpace(m.ServedModelName); s != "" {
		return s
	}
	return fallback
}

// extraArgsHasServedModelName reports whether the model's extra_args
// contains a --served-model-name override. Used by the validator to
// detect the HIGH-3 footgun where the operator overrode the vLLM
// served name but forgot to mirror it in jukebox's served_model_name
// field.
func extraArgsHasServedModelName(args []string) bool {
	for _, a := range args {
		if a == "--served-model-name" {
			return true
		}
		if strings.HasPrefix(a, "--served-model-name=") {
			return true
		}
	}
	return false
}

// EffectiveColdLoadTimeout returns the configured ColdLoadTimeoutSeconds,
// or DefaultColdLoadTimeout if unset / <= 0. Used by the cold-load
// polling sites (wake-from-Stopped, redeploy-member) to bound the
// /is_sleeping wait. The validator already rejects out-of-range values
// (see Validate), so this can trust whatever is set.
func (m ModelConfig) EffectiveColdLoadTimeout() time.Duration {
	if m.ColdLoadTimeoutSeconds <= 0 {
		return DefaultColdLoadTimeout
	}
	return time.Duration(m.ColdLoadTimeoutSeconds) * time.Second
}

// EffectiveCumemDrainTimeout returns the configured
// CumemDrainTimeoutSeconds as a Duration when explicitly set (>0), or
// 0 when unset/disabled. A return of 0 disables the cumem-drain guard
// for this model entirely — the existing best-effort
// VLLM.DrainTimeout poll still runs, but no /sleep call is refused on
// drain-timeout grounds. Operators opt INTO the guard by setting the
// per-model knob (DefaultCumemDrainTimeout = 5s is the recommended
// starting value); the package default is no-op so existing configs
// keep their pre-guard behavior. Defends vllm-project/vllm#45520.
func (m ModelConfig) EffectiveCumemDrainTimeout() time.Duration {
	if m.CumemDrainTimeoutSeconds <= 0 {
		return 0
	}
	return time.Duration(m.CumemDrainTimeoutSeconds) * time.Second
}

// EffectiveWakeSettle returns the configured WakeSettleSeconds as a
// Duration when explicitly set (>0), or 0 when unset/disabled. A
// return of 0 disables the post-wake settle barrier for this model
// entirely. Operators opt INTO the guard by setting the per-model
// knob (DefaultWakeSettleSeconds = 2s is the recommended starting
// value); the package default is no-op so existing configs keep their
// pre-guard behavior. Defends vllm-project/vllm#45519.
func (m ModelConfig) EffectiveWakeSettle() time.Duration {
	if m.WakeSettleSeconds <= 0 {
		return 0
	}
	return time.Duration(m.WakeSettleSeconds) * time.Second
}

// EffectiveMinTimeSinceWake returns the configured
// MinTimeSinceWakeSeconds as a Duration. Unlike EffectiveWakeSettle
// (which is opt-IN, returning 0 when unset), this gate is opt-OUT:
// when MinTimeSinceWakeSeconds is unset (0), the function returns
// DefaultMinTimeSinceWake (5s). To explicitly DISABLE the gate set
// the field to -1; the validator allows -1 for that purpose. The
// asymmetry (default-on, opt-out) is deliberate — see
// project_jukebox_rapid_sleep_wake_race for why we don't want to
// regress to the wedge-on-cycle-23 stress mode by accident.
func (m ModelConfig) EffectiveMinTimeSinceWake() time.Duration {
	if m.MinTimeSinceWakeSeconds < 0 {
		return 0
	}
	if m.MinTimeSinceWakeSeconds == 0 {
		return DefaultMinTimeSinceWake
	}
	return time.Duration(m.MinTimeSinceWakeSeconds) * time.Second
}

// EffectivePriority returns the configured Priority, defaulting to
// DefaultPriority ("normal") when unset. Pinned models are forced to
// "critical" by the admission controller regardless of this value.
func (m ModelConfig) EffectivePriority() string {
	if m.Priority == "" {
		return DefaultPriority
	}
	return m.Priority
}

// EffectiveEvictAction returns the configured EvictAction, defaulting
// to DefaultEvictAction ("sleep") when unset. The validator already
// rejects stop on pinned / non-external / non-swap-group models, so
// callers can trust this returns a value the rest of the stack can act
// on.
func (m ModelConfig) EffectiveEvictAction() string {
	if m.EvictAction == "" {
		return DefaultEvictAction
	}
	return m.EvictAction
}

// AdmissionEnabled reports whether the admission controller should track
// this model. A model with no declared VRAM footprint is invisible to
// admission — its wake is unmediated and any peer's wake won't evict it.
func (m ModelConfig) AdmissionEnabled() bool {
	return m.ExpectedVRAMMBPerGPU > 0 || len(m.ExpectedVRAMMBByGPU) > 0
}

// EffectiveExpectedVRAMMB returns the expected awake VRAM footprint for
// this model on the specified GPU. When the per-GPU map is set, the
// per-GPU value wins (validator guarantees every model GPU has an entry
// and each is > 0). Otherwise falls back to the flat scalar. For models
// where AdmissionEnabled() is false, returns 0.
func (m ModelConfig) EffectiveExpectedVRAMMB(gpu int) int {
	if v, ok := m.ExpectedVRAMMBByGPU[gpu]; ok {
		return v
	}
	return m.ExpectedVRAMMBPerGPU
}

// EffectiveMaxConcurrentRequests returns the configured per-model
// in-flight concurrency cap. 0 means "unlimited" (the cap is disabled);
// any positive value is the hard ceiling enforced at the proxy-dispatch
// admission point. PREVENTION half of vllm#45094. Hot-reloadable.
func (m ModelConfig) EffectiveMaxConcurrentRequests() int {
	if m.MaxConcurrentRequests < 0 {
		return 0
	}
	return m.MaxConcurrentRequests
}

// EffectiveDecodeProbeInterval returns the configured
// behavior.decode_probe_interval. 0 means "disabled" — callers must not
// launch the probe. Any positive value is the probe cadence. DETECTION
// half of vllm#45094.
func (b BehaviorConfig) EffectiveDecodeProbeInterval() time.Duration {
	if b.DecodeProbeInterval.Duration <= 0 {
		return 0
	}
	return b.DecodeProbeInterval.Duration
}

// AdmissionEnabled at the config level reports whether ANY model has
// admission enabled. When false the scheduler skips controller setup
// entirely.
func (c *Config) AdmissionEnabled() bool {
	for _, m := range c.Models {
		if m.AdmissionEnabled() {
			return true
		}
	}
	return false
}

func (c *Config) ResolveModel(name string) (resolvedName string, model ModelConfig, err error) {
	cfg, ok := c.Models[name]
	if !ok {
		return "", ModelConfig{}, fmt.Errorf("model %q not found", name)
	}
	if cfg.Alias == "" {
		return name, cfg, nil
	}
	target, ok := c.Models[cfg.Alias]
	if !ok {
		return "", ModelConfig{}, fmt.Errorf("alias %q references unknown model %q", name, cfg.Alias)
	}
	return cfg.Alias, target, nil
}

// ResolveLogPath returns the log file path for a model.
// Returns empty string if logging is not configured.
func (c *Config) ResolveLogPath(modelName string) string {
	model, ok := c.Models[modelName]
	if !ok {
		return ""
	}

	// Per-model override takes precedence
	if model.LogFile != "" {
		return model.LogFile
	}

	// Fall back to log_dir/<model>.log
	if c.VLLM.LogDir != "" {
		return filepath.Join(c.VLLM.LogDir, modelName+".log")
	}

	return ""
}

func (c *Config) validatePowerLimits() error {
	// gpu_power_limits and default_power_limit are mutually exclusive
	if len(c.GPUPowerLimits) > 0 && c.DefaultPowerLimit != nil {
		return fmt.Errorf("gpu_power_limits and default_power_limit are mutually exclusive")
	}

	// Validate global gpu_power_limits values are positive
	for gpu, limit := range c.GPUPowerLimits {
		if limit <= 0 {
			return fmt.Errorf("gpu_power_limits[%d] must be > 0, got %d", gpu, limit)
		}
	}

	// Validate default_power_limit is positive
	if c.DefaultPowerLimit != nil && *c.DefaultPowerLimit <= 0 {
		return fmt.Errorf("default_power_limit must be > 0, got %d", *c.DefaultPowerLimit)
	}

	return nil
}

func (c *Config) validateModelPowerLimits(name string, model ModelConfig) error {
	// power_limit and power_limits are mutually exclusive
	if model.PowerLimit != nil && len(model.PowerLimits) > 0 {
		return fmt.Errorf("model %q: power_limit and power_limits are mutually exclusive", name)
	}

	// Validate power_limit is positive
	if model.PowerLimit != nil && *model.PowerLimit <= 0 {
		return fmt.Errorf("model %q: power_limit must be > 0, got %d", name, *model.PowerLimit)
	}

	// Validate power_limits values are positive
	for gpu, limit := range model.PowerLimits {
		if limit <= 0 {
			return fmt.Errorf("model %q: power_limits[%d] must be > 0, got %d", name, gpu, limit)
		}
	}

	// If model has power_limits, all GPU indices must be in the model's gpus list
	if len(model.PowerLimits) > 0 && len(model.GPUs) > 0 {
		validGPUs := make(map[int]bool)
		for _, gpu := range model.GPUs {
			validGPUs[gpu] = true
		}

		for gpu := range model.PowerLimits {
			if !validGPUs[gpu] {
				return fmt.Errorf("model %q: power_limits GPU %d not in gpus list %v", name, gpu, model.GPUs)
			}
		}
	}

	return nil
}

func (c *Config) validateLogSettings() error {
	if c.VLLM.LogMaxSizeMB < 0 {
		return fmt.Errorf("vllm.log_max_size_mb must be >= 0, got %d", c.VLLM.LogMaxSizeMB)
	}
	if c.VLLM.LogMaxFiles < 0 {
		return fmt.Errorf("vllm.log_max_files must be >= 0, got %d", c.VLLM.LogMaxFiles)
	}
	return nil
}
