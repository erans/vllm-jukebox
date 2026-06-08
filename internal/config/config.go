package config

import (
	"bytes"
	"fmt"
	"log/slog"
	"path/filepath"
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
		switch model.Priority {
		case "", PriorityCritical, PriorityNormal, PriorityBestEffort:
			// ok
		default:
			return fmt.Errorf("model %q: unknown priority %q (must be one of: critical, normal, best-effort)", name, model.Priority)
		}

		if !model.AdmissionEnabled() {
			continue
		}

		if model.ExpectedVRAMMBPerGPU < 0 {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu must be >= 0", name)
		}
		if model.SleepL1ResidualMB < 0 {
			return fmt.Errorf("model %q: sleep_l1_residual_mb must be >= 0", name)
		}
		if model.SleepL1ResidualMB > model.ExpectedVRAMMBPerGPU {
			return fmt.Errorf("model %q: sleep_l1_residual_mb (%d) must not exceed expected_vram_mb_per_gpu (%d) — a sleeping model cannot hold more VRAM than it took while awake",
				name, model.SleepL1ResidualMB, model.ExpectedVRAMMBPerGPU)
		}
		if model.Alias != "" {
			// Aliases must NOT declare their own admission footprint —
			// they share the target's state, and double-counting would
			// break budget accounting.
			return fmt.Errorf("model %q: aliases must not set expected_vram_mb_per_gpu (admission state is owned by the alias target)", name)
		}
		if c.Scheduler == nil {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu requires scheduler mode (admission control is scheduler-only)", name)
		}
		if len(model.GPUs) == 0 {
			return fmt.Errorf("model %q: expected_vram_mb_per_gpu requires 'gpus' to be set", name)
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

// EffectivePriority returns the configured Priority, defaulting to
// DefaultPriority ("normal") when unset. Pinned models are forced to
// "critical" by the admission controller regardless of this value.
func (m ModelConfig) EffectivePriority() string {
	if m.Priority == "" {
		return DefaultPriority
	}
	return m.Priority
}

// AdmissionEnabled reports whether the admission controller should track
// this model. A model with no declared VRAM footprint is invisible to
// admission — its wake is unmediated and any peer's wake won't evict it.
func (m ModelConfig) AdmissionEnabled() bool {
	return m.ExpectedVRAMMBPerGPU > 0
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
