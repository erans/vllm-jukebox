package config

import (
	"bytes"
	"fmt"
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

	// VerifyForwardPass, when true (the default), runs a short real generation
	// (a genuine decode step, not just a prefill) against the backend before an
	// instance is marked ready. This catches a poisoned engine that passes
	// /health and /v1/models but crashes on the first real forward pass
	// (notably after a cumem sleep/wake cycle). Set to false only to opt out of
	// the extra probe. The probe is only run for generative models — pooling /
	// embedding / reranker models don't serve /v1/completions and are skipped
	// (see ModelConfig.Task).
	VerifyForwardPass *bool `yaml:"verify_forward_pass"`

	// VerifyForwardPassTimeout bounds the forward-pass probe specifically,
	// separately from StartupTimeout. The whole VerifyReady path (health-wait +
	// model-loaded + forward-pass) shares StartupTimeout; carving a dedicated
	// per-probe budget here means a slow-but-healthy large-model wake that
	// already consumed most of StartupTimeout still gets a full window to do its
	// one decode step instead of being torn down on a near-exhausted deadline.
	VerifyForwardPassTimeout Duration `yaml:"verify_forward_pass_timeout"`
}

type SchedulerConfig struct {
	NvidiaSMIBinary   string    `yaml:"nvidia_smi_binary"`
	PortRangeStart    int       `yaml:"port_range_start"`
	PortRangeEnd      int       `yaml:"port_range_end"`
	MaxInstances      *int      `yaml:"max_instances"`
	MinInstanceUptime *Duration `yaml:"min_instance_uptime"`
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

	// Task declares what the served model does. It mirrors vLLM's --task flag.
	// Empty (the default) means a generative model that serves /v1/completions
	// and /v1/chat/completions. Non-generative tasks (embed / embedding /
	// rerank / classify / score / reward / pooling) serve /v1/embeddings or
	// /v1/rerank instead and do NOT expose /v1/completions, so the
	// generate-style forward-pass probe must be skipped for them (it would
	// 404/405 on every cold-start and wake).
	Task string `yaml:"task"`
}

// IsGenerative reports whether the model serves the OpenAI /v1/completions /
// /v1/chat/completions surface (the only surface the forward-pass probe knows
// how to exercise). Pooling models — embeddings, rerankers, classifiers —
// serve /v1/embeddings or /v1/rerank instead and must not be probed with a
// generate request. Empty Task defaults to generative for backward
// compatibility with existing configs.
func (m ModelConfig) IsGenerative() bool {
	switch strings.ToLower(strings.TrimSpace(m.Task)) {
	case "", "generate", "generation", "draft", "transcription":
		return true
	default:
		// embed, embedding, embeddings, classify, score, reward, rerank,
		// reranker, pooling, etc. — non-generative.
		return false
	}
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
	if c.VLLM.VerifyForwardPass == nil {
		// Default on: a short real generate (with a genuine decode step)
		// catches a poisoned post-wake engine that /health and /v1/models miss.
		v := true
		c.VLLM.VerifyForwardPass = &v
	}
	if c.VLLM.VerifyForwardPassTimeout.Duration == 0 {
		// Dedicated per-probe budget, independent of StartupTimeout. 30s is
		// generous for a 2-token generate even on a large model that just woke.
		c.VLLM.VerifyForwardPassTimeout = Duration{Duration: 30 * time.Second}
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
		if model.Alias == "" && model.Path == "" {
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
	}

	return nil
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
