package config

import (
	"bytes"
	"fmt"
	"time"

	"gopkg.in/yaml.v3"
)

type Duration struct {
	time.Duration
}

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
	Server   ServerConfig           `yaml:"server"`
	VLLM     VLLMConfig             `yaml:"vllm"`
	Behavior BehaviorConfig         `yaml:"behavior"`
	Models   map[string]ModelConfig `yaml:"models"`
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
	Alias                string            `yaml:"alias"`
	TensorParallelSize   *int              `yaml:"tensor_parallel_size"`
	PipelineParallelSize *int              `yaml:"pipeline_parallel_size"`
	MaxModelLen          *int              `yaml:"max_model_len"`
	GPUMemoryUtilization *float64          `yaml:"gpu_memory_utilization"`
	DType                string            `yaml:"dtype"`
	Quantization         string            `yaml:"quantization"`
	ExtraArgs            []string          `yaml:"extra_args"`
	Env                  map[string]string `yaml:"env"`
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

	if c.Behavior.DefaultModel != "" {
		if _, ok := c.Models[c.Behavior.DefaultModel]; !ok {
			return fmt.Errorf("default_model %q not found in models", c.Behavior.DefaultModel)
		}
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
	}

	if err := c.validateAliases(); err != nil {
		return err
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
