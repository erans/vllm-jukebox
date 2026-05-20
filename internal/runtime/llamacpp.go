package runtime

import (
	"context"
	"errors"

	"vllm-jukebox/internal/config"
)

type llamaCppRuntime struct{}

func (llamaCppRuntime) Name() string { return config.RuntimeLlamaCpp }

func (llamaCppRuntime) Binary(cfg *config.Config) string {
	if cfg.LlamaCpp == nil {
		return ""
	}
	return cfg.LlamaCpp.Binary
}

func (llamaCppRuntime) BuildArgs(cfg *config.Config, model config.ModelConfig, resolvedName string, port int) ([]string, error) {
	return nil, errors.New("llama_cpp runtime: BuildArgs not implemented yet")
}

func (llamaCppRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	return errors.New("llama_cpp runtime: VerifyModelLoaded not implemented yet")
}
