package runtime

import (
	"context"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/vllm"
)

type vllmRuntime struct{}

func (vllmRuntime) Name() string { return config.RuntimeVLLM }

func (vllmRuntime) Binary(cfg *config.Config) string { return cfg.VLLM.Binary }

func (vllmRuntime) BuildArgs(cfg *config.Config, _ config.ModelConfig, resolvedName string, port int) ([]string, error) {
	return vllm.BuildServeArgsForPort(cfg, resolvedName, port)
}

func (vllmRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	return vllm.VerifyModelLoaded(ctx, baseURL, expectedID, expectedPath)
}
