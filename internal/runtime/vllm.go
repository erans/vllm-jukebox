package runtime

import (
	"context"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/vllmcli"
)

type vllmRuntime struct{}

func (vllmRuntime) Name() string { return config.RuntimeVLLM }

func (vllmRuntime) Binary(cfg *config.Config) string { return cfg.VLLM.Binary }

func (vllmRuntime) BuildArgs(cfg *config.Config, _ config.ModelConfig, resolvedName string, port int) ([]string, error) {
	return vllmcli.BuildServeArgsForPort(cfg, resolvedName, port)
}

func (vllmRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	return vllmcli.VerifyModelLoaded(ctx, baseURL, expectedID, expectedPath)
}

func (vllmRuntime) VerifyForwardPass(ctx context.Context, baseURL, expectedID string) error {
	return vllmcli.VerifyForwardPass(ctx, baseURL, expectedID)
}
