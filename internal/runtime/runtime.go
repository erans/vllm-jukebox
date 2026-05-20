package runtime

import (
	"context"
	"fmt"

	"vllm-jukebox/internal/config"
)

// Runtime abstracts the per-backend bits of process startup and readiness checks
// so the Manager can drive either vLLM or llama.cpp behind a common interface.
type Runtime interface {
	Name() string
	Binary(cfg *config.Config) string
	BuildArgs(cfg *config.Config, model config.ModelConfig, resolvedName string, port int) ([]string, error)
	VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error
}

// For returns the Runtime for the given name. Empty string maps to vllm so
// every existing config continues to work unchanged.
func For(name string) (Runtime, error) {
	switch name {
	case "", config.RuntimeVLLM:
		return vllmRuntime{}, nil
	case config.RuntimeLlamaCpp:
		return llamaCppRuntime{}, nil
	default:
		return nil, fmt.Errorf("unknown runtime %q", name)
	}
}
