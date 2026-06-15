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
	// VerifyForwardPass issues a minimal real generation to confirm the
	// inference engine actually serves a request — not just that the HTTP
	// server and model registry are up. This closes the gap where /health
	// and /v1/models report ready but the engine is poisoned (e.g. after a
	// cumem sleep/wake cycle) and crashes on the first real forward pass.
	VerifyForwardPass(ctx context.Context, baseURL, expectedID string) error
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
