// Package vllmcli builds vLLM CLI arguments and verifies model readiness
// against vLLM's HTTP surface. It lives outside internal/vllm so that the
// internal/runtime adapter layer can call it without producing an import
// cycle with internal/vllm's process Manager.
package vllmcli

import (
	"strconv"

	"vllm-jukebox/internal/config"
)

// BuildServeArgs returns the vLLM `serve` args using the global VLLM port.
func BuildServeArgs(cfg *config.Config, requestedModel string) ([]string, error) {
	return BuildServeArgsForPort(cfg, requestedModel, cfg.VLLM.Port)
}

// BuildServeArgsForPort returns the vLLM `serve` args bound to the given port.
func BuildServeArgsForPort(cfg *config.Config, requestedModel string, port int) ([]string, error) {
	resolvedName, model, err := cfg.ResolveModel(requestedModel)
	if err != nil {
		return nil, err
	}

	_ = resolvedName // currently unused, but kept for future model verification/logging

	args := []string{
		"serve",
		model.Path,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
	}

	if model.TensorParallelSize != nil {
		args = append(args, "--tensor-parallel-size", strconv.Itoa(*model.TensorParallelSize))
	}
	if model.PipelineParallelSize != nil {
		args = append(args, "--pipeline-parallel-size", strconv.Itoa(*model.PipelineParallelSize))
	}

	gpu := model.GPUMemoryUtilization
	if gpu == nil {
		gpu = cfg.VLLM.Defaults.GPUMemoryUtilization
	}
	if gpu != nil {
		args = append(args, "--gpu-memory-utilization", strconv.FormatFloat(*gpu, 'f', -1, 64))
	}

	maxLen := model.MaxModelLen
	if maxLen == nil {
		maxLen = cfg.VLLM.Defaults.MaxModelLen
	}
	if maxLen != nil {
		args = append(args, "--max-model-len", strconv.Itoa(*maxLen))
	}

	dtype := model.DType
	if dtype == "" {
		dtype = cfg.VLLM.Defaults.DType
	}
	if dtype != "" {
		args = append(args, "--dtype", dtype)
	}

	if model.Quantization != "" {
		args = append(args, "--quantization", model.Quantization)
	}

	args = append(args, model.ExtraArgs...)
	return args, nil
}
