package vllm

import (
	"fmt"
	"strconv"
	"strings"

	"vllm-jukebox/internal/config"
)

func BuildServeArgs(cfg *config.Config, requestedModel string) ([]string, error) {
	return BuildServeArgsForPort(cfg, requestedModel, cfg.VLLM.Port)
}

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

func BuildEnv(base []string, defaultEnv map[string]string, modelEnv map[string]string) []string {
	envMap := map[string]string{}
	for _, kv := range base {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		envMap[k] = v
	}

	for k, v := range defaultEnv {
		envMap[k] = v
	}
	for k, v := range modelEnv {
		envMap[k] = v
	}

	out := make([]string, 0, len(envMap))
	for k, v := range envMap {
		out = append(out, fmt.Sprintf("%s=%s", k, v))
	}
	return out
}
