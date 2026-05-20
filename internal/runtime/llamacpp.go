package runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
	if model.Path == "" {
		return nil, fmt.Errorf("model %q: path is required for llama_cpp runtime", resolvedName)
	}
	warnIgnoredVLLMFields(resolvedName, model)

	args := []string{
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
	}

	if looksLikeLocalGGUF(model.Path) {
		args = append(args, "-m", model.Path)
	} else {
		args = append(args, "--hf-repo", model.Path)
	}

	if model.MaxModelLen != nil {
		args = append(args, "--ctx-size", strconv.Itoa(*model.MaxModelLen))
	}

	if cfg.LlamaCpp != nil {
		args = append(args, cfg.LlamaCpp.DefaultArgs...)
	}
	args = append(args, model.ExtraArgs...)

	return args, nil
}

func (llamaCppRuntime) VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	url := strings.TrimRight(baseURL, "/") + "/v1/models"
	client := &http.Client{}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("GET /v1/models returned %s", resp.Status)
	}

	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return err
	}

	base := ""
	if expectedPath != "" {
		base = filepath.Base(expectedPath)
	}
	for _, m := range decoded.Data {
		if m.ID == expectedID {
			return nil
		}
		if expectedPath != "" && m.ID == expectedPath {
			return nil
		}
		if base != "" && m.ID == base {
			return nil
		}
	}
	return fmt.Errorf("expected model %q (or path %q) not found in /v1/models", expectedID, expectedPath)
}

// looksLikeLocalGGUF matches the heuristic from scripts/llama-server-wrapper.sh:
// a value is treated as a local file if it is an absolute or explicitly-relative
// path, carries a .gguf suffix, or exists on disk. Anything else is assumed to
// be a Hugging Face repo coordinate like "org/repo[:quant]".
func looksLikeLocalGGUF(p string) bool {
	if strings.HasPrefix(p, "/") || strings.HasPrefix(p, "./") || strings.HasPrefix(p, "../") {
		return true
	}
	if strings.HasSuffix(strings.ToLower(p), ".gguf") {
		return true
	}
	if _, err := os.Stat(p); err == nil {
		return true
	}
	return false
}

func warnIgnoredVLLMFields(name string, m config.ModelConfig) {
	ignored := []string{}
	if m.TensorParallelSize != nil {
		ignored = append(ignored, "tensor_parallel_size")
	}
	if m.PipelineParallelSize != nil {
		ignored = append(ignored, "pipeline_parallel_size")
	}
	if m.GPUMemoryUtilization != nil {
		ignored = append(ignored, "gpu_memory_utilization")
	}
	if m.DType != "" {
		ignored = append(ignored, "dtype")
	}
	if m.Quantization != "" {
		ignored = append(ignored, "quantization")
	}
	if len(ignored) > 0 {
		slog.Warn("llama_cpp runtime: ignoring vLLM-only model fields",
			"model", name,
			"fields", ignored,
			"hint", "translate equivalents into extra_args (e.g. --n-gpu-layers, --split-mode)")
	}
}
