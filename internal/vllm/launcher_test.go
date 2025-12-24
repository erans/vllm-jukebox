package vllm

import "testing"

func TestWrapBinaryArgs_UsesUVXVLLMWhenBinaryIsUVX(t *testing.T) {
	bin, args := wrapBinaryArgs("uvx", []string{"serve", "/models/m"})
	if bin != "uvx" {
		t.Fatalf("expected bin=uvx, got %q", bin)
	}
	if len(args) < 2 || args[0] != "vllm" || args[1] != "serve" {
		t.Fatalf("expected args to start with [vllm serve], got %v", args)
	}
}

func TestWrapBinaryArgs_UsesUVXVLLMWhenBinaryIsPathToUVX(t *testing.T) {
	bin, args := wrapBinaryArgs("/usr/local/bin/uvx", []string{"serve", "/models/m"})
	if bin != "/usr/local/bin/uvx" {
		t.Fatalf("expected bin to be unchanged, got %q", bin)
	}
	if len(args) < 2 || args[0] != "vllm" || args[1] != "serve" {
		t.Fatalf("expected args to start with [vllm serve], got %v", args)
	}
}

func TestWrapBinaryArgs_PassesThroughNonUVX(t *testing.T) {
	bin, args := wrapBinaryArgs("vllm", []string{"serve", "/models/m"})
	if bin != "vllm" {
		t.Fatalf("expected bin=vllm, got %q", bin)
	}
	if len(args) < 1 || args[0] != "serve" {
		t.Fatalf("expected args to start with [serve], got %v", args)
	}
}
