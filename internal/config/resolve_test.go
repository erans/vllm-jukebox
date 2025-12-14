package config_test

import (
	"testing"

	"vllm-jukebox/internal/config"
)

func TestResolveModel_AliasResolvesToTarget(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  base:
    path: "/models/base"
  alias:
    alias: base
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	resolvedName, model, err := cfg.ResolveModel("alias")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if resolvedName != "base" {
		t.Fatalf("expected resolvedName=base, got %q", resolvedName)
	}
	if model.Path != "/models/base" {
		t.Fatalf("expected path /models/base, got %q", model.Path)
	}
}

func TestResolveModel_NonAliasResolvesToSelf(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  base:
    path: "/models/base"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	resolvedName, model, err := cfg.ResolveModel("base")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if resolvedName != "base" {
		t.Fatalf("expected resolvedName=base, got %q", resolvedName)
	}
	if model.Path != "/models/base" {
		t.Fatalf("expected path /models/base, got %q", model.Path)
	}
}

func TestResolveModel_UnknownRejected(t *testing.T) {
	cfg, err := config.Load([]byte(`
vllm:
  port: 8000
models:
  base:
    path: "/models/base"
`))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if _, _, err := cfg.ResolveModel("nope"); err == nil {
		t.Fatalf("expected error")
	}
}

