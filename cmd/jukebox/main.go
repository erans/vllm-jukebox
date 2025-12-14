package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
)

func main() {
	var configPath string
	flag.StringVar(&configPath, "config", "", "Path to YAML config file")
	flag.Parse()

	if configPath == "" {
		_, _ = fmt.Fprintln(os.Stderr, "error: -config is required")
		os.Exit(2)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{}))
	slog.SetDefault(logger)

	data, err := os.ReadFile(configPath)
	if err != nil {
		slog.Error("failed to read config", "path", configPath, "err", err)
		os.Exit(1)
	}
	if len(data) == 0 {
		slog.Error("config file is empty", "path", configPath)
		os.Exit(1)
	}

	slog.Info("jukebox bootstrap ok", "config_path", configPath, "config_bytes", len(data))
}
