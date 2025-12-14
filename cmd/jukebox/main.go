package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/vllm"
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

	cfg, err := config.Load(data)
	if err != nil {
		slog.Error("failed to parse config", "path", configPath, "err", err)
		os.Exit(1)
	}

	var tr inflight.Tracker
	mgr := vllm.NewManager(cfg)
	coord := jukebox.NewCoordinator(cfg, mgr, &tr, time.Now)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go coord.Run(ctx)

	app := fiber.New(fiber.Config{
		ReadTimeout:  cfg.Server.ReadTimeout.Duration,
		WriteTimeout: cfg.Server.WriteTimeout.Duration,
	})
	app.Mount("/", httpserver.NewApp(httpserver.Options{
		Config:      cfg,
		Coordinator: coord,
		InFlight:    &tr,
	}))

	addr := net.JoinHostPort(cfg.Server.Host, fmt.Sprintf("%d", cfg.Server.Port))
	slog.Info("jukebox listening", "addr", addr)

	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
		<-ch
		slog.Info("shutdown signal received")
		cancel()
		_ = app.Shutdown()
	}()

	if err := app.Listen(addr); err != nil && !isExpectedShutdownErr(err) {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}

func isExpectedShutdownErr(err error) bool {
	if err == nil {
		return true
	}
	// Fiber/fasthttp doesn't consistently surface a typed error for shutdown across versions.
	if errors.Is(err, net.ErrClosed) {
		return true
	}
	if strings.Contains(strings.ToLower(err.Error()), "server closed") {
		return true
	}
	return false
}
