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
	"sync"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/config"
	"vllm-jukebox/internal/gpu"
	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/inflight"
	"vllm-jukebox/internal/jukebox"
	"vllm-jukebox/internal/metrics"
	"vllm-jukebox/internal/ports"
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

	metrics.SetState("idle")
	metrics.ConsecutiveFailures.Set(0)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create PowerManager if power limits are configured
	var powerMgr *gpu.PowerManager
	if cfg.GPUPowerLimits != nil || cfg.DefaultPowerLimit != nil {
		defaults := cfg.GPUPowerLimits
		if defaults == nil && cfg.DefaultPowerLimit != nil {
			// Build defaults from inventory if using default_power_limit
			defaults = make(map[int]int)

			nvidiaBinary := "nvidia-smi"
			if cfg.Scheduler != nil && cfg.Scheduler.NvidiaSMIBinary != "" {
				nvidiaBinary = cfg.Scheduler.NvidiaSMIBinary
			}
			inv := gpu.NvidiaSMIInventory{Binary: nvidiaBinary}
			gpus, err := inv.List(context.Background())
			if err != nil {
				slog.Warn("failed to list GPUs for default power limit", "err", err)
			} else {
				for _, g := range gpus {
					defaults[g.Index] = *cfg.DefaultPowerLimit
				}
			}
		}

		nvidiaBinary := "nvidia-smi"
		if cfg.Scheduler != nil && cfg.Scheduler.NvidiaSMIBinary != "" {
			nvidiaBinary = cfg.Scheduler.NvidiaSMIBinary
		}
		powerMgr = gpu.NewPowerManager(nvidiaBinary, defaults, cfg.PowerLimitRequired)

		if err := powerMgr.ApplyStartupLimits(ctx); err != nil {
			slog.Error("failed to apply startup power limits", "err", err)
			os.Exit(1)
		}
	}

	var (
		router jukebox.Router
		stop   func()
	)

	if cfg.Scheduler == nil {
		var tr inflight.Tracker
		tr.OnChange = func(count int64) {
			metrics.InFlightRequests.Set(float64(count))
		}

		mgr := vllm.NewManager(cfg)
		coord := jukebox.NewCoordinator(cfg, mgr, &tr, time.Now)
		go coord.Run(ctx)
		router = jukebox.NewLegacyRouter(cfg, coord, &tr)

		stopOnce := sync.Once{}
		stop = func() {
			stopOnce.Do(func() {
				timeout := cfg.VLLM.ShutdownTimeout.Duration
				if timeout <= 0 {
					timeout = 30 * time.Second
				}
				stopCtx, stopCancel := context.WithTimeout(context.Background(), timeout)
				defer stopCancel()
				if err := mgr.Stop(stopCtx); err != nil {
					slog.Error("failed to stop vLLM", "err", err)
				}
			})
		}

		if cfg.Behavior.DefaultModel != "" {
			go func() {
				preloadTimeout := cfg.VLLM.StartupTimeout.Duration + cfg.VLLM.SwapWaitTimeout.Duration
				if preloadTimeout <= 0 {
					preloadTimeout = 5 * time.Minute
				}
				preCtx, preCancel := context.WithTimeout(context.Background(), preloadTimeout)
				defer preCancel()

				slog.Info("preloading default model", "model", cfg.Behavior.DefaultModel)
				if err := coord.EnsureModel(preCtx, cfg.Behavior.DefaultModel, "startup"); err != nil {
					slog.Error("default model preload failed", "model", cfg.Behavior.DefaultModel, "err", err)
				} else {
					slog.Info("default model preload complete", "model", cfg.Behavior.DefaultModel)
				}
			}()
		}
	} else {
		inv := gpu.NvidiaSMIInventory{Binary: cfg.Scheduler.NvidiaSMIBinary}
		pool := ports.New(cfg.Scheduler.PortRangeStart, cfg.Scheduler.PortRangeEnd)
		sched := jukebox.NewSchedulerWithFactory(cfg, inv, pool, time.Now, nil, powerMgr)
		router = sched

		// Fail fast if configured GPU IDs do not exist.
		{
			checkTimeout := 5 * time.Second
			checkCtx, checkCancel := context.WithTimeout(context.Background(), checkTimeout)
			defer checkCancel()

			gpus, err := inv.List(checkCtx)
			if err != nil {
				slog.Error("failed to read GPU inventory (scheduler mode)", "err", err)
				os.Exit(1)
			}
			exists := map[int]bool{}
			for _, g := range gpus {
				exists[g.Index] = true
			}
			for name, model := range cfg.Models {
				if model.Alias != "" {
					continue
				}
				for _, id := range model.GPUs {
					if !exists[id] {
						slog.Error("configured GPU id not found (scheduler mode)", "model", name, "gpu", id)
						os.Exit(1)
					}
				}
			}
		}

		stopOnce := sync.Once{}
		stop = func() {
			stopOnce.Do(func() {
				timeout := cfg.VLLM.ShutdownTimeout.Duration
				if timeout <= 0 {
					timeout = 30 * time.Second
				}
				stopCtx, stopCancel := context.WithTimeout(context.Background(), timeout)
				defer stopCancel()
				_ = sched.StopAll(stopCtx)
			})
		}

		if cfg.Behavior.DefaultModel != "" {
			go func() {
				preloadTimeout := cfg.VLLM.StartupTimeout.Duration + cfg.VLLM.SwapWaitTimeout.Duration
				if preloadTimeout <= 0 {
					preloadTimeout = 5 * time.Minute
				}
				preCtx, preCancel := context.WithTimeout(context.Background(), preloadTimeout)
				defer preCancel()

				slog.Info("preloading default model", "model", cfg.Behavior.DefaultModel)
				route, err := sched.AcquireRoute(preCtx, cfg.Behavior.DefaultModel, "startup")
				if err != nil {
					slog.Error("default model preload failed", "model", cfg.Behavior.DefaultModel, "err", err)
					return
				}
				if route.Done != nil {
					route.Done()
				}
				slog.Info("default model preload complete", "model", cfg.Behavior.DefaultModel)
			}()
		}
	}

	app := fiber.New(fiber.Config{
		ReadTimeout:  cfg.Server.ReadTimeout.Duration,
		WriteTimeout: cfg.Server.WriteTimeout.Duration,
	})
	app.Mount("/", httpserver.NewApp(httpserver.Options{
		Config: cfg,
		Router: router,
	}))

	if stop != nil {
		defer stop()
	}

	addr := net.JoinHostPort(cfg.Server.Host, fmt.Sprintf("%d", cfg.Server.Port))
	slog.Info("jukebox listening", "addr", addr)

	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
		<-ch
		slog.Info("shutdown signal received")
		cancel()
		_ = app.Shutdown()
		if stop != nil {
			stop()
		}
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
