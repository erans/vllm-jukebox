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
	// Subcommand dispatch. Keep this above flag.Parse so the subcommand
	// gets its own arg slice — the default daemon mode still parses
	// flags exactly as before. Currently `health` is the only
	// subcommand; it short-circuits and exits with the probe's result.
	//
	// IMPORTANT: subcommand detection ONLY fires when the first
	// non-program arg does not start with "-", so existing invocations
	// like `jukebox -config foo.yaml` keep working unchanged.
	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		switch os.Args[1] {
		case "health":
			os.Exit(runHealthCheck(os.Args[2:]))
		}
	}

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

	// Install the validated config in the package-level atomic pointer
	// so subsystems that opt into hot-reload (via config.Current()) start
	// with a non-nil view. Subsystems instantiated below capture the
	// initial *Config by value for their construction params; for those
	// reads, hot-reload only takes effect on next process restart.
	// Future opt-in callers (future feature flags, soft thresholds,
	// log-level toggles, etc.) can read the live view via config.Current()
	// and pick up changes within ~500ms of an active.yaml write.
	config.SetCurrent(cfg)

	metrics.SetState("idle")
	metrics.ConsecutiveFailures.Set(0)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the active.yaml hot-reload watcher. Failures here log loud
	// but are NOT fatal — the existing parse-crash safety net (operator
	// fixes the file + restarts the container) still works.
	//
	// onReload is set AFTER subsystem construction below, so this
	// closure dispatches into a (later-populated) reloadSubscribers
	// slice. Until subsystems register, the watcher just records the
	// metric — the atomic Current() pointer swap already happened
	// inside config.Watch's doReload.
	var (
		reloadMu          sync.Mutex
		reloadSubscribers []func(*config.Config)
	)
	subscribeReload := func(fn func(*config.Config)) {
		reloadMu.Lock()
		defer reloadMu.Unlock()
		reloadSubscribers = append(reloadSubscribers, fn)
	}
	dispatchReload := func() {
		reloadMu.Lock()
		fns := append([]func(*config.Config){}, reloadSubscribers...)
		reloadMu.Unlock()
		live := config.Current()
		if live == nil {
			return
		}
		for _, fn := range fns {
			fn(live)
		}
	}
	if err := config.Watch(ctx, configPath, func(result config.ReloadResult, _ error) {
		metrics.ConfigReloadsTotal.WithLabelValues(string(result)).Inc()
		if result == config.ReloadSuccess {
			dispatchReload()
		}
	}); err != nil {
		slog.Warn("active.yaml hot-reload watcher disabled", "err", err, "path", configPath)
	} else {
		slog.Info("active.yaml hot-reload watcher started", "path", configPath)
	}

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
		coord := jukebox.NewCoordinatorWithPower(cfg, mgr, &tr, time.Now, powerMgr)
		go coord.Run(ctx)
		go coord.IdleMonitor(ctx)
		legacyRouter := jukebox.NewLegacyRouter(cfg, coord, &tr)
		legacyRouter.StartLiveness(ctx)
		router = legacyRouter

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

		// Fail fast if configured GPU IDs do not exist. SKIPPED entirely
		// when every non-alias model is lifecycle: external — those don't
		// allocate local GPUs and nvidia-smi may not even be installed in
		// the jukebox container.
		anyManaged := false
		for _, model := range cfg.Models {
			if model.Alias != "" {
				continue
			}
			if model.EffectiveLifecycle() != config.LifecycleExternal {
				anyManaged = true
				break
			}
		}
		if anyManaged {
			checkTimeout := 5 * time.Second
			checkCtx, checkCancel := context.WithTimeout(context.Background(), checkTimeout)
			defer checkCancel()

			gpus, err := inv.List(checkCtx)
			switch {
			case err != nil && gpu.IsBinaryNotFound(err):
				// nvidia-smi is not installed in this environment. Operators
				// run jukebox in CPU-only / external-routing setups where
				// nvidia-smi may legitimately be absent. Hard-failing here
				// crash-loops the container. Degrade: skip the GPU-id
				// existence check and let the scheduler boot. Admission
				// control (below) will independently degrade for the same
				// reason and log its own warning.
				slog.Warn("scheduler_mode_no_nvidia_smi",
					"err", err,
					"msg", "nvidia-smi not found; skipping GPU id validation and degrading to legacy routing (no admission VRAM tracking). Operators who want admission MUST make nvidia-smi available to the jukebox container.",
				)
			case err != nil:
				slog.Error("failed to read GPU inventory (scheduler mode)", "err", err)
				os.Exit(1)
			default:
				exists := map[int]bool{}
				for _, g := range gpus {
					exists[g.Index] = true
				}
				for name, model := range cfg.Models {
					if model.Alias != "" {
						continue
					}
					// External-lifecycle instances may declare GPUs that live on
					// a different host (jukebox doesn't own the process). Skip
					// the local nvidia-smi check for them — the gpus field on an
					// external model is informational.
					if model.EffectiveLifecycle() == config.LifecycleExternal {
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
		} else {
			slog.Info("all scheduler models are lifecycle: external — skipping local nvidia-smi GPU check")
		}

		// Build admission controller if any model has it enabled. Requires
		// a live nvidia-smi to read per-GPU TotalMB. If the probe fails
		// (e.g. jukebox container has no nvidia-smi, host driver
		// unreachable), log a loud warning and DEGRADE — the scheduler
		// runs without admission. Operators who want admission MUST make
		// nvidia-smi available to the jukebox container.
		if cfg.AdmissionEnabled() {
			probeTimeout := 5 * time.Second
			probeCtx, probeCancel := context.WithTimeout(context.Background(), probeTimeout)
			gpus, err := inv.List(probeCtx)
			probeCancel()
			if err != nil {
				slog.Warn("admission_control_disabled_no_nvidia_smi",
					"err", err,
					"hint", "expected_vram_mb_per_gpu was set on at least one model but nvidia-smi probe failed — admission control will NOT be active",
				)
			} else {
				totalsByGPU := map[int]int{}
				for _, g := range gpus {
					totalsByGPU[g.Index] = g.TotalMB
				}
				evictor := &jukebox.SchedulerEvictor{S: sched}
				adm := jukebox.NewAdmissionController(cfg, totalsByGPU, evictor)
				sched.SetAdmission(adm)
				slog.Info("admission_control_attached",
					"tracked_models", adm.TrackedModels(),
					"gpus", len(totalsByGPU),
				)
			}
		}

		// Bootstrap lifecycle: external instances + start the auto-suspend
		// idle monitor.
		if err := sched.RegisterExternalInstances(ctx); err != nil {
			slog.Error("failed to register external instances", "err", err)
			os.Exit(1)
		}
		go sched.IdleMonitor(ctx)
		// Issue #5: detect external `docker start <peer>` that bypasses
		// jukebox's coldLoadStoppedMember → evictCrossGroupGPUContenders
		// pipeline. Without this monitor, a bare `docker start vllm-vision`
		// while vllm-main holds an overlapping GPU OOMs the vision peer
		// at vLLM worker init. The monitor observes the misordered start,
		// SIGKILLs the half-initialized container, and re-routes through
		// KickColdLoad which runs cross-group eviction first.
		go sched.ExternalStartMonitor(ctx)
		// Issue #8: periodic reconciliation between admission's recorded
		// state and the actual docker container state. Sibling of
		// ExternalStartMonitor — that one handles the SPECIFIC
		// {admission Stopped/Sleeping + docker running} fault by
		// SIGKILLing. This one handles the BROADER
		// {admission Ready/Sleeping + docker NOT running} drift by
		// reconciling admission down to Stopped (and re-kicking a
		// cold-load for pinned peers that should always be live).
		go sched.StateReconciler(ctx)

		// Subscribe the scheduler to active.yaml hot-reloads so its
		// admission controller picks up per-model field edits without
		// requiring a process restart.
		subscribeReload(sched.OnConfigReloaded)

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
