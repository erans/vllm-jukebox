package gpu

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"sort"
	"strconv"
	"sync"
)

type PowerManager struct {
	binary   string
	defaults map[int]int // GPU index -> power limit in watts
	required bool        // fail on error vs warn

	mu      sync.Mutex
	current map[int]int // GPU index -> current power limit (for revert tracking)
}

func NewPowerManager(binary string, defaults map[int]int, required bool) *PowerManager {
	if binary == "" {
		binary = "nvidia-smi"
	}
	copied := make(map[int]int, len(defaults))
	for k, v := range defaults {
		copied[k] = v
	}
	return &PowerManager{
		binary:   binary,
		defaults: copied,
		required: required,
		current:  make(map[int]int),
	}
}

func (p *PowerManager) SetLimit(ctx context.Context, gpuIndex, watts int) error {
	cmd := exec.CommandContext(ctx, p.binary,
		"-i", strconv.Itoa(gpuIndex),
		"-pl", strconv.Itoa(watts),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("nvidia-smi -i %d -pl %d failed: %w (output: %s)", gpuIndex, watts, err, string(output))
	}

	p.mu.Lock()
	p.current[gpuIndex] = watts
	p.mu.Unlock()

	return nil
}

func (p *PowerManager) RevertToDefault(ctx context.Context, gpuIndex int) error {
	defaultWatts, ok := p.defaults[gpuIndex]
	if !ok {
		// No default configured for this GPU, nothing to revert to
		return nil
	}

	p.mu.Lock()
	currentWatts, hasOverride := p.current[gpuIndex]
	p.mu.Unlock()

	// Only revert if we actually applied an override
	if !hasOverride || currentWatts == defaultWatts {
		return nil
	}

	return p.SetLimit(ctx, gpuIndex, defaultWatts)
}

func (p *PowerManager) ApplyStartupLimits(ctx context.Context) error {
	if len(p.defaults) == 0 {
		return nil
	}

	// Sort GPU indices for deterministic order
	gpuIndices := make([]int, 0, len(p.defaults))
	for idx := range p.defaults {
		gpuIndices = append(gpuIndices, idx)
	}
	sort.Ints(gpuIndices)

	var firstErr error
	for _, gpuIndex := range gpuIndices {
		watts := p.defaults[gpuIndex]
		if err := p.SetLimit(ctx, gpuIndex, watts); err != nil {
			slog.Warn("failed to set GPU power limit",
				"gpu", gpuIndex,
				"watts", watts,
				"err", err,
			)
			if p.required && firstErr == nil {
				firstErr = err
			}
		} else {
			slog.Info("set GPU power limit",
				"gpu", gpuIndex,
				"watts", watts,
			)
		}
	}

	return firstErr
}
