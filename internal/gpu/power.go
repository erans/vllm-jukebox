package gpu

import (
	"context"
	"fmt"
	"os/exec"
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
