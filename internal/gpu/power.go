package gpu

import (
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
