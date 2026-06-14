package gpu

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

// ErrBinaryNotFound is returned by NvidiaSMIInventory.List when the
// nvidia-smi binary itself is missing from the environment (e.g. the
// jukebox container has no GPU userspace installed). Callers should treat
// this as "GPUs are not introspectable here" and degrade gracefully —
// hard-failing startup makes the container crash-loop on hosts that
// intentionally run jukebox without nvidia-smi (CPU-only routing,
// scheduler-mode with all lifecycle:external members, etc.).
var ErrBinaryNotFound = errors.New("nvidia-smi binary not found")

// IsBinaryNotFound reports whether err indicates the nvidia-smi binary
// (or whatever was configured as the inventory binary) does not exist on
// PATH or at the configured path. Wraps both exec.ErrNotFound (PATH
// lookup) and fs.ErrNotExist (absolute path that doesn't exist).
func IsBinaryNotFound(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrBinaryNotFound) {
		return true
	}
	if errors.Is(err, exec.ErrNotFound) {
		return true
	}
	if errors.Is(err, fs.ErrNotExist) {
		return true
	}
	return false
}

type GPU struct {
	Index   int
	TotalMB int
	FreeMB  int
}

func ParseInventoryCSV(out string) ([]GPU, error) {
	out = strings.TrimSpace(out)
	if out == "" {
		return nil, nil
	}

	lines := strings.Split(out, "\n")
	gpus := make([]GPU, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			return nil, fmt.Errorf("invalid inventory row %q", line)
		}
		idx, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return nil, fmt.Errorf("invalid gpu index in %q: %w", line, err)
		}
		total, err := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			return nil, fmt.Errorf("invalid total memory in %q: %w", line, err)
		}
		free, err := strconv.Atoi(strings.TrimSpace(parts[2]))
		if err != nil {
			return nil, fmt.Errorf("invalid free memory in %q: %w", line, err)
		}
		gpus = append(gpus, GPU{Index: idx, TotalMB: total, FreeMB: free})
	}

	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	return gpus, nil
}

type Inventory interface {
	List(ctx context.Context) ([]GPU, error)
}

type NvidiaSMIInventory struct {
	Binary string
}

func (n NvidiaSMIInventory) List(ctx context.Context) ([]GPU, error) {
	bin := strings.TrimSpace(n.Binary)
	if bin == "" {
		bin = "nvidia-smi"
	}

	cmd := exec.CommandContext(ctx, bin,
		"--query-gpu=index,memory.total,memory.free",
		"--format=csv,noheader,nounits",
	)
	b, err := cmd.Output()
	if err != nil {
		// Normalize "binary missing" into a sentinel callers can match
		// with errors.Is(err, gpu.ErrBinaryNotFound) — keep the original
		// error in the chain for diagnostics.
		if IsBinaryNotFound(err) {
			return nil, fmt.Errorf("%w: %v", ErrBinaryNotFound, err)
		}
		return nil, err
	}
	return ParseInventoryCSV(string(b))
}
