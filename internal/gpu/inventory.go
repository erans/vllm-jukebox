package gpu

import (
	"context"
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

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
		return nil, err
	}
	return ParseInventoryCSV(string(b))
}
