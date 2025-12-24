package gpu_test

import (
	"testing"

	"vllm-jukebox/internal/gpu"
)

func TestParseInventoryCSV_ParsesAndSorts(t *testing.T) {
	out, err := gpu.ParseInventoryCSV("1, 100, 90\n0,100,80\n")
	if err != nil {
		t.Fatalf("ParseInventoryCSV: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 gpus, got %d", len(out))
	}
	if out[0].Index != 0 || out[0].TotalMB != 100 || out[0].FreeMB != 80 {
		t.Fatalf("unexpected gpu0: %+v", out[0])
	}
	if out[1].Index != 1 || out[1].TotalMB != 100 || out[1].FreeMB != 90 {
		t.Fatalf("unexpected gpu1: %+v", out[1])
	}
}

func TestParseInventoryCSV_AllowsEmpty(t *testing.T) {
	out, err := gpu.ParseInventoryCSV("")
	if err != nil {
		t.Fatalf("ParseInventoryCSV: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("expected empty, got %v", out)
	}
}

func TestParseInventoryCSV_RejectsMalformed(t *testing.T) {
	cases := []string{
		"0,1\n",
		"nope, 1, 2\n",
		"0, nope, 2\n",
		"0, 1, nope\n",
	}
	for _, tc := range cases {
		_, err := gpu.ParseInventoryCSV(tc)
		if err == nil {
			t.Fatalf("expected error for %q", tc)
		}
	}
}
