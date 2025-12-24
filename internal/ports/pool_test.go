package ports_test

import (
	"testing"

	"vllm-jukebox/internal/ports"
)

func TestPool_AcquireExhaustsRange(t *testing.T) {
	p := ports.New(8100, 8101)

	a, ok := p.Acquire()
	if !ok || a != 8100 {
		t.Fatalf("expected 8100, got %d ok=%v", a, ok)
	}
	b, ok := p.Acquire()
	if !ok || b != 8101 {
		t.Fatalf("expected 8101, got %d ok=%v", b, ok)
	}
	c, ok := p.Acquire()
	if ok || c != 0 {
		t.Fatalf("expected exhausted, got %d ok=%v", c, ok)
	}
}

func TestPool_ReleaseAllowsReuse(t *testing.T) {
	p := ports.New(8100, 8101)
	a, _ := p.Acquire()
	b, _ := p.Acquire()
	p.Release(a)

	c, ok := p.Acquire()
	if !ok || c != a {
		t.Fatalf("expected reuse %d, got %d ok=%v (b=%d)", a, c, ok, b)
	}
}
