package ports

import "sync"

type Pool struct {
	start int
	end   int

	mu   sync.Mutex
	used map[int]bool
}

func New(start, end int) *Pool {
	return &Pool{
		start: start,
		end:   end,
		used:  map[int]bool{},
	}
}

func (p *Pool) Acquire() (int, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for port := p.start; port <= p.end; port++ {
		if !p.used[port] {
			p.used[port] = true
			return port, true
		}
	}
	return 0, false
}

func (p *Pool) Release(port int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.used, port)
}
