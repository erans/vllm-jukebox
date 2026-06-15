package proxy

import (
	"net/http"
	"testing"
)

// TestSharedTransport_RaisedPerHostIdleCap asserts the process-wide
// upstream transport keeps a per-host idle pool large enough for the real
// fan-in workload (tens of concurrent requests at one resident model). The
// stdlib default (http.DefaultTransport) caps this at 2, which forces
// connection churn under burst load to a single host; the gateway raises
// it. This is a unit guard against a regression that silently reverts to
// the default-2 behavior (e.g. someone reinstating `&http.Client{}`).
func TestSharedTransport_RaisedPerHostIdleCap(t *testing.T) {
	tr, ok := sharedTransport.(*http.Transport)
	if !ok {
		t.Fatalf("sharedTransport is not *http.Transport: %T", sharedTransport)
	}
	if tr.MaxIdleConnsPerHost <= http.DefaultMaxIdleConnsPerHost {
		t.Fatalf("MaxIdleConnsPerHost=%d must exceed the stdlib default %d "+
			"to avoid connection churn under concurrent fan-in to one host",
			tr.MaxIdleConnsPerHost, http.DefaultMaxIdleConnsPerHost)
	}
}
