package vllmcli_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"vllm-jukebox/internal/vllmcli"
)

// TestVerifyForwardPass exercises the real-generation readiness probe. The
// whole point of this probe is that /health and /v1/models can lie after a
// cumem wake — only a real forward pass reveals a poisoned engine. These cases
// assert the probe PASSES on a healthy generation and FAILS on every shape a
// poisoned/wedged engine produces (5xx, empty output, connection reset).
func TestVerifyForwardPass(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr bool
	}{
		{
			name: "healthy generation passes",
			handler: func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/completions" {
					http.NotFound(w, r)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"text":" yes","finish_reason":"length"}]}`))
			},
			wantErr: false,
		},
		{
			name: "engine 500 fails (poisoned post-wake)",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error":"CUDA error: an illegal memory access was encountered"}`))
			},
			wantErr: true,
		},
		{
			name: "200 with no choices fails (engine produced nothing)",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[]}`))
			},
			wantErr: true,
		},
		{
			// MED #5: a poisoned engine can 200 with an empty completion (no
			// tokens actually decoded). That must NOT be accepted as a pass.
			name: "200 with empty text fails (no tokens decoded)",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"text":"","finish_reason":"length"}]}`))
			},
			wantErr: true,
		},
		{
			// MED #5: text present but finish_reason is not a healthy terminal
			// state — reject (covers a truncated/garbage response shape).
			name: "200 with text but bad finish_reason fails",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"text":" yes","finish_reason":"abort"}]}`))
			},
			wantErr: true,
		},
		{
			// MED #5: text present but finish_reason missing entirely — reject.
			name: "200 with text but empty finish_reason fails",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"text":" yes"}]}`))
			},
			wantErr: true,
		},
		{
			// finish_reason "stop" is also a healthy terminal state.
			name: "200 with text and finish_reason stop passes",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"text":" ok","finish_reason":"stop"}]}`))
			},
			wantErr: false,
		},
		{
			name: "malformed body fails",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`not-json`))
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(tc.handler)
			t.Cleanup(srv.Close)

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			err := vllmcli.VerifyForwardPass(ctx, srv.URL, "m")
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected success, got %v", err)
			}
		})
	}
}

// TestVerifyForwardPass_ForcesDecodeStep asserts the probe request body is
// shaped to force at least one genuine DECODE step (not just a prefill). The
// post-cumem-wake crash fires on the decode cudagraph replay path; a
// max_tokens=1 request returns after prefill and never triggers it, so a
// poisoned engine would falsely pass. The probe must request >=2 tokens AND
// pin min_tokens>=2 + ignore_eos so the engine cannot short-circuit at token 1
// via EOS and skip decode. This test captures the outbound JSON and verifies
// those fields.
func TestVerifyForwardPass_ForcesDecodeStep(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dec := json.NewDecoder(r.Body)
		_ = dec.Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"text":" ok","finish_reason":"length"}]}`))
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := vllmcli.VerifyForwardPass(ctx, srv.URL, "m"); err != nil {
		t.Fatalf("probe failed: %v", err)
	}

	maxTok, ok := gotBody["max_tokens"].(float64)
	if !ok || maxTok < 2 {
		t.Fatalf("max_tokens must be >=2 to force a decode step, got %v", gotBody["max_tokens"])
	}
	minTok, ok := gotBody["min_tokens"].(float64)
	if !ok || minTok < 2 {
		t.Fatalf("min_tokens must be >=2 so EOS cannot skip decode, got %v", gotBody["min_tokens"])
	}
	if ignoreEOS, ok := gotBody["ignore_eos"].(bool); !ok || !ignoreEOS {
		t.Fatalf("ignore_eos must be true so the engine cannot stop at token 1, got %v", gotBody["ignore_eos"])
	}
}

// TestVerifyForwardPass_TransportError ensures a wedged/crashed engine (server
// closed, connection refused) is treated as a verification failure rather than
// a pass. We point at a closed server to simulate the engine being down.
func TestVerifyForwardPass_TransportError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close() // now connections are refused

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := vllmcli.VerifyForwardPass(ctx, url, "m"); err == nil {
		t.Fatalf("expected transport error to fail verification, got nil")
	}
}
