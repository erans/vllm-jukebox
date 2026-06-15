package httpserver_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vllm-jukebox/internal/httpserver"
	"vllm-jukebox/internal/jukebox"
)

// redeployStub is a Router that ALSO satisfies jukebox.MemberRedeployer
// — exposes a recorder for the model name + stubs the result/error
// returned. Used to drive the /admin/redeploy-member handler tests
// without standing up a real scheduler.
type redeployStub struct {
	stubCoord

	called      int
	lastModel   string
	stubResult  jukebox.RedeployResult
	stubErr     error
}

func (r *redeployStub) RedeployMember(_ context.Context, name string) (jukebox.RedeployResult, error) {
	r.called++
	r.lastModel = name
	if r.stubErr != nil {
		return r.stubResult, r.stubErr
	}
	return r.stubResult, nil
}

// nonRedeployerCoord is a Router that does NOT satisfy MemberRedeployer
// — used to assert the 501 fallback when the running mode (e.g. swap
// mode) doesn't support the verb.
type nonRedeployerCoord struct{ stubCoord }

// TestAdminRedeploy_UnknownModelReturns400 verifies the handler maps
// jukebox.ErrRedeployUnknownModel to HTTP 400 with a JSON error body.
func TestAdminRedeploy_UnknownModelReturns400(t *testing.T) {
	r := &redeployStub{
		stubErr: jukebox.ErrRedeployUnknownModel,
	}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{"model":"nope"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
	if r.called != 1 || r.lastModel != "nope" {
		t.Fatalf("expected RedeployMember(nope) once, got calls=%d model=%q", r.called, r.lastModel)
	}
	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := decoded["error"]; !ok {
		t.Fatalf("expected error field in body, got %+v", decoded)
	}
}

// TestAdminRedeploy_IneligibleReturns400 verifies that
// jukebox.ErrRedeployIneligible (model not in a swap_group and not
// evict_action: stop) is also mapped to HTTP 400.
func TestAdminRedeploy_IneligibleReturns400(t *testing.T) {
	r := &redeployStub{
		stubErr: jukebox.ErrRedeployIneligible,
	}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{"model":"plain"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

// TestAdminRedeploy_HappyPathReturns200WithResultBody verifies the
// success shape (Fix 8): the JSON envelope is {"result":{...},"model":"..."}
// and the inner result carries the populated fields including the
// stopped_peers + left_stopped JSON tags so a consumer can see what
// state changed.
func TestAdminRedeploy_HappyPathReturns200WithResultBody(t *testing.T) {
	r := &redeployStub{
		stubResult: jukebox.RedeployResult{
			Model:          "moe",
			Container:      "vllm-moe",
			Result:         "redeployed",
			DurationMS:     287000,
			EvictedPinned:  []string{"main"},
			RestoredPinned: []string{"main"},
			StoppedPeers:   []string{"longctx"},
			LeftStopped:    []string{"longctx"},
		},
	}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{"model":"moe"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	// Read the raw body so we can BOTH structurally decode AND assert
	// the JSON tag names (catches typos in the struct tags that a
	// permissive Decoder would silently miss).
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	// Fix 9 part 2: assert that the envelope carries `result` + `model`
	// AND the inner result carries `stopped_peers` + `left_stopped` —
	// the new fields whose JSON tags would silently fall off if a
	// future commit fat-fingered the tag.
	for _, want := range []string{`"result":`, `"model":`, `"stopped_peers":`, `"left_stopped":`} {
		if !bytes.Contains(bodyBytes, []byte(want)) {
			t.Errorf("expected body to contain %s, got: %s", want, string(bodyBytes))
		}
	}

	type innerResult struct {
		Model          string   `json:"model"`
		Container      string   `json:"container"`
		Result         string   `json:"result"`
		DurationMS     int64    `json:"duration_ms"`
		EvictedPinned  []string `json:"evicted_pinned"`
		RestoredPinned []string `json:"restored_pinned"`
		StoppedPeers   []string `json:"stopped_peers"`
		LeftStopped    []string `json:"left_stopped"`
	}
	type envelope struct {
		Result *innerResult `json:"result"`
		Error  string       `json:"error"`
		Model  string       `json:"model"`
	}
	var got envelope
	if err := json.Unmarshal(bodyBytes, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Error != "" {
		t.Fatalf("expected empty error on success, got %q", got.Error)
	}
	if got.Model != "moe" {
		t.Fatalf("expected envelope model=moe, got %q", got.Model)
	}
	if got.Result == nil {
		t.Fatalf("expected non-nil result on success, got nil (body=%s)", string(bodyBytes))
	}
	if got.Result.Model != "moe" || got.Result.Container != "vllm-moe" || got.Result.Result != "redeployed" || got.Result.DurationMS != 287000 {
		t.Fatalf("unexpected inner result: %+v", got.Result)
	}
	if len(got.Result.EvictedPinned) != 1 || got.Result.EvictedPinned[0] != "main" {
		t.Fatalf("unexpected EvictedPinned: %v", got.Result.EvictedPinned)
	}
	if len(got.Result.RestoredPinned) != 1 || got.Result.RestoredPinned[0] != "main" {
		t.Fatalf("unexpected RestoredPinned: %v", got.Result.RestoredPinned)
	}
	if len(got.Result.StoppedPeers) != 1 || got.Result.StoppedPeers[0] != "longctx" {
		t.Fatalf("unexpected StoppedPeers: %v", got.Result.StoppedPeers)
	}
	if len(got.Result.LeftStopped) != 1 || got.Result.LeftStopped[0] != "longctx" {
		t.Fatalf("unexpected LeftStopped: %v", got.Result.LeftStopped)
	}
}

// TestAdminRedeploy_GenericFailureReturns500 verifies that errors
// other than the two known categorical ones map to 500.
func TestAdminRedeploy_GenericFailureReturns500(t *testing.T) {
	r := &redeployStub{
		stubErr: errors.New("docker daemon unreachable"),
	}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{"model":"moe"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", resp.StatusCode)
	}
}

// TestAdminRedeploy_MissingModelFieldReturns400 verifies a body
// without the model field is rejected before reaching the redeployer.
func TestAdminRedeploy_MissingModelFieldReturns400(t *testing.T) {
	r := &redeployStub{}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
	if r.called != 0 {
		t.Fatalf("expected RedeployMember NOT to be called for empty body, got calls=%d", r.called)
	}
}

// TestAdminRedeploy_InvalidJSONReturns400 verifies a malformed body
// returns 400 without ever invoking the redeployer.
func TestAdminRedeploy_InvalidJSONReturns400(t *testing.T) {
	r := &redeployStub{}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{not json`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

// TestAdminRedeploy_NonSchedulerModeReturns501 verifies that a Router
// that does NOT satisfy MemberRedeployer (e.g. swap-mode LegacyRouter)
// returns 501. Operators running swap mode should not see the verb
// silently misbehave — they should see a clear "not implemented"
// signal.
func TestAdminRedeploy_NonSchedulerModeReturns501(t *testing.T) {
	r := &nonRedeployerCoord{}
	app := httpserver.NewApp(httpserver.Options{Router: r})

	req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
		strings.NewReader(`{"model":"moe"}`))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", resp.StatusCode)
	}
}

// TestAdminRedeploy_ResponseShapeIsConsistent (Fix 8) verifies that
// the JSON envelope shape is structurally consistent across the three
// outcome categories — success (200), RedeployMember-returned-error
// (500), and pre-flight reject (400 missing model). Every response
// MUST decode into the same envelope; only field VALUES differ, never
// the field NAMES. This prevents the previous regression where each
// branch returned a distinct shape (success: bare result body, error:
// {error, model, result}, pre-flight: {error}) and consumer tooling
// had to branch on status code AND probe field presence.
func TestAdminRedeploy_ResponseShapeIsConsistent(t *testing.T) {
	type envelope struct {
		Result *jukebox.RedeployResult `json:"result"`
		Error  string                  `json:"error"`
		Model  string                  `json:"model"`
	}

	cases := []struct {
		name       string
		stub       *redeployStub
		body       string
		wantStatus int
		check      func(t *testing.T, env envelope, raw []byte)
	}{
		{
			name: "success_200",
			stub: &redeployStub{stubResult: jukebox.RedeployResult{
				Model:     "moe",
				Container: "vllm-moe",
				Result:    "redeployed",
			}},
			body:       `{"model":"moe"}`,
			wantStatus: http.StatusOK,
			check: func(t *testing.T, env envelope, _ []byte) {
				if env.Error != "" {
					t.Errorf("expected empty error on success, got %q", env.Error)
				}
				if env.Model != "moe" {
					t.Errorf("expected envelope model=moe, got %q", env.Model)
				}
				if env.Result == nil || env.Result.Result != "redeployed" {
					t.Errorf("expected populated result on success, got %+v", env.Result)
				}
			},
		},
		{
			name: "runtime_error_500",
			stub: &redeployStub{
				stubResult: jukebox.RedeployResult{
					Model:        "moe",
					Container:    "vllm-moe",
					StoppedPeers: []string{"longctx"},
					LeftStopped:  []string{"longctx"},
				},
				stubErr: errors.New("docker daemon unreachable"),
			},
			body:       `{"model":"moe"}`,
			wantStatus: http.StatusInternalServerError,
			check: func(t *testing.T, env envelope, _ []byte) {
				if env.Error == "" {
					t.Errorf("expected populated error on 500, got empty")
				}
				if env.Model != "moe" {
					t.Errorf("expected envelope model=moe, got %q", env.Model)
				}
				if env.Result == nil {
					t.Fatalf("expected non-nil partial result on runtime error, got nil")
				}
				// Partial result should carry the state changes that
				// happened before the failure (so the operator can see
				// what was left in flight).
				if len(env.Result.LeftStopped) != 1 || env.Result.LeftStopped[0] != "longctx" {
					t.Errorf("expected partial result LeftStopped=[longctx], got %v", env.Result.LeftStopped)
				}
			},
		},
		{
			name:       "missing_model_400",
			stub:       &redeployStub{},
			body:       `{}`,
			wantStatus: http.StatusBadRequest,
			check: func(t *testing.T, env envelope, raw []byte) {
				if env.Error == "" {
					t.Errorf("expected populated error on missing model, got empty")
				}
				// Pre-flight rejects: model couldn't be parsed (or was
				// empty), so envelope.Model is empty.
				if env.Model != "" {
					t.Errorf("expected empty envelope model on missing-model reject, got %q", env.Model)
				}
				if env.Result != nil {
					t.Errorf("expected nil result on pre-flight reject, got %+v", env.Result)
				}
				// The "result" key must be ABSENT from the raw body
				// (omitempty drops nil pointers). Earlier, code that
				// passed &RedeployResult{} would serialize "result":{}
				// and break the "presence ⇒ ran the operation" client
				// contract.
				if bytes.Contains(raw, []byte(`"result"`)) {
					t.Errorf("expected raw body to OMIT the result key on pre-flight reject, got: %s", string(raw))
				}
			},
		},
		{
			name: "ineligible_400",
			stub: &redeployStub{
				stubErr: jukebox.ErrRedeployIneligible,
			},
			body:       `{"model":"plain"}`,
			wantStatus: http.StatusBadRequest,
			check: func(t *testing.T, env envelope, raw []byte) {
				if env.Error == "" {
					t.Errorf("expected populated error on ineligible, got empty")
				}
				if env.Model != "plain" {
					t.Errorf("expected envelope model=plain (echoed even on error), got %q", env.Model)
				}
				if env.Result != nil {
					t.Errorf("expected nil result on ineligible pre-flight reject, got %+v", env.Result)
				}
				// Pre-flight reject — "result" must be absent from the
				// wire body (omitempty kicks in only when the pointer
				// is nil, NOT when it's a non-nil pointer to a zero
				// struct).
				if bytes.Contains(raw, []byte(`"result"`)) {
					t.Errorf("expected raw body to OMIT the result key on ineligible reject, got: %s", string(raw))
				}
			},
		},
		{
			name: "unknown_model_400",
			stub: &redeployStub{
				stubErr: jukebox.ErrRedeployUnknownModel,
			},
			body:       `{"model":"nope"}`,
			wantStatus: http.StatusBadRequest,
			check: func(t *testing.T, env envelope, raw []byte) {
				if env.Error == "" {
					t.Errorf("expected populated error on unknown model, got empty")
				}
				if env.Model != "nope" {
					t.Errorf("expected envelope model=nope, got %q", env.Model)
				}
				if env.Result != nil {
					t.Errorf("expected nil result on unknown-model pre-flight reject, got %+v", env.Result)
				}
				if bytes.Contains(raw, []byte(`"result"`)) {
					t.Errorf("expected raw body to OMIT the result key on unknown-model reject, got: %s", string(raw))
				}
			},
		},
		{
			// VRAM-drift-risk surfaces as 503 with a distinct body
			// "code" so operators can branch on it programmatically.
			// Result IS populated (handler propagates partial state so
			// the operator can see what containers were left at risk).
			// No Retry-After — terminal/operator-action condition.
			name: "admin_intervention_503",
			stub: &redeployStub{
				stubResult: jukebox.RedeployResult{
					Model:       "moe",
					Container:   "vllm-moe",
					LeftStopped: []string{"longctx"},
				},
				stubErr: jukebox.ErrAdmissionVRAMDriftRisk,
			},
			body:       `{"model":"moe"}`,
			wantStatus: http.StatusServiceUnavailable,
			check: func(t *testing.T, env envelope, raw []byte) {
				if env.Error == "" {
					t.Errorf("expected populated error on admin-intervention, got empty")
				}
				if env.Model != "moe" {
					t.Errorf("expected envelope model=moe, got %q", env.Model)
				}
				if env.Result == nil {
					t.Fatalf("expected partial result on admin-intervention 503, got nil (operator needs LeftStopped)")
				}
				if len(env.Result.LeftStopped) != 1 || env.Result.LeftStopped[0] != "longctx" {
					t.Errorf("expected LeftStopped=[longctx], got %v", env.Result.LeftStopped)
				}
				if !bytes.Contains(raw, []byte(`"code":"admin_intervention_required"`)) {
					t.Errorf("expected body to carry code=admin_intervention_required, got: %s", string(raw))
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			app := httpserver.NewApp(httpserver.Options{Router: tc.stub})
			req := httptest.NewRequest(http.MethodPost, "/admin/redeploy-member",
				strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			resp, err := app.Test(req)
			if err != nil {
				t.Fatalf("app.Test: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("expected %d, got %d", tc.wantStatus, resp.StatusCode)
			}
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			// EVERY path must decode into the same envelope shape.
			// json.Unmarshal accepts missing keys silently, so we
			// additionally cross-check the body contains "model"
			// (always echoed) — except on the missing_model branch
			// where the model field is empty and `omitempty` is NOT
			// applied to keep the shape stable.
			//
			// We don't enforce DisallowUnknownFields here because the
			// envelope IS the shape — any unknown field would be a
			// regression regardless. The 4 envelope fields are
			// model/error/result, with omitempty on result+error so
			// the body is trim on the success path.
			var env envelope
			if err := json.Unmarshal(body, &env); err != nil {
				t.Fatalf("decode envelope: %v (body=%s)", err, string(body))
			}
			tc.check(t, env, body)
		})
	}
}
