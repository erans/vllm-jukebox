package httpserver

import (
	"errors"
	"log/slog"
	"net/http"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/jukebox"
)

// redeployMemberRequest is the JSON body shape accepted by
// POST /admin/redeploy-member.
type redeployMemberRequest struct {
	Model string `json:"model"`
}

// redeployResponse is the unified envelope returned by the redeploy-member
// endpoint regardless of status code (Fix 8). Previously the handler
// returned three structurally different JSON shapes — bare RedeployResult
// on success (200), {"error", "model", "result"} on RedeployMember-returned
// error (400/500), and {"error"} on pre-flight reject (501/400) — forcing
// every client to branch on status code AND probe field presence to
// decode the body. The unified envelope is:
//
//	{
//	  "result": {...},   // key OMITTED on pre-flight rejects (501,
//	                     //   400 missing/invalid model, 400 unknown
//	                     //   model, 400 ineligible). Handler passes
//	                     //   Result=nil for these paths so omitempty
//	                     //   actually drops the key — earlier versions
//	                     //   passed &RedeployResult{} which serialized
//	                     //   as "result": {}, violating the
//	                     //   "presence ⇒ ran the operation" contract.
//	                     //   Clients should test field PRESENCE, not
//	                     //   nullity: `if "result" in body` rather
//	                     //   than `if body["result"] is not None`.
//	  "error":  "...",   // empty string on success
//	  "model":  "...",  // always echoed; empty if model couldn't be parsed
//	  "code":   "..."    // categorical error code, set ONLY for known
//	                     //   error classes operators can act on
//	                     //   (currently: "admin_intervention_required"
//	                     //   for ErrAdmissionVRAMDriftRisk → 503).
//	                     //   Empty/absent on success and on generic
//	                     //   500/400 paths.
//	}
//
// Same shape on every code path; clients decode once and check the two
// presence fields. The "result" pointer-or-nil keeps the success body
// trim (no empty {} envelope on success).
type redeployResponse struct {
	Result *jukebox.RedeployResult `json:"result,omitempty"`
	Error  string                  `json:"error,omitempty"`
	Model  string                  `json:"model"`
	Code   string                  `json:"code,omitempty"`
}

// redeployMemberHandler dispatches an operator-initiated stop+start of
// the container backing a swap-group / evict_action:stop member. Per
// Kagi PATCH 16, this verb cannot be a CLI subcommand — a separate
// process cannot acquire the in-memory cold-load mutex or mutate
// admission state — so it lives as an HTTP endpoint on the running
// daemon. Auth is intentionally NOT enforced here: the compose binds
// jukebox port 9000 loopback-only and the operator reaches it via
// localhost or the existing front-door proxy (reverse proxy).
//
// Synchronous from the operator's POV — a cold-load is 5-8 min. There
// is no async-503 path; the operator explicitly invoked this knowing
// it will block.
//
// The router must satisfy jukebox.MemberRedeployer. *Scheduler is the
// only implementation today; swap-mode (LegacyRouter) returns 501
// because the verb has no meaning when there's a single coordinator-
// owned process rather than per-model containers.
//
// Response shape: ALWAYS redeployResponse (see Fix 8 comment above).
// Status codes:
//
//	200 — success; result populated, error empty.
//	400 — bad input (missing/empty model, invalid JSON, unknown model,
//	      ineligible model); error populated, result key OMITTED
//	      (handler passes nil so omitempty drops it — clients can
//	      reliably treat "result-key-present" as "operation ran").
//	500 — runtime failure (docker stop/start, /is_sleeping poll
//	      timeout, pinned-peer pause failure); error populated AND
//	      result populated with the partial outcome (StoppedPeers,
//	      LeftStopped, etc.) so the operator can see what state changed.
//	503 — admission books may have drifted from physical container
//	      state (ErrAdmissionVRAMDriftRisk). Body code field is
//	      "admin_intervention_required". No Retry-After header — this
//	      is a terminal/operator-action condition. Result populated
//	      with partial outcome (LeftStopped lists at-risk containers).
//	501 — router doesn't satisfy MemberRedeployer (swap mode); error
//	      populated, result key omitted, model empty (no body parsed
//	      before this branch).
func redeployMemberHandler(router jukebox.Router) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Reject anything that isn't a swap-group-capable router up
		// front so we don't parse a body just to 501 on it.
		redeployer, ok := router.(jukebox.MemberRedeployer)
		if !ok {
			return c.Status(http.StatusNotImplemented).JSON(redeployResponse{
				Error: "redeploy-member is only available in scheduler mode (multi-instance with admission)",
			})
		}

		// Parse the body so we can echo `model` in the response even on
		// failure. If parsing fails the `model` field stays empty —
		// clients should still be able to round-trip the envelope.
		var req redeployMemberRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(http.StatusBadRequest).JSON(redeployResponse{
				Error: "invalid JSON body: " + err.Error(),
			})
		}
		if req.Model == "" {
			return c.Status(http.StatusBadRequest).JSON(redeployResponse{
				Error: "missing required field: model",
			})
		}

		slog.Info("redeploy_member_request",
			"model", req.Model,
			"request_id", c.Get(requestIDHeader),
		)

		result, err := redeployer.RedeployMember(c.UserContext(), req.Model)
		if err != nil {
			// Map known categorical errors to HTTP status codes.
			//   - 400: input-shape errors the operator can fix from the API
			//          (unknown model, ineligible model).
			//   - 503: ErrAdmissionVRAMDriftRisk — admission books may have
			//          drifted from physical container state because a
			//          cleanup `docker stop` failed during a redeploy. This
			//          requires operator reconciliation (see audit log +
			//          runbook), NOT an automated retry. The body code
			//          "admin_intervention_required" mirrors the
			//          RejectAdminIntervention surface used on the proxy
			//          handlers, and Retry-After is intentionally absent
			//          so retry-aware clients treat it as terminal.
			//   - 500: everything else (docker stop/start, /is_sleeping
			//          poll timeout, pinned-peer pause failure) — runtime
			//          state the operator cannot fix from the API surface.
			status := http.StatusInternalServerError
			errorCode := ""
			switch {
			case errors.Is(err, jukebox.ErrRedeployUnknownModel),
				errors.Is(err, jukebox.ErrRedeployIneligible):
				status = http.StatusBadRequest
			case errors.Is(err, jukebox.ErrAdmissionVRAMDriftRisk):
				status = http.StatusServiceUnavailable
				errorCode = "admin_intervention_required"
			}
			slog.Warn("redeploy_member_failed",
				"model", req.Model,
				"err", err,
				"status", status,
			)
			// Carry the partial RedeployResult so the operator can see
			// what state changed (e.g. peers we stopped before failing).
			// Pre-flight rejects (unknown-model, ineligible) pass nil so
			// omitempty actually drops the "result" key — see the
			// redeployResponse docstring.
			resp := redeployResponse{
				Error: err.Error(),
				Model: req.Model,
				Code:  errorCode,
			}
			if !errors.Is(err, jukebox.ErrRedeployUnknownModel) &&
				!errors.Is(err, jukebox.ErrRedeployIneligible) {
				resp.Result = &result
			}
			return c.Status(status).JSON(resp)
		}

		return c.Status(http.StatusOK).JSON(redeployResponse{
			Result: &result,
			Model:  req.Model,
		})
	}
}
