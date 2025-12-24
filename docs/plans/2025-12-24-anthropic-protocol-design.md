# Anthropic Protocol Support Design

## Goal

Add native Anthropic API protocol support to vllm-jukebox, allowing clients using Anthropic SDKs to connect directly without code changes.

## Architecture

Support both protocols natively side-by-side:
- OpenAI: `/v1/chat/completions`, `/v1/completions`, etc.
- Anthropic: `/v1/messages`

Both protocols proxy directly to vLLM, which handles the actual request/response format natively.

## Request Flow

Same for both protocols:
1. Request arrives at jukebox
2. Extract `model` from JSON body (works for both - same field name)
3. Resolve model via config (aliases, validation)
4. `AcquireRoute()` to get/start vLLM instance
5. Proxy request to vLLM backend
6. Rewrite `model` field in response if configured
7. Return response with protocol-appropriate error format if anything fails

## New Endpoints

- `POST /v1/messages` - Main Anthropic chat completion endpoint

Additional Anthropic endpoints will be added as discovered in vLLM's implementation.

## Error Response Format

Anthropic errors use a different format than OpenAI:

**OpenAI:**
```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

**Anthropic:**
```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}
```

### Error Type Mapping

| Jukebox condition | Anthropic error type | HTTP status |
|-------------------|---------------------|-------------|
| Invalid JSON | `invalid_request_error` | 400 |
| Model not found | `invalid_request_error` | 400 |
| Model switching | `overloaded_error` | 503 |
| vLLM unavailable | `api_error` | 500 |

## Model Name Rewriting

The existing `behavior.rewrite_model_name` setting applies to both protocols. When enabled, the `model` field in responses is rewritten to match the requested model name.

For Anthropic SSE streaming, the event format differs from OpenAI but model rewriting follows the same logic.

## File Changes

1. **`internal/httpserver/server.go`** - Add Anthropic routes:
   ```go
   app.Post("/v1/messages", anthropicProxyHandler(opts))
   ```

2. **`internal/httpserver/handlers_anthropic.go`** (new file):
   - Anthropic-specific handler
   - Uses same `extractModel()` - Anthropic also has `"model"` in body
   - Uses same `AcquireRoute()` for routing
   - Maps errors to Anthropic format via `writeAnthropicError()`

3. **`internal/httpserver/errors.go`** (new file):
   - `writeOpenAIError()` - existing logic moved here
   - `writeAnthropicError()` - new Anthropic format

4. **`internal/proxy/rewrite.go`** - Add Anthropic model rewriting:
   - `RewriteAnthropicModel()` for JSON responses
   - `RewriteAnthropicSSE()` for streaming

## Testing

### Unit Tests

**`internal/httpserver/handlers_anthropic_test.go`:**
- `TestAnthropicProxy_ExtractsModelAndRoutes`
- `TestAnthropicProxy_InvalidJSONReturnsAnthropicError`
- `TestAnthropicProxy_MissingModelReturns400`
- `TestAnthropicProxy_UnknownModelReturns400`
- `TestAnthropicProxy_SwapInProgressReturns503`

**`internal/proxy/rewrite_test.go`:**
- `TestRewriteAnthropicModel_RewritesTopLevelModel`
- `TestRewriteAnthropicSSE_RewritesEventData`

### Smoke Test

**`scripts/smoke_anthropic.sh`:**
- Fake vLLM server responding with Anthropic format
- Test `/v1/messages` endpoint
- Verify error responses are Anthropic-formatted

### Integration Test (Manual)

- Real vLLM with Anthropic endpoint enabled
- Test with Anthropic Python SDK

## Configuration

No new configuration required. Existing settings work for both protocols:
- `behavior.rewrite_model_name` applies to both
- Model definitions work for both (same `model` field)
- Aliases work for both protocols

## Non-Goals

- `/v1/models` endpoint for Anthropic (Anthropic clients don't expect it)
- Protocol translation/bridging (both protocols native to vLLM)
