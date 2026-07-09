package httpserver_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v2"

	"vllm-jukebox/internal/httpserver"
)

func newAuthApp(key string) *fiber.App {
	app := fiber.New()
	app.Use(httpserver.AuthAPIKey(key))
	app.Get("/protected", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})
	return app
}

func requireOpenAIAuthError(t *testing.T, resp *http.Response) {
	t.Helper()
	if resp.StatusCode != http.StatusUnauthorized {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 401, got %d: %s", resp.StatusCode, body)
	}

	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("expected OpenAI-style JSON error: %v", err)
	}
	if body.Error.Message == "" || body.Error.Type == "" || body.Error.Code == "" {
		t.Fatalf("expected OpenAI-style error fields, got %+v", body.Error)
	}
}

func TestAuthAPIKey_EmptyExpectedKeyAllowsRequestWithoutHeader(t *testing.T) {
	app := newAuthApp("")
	req := httptest.NewRequest("GET", "/protected", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}
}

func TestAuthAPIKey_MissingHeader_401(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	requireOpenAIAuthError(t, resp)
}

func TestAuthAPIKey_WrongKey_401(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	requireOpenAIAuthError(t, resp)
}

func TestAuthAPIKey_CorrectKey_200(t *testing.T) {
	app := newAuthApp("secret")
	req := httptest.NewRequest("GET", "/protected", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("test: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}
}

func TestAuthAPIKey_MalformedHeader_401(t *testing.T) {
	app := newAuthApp("secret")
	for _, h := range []string{"secret", "Token secret", "Bearer", "bearer secret"} {
		req := httptest.NewRequest("GET", "/protected", strings.NewReader(""))
		req.Header.Set("Authorization", h)
		resp, err := app.Test(req)
		if err != nil {
			t.Fatalf("test: %v", err)
		}
		requireOpenAIAuthError(t, resp)
	}
}
