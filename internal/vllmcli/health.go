package vllmcli

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// VerifyModelLoaded calls GET /v1/models on the vLLM backend and confirms
// that the expected model id (or its underlying path) is present.
func VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	url := strings.TrimRight(baseURL, "/") + "/v1/models"
	client := &http.Client{}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("GET /v1/models returned %s", resp.Status)
	}

	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return err
	}

	for _, m := range decoded.Data {
		if m.ID == expectedID || (expectedPath != "" && m.ID == expectedPath) {
			return nil
		}
	}
	return fmt.Errorf("expected model %q not found in /v1/models", expectedID)
}
