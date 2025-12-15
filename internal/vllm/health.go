package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

var ErrProcessExited = errors.New("process exited")

func WaitForHealth(ctx context.Context, baseURL string) error {
	url := strings.TrimRight(baseURL, "/") + "/health"
	client := &http.Client{}

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ticker.C:
			continue
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func WaitForHealthOrExit(ctx context.Context, baseURL string, exited <-chan struct{}) error {
	if exited == nil {
		return WaitForHealth(ctx, baseURL)
	}

	url := strings.TrimRight(baseURL, "/") + "/health"
	client := &http.Client{}

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-exited:
			return ErrProcessExited
		default:
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ticker.C:
			continue
		case <-exited:
			return ErrProcessExited
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

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
