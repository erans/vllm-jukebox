package vllm

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"vllm-jukebox/internal/vllmcli"
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

// VerifyModelLoaded is a thin re-export of vllmcli.VerifyModelLoaded so
// existing callers (and tests) inside this package keep compiling. The real
// implementation lives in internal/vllmcli to avoid an import cycle with
// internal/runtime.
func VerifyModelLoaded(ctx context.Context, baseURL, expectedID, expectedPath string) error {
	return vllmcli.VerifyModelLoaded(ctx, baseURL, expectedID, expectedPath)
}
