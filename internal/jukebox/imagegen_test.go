package jukebox

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseImageSize_Defaults(t *testing.T) {
	cases := []struct {
		in     string
		wantW  int
		wantH  int
		reason string
	}{
		{"", defaultImageW, defaultImageH, "empty falls to default"},
		{"   ", defaultImageW, defaultImageH, "whitespace falls to default"},
		{"garbage", defaultImageW, defaultImageH, "non-WxH falls to default"},
		{"512x512", 512, 512, "exact parse"},
		{"1024X1024", 1024, 1024, "case-insensitive"},
		{"515x515", 512, 512, "snap-to-8"},
		{"40x40", 64, 64, "clamp-to-min"},
		{"4000x4000", 2048, 2048, "clamp-to-max"},
		{"-5x-5", defaultImageW, defaultImageH, "negative falls to default"},
		{"0x0", defaultImageW, defaultImageH, "zero falls to default"},
	}
	for _, c := range cases {
		t.Run(c.reason, func(t *testing.T) {
			gotW, gotH := parseImageSize(c.in)
			if gotW != c.wantW || gotH != c.wantH {
				t.Fatalf("parseImageSize(%q) = (%d, %d), want (%d, %d)",
					c.in, gotW, gotH, c.wantW, c.wantH)
			}
		})
	}
}

func TestBuildQwenImageWorkflow_Shape(t *testing.T) {
	wf := buildQwenImageWorkflow("a red apple", 512, 768, 42)
	// All 11 nodes present + class_type wired up.
	want := map[string]string{
		"1":  "VAELoader",
		"2":  "CLIPLoader",
		"3":  "UNETLoader",
		"4":  "LoraLoaderModelOnly",
		"5":  "ModelSamplingAuraFlow",
		"6":  "CLIPTextEncode",
		"7":  "CLIPTextEncode",
		"8":  "EmptySD3LatentImage",
		"9":  "KSampler",
		"10": "VAEDecode",
		"11": "SaveImage",
	}
	for id, ct := range want {
		node, ok := wf[id].(map[string]any)
		if !ok {
			t.Fatalf("node %s missing or wrong shape", id)
		}
		if node["class_type"] != ct {
			t.Errorf("node %s: class_type = %v, want %s", id, node["class_type"], ct)
		}
	}
	// Prompt is wired into node 6.
	n6 := wf["6"].(map[string]any)["inputs"].(map[string]any)
	if n6["text"] != "a red apple" {
		t.Errorf("prompt not wired to node 6: %v", n6["text"])
	}
	// Negative prompt is empty at node 7.
	n7 := wf["7"].(map[string]any)["inputs"].(map[string]any)
	if n7["text"] != "" {
		t.Errorf("negative prompt should be empty, got %v", n7["text"])
	}
	// Latent image dims.
	n8 := wf["8"].(map[string]any)["inputs"].(map[string]any)
	if n8["width"] != 512 || n8["height"] != 768 {
		t.Errorf("EmptySD3LatentImage dims = %vx%v, want 512x768", n8["width"], n8["height"])
	}
	// KSampler seed + step count.
	n9 := wf["9"].(map[string]any)["inputs"].(map[string]any)
	if n9["seed"].(int64) != 42 {
		t.Errorf("KSampler seed = %v, want 42", n9["seed"])
	}
	if n9["steps"] != defaultSamplerSteps {
		t.Errorf("KSampler steps = %v, want %d", n9["steps"], defaultSamplerSteps)
	}
	// JSON-marshalable round-trip (catches any non-serializable values).
	if _, err := json.Marshal(wf); err != nil {
		t.Fatalf("workflow not JSON-marshalable: %v", err)
	}
}

func TestEncodeImageDataURIPayload(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0xde, 0xad, 0xbe, 0xef}
	resp := EncodeImageDataURIPayload(png, 1718294400)
	if resp.Created != 1718294400 {
		t.Errorf("Created = %d, want 1718294400", resp.Created)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("Data length = %d, want 1", len(resp.Data))
	}
	got, err := base64.StdEncoding.DecodeString(resp.Data[0].B64JSON)
	if err != nil {
		t.Fatalf("Data[0].B64JSON not base64: %v", err)
	}
	if string(got) != string(png) {
		t.Errorf("decoded bytes differ from input")
	}
}

func TestEncodeImageDataURIPayload_DefaultsTimestamp(t *testing.T) {
	resp := EncodeImageDataURIPayload([]byte("x"), 0)
	if resp.Created == 0 {
		t.Errorf("Created should default to time.Now().Unix(), got 0")
	}
}

// TestImageGenerate_EndToEnd_HappyPath spins up a fake ComfyUI server
// covering /prompt, /history, and /view; verifies that ImageGenerate
// submits, polls, and fetches the PNG.
func TestImageGenerate_EndToEnd_HappyPath(t *testing.T) {
	const promptID = "test-prompt-id-123"
	const pngBody = "\x89PNG\r\n\x1a\nFAKEPNG"

	var historyCalls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/prompt", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "wrong method", http.StatusMethodNotAllowed)
			return
		}
		// Sanity-check the body has a "prompt" key with our node graph.
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if _, ok := body["prompt"].(map[string]any); !ok {
			http.Error(w, "missing prompt field", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"prompt_id": promptID})
	})
	mux.HandleFunc("/history/"+promptID, func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&historyCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		if n < 2 {
			// First poll: queued, no outputs yet.
			_, _ = w.Write([]byte(`{}`))
			return
		}
		// Second poll: SaveImage produced output.
		resp := map[string]any{
			promptID: map[string]any{
				"status": map[string]any{"status_str": "success"},
				"outputs": map[string]any{
					"11": map[string]any{
						"images": []any{
							map[string]any{
								"filename":  "jukebox_00001_.png",
								"subfolder": "",
								"type":      "output",
							},
						},
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	})
	mux.HandleFunc("/view", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("filename") != "jukebox_00001_.png" {
			http.Error(w, "wrong filename", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte(pngBody))
	})

	srv := httptest.NewServer(mux)
	defer srv.Close()

	png, err := ImageGenerate(
		context.Background(),
		ImageGenOptions{
			BaseURL:      srv.URL,
			PollTimeout:  5 * time.Second,
			PollInterval: 50 * time.Millisecond,
			HTTPTimeout:  2 * time.Second,
		},
		ImageGenRequest{
			Model:  "Qwen-Image-Lightning",
			Prompt: "a small red apple on a white plate",
			N:      1,
			Size:   "512x512",
		},
	)
	if err != nil {
		t.Fatalf("ImageGenerate failed: %v", err)
	}
	if string(png) != pngBody {
		t.Errorf("returned PNG = %q, want %q", png, pngBody)
	}
	if atomic.LoadInt32(&historyCalls) < 2 {
		t.Errorf("expected at least 2 /history polls, got %d", atomic.LoadInt32(&historyCalls))
	}
}

func TestImageGenerate_RejectsBadRequest(t *testing.T) {
	cases := []struct {
		name   string
		req    ImageGenRequest
		opts   ImageGenOptions
		status int
		substr string
	}{
		{
			name:   "empty base URL",
			req:    ImageGenRequest{Prompt: "x"},
			opts:   ImageGenOptions{BaseURL: ""},
			status: http.StatusInternalServerError,
			substr: "empty base URL",
		},
		{
			name:   "empty prompt",
			req:    ImageGenRequest{Prompt: ""},
			opts:   ImageGenOptions{BaseURL: "http://example"},
			status: http.StatusBadRequest,
			substr: "missing required field 'prompt'",
		},
		{
			name:   "n>1 unsupported",
			req:    ImageGenRequest{Prompt: "x", N: 2},
			opts:   ImageGenOptions{BaseURL: "http://example"},
			status: http.StatusBadRequest,
			substr: "n must be 1",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ImageGenerate(context.Background(), tc.opts, tc.req)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			ige, ok := err.(*ImageGenError)
			if !ok {
				t.Fatalf("error is not *ImageGenError: %T", err)
			}
			if ige.Status != tc.status {
				t.Errorf("status = %d, want %d", ige.Status, tc.status)
			}
			if !strings.Contains(ige.Message, tc.substr) {
				t.Errorf("message %q does not contain %q", ige.Message, tc.substr)
			}
		})
	}
}

func TestImageGenerate_ComfyUIErrorStatus(t *testing.T) {
	const promptID = "err-test"
	mux := http.NewServeMux()
	mux.HandleFunc("/prompt", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"prompt_id": promptID})
	})
	mux.HandleFunc("/history/"+promptID, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			promptID: map[string]any{
				"status":  map[string]any{"status_str": "error"},
				"outputs": map[string]any{},
			},
		})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	_, err := ImageGenerate(
		context.Background(),
		ImageGenOptions{
			BaseURL:      srv.URL,
			PollTimeout:  2 * time.Second,
			PollInterval: 50 * time.Millisecond,
			HTTPTimeout:  1 * time.Second,
		},
		ImageGenRequest{Prompt: "x"},
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	ige, ok := err.(*ImageGenError)
	if !ok {
		t.Fatalf("error is not *ImageGenError: %T", err)
	}
	if ige.Status != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", ige.Status)
	}
	if !strings.Contains(ige.Message, "workflow errored") {
		t.Errorf("message %q does not mention workflow error", ige.Message)
	}
}
