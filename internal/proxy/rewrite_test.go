package proxy_test

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"vllm-jukebox/internal/proxy"
)

func TestRewriteJSONModel_RewritesTopLevelModel(t *testing.T) {
	in := []byte(`{"id":"x","model":"base","other":1}`)
	out, ok, err := proxy.RewriteJSONModel(in, "alias")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !ok {
		t.Fatalf("expected ok=true")
	}
	if !strings.Contains(string(out), `"model":"alias"`) {
		t.Fatalf("expected rewritten model, got %s", string(out))
	}
}

func TestRewriteSSEModel_RewritesDataJSONChunks(t *testing.T) {
	in := "event: message\n" +
		"data: {\"model\":\"base\",\"x\":1}\n" +
		"\n" +
		"data: [DONE]\n\n"

	var out bytes.Buffer
	if err := proxy.RewriteSSEModel(strings.NewReader(in), &out, "alias"); err != nil {
		t.Fatalf("err: %v", err)
	}

	s := out.String()
	if !strings.Contains(s, `"model":"alias"`) {
		t.Fatalf("expected rewritten model, got %q", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Fatalf("expected DONE passthrough, got %q", s)
	}
}

func TestRewriteSSEModel_PassthroughNonDataLines(t *testing.T) {
	in := "event: ping\n\n"
	var out bytes.Buffer
	if err := proxy.RewriteSSEModel(strings.NewReader(in), &out, "alias"); err != nil {
		t.Fatalf("err: %v", err)
	}
	if got := out.String(); got != in {
		t.Fatalf("expected passthrough, got %q", got)
	}
}

func TestRewriteSSEModel_RejectsNilReaderWriter(t *testing.T) {
	if err := proxy.RewriteSSEModel(nil, io.Discard, "x"); err == nil {
		t.Fatalf("expected error")
	}
	if err := proxy.RewriteSSEModel(strings.NewReader(""), nil, "x"); err == nil {
		t.Fatalf("expected error")
	}
}

