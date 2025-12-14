package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// RewriteJSONModel rewrites a top-level {"model": "..."} field (if present) to requestedModel.
// Returns (newBody, true, nil) if a rewrite occurred, otherwise (originalBody, false, nil).
func RewriteJSONModel(body []byte, requestedModel string) ([]byte, bool, error) {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, false, err
	}

	if _, ok := obj["model"]; !ok {
		return body, false, nil
	}
	obj["model"] = requestedModel

	out, err := json.Marshal(obj)
	if err != nil {
		return nil, false, err
	}
	return out, true, nil
}

// RewriteSSEModel reads an SSE stream and rewrites JSON objects in "data: <json>" lines to set model=requestedModel.
// "data: [DONE]" lines are passed through unchanged.
func RewriteSSEModel(r io.Reader, w io.Writer, requestedModel string) error {
	if r == nil {
		return fmt.Errorf("reader is nil")
	}
	if w == nil {
		return fmt.Errorf("writer is nil")
	}

	br := bufio.NewReader(r)
	for {
		line, err := br.ReadString('\n')
		if line != "" {
			if strings.HasPrefix(line, "data:") {
				rest := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if rest != "[DONE]" && rest != "" && strings.HasPrefix(rest, "{") {
					rewritten, _, rewriteErr := RewriteJSONModel([]byte(rest), requestedModel)
					if rewriteErr == nil {
						_, _ = io.WriteString(w, "data: ")
						_, _ = w.Write(bytes.TrimSpace(rewritten))
						_, _ = io.WriteString(w, "\n")
						flushIfPossible(w)
						goto next
					}
				}
			}
			if _, writeErr := io.WriteString(w, line); writeErr != nil {
				return writeErr
			}
			flushIfPossible(w)
		}
	next:
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

type flusher interface {
	Flush() error
}

func flushIfPossible(w io.Writer) {
	if f, ok := w.(flusher); ok {
		_ = f.Flush()
	}
}
