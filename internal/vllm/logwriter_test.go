package vllm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRotatingFileWriter_Write(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	w, err := NewRotatingFileWriter(path, 1, 3) // 1MB max, keep 3 files
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer w.Close()

	// Write some data
	msg := "hello world\n"
	n, err := w.Write([]byte(msg))
	if err != nil {
		t.Fatalf("write failed: %v", err)
	}
	if n != len(msg) {
		t.Fatalf("expected %d bytes written, got %d", len(msg), n)
	}

	// Verify file exists and contains data
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if string(data) != msg {
		t.Fatalf("expected %q, got %q", msg, string(data))
	}
}

func TestRotatingFileWriter_Rotation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	// Use very small max size to trigger rotation
	w, err := NewRotatingFileWriter(path, 0, 3) // 0 MB = will rotate on any write > 0
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	// Manually set maxBytes to something small for testing
	w.maxBytes = 100

	// Write enough to trigger rotation
	for i := 0; i < 5; i++ {
		msg := strings.Repeat("x", 50) + "\n"
		_, err := w.Write([]byte(msg))
		if err != nil {
			t.Fatalf("write %d failed: %v", i, err)
		}
	}
	w.Close()

	// Check that rotated files exist
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("main log file should exist")
	}
	if _, err := os.Stat(path + ".1"); os.IsNotExist(err) {
		t.Fatal("rotated file .1 should exist")
	}
}

func TestRotatingFileWriter_CreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "subdir", "nested", "test.log")

	w, err := NewRotatingFileWriter(path, 50, 5)
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}
	defer w.Close()

	// Verify directory was created
	if _, err := os.Stat(filepath.Dir(path)); os.IsNotExist(err) {
		t.Fatal("directory should have been created")
	}
}

func TestRotatingFileWriter_MaxFilesRespected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.log")

	w, err := NewRotatingFileWriter(path, 0, 2) // keep only 2 rotated files
	if err != nil {
		t.Fatalf("failed to create writer: %v", err)
	}

	w.maxBytes = 50 // small size for testing

	// Write enough to trigger multiple rotations
	for i := 0; i < 10; i++ {
		msg := strings.Repeat("y", 60) + "\n"
		w.Write([]byte(msg))
	}
	w.Close()

	// .1 and .2 should exist, .3 should not
	if _, err := os.Stat(path + ".1"); os.IsNotExist(err) {
		t.Fatal(".1 should exist")
	}
	if _, err := os.Stat(path + ".2"); os.IsNotExist(err) {
		t.Fatal(".2 should exist")
	}
	if _, err := os.Stat(path + ".3"); !os.IsNotExist(err) {
		t.Fatal(".3 should NOT exist")
	}
}
