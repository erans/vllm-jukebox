package vllm

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RotatingFileWriter writes to a file with size-based rotation.
// When the file exceeds maxBytes, it rotates: file.log -> file.log.1, etc.
// Old files beyond maxFiles are deleted.
type RotatingFileWriter struct {
	mu          sync.Mutex
	path        string
	maxBytes    int64
	maxFiles    int
	currentSize int64
	file        *os.File
}

// NewRotatingFileWriter creates a new rotating file writer.
// maxSizeMB is the maximum size in megabytes before rotation.
// maxFiles is the number of rotated files to keep.
func NewRotatingFileWriter(path string, maxSizeMB, maxFiles int) (*RotatingFileWriter, error) {
	// Create parent directories if needed
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory %s: %w", dir, err)
	}

	w := &RotatingFileWriter{
		path:     path,
		maxBytes: int64(maxSizeMB) * 1024 * 1024,
		maxFiles: maxFiles,
	}

	if err := w.openFile(); err != nil {
		return nil, err
	}

	return w, nil
}

func (w *RotatingFileWriter) openFile() error {
	f, err := os.OpenFile(w.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open log file %s: %w", w.path, err)
	}

	// Get current file size
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return fmt.Errorf("failed to stat log file %s: %w", w.path, err)
	}

	w.file = f
	w.currentSize = info.Size()
	return nil
}

func (w *RotatingFileWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return 0, fmt.Errorf("log file is closed")
	}

	// Check if we need to rotate before writing
	if w.maxBytes > 0 && w.currentSize+int64(len(p)) > w.maxBytes {
		if err := w.rotate(); err != nil {
			// Log rotation failed, but continue writing to current file
			// This is graceful degradation
		}
	}

	n, err := w.file.Write(p)
	w.currentSize += int64(n)
	return n, err
}

func (w *RotatingFileWriter) rotate() error {
	// Close current file
	if w.file != nil {
		w.file.Close()
		w.file = nil
	}

	// Delete oldest file if it exists
	oldest := fmt.Sprintf("%s.%d", w.path, w.maxFiles)
	os.Remove(oldest)

	// Rotate existing files: .4 -> .5, .3 -> .4, etc.
	for i := w.maxFiles - 1; i >= 1; i-- {
		oldPath := fmt.Sprintf("%s.%d", w.path, i)
		newPath := fmt.Sprintf("%s.%d", w.path, i+1)
		os.Rename(oldPath, newPath)
	}

	// Rotate current file to .1
	if err := os.Rename(w.path, w.path+".1"); err != nil && !os.IsNotExist(err) {
		// If rename fails, try to reopen the original file
		return w.openFile()
	}

	// Open new file
	w.currentSize = 0
	return w.openFile()
}

func (w *RotatingFileWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file == nil {
		return nil
	}

	err := w.file.Close()
	w.file = nil
	return err
}
