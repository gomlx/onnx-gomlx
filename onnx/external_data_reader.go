package onnx

import (
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
)

// ExternalDataReader manages external data files for tensor loading.
// It caches file handles by path since multiple tensors often share the same external file,
// avoiding repeated open/close overhead during model loading.
type ExternalDataReader struct {
	baseDir string
	files   map[string]*os.File
	mu      sync.Mutex
}

// NewExternalDataReader creates a reader for the given model directory.
// baseDir is the directory containing the ONNX model file, used to resolve external data paths.
func NewExternalDataReader(baseDir string) *ExternalDataReader {
	return &ExternalDataReader{
		baseDir: baseDir,
		files:   make(map[string]*os.File),
	}
}

// getOrOpenFile returns the file handle for the given location, opening it if necessary.
func (r *ExternalDataReader) getOrOpenFile(location string) (*os.File, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if already open
	if file, ok := r.files[location]; ok {
		return file, nil
	}

	// Resolve the external file path relative to the model directory
	externalPath := filepath.Join(r.baseDir, location)

	// Open the file
	file, err := os.Open(externalPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open external data file %q", externalPath)
	}

	r.files[location] = file
	return file, nil
}

// ReadInto reads external data directly into the provided byte slice.
func (r *ExternalDataReader) ReadInto(info *externalDataInfo, dst []byte) error {
	if r.baseDir == "" {
		return errors.New("base directory is required for reading external data")
	}

	file, err := r.getOrOpenFile(info.location)
	if err != nil {
		return err
	}

	// Determine the length to read
	length := int64(len(dst))
	if info.length > 0 {
		// Explicit length specified
		if info.length != int64(len(dst)) {
			return errors.Errorf("external data length %d doesn't match destination size %d", info.length, len(dst))
		}
		length = info.length
	}

	// Read at the specified offset (ReadAt is safe for concurrent use)
	n, err := file.ReadAt(dst, info.offset)
	if err != nil && err != io.EOF {
		return errors.Wrapf(err, "failed to read %d bytes at offset %d from external data file %q",
			length, info.offset, info.location)
	}
	if int64(n) != length {
		return errors.Errorf("read %d bytes but expected %d from external data file %q",
			n, length, info.location)
	}

	return nil
}

// Close closes all cached file handles and releases resources.
// After Close is called, the reader should not be used.
func (r *ExternalDataReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var firstErr error
	for path, file := range r.files {
		if err := file.Close(); err != nil && firstErr == nil {
			firstErr = errors.Wrapf(err, "failed to close file %q", path)
		}
	}
	r.files = nil
	return firstErr
}

// readExternalDataDirect reads tensor data from an external file directly into dst.
// This is the fallback path when mmap is not available or for simple cases.
// baseDir is the directory containing the ONNX model file.
func readExternalDataDirect(baseDir string, info *externalDataInfo, dst []byte) error {
	if baseDir == "" {
		return errors.New("base directory is required for reading external data")
	}

	// Resolve the external file path relative to the model directory
	externalPath := filepath.Join(baseDir, info.location)

	file, err := os.Open(externalPath)
	if err != nil {
		return errors.Wrapf(err, "failed to open external data file %q", externalPath)
	}
	defer file.Close()

	// Seek to offset if specified
	if info.offset > 0 {
		_, err = file.Seek(info.offset, io.SeekStart)
		if err != nil {
			return errors.Wrapf(err, "failed to seek to offset %d in external data file %q", info.offset, externalPath)
		}
	}

	// Read directly into destination
	n, err := io.ReadFull(file, dst)
	if err != nil {
		return errors.Wrapf(err, "failed to read %d bytes from external data file %q (read %d)", len(dst), externalPath, n)
	}

	return nil
}
