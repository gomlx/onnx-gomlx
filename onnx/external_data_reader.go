package onnx

import (
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// ExternalDataReader manages memory-mapped external data files for efficient tensor loading.
// It caches mmap regions by file path since multiple tensors often share the same external file.
// This eliminates the need for intermediate buffer allocations when loading tensor data.
type ExternalDataReader struct {
	baseDir  string
	mappings map[string]*mmapRegion
	mu       sync.Mutex
}

// mmapRegion holds a memory-mapped file region.
type mmapRegion struct {
	reader *mmap.ReaderAt
}

// NewExternalDataReader creates a reader for the given model directory.
// baseDir is the directory containing the ONNX model file, used to resolve external data paths.
func NewExternalDataReader(baseDir string) *ExternalDataReader {
	return &ExternalDataReader{
		baseDir:  baseDir,
		mappings: make(map[string]*mmapRegion),
	}
}

// getOrCreateMapping returns the mmap region for the given file path, creating it if necessary.
func (r *ExternalDataReader) getOrCreateMapping(location string) (*mmapRegion, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if already mapped
	if region, ok := r.mappings[location]; ok {
		return region, nil
	}

	// Resolve the external file path relative to the model directory
	externalPath := filepath.Join(r.baseDir, location)

	// Open and mmap the file
	reader, err := mmap.Open(externalPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to mmap external data file %q", externalPath)
	}

	region := &mmapRegion{reader: reader}
	r.mappings[location] = region
	return region, nil
}

// ReadInto reads external data directly into the provided byte slice.
// This avoids intermediate allocations by copying directly from the mmap region
// into the destination buffer (typically the tensor's backing memory).
func (r *ExternalDataReader) ReadInto(info *externalDataInfo, dst []byte) error {
	if r.baseDir == "" {
		return errors.New("base directory is required for reading external data")
	}

	region, err := r.getOrCreateMapping(info.location)
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

	// Read from mmap region directly into destination
	n, err := region.reader.ReadAt(dst, info.offset)
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

// Close unmaps all memory regions and releases resources.
// After Close is called, the reader should not be used.
func (r *ExternalDataReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var firstErr error
	for path, region := range r.mappings {
		if err := region.reader.Close(); err != nil && firstErr == nil {
			firstErr = errors.Wrapf(err, "failed to close mmap for %q", path)
		}
	}
	r.mappings = nil
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
