// Package parser provides functions to parse ONNX models into GoMLX.
package parser

import (
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/gomlx/onnx-gomlx/onnx"
)

// FromProto parses an ONNX model into an internal representation that can be used to build a GoMLX graph.
func FromProto(contents []byte) (onnx.Model, error) {
	return onnxgomlx.Parse(contents)
}

// FromFile parses an ONNX model file into an internal representation that can be used to build a GoMLX graph.
func FromFile(filePath string) (onnx.Model, error) {
	return onnxgomlx.ReadFile(filePath)
}
