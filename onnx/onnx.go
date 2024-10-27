// Package onnx provides functionality to parse ONNX models and generate the corresponding GoMLX.
//
//   - Parse: converts a serialized ONNX ModelProto to a Model.
//   - ReadFile: reads a file and calls Parse. It returns a Model.
//   - Model: object holding information about an ONNX model. It can be used to generate the corresponding GoMLX
//     model graph and executed for inference or used on a training loop for fine-tuning. It can also be used to
//     populate a context with the variables of the ONNX model.
package onnx

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"os"
)

// Model represents a parsed ONNX file.
type Model struct {
	Proto                       protos.ModelProto
	InputsNames, OutputsNames   []string
	InputsShapes, OutputsShapes []DynamicShape
}

// Parse parses an ONNX model into an internal representation that can be used to build a GoMLX graph.
func Parse(contents []byte) (*Model, error) {
	m := &Model{}
	err := proto.Unmarshal(contents, &m.Proto)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse ONNX model proto")
	}

	// Parse inputs and outputs.
	m.InputsNames = make([]string, len(m.Proto.Graph.Input))
	m.InputsShapes = make([]DynamicShape, len(m.Proto.Graph.Input))
	for ii, input := range m.Proto.Graph.Input {
		m.InputsNames[ii] = input.Name
		tensorType, ok := input.Type.Value.(*protos.TypeProto_TensorType)
		if !ok {
			return nil, errors.Errorf("output #%d (%q) is not a tensor, not sure how to handle it", ii, input.Name)
		}
		m.InputsShapes[ii], err = makeDynamicShapeFromProto(tensorType.TensorType)
		if err != nil {
			return nil, errors.WithMessagef(err, "while parsing output #%d (%q)", ii, input.Name)
		}
	}
	m.OutputsNames = make([]string, len(m.Proto.Graph.Output))
	m.OutputsShapes = make([]DynamicShape, len(m.Proto.Graph.Output))
	for ii, output := range m.Proto.Graph.Output {
		m.OutputsNames[ii] = output.Name
		tensorType, ok := output.Type.Value.(*protos.TypeProto_TensorType)
		if !ok {
			return nil, errors.Errorf("output #%d (%q) is not a tensor, not sure how to handle it", ii, output.Name)
		}
		m.OutputsShapes[ii], err = makeDynamicShapeFromProto(tensorType.TensorType)
		if err != nil {
			return nil, errors.WithMessagef(err, "while parsing output #%d (%q)", ii, output.Name)
		}
	}

	return m, nil
}

// ReadFile parses an ONNX model file into an internal representation that can be used to build a GoMLX graph.
// Notice any large constant is converted to variables.
func ReadFile(filePath string) (*Model, error) {
	contents, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read ONNX model file in %s", filePath)
	}
	return Parse(contents)
}

// Inputs returns the names and DynamicShapes of the inputs.
func (m *Model) Inputs() (names []string, dshapes []DynamicShape) {
	return m.InputsNames, m.InputsShapes
}

// Outputs returns a description of the outputs.
func (m *Model) Outputs() (names []string, dshapes []DynamicShape) {
	return m.OutputsNames, m.OutputsShapes
}

// BuildGraph that can be used both for inference and training.
// ctx can be set to nil if the model doesn't have any variables.
//
// As in GoMLX graph functions, it panics (throw exceptions) in case of errors.
func (m *Model) BuildGraph(ctx *context.Context, inputs []*Node) (outputs []*Node) {
	// Sanity check of things we don't support yet.
	if len(m.Proto.Functions) > 0 {
		exceptions.Panicf("onnx.BuildGraph does not support yet ONNX functions")
	}
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.BuildGraph does not support yet ONNX SparseTensors")
	}

	return nil
}
