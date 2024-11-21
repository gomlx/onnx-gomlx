// Package onnx provides functionality to parse ONNX models and generate the corresponding GoMLX.
//
//   - Parse: converts a serialized ONNX ModelProto to a Model.
//   - ReadFile: reads a file and calls Parse. It returns a Model.
//   - Model: object holding information about an ONNX model. It can be used to generate the corresponding GoMLX
//     model graph and executed for inference or used on a training loop for fine-tuning. It can also be used to
//     populate a context with the variables of the ONNX model.
package onnx

import (
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"io"
	"os"
)

// Model represents a parsed ONNX file.
type Model struct {
	onnxFileName     string
	Proto            protos.ModelProto
	nodeOutputToNode map[string]*protos.NodeProto

	// names used for variables and inputs: these are like internal outputs, but they come not from a node,
	// but from an input or variable. Used to introspect the graph.
	inputsNameSet       types.Set[string]
	variableNameToValue map[string]*protos.TensorProto

	name                        string
	InputsNames, OutputsNames   []string
	InputsShapes, OutputsShapes []DynamicShape

	// inputsAsConstants: see WithInputsAsConstants
	inputsAsConstants map[string]any
}

// Parse parses an ONNX model into an internal representation that can be used to build a GoMLX graph.
func Parse(contents []byte) (*Model, error) {
	m := &Model{}
	err := proto.Unmarshal(contents, &m.Proto)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse ONNX model proto")
	}

	// Parse inputs and outputs.
	m.name = m.Proto.Graph.Name
	m.inputsNameSet = types.MakeSet[string]()
	m.InputsNames = make([]string, len(m.Proto.Graph.Input))
	m.InputsShapes = make([]DynamicShape, len(m.Proto.Graph.Input))
	for ii, input := range m.Proto.Graph.Input {
		m.InputsNames[ii] = input.Name
		m.inputsNameSet.Insert(input.Name)

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

	// Set of variable names.
	m.variableNameToValue = make(map[string]*protos.TensorProto)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		m.variableNameToValue[tensorProto.Name] = tensorProto
	}

	// Maps the intermediary node outputs to the nodes that create them.
	m.nodeOutputToNode = make(map[string]*protos.NodeProto)
	for _, node := range m.Proto.Graph.Node {
		for _, outputName := range node.GetOutput() {
			if otherNode, found := m.nodeOutputToNode[outputName]; found {
				return nil, errors.Errorf("invalid graph: node output name %q used by 2 different nodes: (1) %s, (2) %s",
					outputName, nodeToString(otherNode), nodeToString(node))
			}
			m.nodeOutputToNode[outputName] = node
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
	m, err := Parse(contents)
	if err != nil {
		return nil, err
	}
	m.onnxFileName = filePath
	return m, nil
}

// Name of the model graph.
func (m *Model) Name() string { return m.name }

// Inputs returns the names and DynamicShapes of the inputs.
func (m *Model) Inputs() (names []string, dshapes []DynamicShape) {
	return m.InputsNames, m.InputsShapes
}

// Outputs returns a description of the outputs.
func (m *Model) Outputs() (names []string, dshapes []DynamicShape) {
	return m.OutputsNames, m.OutputsShapes
}

// NumInputs returns the number of inputs this graph takes.
func (m *Model) NumInputs() int {
	return len(m.InputsNames)
}

// WithInputsAsConstants marks inputs to be considered as constants, and not vary for different examples in training
// or inference.
// Use this just immediately after the creation of the Model. Later changes can cause inconsistencies.
//
// This makes them become constants in the graph, and they shouldn't be passed to CallGraph as inputs.
//
// The value each input maps to will be converted to a tensors.FromAnyValue.
func (m *Model) WithInputsAsConstants(inputsAsConstants map[string]any) *Model {
	m.inputsAsConstants = inputsAsConstants
	return m
}

// Write will write the ONNX model to the given writer (usually a file).
//
// This is useful, if the model variables were updated (e.g.: fine-tuning in GoMLX) and one wants to save the
// model.
// See ContextToONNX to copy over the variables in GoMLX's Context (presumably after some training/update) to the
// ONNX's model proto.
//
// See also Model.SaveToFile.
func (m *Model) Write(w io.Writer) error {
	content, err := proto.Marshal(m.Proto)
	if err != nil {
		return errors.Wrapf(err, "failed to serialize ONNX model proto")
	}
	_, err = w.Write(content)
	if err != nil {
		return errors.Wrapf(err, "failed to write serialized ONNX model proto")
	}
	return nil
}

// SaveToFile serializes the ONNX model to the given file.
//
// This is useful, if the model variables were updated (e.g.: fine-tuning in GoMLX) and one wants to save the
// model.
// See ContextToONNX to copy over the variables in GoMLX's Context (presumably after some training/update) to the
// ONNX's model proto.
func (m *Model) SaveToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return errors.Wrapf(err, "failed to save ONNX model proto to %s", path)
	}
	err = m.Write(f)
	if err != nil {
		_ = f.Close()
		return err
	}
	err = f.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to save ONNX model proto to %s", path)
	}
	return nil
}
