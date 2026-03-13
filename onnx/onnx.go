// Package onnx provides the public interface for ONNX models in GoMLX.
//
// Use onnx/parser to parse ONNX models from either the proto contents, or reading from a file.
package onnx

import (
	"io"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// DynamicDim is used for dynamic axes in a shape.
const DynamicDim = -1

// Model interface represents a parsed ONNX file.
// It can be used to generate the corresponding GoMLX model graph and executed for inference or used on a training loop for fine-tuning.
// It can also be used to populate a context with the variables of the ONNX model.
type Model interface {
	// Name of the model graph.
	Name() string

	// Close releases resources held by the model.
	Close() error

	// Inputs return the names and shapes of the inputs.
	// Shapes will return with a dimension set to DynamicDim (-1) for dynamic axes.
	Inputs() (names []string, dshapes []shapes.Shape)

	// Outputs return a description of the outputs.
	// Shapes will return with a dimension set to DynamicDim (-1) for dynamic axes.
	Outputs() (names []string, dshapes []shapes.Shape)

	// NumInputs returns the number of inputs this graph takes.
	NumInputs() int

	// WithInputsAsConstants marks inputs to be considered as constants and not vary for different examples in training or inference.
	WithInputsAsConstants(inputsAsConstants map[string]any)

	// AllowDTypePromotion enables automatic dtype promotion for operations with mismatched types.
	AllowDTypePromotion()

	// PrioritizeFloat16 configures dtype promotion to prefer Float16 over Float32.
	PrioritizeFloat16()

	// Write will write the ONNX model to the given writer (usually a file).
	Write(w io.Writer) error

	// SaveToFile serializes the ONNX model to the given file.
	SaveToFile(path string) error

	// CallGraph calls the ONNX graph, and hence are building it with GoMLX ops.
	CallGraph(ctx *context.Context, g *Graph, inputs map[string]*Node, outputNames ...string) (outputs []*Node)

	// VariablesToContext uploads all variable values from the ONNX model to the context.
	VariablesToContext(ctx *context.Context) error

	// FreeUnusedVariables frees variables that are not used in the graph.
	FreeUnusedVariables()

	// ContextToONNX copies over the variables in GoMLX's Context to the ONNX's model proto.
	ContextToONNX(ctx *context.Context) error

	// String implements fmt.Stringer.
	String() string

	// PrintGraph prints the model graph to the given writer.
	PrintGraph(writer io.Writer) error

	// PrintVariables prints the model variables to the given writer.
	PrintVariables(writer io.Writer) error

	// PrintGraphviz prints the model graph in Graphviz format to the given writer.
	PrintGraphviz(writer io.Writer, targets ...string) error

	// ShapeForName returns the shape of the given node output name.
	ShapeForName(name string) shapes.Shape

	// ValidateInputs checks the inputs has a shape that is compatible with the DynamicShapes of the inputs for the model.
	ValidateInputs(inputsShapes ...shapes.Shape) error

	// DisableFusion clears all detected fusions, forcing normal (unfused) conversion.
	DisableFusion()
}

// ModelScope is the default scope used for ONNX model variables in a GoMLX context.
const ModelScope = "ONNX"
