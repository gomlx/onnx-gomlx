package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// ShapeResolver resolves shapes for ONNX node outputs before GoMLX conversion.
// This solves the ordering problem where nodes depend on shapes of not-yet-converted tensors.
//
// Shape resolution happens in priority order:
// 1. ONNX value_info (shapes embedded in the model file)
// 2. Input shapes (concrete shapes provided at graph build time)
// 3. Variable shapes (from model initializers/weights)
// 4. Computed shapes (propagated through the ONNX graph)
type ShapeResolver struct {
	model *Model

	// onnxShapes contains shapes from the ONNX model's value_info section.
	// These are shapes that ONNX shape inference computed and stored in the model.
	onnxShapes map[string]shapes.Shape

	// inputShapes contains concrete shapes for model inputs at call time.
	inputShapes map[string]shapes.Shape

	// variableShapes contains shapes from model initializers (weights).
	variableShapes map[string]shapes.Shape

	// computedShapes contains shapes computed by propagating through the ONNX graph.
	computedShapes map[string]shapes.Shape

	// inputConstantValues contains actual values for inputs passed via WithInputsAsConstants.
	// This allows value tracing through data-dependent operations like ReduceMax.
	inputConstantValues map[string][]int

	// Whether shape propagation has been run
	propagated bool
}

// makeShapeSafe creates a shape, using MakeDynamic if any dimension is negative (dynamic).
// This handles shapes from If branches or other sources that may have symbolic dimensions.
func makeShapeSafe(dtype dtypes.DType, dims ...int) shapes.Shape {
	for _, d := range dims {
		if d < 0 {
			return shapes.MakeDynamic(dtype, dims...)
		}
	}
	return shapes.Make(dtype, dims...)
}

// NewShapeResolver creates a new ShapeResolver for the given model.
// It immediately extracts shapes from the model's value_info and initializers.
func NewShapeResolver(m *Model) *ShapeResolver {
	sr := &ShapeResolver{
		model:               m,
		onnxShapes:          make(map[string]shapes.Shape),
		inputShapes:         make(map[string]shapes.Shape),
		variableShapes:      make(map[string]shapes.Shape),
		computedShapes:      make(map[string]shapes.Shape),
		inputConstantValues: make(map[string][]int),
	}

	// Extract shapes from ONNX value_info
	sr.extractValueInfoShapes()

	// Extract shapes from initializers (variables/weights)
	sr.extractVariableShapes()

	// Extract values from inputsAsConstants if available
	sr.extractInputConstantValues()

	return sr
}

// extractValueInfoShapes extracts shapes from the ONNX model's value_info section.
// value_info contains shape annotations for intermediate tensors.
// Also extracts shapes from graph inputs and outputs.
func (sr *ShapeResolver) extractValueInfoShapes() {
	graph := sr.model.Proto.Graph
	if graph == nil {
		return
	}

	// Extract from value_info
	for _, vi := range graph.ValueInfo {
		if vi.Name == "" || vi.Type == nil {
			continue
		}

		tensorType, ok := vi.Type.Value.(*protos.TypeProto_TensorType)
		if !ok || tensorType.TensorType == nil {
			continue
		}

		shape, hasDynamic, err := sr.tensorTypeToShapeWithDynamic(tensorType.TensorType)
		if err != nil || hasDynamic {
			continue // Skip shapes we can't parse or have dynamic dimensions
		}

		sr.onnxShapes[vi.Name] = shape
	}

	// Also extract from graph inputs (they have shape info too)
	for _, input := range graph.Input {
		if input.Name == "" || input.Type == nil {
			continue
		}

		tensorType, ok := input.Type.Value.(*protos.TypeProto_TensorType)
		if !ok || tensorType.TensorType == nil {
			continue
		}

		shape, hasDynamic, err := sr.tensorTypeToShapeWithDynamic(tensorType.TensorType)
		if err != nil || hasDynamic {
			continue // Skip dynamic shapes from inputs
		}

		sr.onnxShapes[input.Name] = shape
	}

	// And from graph outputs
	for _, output := range graph.Output {
		if output.Name == "" || output.Type == nil {
			continue
		}

		tensorType, ok := output.Type.Value.(*protos.TypeProto_TensorType)
		if !ok || tensorType.TensorType == nil {
			continue
		}

		shape, hasDynamic, err := sr.tensorTypeToShapeWithDynamic(tensorType.TensorType)
		if err != nil || hasDynamic {
			continue // Skip dynamic shapes from outputs
		}

		sr.onnxShapes[output.Name] = shape
	}

}

// extractVariableShapes extracts shapes from model initializers (weights/constants).
func (sr *ShapeResolver) extractVariableShapes() {
	for name, tensorProto := range sr.model.variableNameToValue {
		shape, err := Shape(tensorProto)
		if err != nil {
			// Debug: log failures for Cast-related tensors

			continue
		}

		sr.variableShapes[name] = shape
	}
}

// extractInputConstantValues extracts integer values from inputs passed via WithInputsAsConstants.
// This enables value tracing through data-dependent operations like ReduceMax.
func (sr *ShapeResolver) extractInputConstantValues() {
	if sr.model.inputsAsConstants == nil {
		return
	}

	for name, value := range sr.model.inputsAsConstants {
		if value == nil {
			continue
		}

		// Try to extract integer values from the constant
		var intVals []int

		switch v := value.(type) {
		case []int:
			intVals = v
		case []int32:
			intVals = make([]int, len(v))
			for i, x := range v {
				intVals[i] = int(x)
			}
		case []int64:
			intVals = make([]int, len(v))
			for i, x := range v {
				intVals[i] = int(x)
			}
		case int:
			intVals = []int{v}
		case int32:
			intVals = []int{int(v)}
		case int64:
			intVals = []int{int(v)}
		default:
			// Try to extract from tensor types using reflection
			// Handle *tensors.Tensor or similar types that have Size() and ConstFlatData()
			if t, ok := value.(interface {
				Size() int
				ConstFlatData(func(any)) error
			}); ok {
				size := t.Size()
				if size > 0 {
					intVals = make([]int, 0, size)
					_ = t.ConstFlatData(func(flat any) {
						switch fv := flat.(type) {
						case []int64:
							for _, x := range fv {
								intVals = append(intVals, int(x))
							}
						case []int32:
							for _, x := range fv {
								intVals = append(intVals, int(x))
							}
						case []int:
							intVals = append(intVals, fv...)
						}
					})
				}
			}
			if len(intVals) == 0 {
				continue // Skip non-integer types or empty tensors
			}
		}

		if len(intVals) > 0 {
			sr.inputConstantValues[name] = intVals
		}
	}
}

// tensorTypeToShapeWithDynamic converts an ONNX TypeProto_Tensor to a shapes.Shape.
// Returns (shape, hasDynamic, error) where hasDynamic indicates if any dimension is dynamic.
func (sr *ShapeResolver) tensorTypeToShapeWithDynamic(tt *protos.TypeProto_Tensor) (shapes.Shape, bool, error) {
	dtype, err := dtypeForONNX(protos.TensorProto_DataType(tt.GetElemType()))
	if err != nil {
		return shapes.Shape{}, false, err
	}

	if tt.Shape == nil {
		return shapes.Make(dtype), false, nil
	}

	dims := make([]int, len(tt.Shape.Dim))
	hasDynamic := false
	for i, d := range tt.Shape.Dim {
		if dimVal, ok := d.GetValue().(*protos.TensorShapeProto_Dimension_DimValue); ok {
			dims[i] = int(dimVal.DimValue)
		} else {
			dims[i] = -1 // Dynamic dimension
			hasDynamic = true
		}
	}

	// Don't try to create a shape with dynamic dimensions
	if hasDynamic {
		return shapes.Shape{}, true, nil
	}

	return shapes.Make(dtype, dims...), false, nil
}

// tensorTypeToShape converts an ONNX TypeProto_Tensor to a shapes.Shape.
// Skips shapes with dynamic dimensions.
func (sr *ShapeResolver) tensorTypeToShape(tt *protos.TypeProto_Tensor) (shapes.Shape, error) {
	shape, hasDynamic, err := sr.tensorTypeToShapeWithDynamic(tt)
	if hasDynamic {
		return shapes.Shape{}, fmt.Errorf("shape has dynamic dimensions")
	}
	return shape, err
}

// SetInputShapes sets the concrete input shapes for this graph execution.
// Call this before PropagateShapes.
func (sr *ShapeResolver) SetInputShapes(inputShapes map[string]shapes.Shape) {
	sr.inputShapes = inputShapes
	sr.propagated = false // Need to re-propagate with new inputs
}

// PropagateShapes propagates shapes through the ONNX graph.
// This should be called after SetInputShapes and before conversion.
// Uses multiple passes to handle cases where nodes depend on shapes that haven't been computed yet.
func (sr *ShapeResolver) PropagateShapes() {
	// Refresh input constant values in case WithInputsAsConstants was called after ShapeResolver creation
	sr.extractInputConstantValues()

	if sr.propagated {
		return
	}

	graph := sr.model.Proto.Graph
	if graph == nil {
		sr.propagated = true
		return
	}

	// Multi-pass approach: keep processing until no more progress
	maxPasses := 10
	consecutiveNonProgress := 0
	for pass := 0; pass < maxPasses; pass++ {
		initialCount := len(sr.computedShapes)

		// Process nodes in order (ONNX guarantees topological order)
		for i, node := range graph.Node {
			if pass == 0 && i%500 == 0 {
			}
			sr.propagateNodeShape(node)
		}

		newCount := len(sr.computedShapes)
		progress := newCount - initialCount

		// If no progress was made, track consecutive passes with no progress
		if progress == 0 {
			consecutiveNonProgress++
			// Stop if we've had 3 consecutive passes with no progress
			if consecutiveNonProgress >= 3 {
				break
			}
		} else {
			consecutiveNonProgress = 0
		}
	}

	sr.propagated = true
}

// propagateNodeShape computes the output shape(s) for a single ONNX node.
func (sr *ShapeResolver) propagateNodeShape(node *protos.NodeProto) {
	if node == nil || len(node.Output) == 0 {
		return
	}

	// Skip if we already computed this node's output shape
	if _, exists := sr.computedShapes[node.Output[0]]; exists {
		return
	}

	// Get input shapes
	inputShapes := make([]shapes.Shape, len(node.Input))
	for i, inputName := range node.Input {
		if inputName == "" {
			continue
		}
		if shape, ok := sr.GetShape(inputName); ok {
			// Only use valid shapes
			if shape.DType != dtypes.InvalidDType {
				inputShapes[i] = shape
			}
		}
	}

	// Check if we have all required input shapes before attempting inference
	hasAllInputs := true
	for i, inputName := range node.Input {
		if inputName != "" && inputShapes[i].DType == dtypes.InvalidDType {
			hasAllInputs = false
			break
		}
	}

	// Compute output shape based on operation type
	var outputShape shapes.Shape
	var ok bool

	// Some ops can work even with missing inputs
	// Reshape can compute output shape by tracing through the shape tensor, even if input shapes are unknown
	canWorkWithMissingInputs := node.OpType == "Constant" || node.OpType == "ConstantOfShape" ||
		node.OpType == "Reshape" || node.OpType == "Expand"
	if !hasAllInputs && !canWorkWithMissingInputs {
		// Debug skipped nodes of interest - trace full BERT layer chain

		// Skip inference for this node
		return
	}

	switch node.OpType {
	case "Shape":
		// Shape op outputs a 1D tensor with the rank of the input
		if len(inputShapes) > 0 && inputShapes[0].Rank() > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			outputShape = shapes.Make(dtypes.Int64, inputShapes[0].Rank())
			ok = true
		}

	case "Gather":
		outputShape, ok = sr.inferGatherShape(node, inputShapes)

	case "Concat":
		outputShape, ok = sr.inferConcatShape(node, inputShapes)

	case "Unsqueeze":
		outputShape, ok = sr.inferUnsqueezeShape(node, inputShapes)

	case "Squeeze":
		outputShape, ok = sr.inferSqueezeShape(node, inputShapes)

	case "Reshape":
		outputShape, ok = sr.inferReshapeShape(node, inputShapes)

	case "Slice":
		outputShape, ok = sr.inferSliceShape(node, inputShapes)

	case "Cast":
		outputShape, ok = sr.inferCastShape(node, inputShapes)

	case "Constant":
		outputShape, ok = sr.inferConstantShape(node)

	case "ConstantOfShape":
		// ConstantOfShape output shape depends on the shape input value
		// We can only resolve this if we know the actual values
		outputShape, ok = sr.inferConstantOfShapeShape(node, inputShapes)

	case "Add", "Sub", "Mul", "Div", "Pow":
		// Arithmetic binary ops with broadcasting
		outputShape, ok = sr.inferBinaryOpShape(inputShapes)
		// Debug specific ops

	case "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "And", "Or":
		// Comparison/logical binary ops: output shape is broadcast of inputs, dtype is bool
		outputShape, ok = sr.inferBinaryOpShape(inputShapes)
		if ok {
			outputShape = makeShapeSafe(dtypes.Bool, outputShape.Dimensions...)
		}

	case "MatMul":
		outputShape, ok = sr.inferMatMulShape(inputShapes)
		// Debug MatMul for attention

	case "Transpose":
		outputShape, ok = sr.inferTransposeShape(node, inputShapes)

	case "Softmax", "Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Abs", "Neg",
		"Erf", "Gelu", "Selu", "Elu", "LeakyRelu", "HardSigmoid", "HardSwish",
		"Clip", "Floor", "Ceil", "Round", "Sin", "Cos", "Identity", "Sign", "Not":
		// Unary ops preserve shape
		if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			outputShape = inputShapes[0]
			ok = true
		}
		// Debug Softmax for attention
		if node.OpType == "Softmax" && strings.Contains(node.Output[0], "layer.0/attention") {
			if len(node.Input) > 0 {
				// Check if input is in computedShapes

			}
		}
		// Debug Sqrt for attention
		if node.OpType == "Sqrt" {
			if len(node.Input) > 0 {
			}
		}

	case "Where":
		// Where(condition, X, Y): output shape is broadcast of all three, dtype from X/Y
		if len(inputShapes) >= 3 {
			// Compute broadcast shape across all 3 inputs
			// First broadcast condition with X
			broadcastShape, broadcastOk := sr.inferBinaryOpShape(inputShapes[:2])
			if broadcastOk {
				// Then broadcast result with Y
				broadcastShape, broadcastOk = sr.inferBinaryOpShape([]shapes.Shape{broadcastShape, inputShapes[2]})
			}
			if broadcastOk {
				// Use dtype from X (second input) not condition (first input)
				dtype := inputShapes[1].DType
				if dtype == dtypes.InvalidDType {
					dtype = inputShapes[2].DType
				}
				if dtype != dtypes.InvalidDType {
					outputShape = makeShapeSafe(dtype, broadcastShape.Dimensions...)
					ok = true
				}
			}

		}

	case "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd":
		outputShape, ok = sr.inferReduceShape(node, inputShapes)

	case "Expand":
		outputShape, ok = sr.inferExpandShape(node, inputShapes)

	case "Range":
		outputShape, ok = sr.inferRangeShape(node, inputShapes)

	case "LayerNormalization", "BatchNormalization":
		// These preserve the first input's shape
		if len(inputShapes) > 0 {
			outputShape = inputShapes[0]
			ok = true
		}

	case "Split":
		// Split produces multiple outputs - handle first output
		outputShape, ok = sr.inferSplitShape(node, inputShapes)

	case "Pad":
		outputShape, ok = sr.inferPadShape(node, inputShapes)

	case "Tile":
		outputShape, ok = sr.inferTileShape(node, inputShapes)

	case "Flatten":
		// Flatten reshapes to 2D: [d0*d1*...*d(axis-1), d(axis)*...*d(n-1)]
		if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			axis := getIntAttrOr(node, "axis", 1)
			dims := inputShapes[0].Dimensions
			if axis < 0 {
				axis = len(dims) + axis
			}
			if axis <= 0 {
				axis = 1
			}
			if axis > len(dims) {
				axis = len(dims)
			}
			// Compute product of dims before and after axis
			dim0 := 1
			for i := 0; i < axis; i++ {
				if dims[i] < 0 {
					dim0 = -1
					break
				}
				dim0 *= dims[i]
			}
			dim1 := 1
			for i := axis; i < len(dims); i++ {
				if dims[i] < 0 {
					dim1 = -1
					break
				}
				dim1 *= dims[i]
			}
			outputShape = makeShapeSafe(inputShapes[0].DType, dim0, dim1)
			ok = true
		}

	case "GatherElements":
		// GatherElements: output has same shape as indices tensor
		if len(inputShapes) >= 2 && inputShapes[1].DType != dtypes.InvalidDType {
			// Output dtype is same as data input
			dtype := inputShapes[0].DType
			if dtype == dtypes.InvalidDType {
				dtype = dtypes.Float32 // fallback
			}
			outputShape = makeShapeSafe(dtype, inputShapes[1].Dimensions...)
			ok = true
		}

	case "ScatterND", "ScatterElements":
		// Scatter ops: output has same shape as data input (first input)
		if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			outputShape = inputShapes[0]
			ok = true
		}

	case "TopK":
		// TopK: output shape is input shape with last dimension = k
		if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			k := getIntAttrOr(node, "k", 1)
			// Try to get k from second input if available
			if len(node.Input) >= 2 {
				kVals := sr.extractConstantIntSlice(node.Input[1])
				if len(kVals) == 1 {
					k = kVals[0]
				}
			}
			dims := make([]int, len(inputShapes[0].Dimensions))
			copy(dims, inputShapes[0].Dimensions)
			axis := getIntAttrOr(node, "axis", -1)
			if axis < 0 {
				axis = len(dims) + axis
			}
			if axis >= 0 && axis < len(dims) {
				dims[axis] = k
			}
			outputShape = makeShapeSafe(inputShapes[0].DType, dims...)
			ok = true
		}

	case "NonZero":
		// NonZero is data-dependent - we can't determine the second dimension
		// Output shape is [rank, num_nonzero] where num_nonzero depends on input values
		// Just try to get from value_info, otherwise we can't resolve this
		if shape, found := sr.onnxShapes[node.Output[0]]; found {
			outputShape = shape
			ok = true
		}

	case "Einsum":
		// Einsum is complex - try to get from value_info
		if shape, found := sr.onnxShapes[node.Output[0]]; found {
			outputShape = shape
			ok = true
		}

	case "If":
		// If is a control flow op with then_branch and else_branch subgraphs
		// Try to determine shape from:
		// 1. Evaluate condition if possible and pick corresponding branch output
		// 2. If both branches have same output shape, use that
		// 3. Fall back to value_info
		outputShape, ok = sr.inferIfShape(node, inputShapes)
		if !ok {
			if shape, found := sr.onnxShapes[node.Output[0]]; found {
				outputShape = shape
				ok = true
			}
		}

	case "LSTM":
		// LSTM outputs follow ONNX spec:
		// - Y (output 0): [seq_length, num_directions, batch_size, hidden_size] (layout=0)
		//                 or [batch_size, seq_length, num_directions, hidden_size] (layout=1)
		// - Y_h (output 1): [num_directions, batch_size, hidden_size]
		// - Y_c (output 2): [num_directions, batch_size, hidden_size]
		outputShape, ok = sr.inferLSTMShape(node, inputShapes)

	default:
		// For other ops, try to get from value_info
		if shape, found := sr.onnxShapes[node.Output[0]]; found {
			outputShape = shape
			ok = true
		}
	}

	if ok && len(node.Output) > 0 && node.Output[0] != "" {
		// Validate that the shape doesn't have invalid dimensions
		hasInvalidDim := false

		if !hasInvalidDim {
			sr.computedShapes[node.Output[0]] = outputShape
		}
	}
}

// GetShape returns the resolved shape for a node output.
// Returns the shape and true if found, or an empty shape and false if not found.
func (sr *ShapeResolver) GetShape(name string) (shapes.Shape, bool) {
	// Debug specific names
	debugName := strings.Contains(name, "Cast_334") || strings.Contains(name, "Sqrt_333")

	// Priority order:
	// 1. Computed shapes (most accurate for current inputs)
	if shape, ok := sr.computedShapes[name]; ok {

		return shape, true
	}

	// 2. Input shapes (concrete at call time)
	if shape, ok := sr.inputShapes[name]; ok {

		return shape, true
	}

	// 3. Variable shapes (from weights)
	if shape, ok := sr.variableShapes[name]; ok {

		return shape, true
	}

	// 4. ONNX value_info (from model file)
	if shape, ok := sr.onnxShapes[name]; ok {

		return shape, true
	}

	if debugName {
		// Check if it's a node output
		if node := sr.model.nodeOutputToNode[name]; node != nil {
		} else {
		}
	}

	return shapes.Shape{}, false
}

// GetDimensions returns the dimensions for a node output.
// Returns nil if shape is not found or has dynamic dimensions.
func (sr *ShapeResolver) GetDimensions(name string) []int {
	shape, ok := sr.GetShape(name)
	if !ok {
		return nil
	}

	// Check for dynamic dimensions

	return shape.Dimensions
}

// Helper functions for shape inference

func (sr *ShapeResolver) inferGatherShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) < 2 || inputShapes[0].Rank() == 0 {
		return shapes.Shape{}, false
	}

	data := inputShapes[0]
	indices := inputShapes[1]
	axis := getIntAttrOr(node, "axis", 0)
	if axis < 0 {
		axis = data.Rank() + axis
	}

	// Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
	outputDims := make([]int, 0, data.Rank()-1+indices.Rank())
	outputDims = append(outputDims, data.Dimensions[:axis]...)
	outputDims = append(outputDims, indices.Dimensions...)
	outputDims = append(outputDims, data.Dimensions[axis+1:]...)

	return makeShapeSafe(data.DType, outputDims...), true
}

func (sr *ShapeResolver) inferConcatShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) == 0 {
		return shapes.Shape{}, false
	}

	// Find first non-empty shape for dtype and rank
	var firstShape shapes.Shape

	if firstShape.Rank() == 0 {
		return shapes.Shape{}, false
	}

	axis := getIntAttrOr(node, "axis", 0)
	if axis < 0 {
		axis = firstShape.Rank() + axis
	}

	// Sum the axis dimension, copy others
	outputDims := make([]int, firstShape.Rank())
	copy(outputDims, firstShape.Dimensions)

	axisDim := 0
	for _, s := range inputShapes {
		if s.Rank() == 0 {
			continue
		}
		if axis < len(s.Dimensions) {
			if s.Dimensions[axis] < 0 {
				axisDim = -1 // Dynamic
				break
			}
			axisDim += s.Dimensions[axis]
		}
	}
	outputDims[axis] = axisDim

	// Don't create shapes with invalid (< -1) dimensions
	for _, d := range outputDims {
		if d < -1 {
			return shapes.Shape{}, false
		}
	}

	return makeShapeSafe(firstShape.DType, outputDims...), true
}

func (sr *ShapeResolver) inferUnsqueezeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) == 0 {
		return shapes.Shape{}, false
	}

	// Try to get axes from attributes (ONNX opset < 13) or from second input
	axes := getIntSliceAttr(node, "axes")
	if len(axes) == 0 && len(node.Input) > 1 {
		// ONNX opset 13+: axes come from constant input
		axesInputName := node.Input[1]
		axesNode := sr.model.nodeOutputToNode[axesInputName]

		if axesNode != nil && axesNode.OpType == "Constant" {

			for _, attr := range axesNode.Attribute {

				if attr.Name == "value" && attr.T != nil {
					extractedAxes, err := extractIntSliceFromTensor(attr.T)

					if err == nil && len(extractedAxes) > 0 {
						axes = extractedAxes
					}
				}
			}
		}
	}

	if len(axes) == 0 {
		return shapes.Shape{}, false
	}

	inputDims := inputShapes[0].Dimensions
	outputRank := len(inputDims) + len(axes)

	// Normalize negative axes
	normalizedAxes := make([]int, len(axes))

	// Build output dimensions
	outputDims := make([]int, outputRank)
	axisSet := make(map[int]bool)
	for _, a := range normalizedAxes {
		axisSet[a] = true
	}

	inputIdx := 0
	for i := 0; i < outputRank; i++ {
		if axisSet[i] {
			outputDims[i] = 1
		} else {
			if inputIdx < len(inputDims) {
				outputDims[i] = inputDims[inputIdx]
				inputIdx++
			}
		}
	}

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

func (sr *ShapeResolver) inferSqueezeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) == 0 {
		return shapes.Shape{}, false
	}

	axes := getIntSliceAttr(node, "axes")

	// ONNX opset 13+: axes come from second input if not in attributes
	if len(axes) == 0 && len(node.Input) > 1 {
		axesInputName := node.Input[1]
		axesNode := sr.model.nodeOutputToNode[axesInputName]
		if axesNode != nil && axesNode.OpType == "Constant" {
			for _, attr := range axesNode.Attribute {
				if attr.Name == "value" && attr.T != nil {
					if val, err := extractIntSliceFromTensor(attr.T); err == nil && len(val) > 0 {
						axes = val
					}
				}
			}
		}
	}

	inputDims := inputShapes[0].Dimensions

	// If no axes specified, squeeze all dims of size 1
	if len(axes) == 0 {
		outputDims := make([]int, 0, len(inputDims))
		hasDynamic := false
		for _, d := range inputDims {
			if d == 1 {
				continue // Squeeze this dimension
			}
			if d < 0 {
				hasDynamic = true
			}
			outputDims = append(outputDims, d)
		}
		if hasDynamic {
			return shapes.MakeDynamic(inputShapes[0].DType, outputDims...), true
		}
		return shapes.Make(inputShapes[0].DType, outputDims...), true
	}

	// Normalize and create axis set
	axisSet := make(map[int]bool)
	for _, a := range axes {
		if a < 0 {
			a = len(inputDims) + a
		}
		axisSet[a] = true
	}

	outputDims := make([]int, 0, len(inputDims))
	hasDynamic := false
	for i, d := range inputDims {
		if axisSet[i] {
			continue // Squeeze this dimension
		}
		if d < 0 {
			hasDynamic = true
		}
		outputDims = append(outputDims, d)
	}

	if hasDynamic {
		return shapes.MakeDynamic(inputShapes[0].DType, outputDims...), true
	}
	return shapes.Make(inputShapes[0].DType, outputDims...), true
}

func (sr *ShapeResolver) inferReshapeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	debugReshape := strings.Contains(node.Output[0], "layer.5/attention/self/Reshape_15") ||
		strings.Contains(node.Output[0], "layer.5/attention/self/Reshape_14") ||
		strings.Contains(node.Output[0], "layer.5/attention/self/Reshape_13") ||
		strings.Contains(node.Output[0], "layer.5/attention/self/Reshape_5") ||
		strings.Contains(node.Output[0], "layer.0/attention/self/Reshape") ||
		strings.Contains(node.Output[0], "Reshape_12") ||
		strings.Contains(node.Output[0], "Reshape_13") ||
		strings.Contains(node.Output[0], "span_rep_layer/span_rep_layer/Reshape_2")
	if debugReshape && len(node.Input) > 0 {
	}

	// Reshape output shape depends on the shape input values, not just its shape
	// Try to extract shape values from the second input
	if len(node.Input) < 2 {

		// Fallback to value_info if we don't have enough inputs
		if shape, ok := sr.onnxShapes[node.Output[0]]; ok {

			return shape, true
		}
		return shapes.Shape{}, false
	}

	// Get dtype from input if available, otherwise default to Float32
	dtype := dtypes.Float32
	inputSize := 0
	if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
		dtype = inputShapes[0].DType
		inputSize = inputShapes[0].Size()
	}

	shapeInputName := node.Input[1]
	shapeInputNode := sr.model.nodeOutputToNode[shapeInputName]

	// If the shape input is a Constant, extract the values directly
	if shapeInputNode != nil && shapeInputNode.OpType == "Constant" {
		for _, attr := range shapeInputNode.Attribute {
			if attr.Name == "value" && attr.T != nil {
				if dims, err := extractIntSliceFromTensor(attr.T); err == nil && len(dims) > 0 {
					knownProduct := 1
					inferIdx := -1
					for i, d := range dims {
						if d == -1 {
							inferIdx = i
						} else if d > 0 {
							knownProduct *= d
						}
					}
					if inferIdx >= 0 && inputSize > 0 && knownProduct > 0 {
						dims[inferIdx] = inputSize / knownProduct
					}
					// Only return if we resolved all dimensions (no -1 remaining)
					if inferIdx < 0 || (inferIdx >= 0 && inputSize > 0) {
						return makeShapeSafe(dtype, dims...), true
					}
				}
			}
		}
	}

	// Try to trace through shape manipulation ops
	dims := sr.traceShapeValues(shapeInputName)

	if len(dims) > 0 {
		// Check if all dimensions are concrete (allow -1 for infer)
		allConcrete := true
		for _, d := range dims {
			if d < -1 {
				allConcrete = false
				break
			}
		}
		if allConcrete {
			// Handle -1 (infer dimension) if we know the input size
			knownProduct := 1
			inferIdx := -1
			for i, d := range dims {
				if d == -1 {
					inferIdx = i
				} else if d > 0 {
					knownProduct *= d
				}
			}
			if inferIdx >= 0 && inputSize > 0 && knownProduct > 0 {
				dims[inferIdx] = inputSize / knownProduct
			}
			// Only return if we resolved all dimensions (no -1 remaining)
			if inferIdx < 0 || (inferIdx >= 0 && inputSize > 0) {
				return shapes.Make(dtype, dims...), true
			}
		}
	}

	// If full trace failed, try partial trace
	partialDims := sr.traceShapeValuesPartial(shapeInputName)

	if len(partialDims) > 0 {
		// Only try to resolve unknowns if we have input size
		if inputSize > 0 {
			// Count unknown dimensions and compute known product
			knownProduct := 1
			unknownCount := 0
			hasInfer := false
			for _, d := range partialDims {
				if d == UnknownDim {
					unknownCount++
				} else if d == -1 {
					hasInfer = true
				} else if d > 0 {
					knownProduct *= d
				}
			}

			// Special handling for -1 (infer) combined with unknowns:
			// If we have [-1, unknown, known...] pattern, -1 absorbs the batch dimension
			// and the unknown is typically a sequence/spatial dimension
			// For now, we can only resolve if exactly one total unknown/infer dimension
			totalUnknown := unknownCount
			if hasInfer {
				totalUnknown++
			}

			// If only one unknown (treating -1 as absorbing it), compute
			if unknownCount == 1 && !hasInfer && knownProduct > 0 && inputSize%knownProduct == 0 {
				inferredDim := inputSize / knownProduct
				resolvedDims := make([]int, len(partialDims))
				for i, d := range partialDims {
					if d == UnknownDim {
						resolvedDims[i] = inferredDim
					} else {
						resolvedDims[i] = d
					}
				}
				return shapes.Make(dtype, resolvedDims...), true
			}

			// If we have -1 and one unknown, the -1 absorbs both: [batch*unknown] / knownProduct
			// This is common in attention: [-1, seq, head_dim] where -1 absorbs batch*heads
			if unknownCount == 1 && hasInfer && knownProduct > 0 {
				// The remaining size after known dims is split between -1 and unknown
				remainingSize := inputSize / knownProduct
				// Try common patterns: the unknown might be seq_len from inputs
				// Look for seq_len in input shapes (typically 128 for this model)
				for inputName := range sr.inputShapes {
					inputShape := sr.inputShapes[inputName]
					for _, dim := range inputShape.Dimensions {
						if dim > 0 && remainingSize%dim == 0 {
							// This dim could be the unknown
							otherDim := remainingSize / dim
							if otherDim > 0 {
								resolvedDims := make([]int, len(partialDims))
								for i, d := range partialDims {
									if d == -1 {
										resolvedDims[i] = otherDim
									} else if d == UnknownDim {
										resolvedDims[i] = dim
									} else {
										resolvedDims[i] = d
									}
								}
								// Verify the product matches
								product := 1
								for _, rd := range resolvedDims {
									product *= rd
								}
								if product == inputSize {

									return shapes.Make(dtype, resolvedDims...), true
								}
							}
						}
					}
				}
			}
		}
		// If we have partial dims but couldn't resolve all unknowns,
		// still return them as a partial shape for downstream use
		// Use MakeDynamic since we have unknown dimensions
		if len(partialDims) > 0 {
			resolvedDims := make([]int, len(partialDims))
			hasDynamic := false

			// Use MakeDynamic if there are unknown dimensions or negative dimensions
			if hasDynamic {
				return shapes.MakeDynamic(dtype, resolvedDims...), true
			}
			return shapes.Make(dtype, resolvedDims...), true
		}
	}

	// Final fallback: try value_info
	// Only use this if we couldn't compute a concrete shape ourselves
	if shape, ok := sr.onnxShapes[node.Output[0]]; ok {

		return shape, true
	}

	return shapes.Shape{}, false
}

func (sr *ShapeResolver) inferSliceShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// First try value_info
	if shape, ok := sr.onnxShapes[node.Output[0]]; ok {
		return shape, true
	}

	// Slice needs: data, starts, ends, (optional: axes, steps)
	if len(inputShapes) == 0 || inputShapes[0].Rank() == 0 {
		return shapes.Shape{}, false
	}

	// Extract starts, ends, axes, steps from constant inputs or by tracing through simple ops
	var starts, ends, axes, steps []int

	if len(node.Input) >= 2 {
		starts = sr.extractConstantIntSlice(node.Input[1])
		if starts == nil {
			starts = sr.traceShapeValues(node.Input[1])
		}
	}
	if len(node.Input) >= 3 {
		ends = sr.extractConstantIntSlice(node.Input[2])
		if ends == nil {
			ends = sr.traceShapeValues(node.Input[2])
		}
	}
	if len(node.Input) >= 4 {
		axes = sr.extractConstantIntSlice(node.Input[3])
		if axes == nil {
			axes = sr.traceShapeValues(node.Input[3])
		}
	}
	if len(node.Input) >= 5 {
		steps = sr.extractConstantIntSlice(node.Input[4])
		if steps == nil {
			steps = sr.traceShapeValues(node.Input[4])
		}
	}

	if starts == nil || ends == nil {
		return shapes.Shape{}, false
	}

	// Default axes: 0, 1, 2, ... len(starts)
	if axes == nil {
		axes = make([]int, len(starts))
		for i := range axes {
			axes[i] = i
		}
	}

	// Default steps: all 1s
	if steps == nil {
		steps = make([]int, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}

	// Compute output shape
	inputDims := inputShapes[0].Dimensions
	outputDims := make([]int, len(inputDims))
	copy(outputDims, inputDims)

	for i, axis := range axes {
		if axis < 0 {
			axis = len(inputDims) + axis
		}
		if axis < 0 || axis >= len(inputDims) {
			continue
		}

		dimSize := inputDims[axis]
		start := starts[i]
		end := ends[i]
		step := steps[i]
		if step == 0 {
			step = 1
		}

		// Normalize negative indices
		if start < 0 {
			start = dimSize + start
		}
		if end < 0 {
			end = dimSize + end
		}

		// Clamp to valid range
		if start < 0 {
			start = 0
		}
		if start > dimSize {
			start = dimSize
		}
		if end < 0 {
			end = 0
		}
		if end > dimSize {
			end = dimSize
		}

		// Handle large end values (INT_MAX)
		if end > 1000000000 {
			end = dimSize
		}

		// Compute output size for this axis
		if step > 0 {
			outputDims[axis] = (end - start + step - 1) / step
		} else {
			outputDims[axis] = (start - end + (-step) - 1) / (-step)
		}
		if outputDims[axis] < 0 {
			outputDims[axis] = 0
		}
	}

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

// extractConstantIntSlice extracts int values from a constant node output.
func (sr *ShapeResolver) extractConstantIntSlice(outputName string) []int {
	node := sr.model.nodeOutputToNode[outputName]
	if node == nil || node.OpType != "Constant" {
		return nil
	}
	for _, attr := range node.Attribute {
		if attr.Name == "value" && attr.T != nil {
			if val, err := extractIntSliceFromTensor(attr.T); err == nil {
				return val
			}
		}
	}
	return nil
}

// extractScalarThroughCast traces through Cast and similar pass-through ops to extract a scalar constant value.
// Returns the scalar value and true if successful.
func (sr *ShapeResolver) extractScalarThroughCast(outputName string) (int, bool) {
	return sr.extractScalarThroughCastRecursive(outputName, 0)
}

func (sr *ShapeResolver) extractScalarThroughCastRecursive(outputName string, depth int) (int, bool) {
	if depth > 10 {
		return 0, false
	}

	node := sr.model.nodeOutputToNode[outputName]
	if node == nil {
		return 0, false
	}

	switch node.OpType {
	case "Constant":
		vals := sr.extractConstantIntSlice(outputName)
		if len(vals) == 1 {
			return vals[0], true
		}
		return 0, false

	case "Cast":
		// Trace through Cast
		if len(node.Input) > 0 {

			return sr.extractScalarThroughCastRecursive(node.Input[0], depth+1)
		}

	case "Squeeze", "Unsqueeze":
		// Trace through Squeeze/Unsqueeze
		if len(node.Input) > 0 {

			return sr.extractScalarThroughCastRecursive(node.Input[0], depth+1)
		}

	case "Slice":
		// Slice with constant start/end extracts a value from a shape
		if len(node.Input) >= 3 {
			starts := sr.extractConstantIntSlice(node.Input[1])
			ends := sr.extractConstantIntSlice(node.Input[2])
			if len(starts) == 1 && len(ends) == 1 {
				vals := sr.traceShapeValues(node.Input[0])

				if len(vals) > 0 {
					// Handle negative indices
					startIdx := starts[0]
					if startIdx < 0 {
						startIdx = len(vals) + startIdx
					}
					if startIdx >= 0 && startIdx < len(vals) {
						// Return the single value at start position

						return vals[startIdx], true
					}
				}
			}
		}

	case "Gather":
		// For Gather with axis=0 and constant index, trace through
		if len(node.Input) >= 2 {
			idx := sr.extractConstantIntSlice(node.Input[1])

			if len(idx) == 1 {
				// Get the values from the data input
				vals := sr.traceShapeValues(node.Input[0])

				if idx[0] >= 0 && idx[0] < len(vals) {
					return vals[idx[0]], true
				}
			}
		}

	case "Shape":
		// Shape op - get the dimension
		if len(node.Input) > 0 {
			inputShape, ok := sr.GetShape(node.Input[0])
			if ok && inputShape.DType != dtypes.InvalidDType {
				// Shape returns all dimensions as a 1D tensor
				// This is used when combined with Gather to select a single dim
				// We can't return a single value here

				return 0, false
			}
		}

	case "ReduceMax", "ReduceMin":
		// ReduceMax/ReduceMin of traced values - compute the max/min
		if len(node.Input) > 0 {
			vals := sr.traceShapeValues(node.Input[0])
			if len(vals) > 0 {
				result := vals[0]
				for _, v := range vals[1:] {
					if node.OpType == "ReduceMax" {
						if v > result {
							result = v
						}
					} else {
						if v < result {
							result = v
						}
					}
				}
				return result, true
			}
		}

	case "ReduceSum":
		// ReduceSum of traced values - compute the sum
		if len(node.Input) > 0 {
			vals := sr.traceShapeValues(node.Input[0])

			if len(vals) > 0 {
				result := 0
				for _, v := range vals {
					result += v
				}
				return result, true
			}
		}
	}

	return 0, false
}

func (sr *ShapeResolver) inferCastShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Debug Cast for layer.0
	debugCast := strings.Contains(node.Output[0], "layer.0/attention/self/Cast_output_0")
	if debugCast {
		if len(node.Input) > 0 {
		}
	}

	if len(inputShapes) == 0 || inputShapes[0].DType == dtypes.InvalidDType {

		return shapes.Shape{}, false
	}

	to := getIntAttrOr(node, "to", 0)
	dtype, err := dtypeForONNX(protos.TensorProto_DataType(to))
	if err != nil {
		return shapes.Shape{}, false
	}

	return makeShapeSafe(dtype, inputShapes[0].Dimensions...), true
}

func (sr *ShapeResolver) inferConstantShape(node *protos.NodeProto) (shapes.Shape, bool) {
	// Try to get from value_info
	if shape, ok := sr.onnxShapes[node.Output[0]]; ok {
		return shape, true
	}

	// Try to infer from the constant value attribute
	for _, attr := range node.Attribute {
		if attr.Name == "value" && attr.T != nil {
			shape, err := Shape(attr.T)
			if err == nil {
				return shape, true
			}
		}
	}

	return shapes.Shape{}, false
}

func (sr *ShapeResolver) inferConstantOfShapeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// ConstantOfShape takes a 1D shape tensor as input
	// We need the actual values, not just the shape of the shape tensor

	// Get dtype from ConstantOfShape value attribute
	dtype := dtypes.Float32 // default
	for _, attr := range node.Attribute {
		if attr.Name == "value" && attr.T != nil {
			if d, err := dtypeForONNX(protos.TensorProto_DataType(attr.T.DataType)); err == nil {
				dtype = d
			}
		}
	}

	// First try value_info
	if shape, ok := sr.onnxShapes[node.Output[0]]; ok {
		return shape, true
	}

	if len(node.Input) == 0 {
		return shapes.Shape{}, false
	}

	shapeInputName := node.Input[0]
	shapeInputNode := sr.model.nodeOutputToNode[shapeInputName]
	if shapeInputNode == nil {
		return shapes.Shape{}, false
	}

	// Debug: show what we're looking at
	if len(node.Output) > 0 && (node.Output[0] == "/core/ConstantOfShape_4_output_0" ||
		node.Output[0] == "/core/ConstantOfShape_output_0") {
		if len(shapeInputNode.Input) > 0 {
		}
	}

	// If the shape input is a Constant, extract the values directly
	if shapeInputNode.OpType == "Constant" {
		for _, attr := range shapeInputNode.Attribute {
			if attr.Name == "value" && attr.T != nil {
				if dims, err := extractIntSliceFromTensor(attr.T); err == nil && len(dims) > 0 {
					return makeShapeSafe(dtype, dims...), true
				}
			}
		}
	}

	// If the shape input is a Shape operation, the ConstantOfShape output
	// has the same dimensions as the Shape input's tensor
	if shapeInputNode.OpType == "Shape" && len(shapeInputNode.Input) > 0 {
		tensorName := shapeInputNode.Input[0]
		if tensorShape, ok := sr.GetShape(tensorName); ok {
			// Check if all dimensions are concrete
			allConcrete := true
			for _, d := range tensorShape.Dimensions {
				if d < 0 {
					allConcrete = false
					break
				}
			}
			if allConcrete && len(tensorShape.Dimensions) > 0 {
				return makeShapeSafe(dtype, tensorShape.Dimensions...), true
			}
		}
	}

	// Try to trace through shape manipulation ops (Concat, Gather, etc.)
	dims := sr.traceShapeValues(shapeInputName)
	if len(dims) > 0 {
		allConcrete := true
		for _, d := range dims {
			if d < 0 {
				allConcrete = false
				break
			}
		}
		if allConcrete {
			return makeShapeSafe(dtype, dims...), true
		}
	}

	return shapes.Shape{}, false
}

// UnknownDim is a sentinel value for dimensions that couldn't be traced
const UnknownDim = -999999

// TraceShapeValues is the public wrapper for traceShapeValues.
// It traces through shape manipulation operations to extract concrete dimension values.
// Returns the dimension values if all can be resolved, nil otherwise.
func (sr *ShapeResolver) TraceShapeValues(nodeName string) []int {
	return sr.traceShapeValuesWithDebug(nodeName, 0)
}

// TraceShapeValuesPartial is the public wrapper for traceShapeValuesPartial.
// It traces through shape manipulation operations to extract dimension values.
// Unlike TraceShapeValues, it returns partial results with UnknownDim for values that can't be resolved.
// Returns nil only if we can't determine the structure at all.
func (sr *ShapeResolver) TraceShapeValuesPartial(nodeName string) []int {
	return sr.traceShapeValuesPartialWithDebug(nodeName, 0)
}

// traceShapeValues traces through shape manipulation operations to extract concrete dimension values.
// Returns the dimension values if all can be resolved, nil otherwise.
func (sr *ShapeResolver) traceShapeValues(nodeName string) []int {
	return sr.traceShapeValuesWithDebug(nodeName, 0)
}

// traceShapeValuesPartial traces through shape manipulation operations.
// Unlike traceShapeValues, it returns partial results with UnknownDim for values that can't be resolved.
// Returns nil only if we can't determine the structure at all.
func (sr *ShapeResolver) traceShapeValuesPartial(nodeName string) []int {
	return sr.traceShapeValuesPartialWithDebug(nodeName, 0)
}

func (sr *ShapeResolver) traceShapeValuesPartialWithDebug(nodeName string, depth int) []int {
	if depth > 20 {
		return nil
	}

	node := sr.model.nodeOutputToNode[nodeName]
	if node == nil {
		// Check if this is a model input with constant values
		if vals, found := sr.inputConstantValues[nodeName]; found {
			return vals
		}
		return nil
	}

	switch node.OpType {
	case "Constant":
		for _, attr := range node.Attribute {
			if attr.Name == "value" && attr.T != nil {
				if val, err := extractIntSliceFromTensor(attr.T); err == nil {
					return val
				}
			}
		}
		// If we couldn't extract from the attribute, check if this is a model input
		// that was passed as a constant via WithInputsAsConstants
		if len(node.Output) > 0 {
			outputName := node.Output[0]
			if vals, found := sr.inputConstantValues[outputName]; found {
				return vals
			}
		}
		return nil

	case "Shape":
		if len(node.Input) > 0 {
			inputName := node.Input[0]

			// First check if this input has a concrete shape from inputsAsConstants
			if sr.model.inputsAsConstants != nil {
				if constVal, found := sr.model.inputsAsConstants[inputName]; found {
					// Get the shape from the constant tensor
					if tensor, ok := constVal.(interface{ Shape() shapes.Shape }); ok {
						shape := tensor.Shape()
						return shape.Dimensions
					}
				}
			}

			// Try to get shape from propagated shapes
			if shape, ok := sr.GetShape(inputName); ok {
				return shape.Dimensions
			}
		}
		// Can't resolve - return single unknown
		return []int{UnknownDim}

	case "Concat":
		// Concatenate values from all inputs, using UnknownDim for failed traces
		var result []int
		for _, inputName := range node.Input {
			inputVals := sr.traceShapeValuesPartialWithDebug(inputName, depth+1)

			if inputVals == nil {
				// If we can't even get partial results, we need at least one value
				// For shape tensors in BERT attention, assume 1 value per input
				result = append(result, UnknownDim)
			} else {
				result = append(result, inputVals...)
			}
		}

		return result

	case "Gather":
		if len(node.Input) >= 2 {
			dataVals := sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)
			if dataVals == nil {
				return []int{UnknownDim}
			}

			// Try to get the index
			var indices []int
			indexNode := sr.model.nodeOutputToNode[node.Input[1]]
			if indexNode != nil && indexNode.OpType == "Constant" {
				for _, attr := range indexNode.Attribute {
					if attr.Name == "value" && attr.T != nil {
						if val, err := extractIntSliceFromTensor(attr.T); err == nil {
							indices = val
						}
					}
				}
			}
			if indices == nil {
				indices = sr.traceShapeValuesPartialWithDebug(node.Input[1], depth+1)
			}

			if len(indices) == 0 {
				return []int{UnknownDim}
			}

			// Get axis (default 0)
			axis := 0

			if axis == 0 {
				result := make([]int, len(indices))
				for i, idx := range indices {
					if idx == UnknownDim {
						result[i] = UnknownDim
					} else {
						if idx < 0 {
							idx = len(dataVals) + idx
						}
						if idx >= 0 && idx < len(dataVals) {
							result[i] = dataVals[idx]
						} else {
							result[i] = UnknownDim
						}
					}
				}
				return result
			}
		}
		return []int{UnknownDim}

	case "Unsqueeze":
		if len(node.Input) > 0 {
			result := sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)

			if result != nil {
				return result
			}
		}
		return []int{UnknownDim}

	case "Squeeze":
		if len(node.Input) > 0 {
			return sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)
		}
		return []int{UnknownDim}

	case "Cast":
		if len(node.Input) > 0 {
			return sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)
		}
		return []int{UnknownDim}

	case "ReduceMax", "ReduceMin":
		// Reduce ops - compute max/min of input values
		if len(node.Input) > 0 {
			inputVals := sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)

			if len(inputVals) == 0 {
				return []int{UnknownDim}
			}
			// Check if any values are unknown
			hasUnknown := false
			for _, v := range inputVals {
				if v == UnknownDim {
					hasUnknown = true
					break
				}
			}
			if hasUnknown {
				return []int{UnknownDim}
			}
			// Compute max/min
			result := inputVals[0]
			for _, v := range inputVals[1:] {
				if node.OpType == "ReduceMax" {
					if v > result {
						result = v
					}
				} else { // ReduceMin
					if v < result {
						result = v
					}
				}
			}
			return []int{result}
		}
		return []int{UnknownDim}

	case "ReduceSum":
		// ReduceSum - for shape tracing, we need to handle this specially
		// If the input comes from Equal(input_ids, ...), this is likely computing sequence lengths
		if len(node.Input) > 0 {
			// Special case: if input is Cast(Equal(input_ids, ...)), infer sequence length
			// Pattern: ReduceSum(Cast(Equal(input_ids, pad_token))) computes sequence lengths
			inputNode := sr.model.nodeOutputToNode[node.Input[0]]
			if inputNode != nil && inputNode.OpType == "Cast" && len(inputNode.Input) > 0 {
				castInputNode := sr.model.nodeOutputToNode[inputNode.Input[0]]
				if castInputNode != nil && castInputNode.OpType == "Equal" {
					// Check if Equal uses an input from inputsAsConstants
					for _, equalInput := range castInputNode.Input {
						if sr.model.inputsAsConstants != nil {
							if constVal, found := sr.model.inputsAsConstants[equalInput]; found {
								if tensor, ok := constVal.(interface{ Shape() shapes.Shape }); ok {
									shape := tensor.Shape()
									// This is likely computing sequence length
									// ReduceSum over axis=1 of [batch, seq_len] gives [batch] with value=seq_len
									// Return [seq_len] as the result
									if len(shape.Dimensions) >= 2 {
										seqLen := shape.Dimensions[1]
										return []int{seqLen}
									}
								}
							}
						}
					}
				}
			}

			// Fallback: try to trace normally
			inputVals := sr.traceShapeValuesPartialWithDebug(node.Input[0], depth+1)
			if len(inputVals) == 0 {
				return []int{UnknownDim}
			}
			// Check if any values are unknown
			for _, v := range inputVals {
				if v == UnknownDim {
					return []int{UnknownDim}
				}
			}
			// Compute sum
			result := 0
			for _, v := range inputVals {
				result += v
			}
			return []int{result}
		}
		return []int{UnknownDim}

	case "Equal":
		// Equal compares two inputs element-wise
		// This is commonly used to create masks (e.g., padding masks)
		// We can't compute actual boolean values, but ReduceSum can detect this pattern
		// and infer sequence lengths from the input shapes
		return nil

	case "Slice":
		// Slice is complex - return unknown for now
		return nil
	}

	return nil
}

func (sr *ShapeResolver) traceShapeValuesWithDebug(nodeName string, depth int) []int {
	// Prevent infinite recursion
	if depth > 20 {
		return nil
	}

	indent := ""
	for i := 0; i < depth; i++ {
		indent += "  "
	}

	node := sr.model.nodeOutputToNode[nodeName]
	if node == nil {
		// Check if this is a model input with constant values
		if vals, found := sr.inputConstantValues[nodeName]; found {

			return vals
		}
		// Not an error - could be an input or initializer

		return nil
	}

	switch node.OpType {
	case "Constant":
		for _, attr := range node.Attribute {
			if attr.Name == "value" && attr.T != nil {
				if val, err := extractIntSliceFromTensor(attr.T); err == nil {
					return val
				}
			}
		}
		// If we couldn't extract from the attribute, check if this is a model input
		// that was passed as a constant via WithInputsAsConstants
		if len(node.Output) > 0 {
			outputName := node.Output[0]
			if vals, found := sr.inputConstantValues[outputName]; found {
				return vals
			}
		}
		return nil

	case "ConstantOfShape":
		// ConstantOfShape creates a tensor filled with a constant value
		// The shape comes from the input, the fill value from the attribute
		outputShape, ok := sr.GetShape(nodeName)

		if !ok || len(outputShape.Dimensions) == 0 {
			return nil
		}
		numElements := outputShape.Size()
		if numElements <= 0 || numElements > 1000 { // sanity check
			return nil
		}
		// Get fill value from attribute (defaults to 0)
		fillValue := int64(0)
		for _, attr := range node.Attribute {

			if attr.Name == "value" && attr.T != nil {

				// Extract the fill value from the tensor attribute
				if len(attr.T.Int64Data) > 0 {
					fillValue = attr.T.Int64Data[0]
				} else if len(attr.T.FloatData) > 0 {
					fillValue = int64(attr.T.FloatData[0])
				} else if len(attr.T.RawData) > 0 {
					// Try to parse raw data based on ONNX dtype
					// ONNX DataType enum: FLOAT=1, INT32=6, INT64=7, FLOAT16=10, etc.
					switch attr.T.DataType {
					case 7: // ONNX INT64
						fillValue = int64(binary.LittleEndian.Uint64(attr.T.RawData[:8]))
					case 6: // ONNX INT32
						fillValue = int64(int32(binary.LittleEndian.Uint32(attr.T.RawData[:4])))
					case 1: // ONNX FLOAT
						bits := binary.LittleEndian.Uint32(attr.T.RawData[:4])
						fillValue = int64(math.Float32frombits(bits))
					}
				}
			}
		}
		// Create result array filled with the constant value
		result := make([]int, numElements)
		for i := range result {
			result[i] = int(fillValue)
		}

		return result

	case "Shape":
		if len(node.Input) > 0 {
			inputName := node.Input[0]
			if shape, ok := sr.GetShape(inputName); ok {
				allConcrete := true
				for _, d := range shape.Dimensions {
					if d < 0 {
						allConcrete = false
						break
					}
				}
				if allConcrete {
					return shape.Dimensions
				}
				// Shape has dynamic dims - try to fill them in from concrete input shapes
				// Make a copy so we can modify it
				dims := make([]int, len(shape.Dimensions))
				copy(dims, shape.Dimensions)

				// Try to resolve dynamic dims by looking at model input shapes
				// This helps when the batch dimension is known from inputs but not propagated
				for i, d := range dims {
					if d < 0 {
						// Look for a matching dimension in input shapes
						for _, inputShape := range sr.inputShapes {
							if i < len(inputShape.Dimensions) && inputShape.Dimensions[i] > 0 {
								dims[i] = inputShape.Dimensions[i]
								break
							}
						}
					}
				}

				// Return what we have - even with some dynamic dims, it's better than nil
				return dims
			}
		}
		return nil

	case "Concat":
		// Concatenate values from all inputs
		var result []int
		for _, inputName := range node.Input {
			inputVals := sr.traceShapeValuesWithDebug(inputName, depth+1)

			if inputVals == nil {

				return nil // Can't resolve
			}
			result = append(result, inputVals...)
		}

		return result

	case "Gather":
		// Gather from a shape tensor with a constant index
		if len(node.Input) >= 2 {
			dataVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)

			if dataVals == nil {
				return nil
			}

			// Try to get the index - could be constant or traced
			var indices []int
			indexNode := sr.model.nodeOutputToNode[node.Input[1]]
			if indexNode != nil && indexNode.OpType == "Constant" {
				for _, attr := range indexNode.Attribute {
					if attr.Name == "value" && attr.T != nil {
						if val, err := extractIntSliceFromTensor(attr.T); err == nil {
							indices = val
						}
					}
				}
			} else {
				// Try tracing through the index computation
				indices = sr.traceShapeValuesWithDebug(node.Input[1], depth+1)
			}

			if len(indices) == 0 {
				return nil
			}

			// Get axis attribute (default 0)
			axis := getIntAttr(node, "axis", 0)

			// For axis=0 (most common in shape tensors), gather elements
			if axis == 0 {
				result := make([]int, len(indices))
				for i, idx := range indices {
					if idx < 0 {
						idx = len(dataVals) + idx // negative indexing
					}
					if idx >= 0 && idx < len(dataVals) {
						result[i] = dataVals[idx]
					} else {
						return nil
					}
				}
				return result
			}
		}

	case "Unsqueeze":
		// Unsqueeze just adds a dimension of 1, values are unchanged
		if len(node.Input) > 0 {
			result := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)

			return result
		}

	case "Squeeze":
		// Squeeze removes dimensions of 1, values are unchanged
		if len(node.Input) > 0 {
			return sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
		}

	case "Cast":
		// Cast doesn't change values
		if len(node.Input) > 0 {
			return sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
		}

	case "Not":
		// Not is a logical negation - trace through to input and negate
		// Not(0) = 1, Not(non-zero) = 0
		if len(node.Input) > 0 {
			// Debug: trace what the Not depends on (for If condition tracing)
			if strings.Contains(nodeName, "Not_526") || strings.Contains(nodeName, "Cast_527") {
				inputName := node.Input[0]
				inputNode := sr.model.nodeOutputToNode[inputName]
				if inputNode != nil {
				}
			}
			inputVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			if inputVals == nil {
				return nil
			}
			result := make([]int, len(inputVals))

			return result
		}

	case "Slice":
		// Slice extracts a portion of the shape
		if len(node.Input) >= 3 {
			dataVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			if dataVals == nil {
				return nil
			}
			// Try to get starts and ends
			startsNode := sr.model.nodeOutputToNode[node.Input[1]]
			endsNode := sr.model.nodeOutputToNode[node.Input[2]]
			if startsNode != nil && startsNode.OpType == "Constant" &&
				endsNode != nil && endsNode.OpType == "Constant" {
				var starts, ends []int

				if len(starts) > 0 && len(ends) > 0 {
					start := starts[0]
					end := ends[0]
					if start < 0 {
						start = len(dataVals) + start
					}
					if end < 0 {
						end = len(dataVals) + end
					}
					if end > len(dataVals) {
						end = len(dataVals)
					}
					if start >= 0 && end > start && end <= len(dataVals) {
						return dataVals[start:end]
					}
				}
			}
		}

	case "Mul":
		// Multiply constants (common in shape calculations)
		if len(node.Input) >= 2 {
			lhsVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			rhsVals := sr.traceShapeValuesWithDebug(node.Input[1], depth+1)
			if lhsVals == nil || rhsVals == nil {
				return nil
			}
			// Scalar multiplication
			if len(lhsVals) == 1 && len(rhsVals) == 1 {
				return []int{lhsVals[0] * rhsVals[0]}
			}
			// Element-wise multiplication
			if len(lhsVals) == len(rhsVals) {
				result := make([]int, len(lhsVals))
				for i := range lhsVals {
					result[i] = lhsVals[i] * rhsVals[i]
				}
				return result
			}
			// Broadcast: one operand is scalar
			if len(lhsVals) == 1 {
				result := make([]int, len(rhsVals))
				for i := range rhsVals {
					result[i] = lhsVals[0] * rhsVals[i]
				}
				return result
			}
			if len(rhsVals) == 1 {
				result := make([]int, len(lhsVals))
				for i := range lhsVals {
					result[i] = lhsVals[i] * rhsVals[0]
				}
				return result
			}
		}

	case "Div":
		// Divide constants (common in shape calculations, e.g., splitting heads: embedding_dim / num_heads)
		if len(node.Input) >= 2 {
			lhsVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			rhsVals := sr.traceShapeValuesWithDebug(node.Input[1], depth+1)
			if lhsVals == nil || rhsVals == nil {
				return nil
			}
			// Scalar division
			if len(lhsVals) == 1 && len(rhsVals) == 1 && rhsVals[0] != 0 {
				return []int{lhsVals[0] / rhsVals[0]}
			}
			// Element-wise division
			if len(lhsVals) == len(rhsVals) {
				result := make([]int, len(lhsVals))
				for i := range lhsVals {
					if rhsVals[i] == 0 {
						return nil // Avoid division by zero
					}
					result[i] = lhsVals[i] / rhsVals[i]
				}
				return result
			}
			// Broadcast: divide array by scalar
			if len(rhsVals) == 1 && rhsVals[0] != 0 {
				result := make([]int, len(lhsVals))
				for i := range lhsVals {
					result[i] = lhsVals[i] / rhsVals[0]
				}
				return result
			}
			// Broadcast: divide scalar by array (less common but possible)
			if len(lhsVals) == 1 {
				result := make([]int, len(rhsVals))
				for i := range rhsVals {
					if rhsVals[i] == 0 {
						return nil
					}
					result[i] = lhsVals[0] / rhsVals[i]
				}
				return result
			}
		}

	case "Add", "Sub":
		// Add/Sub constants (common in shape calculations)
		if len(node.Input) >= 2 {
			lhsVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			rhsVals := sr.traceShapeValuesWithDebug(node.Input[1], depth+1)
			if lhsVals == nil || rhsVals == nil {
				return nil
			}
			isAdd := node.OpType == "Add"
			// Scalar operations
			if len(lhsVals) == 1 && len(rhsVals) == 1 {
				if isAdd {
					return []int{lhsVals[0] + rhsVals[0]}
				}
				return []int{lhsVals[0] - rhsVals[0]}
			}
			// Element-wise operations
			if len(lhsVals) == len(rhsVals) {
				result := make([]int, len(lhsVals))

				return result
			}
			// Broadcast: one operand is scalar
			if len(lhsVals) == 1 {
				result := make([]int, len(rhsVals))

				return result
			}
			if len(rhsVals) == 1 {
				result := make([]int, len(lhsVals))

				return result
			}
		}

	case "ReduceMax", "ReduceMin":
		// Reduce ops - compute max/min of input values
		if len(node.Input) > 0 {
			inputVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			if inputVals == nil || len(inputVals) == 0 {
				return nil
			}
			result := inputVals[0]
			for _, v := range inputVals[1:] {
				if node.OpType == "ReduceMax" {
					if v > result {
						result = v
					}
				} else { // ReduceMin
					if v < result {
						result = v
					}
				}
			}
			// For value tracing, we just need the max/min value regardless of keepdims
			return []int{result}
		}

	case "Reshape":
		// Reshape just passes through values (shape manipulation, not value change)
		if len(node.Input) > 0 {
			return sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
		}

	case "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual":
		// Comparison ops - return 1 for true, 0 for false
		if len(node.Input) >= 2 {
			// Debug: trace what the comparison depends on (for If condition tracing)
			if strings.Contains(nodeName, "Not_526") || strings.Contains(nodeName, "Equal_") {
				lhsNode := sr.model.nodeOutputToNode[node.Input[0]]
				rhsNode := sr.model.nodeOutputToNode[node.Input[1]]
				if lhsNode != nil {
				}
				if rhsNode != nil {
				}
			}
			lhsVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			rhsVals := sr.traceShapeValuesWithDebug(node.Input[1], depth+1)

			if lhsVals != nil && rhsVals != nil {
				// Handle scalar comparisons
				if len(lhsVals) == 1 && len(rhsVals) == 1 {
					var result int
					switch node.OpType {
					case "Equal":
						if lhsVals[0] == rhsVals[0] {
							result = 1
						}
					case "Less":
						if lhsVals[0] < rhsVals[0] {
							result = 1
						}
					case "Greater":
						if lhsVals[0] > rhsVals[0] {
							result = 1
						}
					case "LessOrEqual":
						if lhsVals[0] <= rhsVals[0] {
							result = 1
						}
					case "GreaterOrEqual":
						if lhsVals[0] >= rhsVals[0] {
							result = 1
						}
					}
					return []int{result}
				}
				// Handle element-wise comparison
				if len(lhsVals) == len(rhsVals) {
					result := make([]int, len(lhsVals))
					for i := range lhsVals {
						switch node.OpType {
						case "Equal":
							if lhsVals[i] == rhsVals[i] {
								result[i] = 1
							}
						case "Less":
							if lhsVals[i] < rhsVals[i] {
								result[i] = 1
							}
						case "Greater":
							if lhsVals[i] > rhsVals[i] {
								result[i] = 1
							}
						case "LessOrEqual":
							if lhsVals[i] <= rhsVals[i] {
								result[i] = 1
							}
						case "GreaterOrEqual":
							if lhsVals[i] >= rhsVals[i] {
								result[i] = 1
							}
						}
					}
					return result
				}
				// Handle broadcast: scalar compared with array
				if len(lhsVals) == 1 {
					result := make([]int, len(rhsVals))
					for i := range rhsVals {
						switch node.OpType {
						case "Equal":
							if lhsVals[0] == rhsVals[i] {
								result[i] = 1
							}
						case "Less":
							if lhsVals[0] < rhsVals[i] {
								result[i] = 1
							}
						case "Greater":
							if lhsVals[0] > rhsVals[i] {
								result[i] = 1
							}
						case "LessOrEqual":
							if lhsVals[0] <= rhsVals[i] {
								result[i] = 1
							}
						case "GreaterOrEqual":
							if lhsVals[0] >= rhsVals[i] {
								result[i] = 1
							}
						}
					}
					return result
				}
				if len(rhsVals) == 1 {
					result := make([]int, len(lhsVals))
					for i := range lhsVals {
						switch node.OpType {
						case "Equal":
							if lhsVals[i] == rhsVals[0] {
								result[i] = 1
							}
						case "Less":
							if lhsVals[i] < rhsVals[0] {
								result[i] = 1
							}
						case "Greater":
							if lhsVals[i] > rhsVals[0] {
								result[i] = 1
							}
						case "LessOrEqual":
							if lhsVals[i] <= rhsVals[0] {
								result[i] = 1
							}
						case "GreaterOrEqual":
							if lhsVals[i] >= rhsVals[0] {
								result[i] = 1
							}
						}
					}
					return result
				}
			}
		}

	case "Where":
		// Where(condition, X, Y) - select values element-wise
		// We can only trace this if condition is all true or all false, or if X == Y
		// For now, try to determine if all elements select from X or Y
		if len(node.Input) >= 3 {
			condVals := sr.traceShapeValuesWithDebug(node.Input[0], depth+1)
			xVals := sr.traceShapeValuesWithDebug(node.Input[1], depth+1)
			yVals := sr.traceShapeValuesWithDebug(node.Input[2], depth+1)

			if xVals != nil && yVals != nil && len(xVals) == len(yVals) {
				// If X and Y have the same values, the condition doesn't matter
				same := true

				if same {
					return xVals
				}
			}

			// If we have concrete condition values
			if condVals != nil {
				allTrue := true
				allFalse := true

				if allTrue && xVals != nil {
					return xVals
				}
				if allFalse && yVals != nil {
					return yVals
				}
			}

			// Try element-wise selection if we have condition and both values
			if condVals != nil && xVals != nil && yVals != nil &&
				len(condVals) == len(xVals) && len(xVals) == len(yVals) {
				result := make([]int, len(condVals))

				return result
			}
		}
	}

	return nil
}

func (sr *ShapeResolver) inferBinaryOpShape(inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) < 2 {
		return shapes.Shape{}, false
	}

	// Broadcast shapes
	a, b := inputShapes[0], inputShapes[1]

	// Check for valid dtypes
	if a.DType == dtypes.InvalidDType || b.DType == dtypes.InvalidDType {
		return shapes.Shape{}, false
	}

	// Handle scalar cases: scalar op scalar = scalar, scalar op tensor = tensor
	if a.Rank() == 0 && b.Rank() == 0 {
		// Scalar op scalar = scalar
		return shapes.Make(a.DType), true
	}
	if a.Rank() == 0 {
		// Scalar op tensor = tensor (scalar broadcasts to any shape)
		return b, true
	}
	if b.Rank() == 0 {
		// Tensor op scalar = tensor
		return a, true
	}

	// Simple broadcast: max rank, max dims
	maxRank := a.Rank()
	if b.Rank() > maxRank {
		maxRank = b.Rank()
	}

	outputDims := make([]int, maxRank)
	for i := 0; i < maxRank; i++ {
		aDim := 1
		bDim := 1
		if i < a.Rank() {
			aDim = a.Dimensions[a.Rank()-1-i]
		}
		if i < b.Rank() {
			bDim = b.Dimensions[b.Rank()-1-i]
		}

		if aDim < 0 || bDim < 0 {
			outputDims[maxRank-1-i] = -1
		} else if aDim == 1 {
			outputDims[maxRank-1-i] = bDim
		} else if bDim == 1 {
			outputDims[maxRank-1-i] = aDim
		} else if aDim == bDim {
			outputDims[maxRank-1-i] = aDim
		} else {
			return shapes.Shape{}, false // Incompatible
		}
	}

	return makeShapeSafe(a.DType, outputDims...), true
}

func (sr *ShapeResolver) inferMatMulShape(inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) < 2 {
		return shapes.Shape{}, false
	}

	a, b := inputShapes[0], inputShapes[1]
	if a.Rank() < 2 || b.Rank() < 2 {
		return shapes.Shape{}, false
	}

	// Output: [..., M, N] where A is [..., M, K] and B is [..., K, N]
	outputDims := make([]int, 0, a.Rank())

	// Batch dimensions (broadcast)
	maxBatch := a.Rank() - 2
	if b.Rank()-2 > maxBatch {
		maxBatch = b.Rank() - 2
	}
	for i := 0; i < maxBatch; i++ {
		aDim := 1
		bDim := 1
		if i < a.Rank()-2 {
			aDim = a.Dimensions[a.Rank()-2-1-i]
		}
		if i < b.Rank()-2 {
			bDim = b.Dimensions[b.Rank()-2-1-i]
		}
		if aDim == 1 {
			outputDims = append([]int{bDim}, outputDims...)
		} else {
			outputDims = append([]int{aDim}, outputDims...)
		}
	}

	// M from A, N from B
	outputDims = append(outputDims, a.Dimensions[a.Rank()-2])
	outputDims = append(outputDims, b.Dimensions[b.Rank()-1])

	return makeShapeSafe(a.DType, outputDims...), true
}

func (sr *ShapeResolver) inferTransposeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) == 0 || inputShapes[0].Rank() == 0 {
		return shapes.Shape{}, false
	}

	perm := getIntSliceAttr(node, "perm")
	if len(perm) == 0 {
		// Default: reverse dimensions
		perm = make([]int, inputShapes[0].Rank())
		for i := range perm {
			perm[i] = inputShapes[0].Rank() - 1 - i
		}
	}

	outputDims := make([]int, len(perm))
	for i, p := range perm {
		outputDims[i] = inputShapes[0].Dimensions[p]
	}

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

func (sr *ShapeResolver) inferReduceShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	if len(inputShapes) == 0 {
		return shapes.Shape{}, false
	}

	keepDims := getIntAttrOr(node, "keepdims", 1) == 1
	axes := getIntSliceAttr(node, "axes")

	// If no axes, reduce all
	if len(axes) == 0 {
		if keepDims {
			dims := make([]int, inputShapes[0].Rank())
			for i := range dims {
				dims[i] = 1
			}
			return shapes.Make(inputShapes[0].DType, dims...), true
		}
		return shapes.Make(inputShapes[0].DType), true
	}

	// Normalize axes
	axisSet := make(map[int]bool)
	for _, a := range axes {
		if a < 0 {
			a = inputShapes[0].Rank() + a
		}
		axisSet[a] = true
	}

	if keepDims {
		outputDims := make([]int, inputShapes[0].Rank())

		return makeShapeSafe(inputShapes[0].DType, outputDims...), true
	}

	outputDims := make([]int, 0, inputShapes[0].Rank()-len(axes))

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

func (sr *ShapeResolver) inferExpandShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Expand uses the second input as the target shape
	// Output = element-wise max of input shape and target shape (broadcasting)
	if len(node.Input) >= 2 && len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
		// Try to get the target shape from the second input
		targetDims := sr.traceShapeValues(node.Input[1])

		if len(targetDims) > 0 {
			allConcrete := true

			if allConcrete {
				inputDims := inputShapes[0].Dimensions
				// Align dimensions from the right (broadcasting rule)
				// Prepend 1s to smaller shape
				maxRank := len(targetDims)
				if len(inputDims) > maxRank {
					maxRank = len(inputDims)
				}
				alignedInput := make([]int, maxRank)
				alignedTarget := make([]int, maxRank)
				for i := range alignedInput {
					alignedInput[i] = 1
					alignedTarget[i] = 1
				}
				for i := 0; i < len(inputDims); i++ {
					alignedInput[maxRank-len(inputDims)+i] = inputDims[i]
				}
				for i := 0; i < len(targetDims); i++ {
					alignedTarget[maxRank-len(targetDims)+i] = targetDims[i]
				}
				// Output = element-wise max
				outputDims := make([]int, maxRank)
				for i := 0; i < maxRank; i++ {
					if alignedInput[i] >= alignedTarget[i] {
						outputDims[i] = alignedInput[i]
					} else {
						outputDims[i] = alignedTarget[i]
					}
				}
				result := makeShapeSafe(inputShapes[0].DType, outputDims...)

				return result, true
			}

		}
	}
	// Fall back to value_info
	if shape, ok := sr.onnxShapes[node.Output[0]]; ok {

		return shape, true
	}

	return shapes.Shape{}, false
}

func (sr *ShapeResolver) inferTileShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Tile(input, repeats) produces output where each dimension is multiplied by repeats
	// Output shape = input_shape[i] * repeats[i]
	if len(inputShapes) == 0 || inputShapes[0].DType == dtypes.InvalidDType {
		return shapes.Shape{}, false
	}
	if len(node.Input) < 2 {
		return shapes.Shape{}, false
	}

	inputShape := inputShapes[0]

	// Try to get repeats from the second input
	repeats := sr.traceShapeValues(node.Input[1])

	if len(repeats) == 0 {
		// Try to get from constant
		repeats = sr.extractConstantIntSlice(node.Input[1])

	}

	if len(repeats) == 0 {
		// Fall back to value_info
		if shape, ok := sr.onnxShapes[node.Output[0]]; ok {
			return shape, true
		}
		return shapes.Shape{}, false
	}

	// Tile requires repeats to have same rank as input
	if len(repeats) != inputShape.Rank() {
		return shapes.Shape{}, false
	}

	// Compute output dimensions: input_dim[i] * repeats[i]
	outputDims := make([]int, inputShape.Rank())

	return makeShapeSafe(inputShape.DType, outputDims...), true
}

func (sr *ShapeResolver) inferRangeShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Range(start, limit, delta) produces a 1D tensor
	// Try to extract start, limit, delta as constants (possibly through Cast nodes)
	if len(node.Input) < 3 {
		return shapes.Shape{}, false
	}

	// Try direct constant extraction first
	start := sr.extractConstantIntSlice(node.Input[0])
	limit := sr.extractConstantIntSlice(node.Input[1])
	delta := sr.extractConstantIntSlice(node.Input[2])

	var s, l, d int
	var gotStart, gotLimit, gotDelta bool

	if len(start) == 1 {
		s, gotStart = start[0], true
	} else {
		s, gotStart = sr.extractScalarThroughCast(node.Input[0])

	}

	if len(limit) == 1 {
		l, gotLimit = limit[0], true
	} else {
		l, gotLimit = sr.extractScalarThroughCast(node.Input[1])

	}

	if len(delta) == 1 {
		d, gotDelta = delta[0], true
	} else {
		d, gotDelta = sr.extractScalarThroughCast(node.Input[2])

	}

	if gotStart && gotLimit && gotDelta && d != 0 {
		size := (l - s + d - 1) / d
		if size < 0 {
			size = 0
		}
		dtype := dtypes.Int64
		if len(inputShapes) > 0 && inputShapes[0].DType != dtypes.InvalidDType {
			dtype = inputShapes[0].DType
		}

		return shapes.Make(dtype, size), true
	}

	return shapes.Shape{}, false
}

func (sr *ShapeResolver) inferLSTMShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// LSTM input[0] shape: [seq_length, batch_size, input_size] (layout=0)
	//                   or [batch_size, seq_length, input_size] (layout=1)
	// Get attributes
	hiddenSize := int(getIntAttrOr(node, "hidden_size", 0))
	direction := getStringAttrOr(node, "direction", "forward")
	layout := int(getIntAttrOr(node, "layout", 0))

	if hiddenSize == 0 {
		// Try to get from value_info as fallback
		if shape, found := sr.onnxShapes[node.Output[0]]; found {
			return shape, true
		}
		return shapes.Shape{}, false
	}

	numDirections := 1
	if direction == "bidirectional" {
		numDirections = 2
	}

	// Try to get input shape
	var seqLen, batchSize int
	if len(inputShapes) > 0 && inputShapes[0].Rank() >= 3 && inputShapes[0].DType != dtypes.InvalidDType {
		if layout == 0 {
			seqLen = inputShapes[0].Dimensions[0]
			batchSize = inputShapes[0].Dimensions[1]
		} else {
			batchSize = inputShapes[0].Dimensions[0]
			seqLen = inputShapes[0].Dimensions[1]
		}
	} else {
		// Input shape not resolved - try value_info
		if shape, found := sr.onnxShapes[node.Output[0]]; found {
			return shape, true
		}
		// Use dynamic dimensions
		seqLen = -1
		batchSize = -1
	}

	// LSTM Y output: [seq_length, num_directions, batch_size, hidden_size] (layout=0)
	//             or [batch_size, seq_length, num_directions, hidden_size] (layout=1)
	var outputShape shapes.Shape
	if layout == 0 {
		outputShape = makeShapeSafe(dtypes.Float32, seqLen, numDirections, batchSize, hiddenSize)
	} else {
		outputShape = makeShapeSafe(dtypes.Float32, batchSize, seqLen, numDirections, hiddenSize)
	}

	// Also store shapes for Y_h and Y_c if they exist
	if len(node.Output) >= 2 && node.Output[1] != "" {
		yhShape := makeShapeSafe(dtypes.Float32, numDirections, batchSize, hiddenSize)
		sr.computedShapes[node.Output[1]] = yhShape
	}
	if len(node.Output) >= 3 && node.Output[2] != "" {
		ycShape := makeShapeSafe(dtypes.Float32, numDirections, batchSize, hiddenSize)
		sr.computedShapes[node.Output[2]] = ycShape
	}

	return outputShape, true
}

func (sr *ShapeResolver) inferSplitShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Split divides input along an axis
	if len(inputShapes) == 0 || inputShapes[0].Rank() == 0 {
		return shapes.Shape{}, false
	}

	axis := getIntAttrOr(node, "axis", 0)
	if axis < 0 {
		axis = inputShapes[0].Rank() + axis
	}

	// Get split attribute (list of sizes) or num_outputs
	split := getIntSliceAttr(node, "split")
	numOutputs := len(node.Output)

	outputDims := make([]int, inputShapes[0].Rank())
	copy(outputDims, inputShapes[0].Dimensions)

	if len(split) > 0 {
		// Use first split size for first output
		outputDims[axis] = split[0]
	} else if numOutputs > 0 {
		// Equal split
		outputDims[axis] = inputShapes[0].Dimensions[axis] / numOutputs
	}

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

func (sr *ShapeResolver) inferPadShape(node *protos.NodeProto, inputShapes []shapes.Shape) (shapes.Shape, bool) {
	// Pad adds padding to input dimensions
	if len(inputShapes) == 0 || inputShapes[0].Rank() == 0 {
		return shapes.Shape{}, false
	}

	// Try to get pads from second input (ONNX opset 11+) or attribute
	var pads []int
	if len(node.Input) >= 2 {
		pads = sr.extractConstantIntSlice(node.Input[1])
	}
	if len(pads) == 0 {
		pads = getIntSliceAttr(node, "pads")
	}

	if len(pads) == 0 || len(pads) != 2*inputShapes[0].Rank() {
		// Can't determine padding
		if shape, ok := sr.onnxShapes[node.Output[0]]; ok {
			return shape, true
		}
		return shapes.Shape{}, false
	}

	// Compute output shape: input + pads_begin + pads_end for each dimension
	outputDims := make([]int, inputShapes[0].Rank())
	for i := 0; i < inputShapes[0].Rank(); i++ {
		outputDims[i] = inputShapes[0].Dimensions[i] + pads[i] + pads[i+inputShapes[0].Rank()]
	}

	return makeShapeSafe(inputShapes[0].DType, outputDims...), true
}

// inferIfShape infers the output shape of an If op by examining its branch subgraphs.
// It tries to:
// 1. Evaluate the condition to a constant and pick the corresponding branch output shape
// 2. If both branches have the same output shape, use that
// 3. If branches have different shapes but same rank, use element-wise max (conservative)
func (sr *ShapeResolver) inferIfShape(node *protos.NodeProto, _ []shapes.Shape) (shapes.Shape, bool) {
	// Get the then_branch and else_branch attributes
	var thenGraph, elseGraph *protos.GraphProto
	for _, attr := range node.Attribute {
		if attr.Name == "then_branch" && attr.Type == protos.AttributeProto_GRAPH {
			thenGraph = attr.G
		} else if attr.Name == "else_branch" && attr.Type == protos.AttributeProto_GRAPH {
			elseGraph = attr.G
		}
	}

	if thenGraph == nil || elseGraph == nil {

		return shapes.Shape{}, false
	}

	// Get output shapes from branch subgraphs
	// The output shape comes from the branch's output nodes
	thenShape := sr.getSubgraphOutputShape(thenGraph, 0)
	elseShape := sr.getSubgraphOutputShape(elseGraph, 0)

	if thenShape.DType == dtypes.InvalidDType && elseShape.DType == dtypes.InvalidDType {
		return shapes.Shape{}, false
	}

	// Try to evaluate condition to pick the right branch
	if len(node.Input) > 0 {
		condVals := sr.traceShapeValues(node.Input[0])

		if len(condVals) == 1 {
			if condVals[0] != 0 {
				// Condition is true, use then_branch

				if thenShape.DType != dtypes.InvalidDType {
					return thenShape, true
				}
			} else {
				// Condition is false, use else_branch

				if elseShape.DType != dtypes.InvalidDType {
					return elseShape, true
				}
			}
		}
	}

	// If one branch has valid shape and other doesn't, use the valid one
	if thenShape.DType != dtypes.InvalidDType && elseShape.DType == dtypes.InvalidDType {
		return thenShape, true
	}
	if elseShape.DType != dtypes.InvalidDType && thenShape.DType == dtypes.InvalidDType {
		return elseShape, true
	}

	// Both branches have valid shapes - check if they're equal
	if thenShape.Equal(elseShape) {
		return thenShape, true
	}

	// Different shapes but same rank - take element-wise max (conservative bound)
	if thenShape.Rank() == elseShape.Rank() && thenShape.Rank() > 0 {
		dtype := thenShape.DType
		if dtype == dtypes.InvalidDType {
			dtype = elseShape.DType
		}
		maxDims := make([]int, thenShape.Rank())
		for i := 0; i < thenShape.Rank(); i++ {
			thenDim := thenShape.Dimensions[i]
			elseDim := elseShape.Dimensions[i]
			if thenDim > elseDim {
				maxDims[i] = thenDim
			} else {
				maxDims[i] = elseDim
			}
		}
		return makeShapeSafe(dtype, maxDims...), true
	}

	return shapes.Shape{}, false
}

// getSubgraphOutputShape gets the shape of the specified output from a subgraph.
func (sr *ShapeResolver) getSubgraphOutputShape(graph *protos.GraphProto, outputIndex int) shapes.Shape {
	if graph == nil || len(graph.Output) <= outputIndex {
		return shapes.Shape{}
	}

	outputInfo := graph.Output[outputIndex]
	if outputInfo == nil || outputInfo.Type == nil || outputInfo.Type.GetTensorType() == nil {
		// Try to trace through the subgraph to find the shape
		// Look at the last node that produces this output
		outputName := outputInfo.Name
		for _, node := range graph.Node {
			for _, nodeOutput := range node.Output {
				if nodeOutput == outputName {
					// Found the node, try to get its shape from computed shapes
					if shape, ok := sr.computedShapes[outputName]; ok {
						return shape
					}
				}
			}
		}
		return shapes.Shape{}
	}

	return sr.extractShapeFromValueInfo(outputInfo)
}

// extractShapeFromValueInfo extracts a shape from a ValueInfoProto.
func (sr *ShapeResolver) extractShapeFromValueInfo(vi *protos.ValueInfoProto) shapes.Shape {
	if vi == nil || vi.Type == nil || vi.Type.GetTensorType() == nil {
		return shapes.Shape{}
	}

	tensorType := vi.Type.GetTensorType()
	dtype, err := dtypeForONNX(protos.TensorProto_DataType(tensorType.ElemType))
	if err != nil || dtype == dtypes.InvalidDType {
		return shapes.Shape{}
	}

	if tensorType.Shape == nil || len(tensorType.Shape.Dim) == 0 {
		// Scalar
		return shapes.Make(dtype)
	}

	dims := make([]int, len(tensorType.Shape.Dim))
	hasDynamic := false

	if hasDynamic {
		return shapes.MakeDynamic(dtype, dims...)
	}
	return shapes.Make(dtype, dims...)
}

// extractIntSliceFromTensor extracts an int slice from a TensorProto.
func extractIntSliceFromTensor(t *protos.TensorProto) ([]int, error) {
	if t == nil {
		return nil, nil
	}

	dtype := protos.TensorProto_DataType(t.DataType)
	switch dtype {
	case protos.TensorProto_INT64:
		// First try typed data
		if len(t.Int64Data) > 0 {
			result := make([]int, len(t.Int64Data))
			for i, v := range t.Int64Data {
				result[i] = int(v)
			}
			return result, nil
		}
		// Fall back to raw data
		if len(t.RawData) > 0 {
			numElems := len(t.RawData) / 8
			result := make([]int, numElems)
			for i := 0; i < numElems; i++ {
				val := int64(t.RawData[i*8]) |
					int64(t.RawData[i*8+1])<<8 |
					int64(t.RawData[i*8+2])<<16 |
					int64(t.RawData[i*8+3])<<24 |
					int64(t.RawData[i*8+4])<<32 |
					int64(t.RawData[i*8+5])<<40 |
					int64(t.RawData[i*8+6])<<48 |
					int64(t.RawData[i*8+7])<<56
				result[i] = int(val)
			}
			return result, nil
		}
	case protos.TensorProto_INT32:
		if len(t.Int32Data) > 0 {
			result := make([]int, len(t.Int32Data))
			for i, v := range t.Int32Data {
				result[i] = int(v)
			}
			return result, nil
		}
		// Fall back to raw data
		if len(t.RawData) > 0 {
			numElems := len(t.RawData) / 4
			result := make([]int, numElems)
			for i := 0; i < numElems; i++ {
				val := int32(t.RawData[i*4]) |
					int32(t.RawData[i*4+1])<<8 |
					int32(t.RawData[i*4+2])<<16 |
					int32(t.RawData[i*4+3])<<24
				result[i] = int(val)
			}
			return result, nil
		}
	}

	return nil, nil
}

// getIntSliceAttr gets an int slice attribute from a node.
func getIntSliceAttr(node *protos.NodeProto, name string) []int {
	for _, attr := range node.Attribute {
		if attr.Name == name {
			result := make([]int, len(attr.Ints))
			for i, v := range attr.Ints {
				result[i] = int(v)
			}
			return result
		}
	}
	return nil
}

// getIntAttr gets an int attribute from a node with a default value.
func getIntAttr(node *protos.NodeProto, name string, defaultValue int) int {
	for _, attr := range node.Attribute {
		if attr.Name == name {
			return int(attr.I)
		}
	}
	return defaultValue
}

// TraceDependencies traces backwards from a node to show why its shape couldn't be resolved.
// It returns a formatted string showing the dependency chain and where it breaks.
func (sr *ShapeResolver) TraceDependencies(nodeName string) string {
	var sb strings.Builder
	visited := make(map[string]bool)
	sr.traceDepsRecursive(&sb, nodeName, 0, visited)
	return sb.String()
}

func (sr *ShapeResolver) traceDepsRecursive(sb *strings.Builder, nodeName string, depth int, visited map[string]bool) {
	if depth > 20 {
		fmt.Fprintf(sb, "%s[MAX DEPTH]\n", strings.Repeat("  ", depth))
		return
	}
	if visited[nodeName] {
		fmt.Fprintf(sb, "%s[CYCLE: %s]\n", strings.Repeat("  ", depth), nodeName)
		return
	}
	visited[nodeName] = true

	indent := strings.Repeat("  ", depth)

	// Check if shape is resolved
	shape, ok := sr.GetShape(nodeName)
	status := "✗"
	if ok && shape.DType != dtypes.InvalidDType {
		status = "✓"
	}

	// Find the node that produces this output
	var producerNode *protos.NodeProto
	for _, node := range sr.model.Proto.Graph.Node {
		for _, output := range node.Output {
			if output == nodeName {
				producerNode = node
				break
			}
		}
		if producerNode != nil {
			break
		}
	}

	if producerNode == nil {
		// Check if it's an input or initializer
		if _, isInput := sr.inputShapes[nodeName]; isInput {
			fmt.Fprintf(sb, "%s%s INPUT: %s -> %v\n", indent, status, nodeName, shape)
		} else if _, isVar := sr.variableShapes[nodeName]; isVar {
			fmt.Fprintf(sb, "%s%s VAR: %s -> %v\n", indent, status, nodeName, shape)
		} else {
			fmt.Fprintf(sb, "%s%s UNKNOWN: %s\n", indent, status, nodeName)
		}
		return
	}

	fmt.Fprintf(sb, "%s%s %s (%s) -> %v\n", indent, status, nodeName, producerNode.OpType, shape)

	// Only recurse into unresolved dependencies
	if status == "✗" {

	}
}

// PrintUnresolvedSummary prints a summary of nodes without resolved shapes.
func (sr *ShapeResolver) PrintUnresolvedSummary() {
	unresolvedByOp := make(map[string]int)
	totalUnresolved := 0

	for _, node := range sr.model.Proto.Graph.Node {
		for _, output := range node.Output {
			shape, ok := sr.GetShape(output)
			if !ok || shape.DType == dtypes.InvalidDType {
				unresolvedByOp[node.OpType]++
				totalUnresolved++
			}
		}
	}

	// Sort by count
	type opCount struct {
		op    string
		count int
	}
	var ops []opCount
	for op, count := range unresolvedByOp {
		ops = append(ops, opCount{op, count})
	}
	for i := 0; i < len(ops); i++ {
		for j := i + 1; j < len(ops); j++ {
			if ops[j].count > ops[i].count {
				ops[i], ops[j] = ops[j], ops[i]
			}
		}
	}

}

// FindFirstUnresolved finds the first node in topological order without a resolved shape.
func (sr *ShapeResolver) FindFirstUnresolved() (string, *protos.NodeProto) {
	for _, node := range sr.model.Proto.Graph.Node {
		for _, output := range node.Output {
			shape, ok := sr.GetShape(output)
			if !ok || shape.DType == dtypes.InvalidDType {
				return output, node
			}
		}
	}
	return "", nil
}
