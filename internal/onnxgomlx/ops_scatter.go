package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertScatterND operator
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ScatterND.html
func convertScatterND(_ *Model, _ map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// inputs
	data := inputs[0]
	indices := inputs[1]
	updates := inputs[2]

	// attributes
	reduction := GetStringAttrOr(node, "reduction", "none")

	r := data.Rank()
	if !(r >= 1) {
		exceptions.Panicf("ScatterND: data must have rank >= 1, got %d", r)
	}

	q := indices.Rank()
	if !(q >= 1) {
		exceptions.Panicf("ScatterND: indices must have rank >= 1, got %d", q)
	}

	v := q + r - indices.Shape().Dimensions[len(indices.Shape().Dimensions)-1] - 1

	if updates.Rank() != v {
		exceptions.Panicf("ScatterND: updates has wrong rank")
	}

	indices = normalizeNDIndices(data, indices)
	operand := Identity(data)
	var output *Node
	switch reduction {
	case "add":
		output = ScatterSum(operand, indices, updates, false, false)
	case "mul":
		exceptions.Panicf("ScatterMul has not been implemented yet")
	case "max":
		output = ScatterMax(operand, indices, updates, false, false)
	case "min":
		output = ScatterMin(operand, indices, updates, false, false)
	case "none", "":
		output = ScatterUpdate(operand, indices, updates, false, true)
	default:
		exceptions.Panicf("ScatterND: unrecognized reduction mode %q", reduction)
	}

	if output.Rank() < 1 {
		exceptions.Panicf("ScatterND: output must have rank >= 1, got rank %d", output.Rank())
	}
	return output
}

// normalizeNDIndices converts ONNX negative coordinate tuples for GatherND and ScatterND.
func normalizeNDIndices(data, indices *Node) *Node {
	// Each negative coordinate is relative to the end of its data axis.
	// Broadcast the axis sizes over the leading dimensions of indices.
	indexDepth := indices.Shape().Dimensions[indices.Rank()-1]
	axisSizes := make([]*Node, indexDepth)
	for axis := range axisSizes {
		axisSizes[axis] = ConvertDType(DimensionSize(data, axis), indices.DType())
	}
	isNegative := LessThan(indices, ScalarZero(indices.Graph(), indices.DType()))
	sizes := BroadcastLike(Stack(axisSizes, 0), indices)
	return Where(isNegative, Add(indices, sizes), indices)
}
