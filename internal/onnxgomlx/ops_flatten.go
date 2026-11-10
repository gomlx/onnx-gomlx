package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertFlatten converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Flatten.html
func convertFlatten(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	splitAxis := GetIntAttrOr(node, "axis", 1)
	// Flatten splits dimensions at a boundary, so axis == rank is valid.
	if splitAxis < 0 {
		splitAxis += operand.Rank()
	}
	if splitAxis < 0 || splitAxis > operand.Rank() {
		exceptions.Panicf("Flatten: axis is outside [-rank, rank] for input %s", operand.Shape())
	}
	return onnxFlatten(operand, splitAxis)
}

// onnxFlatten implements the corresponding ONNX operation.
func onnxFlatten(operand *Node, splitAxis int) *Node {
	outerDim, innerDim := 1, 1
	for axis, dim := range operand.Shape().Dimensions {
		if axis < splitAxis {
			outerDim *= dim
		} else {
			innerDim *= dim
		}
	}
	return Reshape(operand, outerDim, innerDim)
}
