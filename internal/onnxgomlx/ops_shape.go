package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// convertShape converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Shape.html
func convertShape(node *protos.NodeProto, inputs []*Node) *Node {
	shape := inputs[0].Shape()
	start := GetIntAttrOr(node, "start", 0)
	if start < 0 {
		start = shape.Rank() + start
	}
	end := GetIntAttrOr(node, "end", shape.Rank())
	if end < 0 {
		end = shape.Rank() + end
	}
	start = min(max(start, 0), shape.Rank())
	end = min(max(end, start), shape.Rank())

	dims := sliceMap(shape.Dimensions[start:end], func(dim int) int64 { return int64(dim) })
	g := inputs[0].Graph()
	value := tensors.FromFlatDataAndDimensions(dims, len(dims))
	defer value.FinalizeAll()
	return ConstTensor(g, value)
}
