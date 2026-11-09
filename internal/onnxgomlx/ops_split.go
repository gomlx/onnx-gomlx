package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertSplit converts the corresponding ONNX node to GoMLX nodes.
//
// Split splits a tensor into multiple outputs along a specified axis.
// This is commonly used in attention mechanisms to split into Q, K, V.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Split.html
func convertSplit(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]

	// Get axis attribute (default is 0)
	axis := GetIntAttrOr(node, "axis", 0)

	// Determine the number of splits from the output count
	numOutputs := len(node.Output)
	if numOutputs == 0 {
		exceptions.Panicf("Split: expected at least 1 output, got 0")
	}

	// Check if split sizes are provided as second input (ONNX opset >= 13)
	// or as attribute (older opset)
	splitSizes := GetIntsAttrOr(node, "split", nil)
	if len(inputs) > 1 {
		// Split sizes provided as input (need to materialize it)
		splitSizesTensor, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			exceptions.Panicf("Split: failed to materialize split sizes for node %s: %v", NodeToString(node), err)
		}
		// Convert tensor to int slice
		splitSizes = tensorToInts(splitSizesTensor)
	}
	if splitSizes == nil {
		// Equal splits - divide dimension evenly
		dim := x.Shape().Dim(axis)
		if dim%numOutputs != 0 {
			exceptions.Panicf("Split: dimension %d (size=%d) not evenly divisible by number of outputs (%d)",
				axis, dim, numOutputs)
		}
		splitSize := dim / numOutputs
		splitSizes = make([]int, numOutputs)
		for i := range splitSizes {
			splitSizes[i] = splitSize
		}
	}

	// Perform the split using SliceAxis
	splits := make([]*Node, numOutputs)
	currentStart := 0
	for i := range numOutputs {
		end := currentStart + splitSizes[i]
		splits[i] = SliceAxis(x, axis, AxisRange(currentStart, end))
		currentStart = end
	}

	// Assign each output to convertedOutputs
	for i, split := range splits {
		convertedOutputs[node.Output[i]] = split
	}

	// Return first output (convention for multi-output ops)
	return splits[0]
}
