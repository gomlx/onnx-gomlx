package onnxgomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/pkg/errors"
)

// convertPad converts an ONNX Pad node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Pad.html
func convertPad(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	mode := GetStringAttrOr(node, "mode", "constant")
	padsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'pads' for node %s", NodeToString(node)))
	}
	pads := tensorToInts(padsT)

	x := inputs[0]

	// Empty pads tensor (e.g., from a zero-sized constant) means no padding.
	if len(pads) == 0 {
		return x
	}

	rank := x.Rank()
	if len(inputs) > 3 && inputs[3] != nil {
		axesTensor, err := m.materializeConstantExpression(node.Input[3], convertedOutputs)
		if err != nil {
			panic(errors.WithMessage(err, "while converting Pad axes"))
		}
		axes := tensorToInts(axesTensor)
		if len(pads) != 2*len(axes) {
			exceptions.Panicf("Pad: expected two padding values per axis")
		}
		expanded := make([]int, 2*rank)
		seen := make([]bool, rank)
		for index, axis := range axes {
			if axis < 0 {
				axis += rank
			}
			if axis < 0 || axis >= rank {
				exceptions.Panicf("Pad: axis %d out of range for rank %d", axis, rank)
			}
			if seen[axis] {
				exceptions.Panicf("Pad: duplicate axis %d", axis)
			}
			seen[axis] = true
			expanded[axis] = pads[index]
			expanded[axis+rank] = pads[index+len(axes)]
		}
		pads = expanded
	}
	if len(pads) != 2*rank {
		exceptions.Panicf("invalid number of padding values: %d axes, got %d padding values -- expected 2 pads per axis", rank, len(pads))
	}

	switch mode {
	case "constant":
		var constantValueNode *Node
		if len(inputs) > 2 && inputs[2] != nil {
			constantValueNode = inputs[2]
		} else {
			constantValueNode = Scalar(x.Graph(), x.DType(), 0)
		}
		paddings := make([]compute.PadAxis, rank)
		for i := range rank {
			paddings[i] = compute.PadAxis{Start: pads[i], End: pads[i+rank]}
		}
		return Pad(x, constantValueNode, paddings...)

	case "reflect":
		return convertPadReflect(x, pads, rank)

	default:
		exceptions.Panicf("Pad: unsupported mode %q", mode)
		return nil
	}
}

// convertPadReflect implements ONNX reflect padding using Slice, Reverse, and Concatenate.
//
// Reflect padding mirrors values at the boundary, excluding the edge element itself.
// For [a, b, c, d] with left=2, right=1: [c, b, a, b, c, d, c].
func convertPadReflect(x *Node, pads []int, rank int) *Node {
	result := x
	for axis := range rank {
		padStart := pads[axis]
		padEnd := pads[axis+rank]
		if padStart == 0 && padEnd == 0 {
			continue
		}

		// Negative padding crops the input before reflecting its remaining edges.
		if padStart < 0 || padEnd < 0 {
			start := max(-padStart, 0)
			end := result.Shape().Dim(axis) + min(padEnd, 0)
			result = SliceAxis(result, axis, AxisRange(start, end))
			padStart = max(padStart, 0)
			padEnd = max(padEnd, 0)
		}

		var parts []*Node

		if padStart > 0 {
			// Slice elements [1, 1+padStart) along axis, then reverse.
			specs := make([]SliceAxisSpec, rank)
			for i := range rank {
				if i == axis {
					specs[i] = AxisRange(1, 1+padStart)
				} else {
					specs[i] = AxisRange()
				}
			}
			leftSlice := Slice(result, specs...)
			parts = append(parts, Reverse(leftSlice, axis))
		}

		parts = append(parts, result)

		if padEnd > 0 {
			// Slice elements [dim-1-padEnd, dim-1) along axis, then reverse.
			// Using negative indices: [-1-padEnd, -1).
			specs := make([]SliceAxisSpec, rank)
			for i := range rank {
				if i == axis {
					specs[i] = AxisRange(-1-padEnd, -1)
				} else {
					specs[i] = AxisRange()
				}
			}
			rightSlice := Slice(result, specs...)
			parts = append(parts, Reverse(rightSlice, axis))
		}

		result = Concatenate(parts, axis)
	}
	return result
}
