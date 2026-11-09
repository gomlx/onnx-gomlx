package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/pkg/errors"
)

// convertSlice converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Slice.html
func convertSlice(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 {
		exceptions.Panicf("Slice requires at least 3 inputs, got %d in node %s", len(inputs), NodeToString(node))
	}

	operand := inputs[0]
	rank := operand.Rank()

	// optional axes param
	var inputAxes []int
	if len(inputs) > 3 && inputs[3] != nil {
		axesT, err := m.materializeConstantExpression(node.Input[3], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", NodeToString(node)))
		}
		inputAxes = tensorToInts(axesT)
	} else {
		// default values according to spec
		inputAxes = make([]int, rank)
		for i := range rank {
			inputAxes[i] = i
		}
	}

	// optional steps param
	var inputSteps []int
	if len(inputs) > 4 && inputs[4] != nil {
		stepsT, err := m.materializeConstantExpression(node.Input[4], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'steps' for node %s", NodeToString(node)))
		}
		inputSteps = tensorToInts(stepsT)
	} else {
		inputSteps = make([]int, len(inputAxes))
		for i := range inputSteps {
			inputSteps[i] = 1
		}
	}

	// Check if this is a dynamic slice (e.g. slicing an iota table by dynamic seq_len):
	sliceAxis := 0
	if len(inputAxes) > 0 {
		sliceAxis = inputAxes[0]
		if sliceAxis < 0 {
			sliceAxis += rank
		}
	}

	operandT, _ := m.materializeConstantExpression(node.Input[0], convertedOutputs)
	dynSource, dynAxis, ok := m.findDynamicDimensionSource(node.Input[2], convertedOutputs)
	if ok {
		if isIotaTensor(operandT, sliceAxis) {
			specs := make([]DimensionSpec, rank)
			for d := range rank {
				if d == sliceAxis {
					specs[d] = DimensionSpecFor(dynSource, dynAxis)
				} else {
					specs[d] = StaticDim(operand.Shape().Dim(d))
				}
			}
			res := DynamicIota(operand.Graph(), operand.DType(), sliceAxis, specs...)
			if startsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs); err == nil {
				starts := tensorToInts(startsT)
				if len(starts) > 0 && starts[0] != 0 {
					res = AddScalar(res, starts[0])
				}
			}
			if len(inputSteps) > 0 && inputSteps[0] != 1 {
				res = MulScalar(res, inputSteps[0])
			}
			return res
		}
	}

	endsT, errEnds := m.materializeConstantExpression(node.Input[2], convertedOutputs)
	if errEnds != nil && convertedOutputs[node.Input[2]] != nil {
		if isIotaTensor(operandT, sliceAxis) {
			endNode := convertedOutputs[node.Input[2]]
			endScalar := Reshape(ConvertDType(endNode, dtypes.Int32))
			specs := make([]DimensionSpec, rank)
			for d := range rank {
				if d == sliceAxis {
					specs[d] = DynamicDim(endScalar)
				} else {
					specs[d] = StaticDim(operand.Shape().Dim(d))
				}
			}
			res := DynamicIota(operand.Graph(), operand.DType(), sliceAxis, specs...)
			if startsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs); err == nil {
				starts := tensorToInts(startsT)
				if len(starts) > 0 && starts[0] != 0 {
					res = AddScalar(res, starts[0])
				}
			}
			if len(inputSteps) > 0 && inputSteps[0] != 1 {
				res = MulScalar(res, inputSteps[0])
			}
			return res
		}
		panic(errors.WithMessagef(errEnds, "while converting 'ends' for node %s", NodeToString(node)))
	}

	startsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'starts' for node %s", NodeToString(node)))
	}
	inputStarts := tensorToInts(startsT)
	inputEnds := tensorToInts(endsT)

	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}

	effectiveStarts := make([]int, rank)
	effectiveEnds := make([]int, rank)
	effectiveSteps := make([]int, rank)

	for i := range rank {
		effectiveStarts[i] = 0
		effectiveEnds[i] = operand.Shape().Dim(i)
		effectiveSteps[i] = 1
	}

	normalizedAxes := make([]int, len(inputAxes))
	for i, axis := range inputAxes {
		if axis < 0 {
			normalizedAxes[i] = axis + rank
		} else {
			normalizedAxes[i] = axis
		}

		if normalizedAxes[i] < 0 || normalizedAxes[i] >= rank {
			exceptions.Panicf("axis %d is out of bounds for tensor of rank %d in node %s",
				inputAxes[i], rank, NodeToString(node))
		}
	}

	// Process each specified axis to override the effective values
	for i := range normalizedAxes {
		axis := normalizedAxes[i]
		start := inputStarts[i]
		end := inputEnds[i]
		step := inputSteps[i]
		dimSize := operand.Shape().Dim(axis)

		// Validate step is not zero
		if step == 0 {
			panic(errors.Errorf("step cannot be 0 for axis %d in node %s", axis, NodeToString(node)))
		}

		// Handle negative start and end indices by adding dimension size
		if start < 0 {
			start += dimSize
		}
		if end < 0 {
			end += dimSize
		}

		if step > 0 {
			// Positive stepping
			// start clamped to [0, dimSize]
			// end clamped to [0, dimSize]
			start = max(0, min(start, dimSize))
			end = max(0, min(end, dimSize))
		} else {
			// Negative stepping (step < 0)
			// start clamped to [0, dimSize-1]
			// end clamped to [-1, dimSize-1]
			start = max(0, min(start, dimSize-1))
			end = max(-1, min(end, dimSize-1))
		}

		effectiveStarts[axis] = start
		effectiveEnds[axis] = end
		effectiveSteps[axis] = step
	}

	// Check if any axis produces an empty slice (start >= end for positive step,
	// or start <= end for negative step). Per the ONNX spec, this should produce
	// a zero-length result along that axis, but GoMLX's Slice doesn't support
	// start == dimSize, so we handle it here.
	emptySlice := false
	outputDims := make([]int, rank)
	for i := range rank {
		start := effectiveStarts[i]
		end := effectiveEnds[i]
		step := effectiveSteps[i]
		if step > 0 {
			if start >= end {
				emptySlice = true
				outputDims[i] = 0
			} else {
				outputDims[i] = (end - start + step - 1) / step
			}
		} else {
			if start <= end {
				emptySlice = true
				outputDims[i] = 0
			} else {
				outputDims[i] = (start - end + (-step) - 1) / (-step)
			}
		}
	}
	if emptySlice {
		return Zeros(operand.Graph(), shapes.Make(operand.DType(), outputDims...))
	}

	// GoMLX's Slice only supports positive strides (and it would interpret an effective end of -1 as
	// "the last element"). So axes with negative steps are reversed first, and the start/end indices mapped to
	// the reversed axis: index i becomes dimSize-1-i.
	// See https://github.com/gomlx/gomlx/issues/472
	var reversedAxes []int
	for i := range rank {
		if effectiveSteps[i] < 0 {
			dimSize := operand.Shape().Dim(i)
			reversedAxes = append(reversedAxes, i)
			effectiveStarts[i], effectiveEnds[i] = dimSize-1-effectiveStarts[i], dimSize-1-effectiveEnds[i]
			effectiveSteps[i] = -effectiveSteps[i]
		}
	}
	if len(reversedAxes) > 0 {
		operand = Reverse(operand, reversedAxes...)
	}

	specs := make([]SliceAxisSpec, rank)
	for i := range rank {
		specs[i] = AxisRange(effectiveStarts[i], effectiveEnds[i]).Stride(effectiveSteps[i])
	}

	return Slice(operand, specs...)
}
