// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertReshape converts an ONNX Reshape node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Reshape.html
func convertReshape(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	if !inputs[1].DType().IsInt() {
		exceptions.Panicf("shape must be integer, got %s for node %s", inputs[1].DType(), NodeToString(node))
	}
	allowZero := GetIntAttrOr(node, "allowzero", 0)

	dimsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		knownDims, isKnown := reshapeTargetStaticDims(m, convertedOutputs, node.Input[1], inputs[1].Shape().Dim(0))
		return runtimeReshape(operand, inputs[1], allowZero, knownDims, isKnown)
	}
	dims := tensorToInts(dimsT)
	if operand.Shape().IsDynamic() {
		specs := make([]DimensionSpec, len(dims))
		inferred := false
		for axis, dim := range dims {
			switch {
			case dim == 0 && allowZero == 0:
				if axis >= operand.Rank() {
					exceptions.Panicf("Reshape: zero-copy axis %d exceeds input rank %d", axis, operand.Rank())
				}
				specs[axis] = DimensionSpecFor(operand, axis)
			case dim == -1:
				if inferred {
					exceptions.Panicf("Reshape: multiple inferred (-1) dimensions in %v", dims)
				}
				inferred = true
				specs[axis] = InferredDim()
			case dim >= 0:
				specs[axis] = StaticDim(dim)
			default:
				exceptions.Panicf("Reshape: invalid dimension %d", dim)
			}
		}
		return DynamicReshape(operand, specs...)
	}
	if allowZero == 0 {
		// If new shape dim is 0, copy over from previous shape.
		for newAxis, dim := range dims {
			if dim == 0 && newAxis < operand.Rank() {
				dims[newAxis] = operand.Shape().Dim(newAxis)
			}
		}
	}
	return Reshape(operand, dims...)
}

// reshapeTargetStaticDims recovers statically known entries from a partially dynamic
// Reshape target such as Concat(Shape(x)[0], 4, 64) so known feature/head widths stay static.
func reshapeTargetStaticDims(m *Model, convertedOutputs map[string]*Node, name string, rank int) ([]int, []bool) {
	if rank < 0 {
		return nil, nil
	}
	dims := make([]int, rank)
	known := make([]bool, rank)
	if value, err := m.materializeConstantExpression(name, convertedOutputs); err == nil {
		values := tensorToInts(value)
		if len(values) == rank {
			copy(dims, values)
			for i := range known {
				known[i] = true
			}
			return dims, known
		}
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok || node.OpType != "Concat" || (GetIntAttrOr(node, "axis", 0) != 0 && GetIntAttrOr(node, "axis", 0) != -1) {
		return dims, known
	}

	index := 0
	for _, inputName := range node.Input {
		if inputName == "" {
			continue
		}
		if value, err := m.materializeConstantExpression(inputName, convertedOutputs); err == nil {
			values := tensorToInts(value)
			for _, v := range values {
				if index >= rank {
					return make([]int, rank), make([]bool, rank)
				}
				dims[index] = v
				known[index] = true
				index++
			}
			continue
		}
		input := convertedOutputs[inputName]
		if input == nil {
			return make([]int, rank), make([]bool, rank)
		}
		count := 1
		for _, dim := range input.Shape().Dimensions {
			if dim < 0 {
				return make([]int, rank), make([]bool, rank)
			}
			count *= dim
		}
		index += count
		if index > rank {
			return make([]int, rank), make([]bool, rank)
		}
	}
	if index != rank {
		return make([]int, rank), make([]bool, rank)
	}
	return dims, known
}

func runtimeReshape(operand, shape *Node, allowZero int, knownDims []int, isKnown []bool) *Node {
	if shape.Rank() != 1 || shape.Shape().Dim(0) < 0 {
		exceptions.Panicf("Reshape: target rank must be statically known, got %s", shape.Shape())
	}
	rank := shape.Shape().Dim(0)
	if rank == 0 {
		return Reshape(operand)
	}

	g := operand.Graph()
	specs := make([]DimensionSpec, rank)
	adjusted := make([]*Node, rank)
	staticInferredAxis := -1
	unknownCount := 0

	for axis := range rank {
		if axis < len(isKnown) && isKnown[axis] {
			dim := knownDims[axis]
			switch {
			case dim > 0:
				specs[axis] = StaticDim(dim)
				adjusted[axis] = Scalar(g, dtypes.Int64, dim)
				continue
			case dim == 0 && allowZero == 0 && axis < operand.Rank():
				specs[axis] = DimensionSpecFor(operand, axis)
				adjusted[axis] = DimensionSize(operand, axis)
				continue
			case dim == 0:
				specs[axis] = StaticDim(0)
				adjusted[axis] = ScalarZero(g, dtypes.Int64)
				continue
			case dim == -1:
				if staticInferredAxis >= 0 {
					exceptions.Panicf("Reshape: multiple inferred (-1) dimensions")
				}
				staticInferredAxis = axis
				specs[axis] = InferredDim()
				continue
			}
		}
		unknownCount++
		val := ConvertDType(Reshape(Slice(shape, AxisRange(axis, axis+1))), dtypes.Int64)
		if allowZero == 0 && axis < operand.Rank() {
			val = Where(Equal(val, ScalarZero(g, dtypes.Int64)), DimensionSize(operand, axis), val)
		}
		adjusted[axis] = val
	}

	// If an axis is already statically -1, no runtime entry can also be -1.
	if unknownCount > 0 && staticInferredAxis < 0 {
		negOne := Scalar(g, dtypes.Int64, -1)
		one := Scalar(g, dtypes.Int64, 1)
		nonNegProduct := one
		for axis := range rank {
			factor := Where(Equal(adjusted[axis], negOne), one, adjusted[axis])
			nonNegProduct = Mul(nonNegProduct, factor)
		}
		totalSize := ReduceMultiply(DynamicShape(operand))
		inferred := Div(totalSize, MaxScalar(nonNegProduct, 1))
		for axis := range rank {
			if axis < len(isKnown) && isKnown[axis] {
				continue
			}
			val := Where(Equal(adjusted[axis], negOne), inferred, adjusted[axis])
			specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_reshape"), val)
		}
	} else if unknownCount > 0 {
		for axis := range rank {
			if axis < len(isKnown) && isKnown[axis] {
				continue
			}
			specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_reshape"), adjusted[axis])
		}
	}

	return DynamicReshape(operand, specs...)
}
