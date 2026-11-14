// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertFlatten converts an ONNX Flatten node to a GoMLX node.
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
	if !operand.Shape().IsDynamic() {
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

	flattenedDim := func(start, end int) DimensionSpec {
		if end-start == 1 && operand.Shape().Dim(start) == shapes.DynamicDim {
			return DimensionSpecFor(operand, start)
		}
		staticSize := 1
		hasDynamicDim := false
		for _, dim := range operand.Shape().Dimensions[start:end] {
			if dim == shapes.DynamicDim {
				hasDynamicDim = true
				continue
			}
			staticSize *= dim
		}
		if !hasDynamicDim {
			return StaticDim(staticSize)
		}
		runtimeShape := Slice(DynamicShape(operand), AxisRange(start, end))
		runtimeSize := ReduceMultiply(runtimeShape, 0)
		return NamedDynamicDim(shapes.UniqueAxisName("#onnx_flatten"), runtimeSize)
	}
	return DynamicReshape(operand,
		flattenedDim(0, splitAxis),
		flattenedDim(splitAxis, operand.Rank()),
	)
}
