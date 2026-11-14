// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// needsRuntimeBroadcast reports whether operands (after onnxImplicitExpansion) require explicit
// runtime broadcasting (e.g., distinct dynamic axis names that may broadcast a runtime
// size-1 axis, or a dynamic axis paired with a non-unit static dimension).
func needsRuntimeBroadcast(operands []*Node) bool {
	if len(operands) < 2 {
		return false
	}
	rank := 0
	for _, op := range operands {
		rank = max(rank, op.Rank())
	}
	for axis := range rank {
		hasDynamic := false
		sharedDynName := ""
		hasNonUnitStatic := false
		for _, op := range operands {
			if op.IsScalar() {
				continue
			}
			dim := op.Shape().Dim(axis)
			if dim == shapes.DynamicDim {
				name := op.Shape().AxisName(axis)
				if !hasDynamic {
					hasDynamic = true
					sharedDynName = name
				} else if name != sharedDynName {
					return true
				}
			} else if dim != 1 {
				hasNonUnitStatic = true
			}
		}
		if hasDynamic && hasNonUnitStatic {
			return true
		}
	}
	return false
}

// runtimeBroadcast makes ONNX multidirectional positional broadcasting explicit when
// dynamic dimensions have distinct symbolic names or broadcast against static dimensions.
func runtimeBroadcast(operands []*Node) []*Node {
	implicit := onnxImplicitExpansion(operands)
	if !needsRuntimeBroadcast(implicit) {
		return implicit
	}
	rank := 0
	for _, operand := range implicit {
		rank = max(rank, operand.Rank())
	}
	if rank == 0 {
		return implicit
	}
	expanded := make([]*Node, len(implicit))
	for i, operand := range implicit {
		expanded[i] = ExpandLeftToRank(operand, rank)
	}

	specs := make([]DimensionSpec, rank)
	for axis := range specs {
		staticSize := 1
		allStatic := true
		hasZeroStatic := false
		for _, operand := range expanded {
			dim := operand.Shape().Dim(axis)
			if dim < 0 {
				allStatic = false
			} else if dim == 0 {
				if staticSize != 1 && staticSize != 0 {
					exceptions.Panicf("ONNX broadcast: incompatible dimensions %d and 0 at axis %d", staticSize, axis)
				}
				staticSize = 0
				hasZeroStatic = true
			} else if staticSize == 1 && !hasZeroStatic {
				staticSize = dim
			} else if dim != 1 && dim != staticSize {
				exceptions.Panicf("ONNX broadcast: incompatible dimensions %d and %d at axis %d", staticSize, dim, axis)
			}
		}
		if allStatic && staticSize >= 0 {
			specs[axis] = StaticDim(staticSize)
			continue
		}
		size := DimensionSize(expanded[0], axis)
		for _, operand := range expanded[1:] {
			other := DimensionSize(operand, axis)
			size = Where(Equal(size, Scalar(size.Graph(), size.DType(), 1)), other, size)
		}
		specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_bcast"), size)
	}
	for i, operand := range expanded {
		expanded[i] = broadcastWithSpecs(operand, specs)
	}
	return expanded
}

func broadcastWithSpecs(operand *Node, specs []DimensionSpec) *Node {
	operand = ExpandLeftToRank(operand, len(specs))
	// Rename dynamic input axes to match target specs while preserving runtime sizes;
	// DynamicBroadcastInDim then expands size-1 axes and validates runtime size compatibility.
	inputSpecs := make([]DimensionSpec, operand.Rank())
	axes := make([]int, operand.Rank())
	for axis := range inputSpecs {
		axes[axis] = axis
		dim := operand.Shape().Dim(axis)
		if dim >= 0 {
			inputSpecs[axis] = StaticDim(dim)
		} else if specs[axis].IsStatic() {
			inputSpecs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_bcast_in"), DimensionSize(operand, axis))
		} else {
			inputSpecs[axis] = NamedDynamicDim(specs[axis].AxisName(), DimensionSize(operand, axis))
		}
	}
	operand = DynamicReshape(operand, inputSpecs...)
	return DynamicBroadcastInDim(operand, axes, specs...)
}

func runtimeExpand(operand, shape *Node) *Node {
	if shape.Rank() != 1 || shape.Shape().Dim(0) < 0 {
		exceptions.Panicf("Expand: target shape rank must be statically known, got %s", shape.Shape())
	}
	shapeRank := shape.Shape().Dim(0)
	rank := max(operand.Rank(), shapeRank)
	if rank == 0 {
		return operand
	}
	operand = ExpandLeftToRank(operand, rank)
	specs := make([]DimensionSpec, rank)
	offset := rank - shapeRank
	for axis := range specs {
		size := DimensionSize(operand, axis)
		if axis >= offset {
			target := ConvertDType(Reshape(Slice(shape, AxisRange(axis-offset, axis-offset+1))), dtypes.Int64)
			size = Where(Equal(target, Scalar(target.Graph(), target.DType(), 1)), size, target)
		}
		specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_expand"), size)
	}
	return broadcastWithSpecs(operand, specs)
}
