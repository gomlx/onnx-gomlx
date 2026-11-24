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

// RuntimeBroadcast makes ONNX multidirectional positional broadcasting explicit when
// dynamic dimensions have distinct symbolic names or broadcast against static dimensions.
func RuntimeBroadcast(operands []*Node) []*Node {
	return runtimeBroadcast(operands)
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
		hasZeroStatic := false
		var firstDynOp *Node
		sharedDynName := ""
		hasDistinctDynNames := false
		for _, operand := range expanded {
			dim := operand.Shape().Dim(axis)
			if dim < 0 {
				name := operand.Shape().AxisName(axis)
				if firstDynOp == nil {
					firstDynOp = operand
					sharedDynName = name
				} else if name != sharedDynName {
					hasDistinctDynNames = true
				}
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
		// If all operands are static, or if any operand has a non-unit static dimension (0 or > 1),
		// the broadcast output dimension is statically fixed to staticSize (matching DynamicBroadcastInDim's
		// static narrowing rule while validating dynamic operands for runtime 1-or-match compatibility).
		if firstDynOp == nil || hasZeroStatic || staticSize != 1 {
			specs[axis] = StaticDim(staticSize)
			continue
		}
		// If all dynamic operands share the same named axis (and static operands are all 1), preserve that axis name.
		if !hasDistinctDynNames && sharedDynName != "" {
			specs[axis] = DimensionSpecFor(firstDynOp, axis)
			continue
		}
		var size *Node
		for _, operand := range expanded {
			if operand.Shape().Dim(axis) >= 0 {
				continue // Static dimension is 1 here and never overrides another operand's size.
			}
			other := DimensionSize(operand, axis)
			if size == nil {
				size = other
			} else {
				size = Where(Equal(size, ScalarOne(size.Graph(), size.DType())), other, size)
			}
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
	alreadyMatches := true
	for axis, spec := range specs {
		dim := operand.Shape().Dim(axis)
		if spec.IsStatic() {
			if dim != spec.Static() {
				alreadyMatches = false
				break
			}
		} else if dim != shapes.DynamicDim || operand.Shape().AxisName(axis) != spec.AxisName() || spec.AxisName() == "" {
			alreadyMatches = false
			break
		}
	}
	if alreadyMatches {
		return operand
	}
	axes := make([]int, operand.Rank())
	for axis := range axes {
		axes[axis] = axis
	}
	return DynamicBroadcastInDim(operand, axes, specs...)
}

func runtimeExpand(operand, shape *Node, knownDims []int, isKnown []bool) *Node {
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
		if axis < offset {
			specs[axis] = DimensionSpecFor(operand, axis)
			continue
		}
		shapeIdx := axis - offset
		if shapeIdx < len(isKnown) && isKnown[shapeIdx] && knownDims[shapeIdx] >= 0 {
			tDim := knownDims[shapeIdx]
			if tDim == 1 {
				specs[axis] = DimensionSpecFor(operand, axis)
			} else {
				inDim := operand.Shape().Dim(axis)
				if inDim >= 0 && inDim != 1 && inDim != tDim {
					exceptions.Panicf("Expand: incompatible dimensions %d and %d at axis %d", inDim, tDim, axis)
				}
				specs[axis] = StaticDim(tDim)
			}
			continue
		}
		size := DimensionSize(operand, axis)
		target := ConvertDType(Reshape(Slice(shape, AxisRange(shapeIdx, shapeIdx+1))), dtypes.Int64)
		size = Where(Equal(target, ScalarOne(target.Graph(), target.DType())), size, target)
		specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_expand"), size)
	}
	return broadcastWithSpecs(operand, specs)
}

func runtimeConstantOfShape(valueN, shape *Node, knownDims []int, isKnown []bool) *Node {
	if shape.Rank() != 1 || shape.Shape().Dim(0) < 0 {
		exceptions.Panicf("ConstantOfShape: target shape rank must be statically known, got %s", shape.Shape())
	}
	rank := shape.Shape().Dim(0)
	if rank == 0 {
		return valueN
	}
	specs := make([]DimensionSpec, rank)
	for axis := range specs {
		if axis < len(isKnown) && isKnown[axis] && knownDims[axis] >= 0 {
			specs[axis] = StaticDim(knownDims[axis])
			continue
		}
		target := ConvertDType(Reshape(Slice(shape, AxisRange(axis, axis+1))), dtypes.Int64)
		specs[axis] = NamedDynamicDim(shapes.UniqueAxisName("#onnx_const_of_shape"), target)
	}
	return DynamicBroadcastInDim(valueN, nil, specs...)
}
