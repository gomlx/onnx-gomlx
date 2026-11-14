// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/pkg/errors"
)

// convertInstanceNormalization converts an ONNX InstanceNormalization node to GoMLX.
//
// Input X has shape [N, C, D1, ..., Dn], with per-channel scale and B of shape [C].
// Normalization across spatial axes [2..rank-1] delegates to nn.LayerNorm so backends
// supporting FusedLayerNorm execute the normalization in a fused kernel.
//
// See ONNX documentation:
// https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html
func convertInstanceNormalization(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 {
		exceptions.Panicf("InstanceNormalization requires X, scale, and B inputs in %s", NodeToString(node))
	}
	x, scale, bias := inputs[0], inputs[1], inputs[2]
	if x.Rank() < 3 {
		exceptions.Panicf("InstanceNormalization expects rank >= 3, got %s in %s", x.Shape(), NodeToString(node))
	}
	if scale.Rank() != 1 || bias.Rank() != 1 || scale.Shape().Dim(0) <= 0 || scale.Shape().Dim(0) != bias.Shape().Dim(0) {
		exceptions.Panicf("InstanceNormalization scale (%s) and B (%s) must be matching static 1D tensors in %s",
			scale.Shape(), bias.Shape(), NodeToString(node))
	}
	channels := scale.Shape().Dim(0)
	if x.Shape().Dim(1) == shapes.DynamicDim {
		specs := DimensionSpecsFor(x)
		specs[1] = StaticDim(channels)
		x = DynamicReshape(x, specs...)
	} else if x.Shape().Dim(1) != channels {
		exceptions.Panicf("InstanceNormalization channel mismatch: X has %d channels, scale has %d in %s",
			x.Shape().Dim(1), channels, NodeToString(node))
	}

	spatialAxes := make([]int, x.Rank()-2)
	paramDims := make([]int, x.Rank())
	paramDims[0] = 1
	paramDims[1] = channels
	for i := range spatialAxes {
		spatialAxes[i] = i + 2
		paramDims[i+2] = 1
	}

	epsilon := float64(GetFloatAttrOr(node, "epsilon", 1e-5))
	normalized := nn.LayerNorm(x, spatialAxes, epsilon, nil, nil, nil)

	if scale.DType() != x.DType() {
		scale = ConvertDType(scale, x.DType())
	}
	if bias.DType() != x.DType() {
		bias = ConvertDType(bias, x.DType())
	}
	scale = Reshape(scale, paramDims...)
	bias = Reshape(bias, paramDims...)
	return Add(Mul(normalized, scale), bias)
}

// convertOneHot converts an ONNX OneHot node to GoMLX.
//
// See ONNX documentation:
// https://onnx.ai/onnx/operators/onnx__OneHot.html
func convertOneHot(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 {
		exceptions.Panicf("OneHot requires indices, depth, and values inputs in %s", NodeToString(node))
	}
	indices, values := inputs[0], inputs[2]
	depthT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "OneHot requires constant depth in %s", NodeToString(node)))
	}
	depthValues := tensorToInts(depthT)
	if len(depthValues) != 1 || depthValues[0] <= 0 {
		exceptions.Panicf("OneHot depth must be a positive scalar, got %v in %s", depthValues, NodeToString(node))
	}
	depth := depthValues[0]
	if values.Rank() != 1 || values.Shape().Dim(0) != 2 {
		exceptions.Panicf("OneHot values must have shape [2], got %s in %s", values.Shape(), NodeToString(node))
	}

	outRank := indices.Rank() + 1
	axis := GetIntAttrOr(node, "axis", -1)
	if axis < 0 {
		axis += outRank
	}
	if axis < 0 || axis >= outRank {
		exceptions.Panicf("OneHot axis %d out of range for output rank %d in %s", axis, outRank, NodeToString(node))
	}

	if !indices.DType().IsInt() {
		indices = ConvertDType(indices, dtypes.Int64)
	}
	zero := ScalarZero(indices.Graph(), indices.DType())
	indices = Where(LessThan(indices, zero), AddScalar(indices, depth), indices)

	mask := OneHot(indices, depth, dtypes.Bool)
	off := Reshape(Slice(values, AxisRange(0, 1)))
	on := Reshape(Slice(values, AxisRange(1, 2)))
	result := Where(mask, on, off)
	if axis == outRank-1 {
		return result
	}
	perm := make([]int, 0, outRank)
	for i := range indices.Rank() {
		if i == axis {
			perm = append(perm, outRank-1)
		}
		perm = append(perm, i)
	}
	return TransposeAllAxes(result, perm...)
}
