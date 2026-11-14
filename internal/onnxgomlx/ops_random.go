// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
)

// convertRandomNormalLike converts an ONNX RandomNormalLike node to a GoMLX node.
// Both static and dynamic input shapes are supported directly via RNGBitGenerator.
//
// See https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html.
func convertRandomNormalLike(scope *model.Scope, node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) != 1 || inputs[0] == nil {
		exceptions.Panicf("RandomNormalLike requires one input in %s", NodeToString(node))
	}

	x := inputs[0]
	shape := x.Shape().Clone()
	shape.DType = GetDTypeAttrOr(node, "dtype", x.DType())
	if !shape.DType.IsFloat() {
		exceptions.Panicf("RandomNormalLike requires a floating-point dtype, got %s in %s", shape.DType, NodeToString(node))
	}

	var values *Node
	if seed := GetNodeAttr(node, "seed", false); seed != nil {
		assertNodeAttrType(node, seed, protos.AttributeProto_FLOAT)
		rngState := RNGStateFromSeedForGraph(x.Graph(), int64(seed.F))
		_, values = RandomNormal(rngState, shape)
	} else if scope != nil {
		values = scope.RandomNormal(x.Graph(), shape)
	} else {
		rngState := RNGStateForGraph(x.Graph())
		_, values = RandomNormal(rngState, shape)
	}

	mean := GetFloatAttrOr(node, "mean", 0)
	stddev := GetFloatAttrOr(node, "scale", 1)
	if stddev < 0 {
		exceptions.Panicf("RandomNormalLike scale must be non-negative, got %g in %s", stddev, NodeToString(node))
	}
	if stddev != 1 {
		values = MulScalar(values, float64(stddev))
	}
	if mean != 0 {
		values = AddScalar(values, float64(mean))
	}
	return values
}
