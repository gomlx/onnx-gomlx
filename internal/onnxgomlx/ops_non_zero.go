// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
)

// convertNonZero converts an ONNX NonZero node to a GoMLX node.
//
// NonZero returns the indices of elements that are non-zero with shape [rank, numNonZero].
// Constant expressions (including expressions derived from concrete input shapes in convertedOutputs)
// are folded at graph-build time so static backends (e.g. XLA) and downstream constant-shape
// subgraphs remain supported; genuinely dynamic runtime inputs delegate to GoMLX's NonZero operator.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It is seeded with the graph inputs and populated
// as nodes are converted.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__NonZero.html
func convertNonZero(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	g := inputs[0].Graph()
	if inputT, err := m.materializeConstantExpression(node.Input[0], convertedOutputs); err == nil {
		res := computeNonZero(inputT)
		defer res.FinalizeAll()
		return ConstTensor(g, res)
	}
	return NonZero(inputs[0], &NonZeroConfig{Transpose: true})
}

// computeNonZero computes the indices of non-zero elements in a tensor.
// Returns a tensor of shape (rank, numNonZero) where each column is an index tuple.
func computeNonZero(t *tensors.Tensor) *tensors.Tensor {
	shape := t.Shape()
	rank := shape.Rank()

	// Get a boolean mask of non-zero elements using flat data access.
	var isNonZero []bool
	t.ConstFlatData(func(flatAny any) {
		isNonZero = nonZeroMaskAny(flatAny)
	})

	// Collect indices of non-zero elements using Shape.Iter().
	var indices [][]int64
	for flatIdx, elemIndices := range shape.Iter() {
		if isNonZero[flatIdx] {
			// Copy the indices since elemIndices is reused.
			idx := make([]int64, rank)
			for i, v := range elemIndices {
				idx[i] = int64(v)
			}
			indices = append(indices, idx)
		}
	}

	numNonZero := len(indices)

	// Create output tensor of shape (rank, numNonZero)
	// Each row contains the indices for one dimension.
	if numNonZero == 0 {
		// Return empty tensor of shape (rank, 0)
		return tensors.FromShape(shapes.Make(dtypes.Int64, rank, 0))
	}

	// Transpose: from (numNonZero, rank) to (rank, numNonZero)
	output := make([][]int64, rank)
	for d := range rank {
		output[d] = make([]int64, numNonZero)
		for i := range numNonZero {
			output[d][i] = indices[i][d]
		}
	}

	return tensors.FromValue(output)
}

// nonZeroMask returns a boolean slice indicating which elements are non-zero.
func nonZeroMask[T gotype.Supported](values []T) []bool {
	res := make([]bool, len(values))
	var zero T
	for i, v := range values {
		res[i] = v != zero
	}
	return res
}

// nonZeroMaskAny converts a flat slice of any supported type to a boolean mask.
func nonZeroMaskAny(valuesAny any) []bool {
	switch values := valuesAny.(type) {
	case []bool:
		// For bool, true is non-zero.
		res := make([]bool, len(values))
		copy(res, values)
		return res
	case []float32:
		return nonZeroMask(values)
	case []float64:
		return nonZeroMask(values)
	case []int8:
		return nonZeroMask(values)
	case []int16:
		return nonZeroMask(values)
	case []int32:
		return nonZeroMask(values)
	case []int64:
		return nonZeroMask(values)
	case []uint8:
		return nonZeroMask(values)
	case []uint16:
		return nonZeroMask(values)
	case []uint32:
		return nonZeroMask(values)
	case []uint64:
		return nonZeroMask(values)
	case []float16.Float16:
		res := make([]bool, len(values))
		for i, v := range values {
			res[i] = v.Float32() != 0
		}
		return res
	case []bfloat16.BFloat16:
		res := make([]bool, len(values))
		for i, v := range values {
			res[i] = v.Float32() != 0
		}
		return res
	default:
		exceptions.Panicf("nonZeroMaskAny: unsupported type %T", valuesAny)
		return nil
	}
}
