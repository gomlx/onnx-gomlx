// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertGatherBlockQuantized expands an ORT block-quantized embedding table and gathers rows.
// The current ORT export uses unsigned packed values, per-row scales and packed zero points.
func convertGatherBlockQuantized(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 4 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil || inputs[3] == nil {
		exceptions.Panicf("GatherBlockQuantized requires packed weights, indices, scales and zero points in %s", NodeToString(node))
	}
	bits := GetIntAttrOr(node, "bits", 4)
	blockSize := GetIntAttrOr(node, "block_size", 32)
	gatherAxis := GetIntAttrOr(node, "gather_axis", 0)
	quantizeAxis := GetIntAttrOr(node, "quantize_axis", 1)
	if (bits != 2 && bits != 4 && bits != 8) || blockSize <= 0 || blockSize&(blockSize-1) != 0 {
		exceptions.Panicf("GatherBlockQuantized: unsupported bits=%d or block_size=%d", bits, blockSize)
	}
	if gatherAxis != 0 || quantizeAxis != 1 || inputs[0].Rank() != 2 || inputs[2].Rank() != 2 || inputs[3].Rank() != 2 {
		exceptions.Panicf("GatherBlockQuantized: only rank-2 tables with gather_axis=0 and quantize_axis=1 are supported")
	}
	weights, indices, scales, zeroPoints := inputs[0], inputs[1], inputs[2], inputs[3]
	packedWidth := weights.Shape().Dim(1)
	if packedWidth <= 0 {
		exceptions.Panicf("GatherBlockQuantized: packed weight width must be static, got %s", weights.Shape())
	}
	hiddenSize := packedWidth * (8 / bits)
	if hiddenSize%blockSize != 0 {
		exceptions.Panicf("GatherBlockQuantized: hidden size %d must be divisible by block size %d", hiddenSize, blockSize)
	}
	blocks := hiddenSize / blockSize
	if weights.Shape().Dim(0) != scales.Shape().Dim(0) || weights.Shape().Dim(0) != zeroPoints.Shape().Dim(0) || scales.Shape().Dim(1) != blocks || zeroPoints.Shape().Dim(1)*(8/bits) != blocks {
		exceptions.Panicf("GatherBlockQuantized: incompatible weights %s, scales %s and zero points %s", weights.Shape(), scales.Shape(), zeroPoints.Shape())
	}

	// Gather the requested packed rows first, keeping the dequantization small for token batches.
	quantized := onnxGather(weights, indices, 0)
	quantized = unpackNBits(quantized, bits)
	qSpecs := DimensionSpecsFor(indices)
	qSpecs = append(qSpecs, StaticDim(blocks), StaticDim(blockSize))
	quantized = DynamicReshape(quantized, qSpecs...)
	quantized = ConvertDType(quantized, dtypes.Float32)

	scaleRows := onnxGather(scales, indices, 0)
	scaleSpecs := DimensionSpecsFor(indices)
	scaleSpecs = append(scaleSpecs, StaticDim(blocks), StaticDim(1))
	scaleRows = DynamicReshape(ConvertDType(scaleRows, dtypes.Float32), scaleSpecs...)

	zeroRows := unpackNBits(onnxGather(zeroPoints, indices, 0), bits)
	zeroSpecs := DimensionSpecsFor(indices)
	zeroSpecs = append(zeroSpecs, StaticDim(blocks), StaticDim(1))
	zeroRows = DynamicReshape(ConvertDType(zeroRows, dtypes.Float32), zeroSpecs...)

	dequantized := Mul(Sub(quantized, zeroRows), scaleRows)
	outputSpecs := DimensionSpecsFor(indices)
	outputSpecs = append(outputSpecs, StaticDim(hiddenSize))
	return DynamicReshape(dequantized, outputSpecs...)
}
