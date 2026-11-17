// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/nn"
)

// convertGatherBlockQuantized converts ORT's com.microsoft::GatherBlockQuantized operator
// via nn.QuantizedGather so backends with OpTypeQuantizedGather run a fused
// row-gather + dequantization kernel, while other backends use GoMLX's decomposed fallback
// (gathering packed Uint8 rows via PackedDType before unpacking).
func convertGatherBlockQuantized(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		exceptions.Panicf("GatherBlockQuantized requires data, indices and scales in %s", NodeToString(node))
	}
	weights, indices, scales := inputs[0], inputs[1], inputs[2]
	var zeroPoints *Node
	if len(inputs) > 3 {
		zeroPoints = inputs[3]
	}

	bits := GetIntAttrOr(node, "bits", 4)
	blockSize := GetIntAttrOr(node, "block_size", 32)
	gatherAxis := GetIntAttrOr(node, "gather_axis", 0)
	quantizeAxis := GetIntAttrOr(node, "quantize_axis", 1)
	if weights.Rank() != 2 || scales.Rank() != 2 || (zeroPoints != nil && zeroPoints.Rank() != 2) {
		exceptions.Panicf("GatherBlockQuantized: only rank-2 tables are supported, got weights=%s scales=%s", weights.Shape(), scales.Shape())
	}
	if gatherAxis < 0 {
		gatherAxis += weights.Rank()
	}
	if quantizeAxis < 0 {
		quantizeAxis += weights.Rank()
	}
	if (bits != 2 && bits != 4 && bits != 8) || blockSize <= 0 || blockSize&(blockSize-1) != 0 {
		exceptions.Panicf("GatherBlockQuantized: unsupported bits=%d or block_size=%d", bits, blockSize)
	}
	if gatherAxis != 0 || quantizeAxis != 1 {
		exceptions.Panicf("GatherBlockQuantized: only gather_axis=0 and quantize_axis=1 are supported, got gather_axis=%d quantize_axis=%d", gatherAxis, quantizeAxis)
	}

	vocabSize := weights.Shape().Dim(0)
	packedWidth := weights.Shape().Dim(1)
	if vocabSize <= 0 || packedWidth <= 0 {
		exceptions.Panicf("GatherBlockQuantized: table dimensions must be static and positive, got %s", weights.Shape())
	}

	packedWeights := bits < 8 && weights.DType() == dtypes.Uint8
	var hiddenSize int
	switch {
	case packedWeights:
		hiddenSize = packedWidth * (8 / bits)
	case bits == 8 && (weights.DType() == dtypes.Uint8 || weights.DType() == dtypes.Int8):
		hiddenSize = packedWidth
	case (bits == 4 && (weights.DType() == dtypes.Uint4 || weights.DType() == dtypes.Int4)) ||
		(bits == 2 && (weights.DType() == dtypes.Uint2 || weights.DType() == dtypes.Int2)):
		hiddenSize = packedWidth
	default:
		exceptions.Panicf("GatherBlockQuantized: unsupported weights dtype %s for bits=%d", weights.DType(), bits)
	}

	blocks := (hiddenSize + blockSize - 1) / blockSize
	if scales.Shape().Dim(0) != vocabSize || scales.Shape().Dim(1) != blocks {
		exceptions.Panicf("GatherBlockQuantized: incompatible weights %s and scales %s (expected [%d, %d])",
			weights.Shape(), scales.Shape(), vocabSize, blocks)
	}
	if zeroPoints != nil {
		if zeroPoints.Shape().Dim(0) != vocabSize {
			exceptions.Panicf("GatherBlockQuantized: incompatible zeroPoints %s for vocabSize=%d", zeroPoints.Shape(), vocabSize)
		}
		if bits < 8 && zeroPoints.DType() == dtypes.Uint8 {
			expectedPackedZPCols := (blocks*bits + 7) / 8
			if zeroPoints.Shape().Dim(1) != expectedPackedZPCols {
				exceptions.Panicf("GatherBlockQuantized: expected packed zeroPoints [%d, %d], got %s",
					vocabSize, expectedPackedZPCols, zeroPoints.Shape())
			}
		} else if zeroPoints.Shape().Dim(1) != blocks {
			exceptions.Panicf("GatherBlockQuantized: expected zeroPoints [%d, %d], got %s",
				vocabSize, blocks, zeroPoints.Shape())
		}
	}

	// Normalize negative ONNX indices; nn.QuantizedGather requires [0, vocabSize) and trailing dim 1.
	normIndices := normalizeNegativeIndicesForAxis(weights, indices, 0)
	gatherIndices := ExpandAxes(normIndices, -1)

	if !packedWeights {
		return nn.QuantizedGather(weights, gatherIndices, &Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     scales,
			ZeroPoint: zeroPoints,
			BlockAxis: 1,
			BlockSize: blockSize,
		})
	}

	subDType := dtypes.Uint4
	if bits == 2 {
		subDType = dtypes.Uint2
	}
	zpNode := zeroPoints
	if zpNode != nil && zpNode.DType() == dtypes.Uint8 {
		backend := weights.Graph().Backend()
		if backend != nil && backend.Capabilities().DTypes[subDType] && zpNode.Shape().Dim(1)*(8/bits) == blocks {
			zpNode = Reshape(Bitcast(zpNode, subDType), vocabSize, blocks)
		} else {
			zpUnpacked := ConvertDType(unpackNBits(zpNode, bits), dtypes.Uint8)
			if zpUnpacked.Shape().Dim(1) != blocks {
				zpUnpacked = Slice(zpUnpacked, AxisRange(), AxisRange(0, blocks))
			}
			zpNode = zpUnpacked
		}
	}
	return nn.QuantizedGather(weights, gatherIndices, &Quantization{
		Scheme:      compute.QuantLinear,
		Scale:       scales,
		ZeroPoint:   zpNode,
		BlockAxis:   1,
		BlockSize:   blockSize,
		PackedDType: subDType,
	})
}
