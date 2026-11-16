// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestGatherBlockQuantized(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	packedWeights := [][]uint8{
		packSubByteValues([]uint8{2, 3, 4, 5, 6, 7, 8, 9}, 4),
		packSubByteValues([]uint8{3, 4, 5, 6, 7, 8, 9, 10}, 4),
	}
	packedZeroPoints := [][]uint8{
		packSubByteValues([]uint8{1, 2}, 4),
		packSubByteValues([]uint8{3, 4}, 4),
	}

	exec := MustNewExec(backend, func(indices *Node) *Node {
		g := indices.Graph()
		weights := Const(g, packedWeights)
		scales := Const(g, [][]float32{{0.5, 2.0}, {1.5, 0.25}})
		zeroPoints := Const(g, packedZeroPoints)
		node := &protos.NodeProto{
			OpType: "GatherBlockQuantized",
			Input:  []string{"weights", "indices", "scales", "zero_points"},
			Output: []string{"output"},
			Attribute: []*protos.AttributeProto{
				intAttr("bits", 4),
				intAttr("block_size", 4),
				intAttr("gather_axis", 0),
				intAttr("quantize_axis", 1),
			},
		}
		return convertTestNode(m, g, node, map[string]*Node{
			"weights": weights, "indices": indices, "scales": scales, "zero_points": zeroPoints,
		})[0]
	}).WithDynamicAxes([]string{"batch", "seq"})
	defer exec.Finalize()

	got := exec.MustExec1([][]int64{{1, 0}})
	defer got.FinalizeAll()
	values := got.Value().([][][]float32)
	require.InDeltaSlice(t, []float32{0, 1.5, 3, 4.5, 0.75, 1, 1.25, 1.5}, values[0][0], 1e-6)
	require.InDeltaSlice(t, []float32{0.5, 1, 1.5, 2, 8, 10, 12, 14}, values[0][1], 1e-6)
}

func TestGatherBlockQuantizedShapeInference(t *testing.T) {
	indices := gatherBlockQuantizedValueInfo("indices", protos.TensorProto_INT64, -1, 3)
	weights := &protos.TensorProto{Name: "weights", DataType: int32(protos.TensorProto_UINT8), Dims: []int64{100, 4}}
	node := &protos.NodeProto{
		OpType: "GatherBlockQuantized",
		Input:  []string{"weights", "indices"},
		Output: []string{"output"},
		Attribute: []*protos.AttributeProto{
			intAttr("bits", 4),
			intAttr("block_size", 4),
			intAttr("gather_axis", 0),
			intAttr("quantize_axis", 1),
		},
	}
	m := &Model{
		Proto: protos.ModelProto{Graph: &protos.GraphProto{
			Input:       []*protos.ValueInfoProto{indices},
			Initializer: []*protos.TensorProto{weights},
		}},
		NodeOutputToNode: map[string]*protos.NodeProto{"output": node},
		VariableNameToValue: map[string]*protos.TensorProto{
			"weights": weights,
		},
	}

	got := m.InferShapeForName("output")
	require.Equal(t, shapes.Shape{
		DType:      dtypes.Float32,
		Dimensions: []int{shapes.DynamicDim, 3, 8},
	}, got)
}

func gatherBlockQuantizedValueInfo(name string, dataType protos.TensorProto_DataType, dims ...int64) *protos.ValueInfoProto {
	shapeDims := make([]*protos.TensorShapeProto_Dimension, len(dims))
	for i, dim := range dims {
		if dim < 0 {
			shapeDims[i] = &protos.TensorShapeProto_Dimension{
				Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: "dynamic"},
			}
			continue
		}
		shapeDims[i] = &protos.TensorShapeProto_Dimension{
			Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: dim},
		}
	}
	return &protos.ValueInfoProto{
		Name: name,
		Type: &protos.TypeProto{
			Value: &protos.TypeProto_TensorType{TensorType: &protos.TypeProto_Tensor{
				ElemType: int32(dataType),
				Shape:    &protos.TensorShapeProto{Dim: shapeDims},
			}},
		},
	}
}
