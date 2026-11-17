// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestGatherBlockQuantized(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		m := createTestModelWithDTypePromoConfig(false, false)

		packedWeights4 := [][]uint8{
			packSubByteValues([]uint8{2, 3, 4, 5, 6, 7, 8, 9}, 4),
			packSubByteValues([]uint8{3, 4, 5, 6, 7, 8, 9, 10}, 4),
		}
		packedZeroPoints4 := [][]uint8{
			packSubByteValues([]uint8{1, 2}, 4),
			packSubByteValues([]uint8{3, 4}, 4),
		}

		for _, fusion := range []string{"true", "false"} {
			t.Run("Q4_WithZP_fusion="+fusion, func(t *testing.T) {
				t.Setenv(FusionEnv, fusion)
				buildFn := func(indices *Node) *Node {
					g := indices.Graph()
					weights := Const(g, packedWeights4)
					scales := Const(g, [][]float32{{0.5, 2.0}, {1.5, 0.25}})
					zeroPoints := Const(g, packedZeroPoints4)
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
				}

				exec := MustNewExec(backend, buildFn)
				if backend.Capabilities().DynamicAxes {
					exec = exec.WithDynamicAxes([]string{"batch", "seq"})
				}
				defer exec.Finalize()

				// Also test negative index (-1 -> row 1).
				got := exec.MustExec1([][]int64{{-1, 0}})
				defer got.FinalizeAll()
				values := got.Value().([][][]float32)
				require.InDeltaSlice(t, []float32{0, 1.5, 3, 4.5, 0.75, 1, 1.25, 1.5}, values[0][0], 1e-6)
				require.InDeltaSlice(t, []float32{0.5, 1, 1.5, 2, 8, 10, 12, 14}, values[0][1], 1e-6)
			})
		}

		t.Run("Q2_Symmetric_Float32", func(t *testing.T) {
			packedWeights2 := [][]uint8{
				packSubByteValues([]uint8{0, 1, 2, 3, 3, 2, 1, 0}, 2),
				packSubByteValues([]uint8{2, 2, 1, 0, 1, 3, 0, 2}, 2),
			}
			scales32 := [][]float32{{0.5, 2.0}, {1.5, 0.25}}
			exec := MustNewExec(backend, func(indices *Node) *Node {
				g := indices.Graph()
				node := &protos.NodeProto{
					OpType: "GatherBlockQuantized",
					Input:  []string{"weights", "indices", "scales"},
					Output: []string{"output"},
					Attribute: []*protos.AttributeProto{
						intAttr("bits", 2),
						intAttr("block_size", 4),
						intAttr("gather_axis", 0),
						intAttr("quantize_axis", 1),
					},
				}
				return convertTestNode(m, g, node, map[string]*Node{
					"weights": Const(g, packedWeights2),
					"indices": indices,
					"scales":  Const(g, scales32),
				})[0]
			})
			defer exec.Finalize()

			got := exec.MustExec1([]int64{1, 0})
			defer got.FinalizeAll()
			values := got.Value().([][]float32)
			require.InDeltaSlice(t, []float32{3, 3, 1.5, 0, 0.25, 0.75, 0, 0.5}, values[0], 1e-6)
			require.InDeltaSlice(t, []float32{0, 0.5, 1, 1.5, 6, 4, 2, 0}, values[1], 1e-6)
		})

		t.Run("Q2_Symmetric_Float16", func(t *testing.T) {
			if !backend.Capabilities().DTypes[dtypes.Float16] {
				t.Skipf("Backend %q does not support Float16", backend.Name())
			}
			packedWeights2 := [][]uint8{
				packSubByteValues([]uint8{0, 1, 2, 3, 3, 2, 1, 0}, 2),
				packSubByteValues([]uint8{2, 2, 1, 0, 1, 3, 0, 2}, 2),
			}
			scales16 := [][]float16.Float16{
				{float16.FromFloat32(0.5), float16.FromFloat32(2.0)},
				{float16.FromFloat32(1.5), float16.FromFloat32(0.25)},
			}
			exec := MustNewExec(backend, func(indices *Node) *Node {
				g := indices.Graph()
				node := &protos.NodeProto{
					OpType: "GatherBlockQuantized",
					Input:  []string{"weights", "indices", "scales"},
					Output: []string{"output"},
					Attribute: []*protos.AttributeProto{
						intAttr("bits", 2),
						intAttr("block_size", 4),
						intAttr("gather_axis", 0),
						intAttr("quantize_axis", 1),
					},
				}
				return convertTestNode(m, g, node, map[string]*Node{
					"weights": Const(g, packedWeights2),
					"indices": indices,
					"scales":  Const(g, scales16),
				})[0]
			})
			defer exec.Finalize()

			got := exec.MustExec1([]int64{1, 0})
			defer got.FinalizeAll()
			require.Equal(t, dtypes.Float16, got.DType())
			raw := got.Value().([][]float16.Float16)
			want0 := []float32{3, 3, 1.5, 0, 0.25, 0.75, 0, 0.5}
			want1 := []float32{0, 0.5, 1, 1.5, 6, 4, 2, 0}
			for c := range 8 {
				require.InDelta(t, want0[c], raw[0][c].Float32(), 1e-3)
				require.InDelta(t, want1[c], raw[1][c].Float32(), 1e-3)
			}
		})

		t.Run("Q8_Int8_NonDivisibleBlock", func(t *testing.T) {
			weights8 := [][]int8{
				{2, 3, 4, 5, 6, 7},
				{3, 4, 5, 6, 7, 8},
			}
			zp8 := [][]int8{
				{1, 2},
				{3, 4},
			}
			exec := MustNewExec(backend, func(indices *Node) *Node {
				g := indices.Graph()
				node := &protos.NodeProto{
					OpType: "GatherBlockQuantized",
					Input:  []string{"weights", "indices", "scales", "zero_points"},
					Output: []string{"output"},
					Attribute: []*protos.AttributeProto{
						intAttr("bits", 8),
						intAttr("block_size", 4),
						intAttr("gather_axis", 0),
						intAttr("quantize_axis", 1),
					},
				}
				return convertTestNode(m, g, node, map[string]*Node{
					"weights":     Const(g, weights8),
					"indices":     indices,
					"scales":      Const(g, [][]float32{{0.5, 2.0}, {1.5, 0.25}}),
					"zero_points": Const(g, zp8),
				})[0]
			})
			defer exec.Finalize()

			got := exec.MustExec1([]int32{1, 0})
			defer got.FinalizeAll()
			values := got.Value().([][]float32)
			require.InDeltaSlice(t, []float32{0, 1.5, 3, 4.5, 0.75, 1}, values[0], 1e-6)
			require.InDeltaSlice(t, []float32{0.5, 1, 1.5, 2, 8, 10}, values[1], 1e-6)
		})
	})
}

func TestGatherBlockQuantizedShapeInference(t *testing.T) {
	indices := gatherBlockQuantizedValueInfo("indices", protos.TensorProto_INT64, -1, 3)
	weights := &protos.TensorProto{Name: "weights", DataType: int32(protos.TensorProto_UINT8), Dims: []int64{100, 4}}
	scales := &protos.TensorProto{Name: "scales", DataType: int32(protos.TensorProto_FLOAT16), Dims: []int64{100, 2}}
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
			Initializer: []*protos.TensorProto{weights, scales},
		}},
		NodeOutputToNode: map[string]*protos.NodeProto{"output": node},
		VariableNameToValue: map[string]*protos.TensorProto{
			"weights": weights,
			"scales":  scales,
		},
	}

	got := m.InferShapeForName("output")
	require.Equal(t, shapes.Shape{
		DType:      dtypes.Float32,
		Dimensions: []int{shapes.DynamicDim, 3, 8},
	}, got)

	node.Input = []string{"weights", "indices", "scales"}
	gotF16 := m.InferShapeForName("output")
	require.Equal(t, shapes.Shape{
		DType:      dtypes.Float16,
		Dimensions: []int{shapes.DynamicDim, 3, 8},
	}, gotF16)
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
