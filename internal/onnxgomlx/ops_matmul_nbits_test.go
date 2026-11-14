// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func packSubByteValues(vals []uint8, bits int) []uint8 {
	elemsPerByte := 8 / bits
	out := make([]uint8, (len(vals)+elemsPerByte-1)/elemsPerByte)
	for i, v := range vals {
		out[i/elemsPerByte] |= (v & ((1 << bits) - 1)) << ((i % elemsPerByte) * bits)
	}
	return out
}

func TestMatMulNBitsFusedAndVariants(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	t.Run("4bit_fused_dynamic_batch", func(t *testing.T) {
		// K=16, N=2, block_size=16, bits=4, default zero point = 8.
		// Row 0 quantized values: all 9 -> dequant = (9 - 8) * 0.5 = 0.5
		// Row 1 quantized values: all 6 -> dequant = (6 - 8) * 2.0 = -4.0
		row0 := make([]uint8, 16)
		row1 := make([]uint8, 16)
		for i := range 16 {
			row0[i] = 9
			row1[i] = 6
		}
		packedBytes := append(packSubByteValues(row0, 4), packSubByteValues(row1, 4)...)

		var graphStr string
		exec := MustNewExec(backend, func(a *Node) *Node {
			g := a.Graph()
			b := Reshape(Const(g, packedBytes), 2, 1, 8)
			scales := Const(g, []float32{0.5, 2.0})
			bias := Const(g, []float32{1.0, -0.5})
			node := &protos.NodeProto{
				OpType: "MatMulNBits",
				Input:  []string{"a", "b", "scales", "", "", "bias"},
				Output: []string{"y"},
				Attribute: []*protos.AttributeProto{
					intAttr("K", 16),
					intAttr("N", 2),
					intAttr("bits", 4),
					intAttr("block_size", 16),
				},
			}
			out := convertTestNode(m, g, node, map[string]*Node{
				"a": a, "b": b, "scales": scales, "bias": bias,
			})[0]
			graphStr = g.String()
			return out
		}).WithDynamicAxes([]string{"batch", "seq", ""})
		defer exec.Finalize()

		ones := make([]float32, 16)
		twos := make([]float32, 16)
		for i := range 16 {
			ones[i] = 1
			twos[i] = 2
		}
		got := exec.MustExec1([][][]float32{{ones, twos}})
		defer got.FinalizeAll()
		require.Contains(t, graphStr, "FusedQuantizedDense")
		// ones: [16*0.5 + 1 = 9, 16*(-4) - 0.5 = -64.5]; twos: [16*1 + 1 = 17, 16*(-8) - 0.5 = -128.5]
		require.InDeltaSlice(t, []float32{9.0, -64.5}, got.Value().([][][]float32)[0][0], 1e-4)
		require.InDeltaSlice(t, []float32{17.0, -128.5}, got.Value().([][][]float32)[0][1], 1e-4)
	})

	t.Run("2bit_non_divisible_K_and_packed_zp", func(t *testing.T) {
		// K=10 (< block_size=16), N=1, bits=2, packed zero_point=1, scale=1.5.
		qVals := make([]uint8, 16)
		for i := range 10 {
			qVals[i] = uint8(i % 4) // 0,1,2,3,0,1,2,3,0,1 -> minus zp(1): -1,0,1,2,-1,0,1,2,-1,0 (sum = 3)
		}
		packedW := packSubByteValues(qVals, 2)
		packedZP := []uint8{1} // 1 block for N=1

		exec := MustNewExec(backend, func(a *Node) *Node {
			g := a.Graph()
			b := Reshape(Const(g, packedW), 1, 1, 4)
			scales := Const(g, []float32{1.5})
			zp := Const(g, packedZP)
			node := &protos.NodeProto{
				OpType: "MatMulNBits",
				Input:  []string{"a", "b", "scales", "zp"},
				Output: []string{"y"},
				Attribute: []*protos.AttributeProto{
					intAttr("K", 10),
					intAttr("N", 1),
					intAttr("bits", 2),
					intAttr("block_size", 16),
				},
			}
			return convertTestNode(m, g, node, map[string]*Node{
				"a": a, "b": b, "scales": scales, "zp": zp,
			})[0]
		})
		defer exec.Finalize()

		in := make([]float32, 10)
		for i := range in {
			in[i] = 1
		}
		got := exec.MustExec1([][]float32{in})
		defer got.FinalizeAll()
		require.InDeltaSlice(t, []float32{4.5}, got.Value().([][]float32)[0], 1e-4)
	})

	t.Run("8bit_float64_activation", func(t *testing.T) {
		qVals := make([]uint8, 16)
		for i := range 16 {
			qVals[i] = 130 // default zp = 128 -> +2
		}
		exec := MustNewExec(backend, func(a *Node) *Node {
			g := a.Graph()
			b := Reshape(Const(g, qVals), 1, 1, 16)
			scales := Const(g, []float32{0.25})
			node := &protos.NodeProto{
				OpType: "MatMulNBits",
				Input:  []string{"a", "b", "scales"},
				Output: []string{"y"},
				Attribute: []*protos.AttributeProto{
					intAttr("K", 16),
					intAttr("N", 1),
					intAttr("bits", 8),
					intAttr("block_size", 16),
				},
			}
			return convertTestNode(m, g, node, map[string]*Node{"a": a, "b": b, "scales": scales})[0]
		})
		defer exec.Finalize()

		in := make([]float64, 16)
		for i := range in {
			in[i] = 1
		}
		got := exec.MustExec1([][]float64{in})
		defer got.FinalizeAll()
		// 16 * ((130 - 128) * 0.25) = 8.0
		require.InDeltaSlice(t, []float64{8.0}, got.Value().([][]float64)[0], 1e-9)
	})
}
