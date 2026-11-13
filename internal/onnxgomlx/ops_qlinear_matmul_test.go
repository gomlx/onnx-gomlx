package onnxgomlx

import (
	"testing"

	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
)

func TestQLinearMatMulSingleElementParameters(t *testing.T) {
	graphtest.RunTestGraphFn(t, "single-element parameters", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]uint8{{129}})
		b := Const(g, [][]int8{{1}})
		inputs = []*Node{a, b}
		outputs = []*Node{onnxQLinearMatMul(a, Const(g, []float32{0.5}), Const(g, []uint8{128}), b, Const(g, []float32{1}), Const(g, []int8{0}), Const(g, []float32{1}), Const(g, []uint8{2}))}
		return
	}, []any{[][]uint8{{2}}}, -1)
}

func TestQLinearMatMulPerRowAndColumnParameters(t *testing.T) {
	graphtest.RunTestGraphFn(t, "per-row and per-column parameters", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]uint8{{5, 7}, {9, 11}})
		b := Const(g, [][]int8{{1, 2, 3}, {4, 5, 6}})
		inputs = []*Node{a, b}
		outputs = []*Node{onnxQLinearMatMul(a, Const(g, []float32{0.5, 1}), Const(g, []uint8{1, 2}), b, Const(g, []float32{0.25, 0.5, 1}), Const(g, []int8{0, 1, 2}), Const(g, float32(0.5)), Const(g, uint8(0)))}
		return
	}, []any{[][]uint8{{7, 14, 28}, {22, 43, 86}}}, -1)
}

func TestQLinearMatMulBroadcastedBatchParameters(t *testing.T) {
	expected := make([][][][]uint8, 2)
	for batch := range expected {
		expected[batch] = make([][][]uint8, 3)
		for inner := range expected[batch] {
			expected[batch][inner] = [][]uint8{{7, 14, 28}, {22, 43, 86}}
		}
	}
	graphtest.RunTestGraphFn(t, "broadcasted batch parameters", func(g *Graph) (inputs, outputs []*Node) {
		a := BroadcastToDims(Const(g, [][][][]uint8{{{{5, 7}, {9, 11}}}}), 2, 1, 2, 2)
		b := BroadcastToDims(Const(g, [][][]int8{{{1, 2, 3}, {4, 5, 6}}}), 3, 2, 3)
		aScale := BroadcastToDims(Const(g, [][][][]float32{{{{0.5}, {1}}}}), 2, 1, 2, 1)
		bScale := BroadcastToDims(Const(g, [][][]float32{{{0.25, 0.5, 1}}}), 3, 1, 3)
		aZero := BroadcastToDims(Const(g, [][][][]uint8{{{{1}, {2}}}}), 2, 1, 2, 1)
		bZero := BroadcastToDims(Const(g, [][][]int8{{{0, 1, 2}}}), 3, 1, 3)
		inputs = []*Node{a, b}
		outputs = []*Node{onnxQLinearMatMul(a, aScale, aZero, b, bScale, bZero, Const(g, float32(0.5)), Const(g, uint8(0)))}
		return
	}, []any{expected}, -1)
}
