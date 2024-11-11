package onnx

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestONNXGather(t *testing.T) {
	graphtest.RunTestGraphFn(t, "onnxGather(axis=0)", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]float32{{1.0, 1.2}, {2.3, 3.4}, {4.5, 5.7}})
		indices := Const(g, [][]int32{{0, 1}, {1, 2}})
		inputs = []*Node{data, indices}
		outputs = []*Node{onnxGather(data, indices, 0)}
		return
	}, []any{
		[][][]float32{
			{
				{1.0, 1.2},
				{2.3, 3.4},
			},
			{
				{2.3, 3.4},
				{4.5, 5.7},
			},
		},
	}, -1)

	graphtest.RunTestGraphFn(t, "onnxGather(axis=1)", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]float32{
			{1.0, 1.2, 1.9},
			{2.3, 3.4, 3.9},
			{4.5, 5.7, 5.9},
		})
		indices := Const(g, [][]int32{{0, 2}})
		inputs = []*Node{data, indices}
		outputs = []*Node{onnxGather(data, indices, 1)}
		return
	}, []any{
		[][][]float32{
			{{1.0, 1.9}},
			{{2.3, 3.9}},
			{{4.5, 5.9}},
		},
	}, -1)
}

func TestTile(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Tile 1D", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, []float32{1, 2})
		inputs = []*Node{operand}
		outputs = []*Node{tile(operand, []int{2})}
		return
	}, []any{
		[]float32{1, 2, 1, 2},
	}, -1)

	graphtest.RunTestGraphFn(t, "Tile 2D", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, [][]float32{{1.0, 1.2}, {2.3, 3.4}, {4.5, 5.7}})
		inputs = []*Node{operand}
		outputs = []*Node{tile(operand, []int{1, 2})}
		return
	}, []any{
		[][]float32{
			{1.0, 1.2, 1.0, 1.2},
			{2.3, 3.4, 2.3, 3.4},
			{4.5, 5.7, 4.5, 5.7},
		},
	}, -1)
}

func TestRangeCount(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	testFn := func(start, limit, delta any, want int) {
		startT := tensors.FromAnyValue(start)
		limitT := tensors.FromAnyValue(limit)
		deltaT := tensors.FromAnyValue(delta)
		got := rangeCount(backend, startT, limitT, deltaT)
		fmt.Printf("\trangeCount(start=%s, limit=%s, delta=%s) = %d (want %d)\n", startT, limitT, deltaT, got, want)
		assert.Equal(t, want, got)
	}

	testFn(uint8(3), uint8(9), uint8(3), 2)
	testFn(uint8(3), uint8(8), uint8(3), 2)
	testFn(uint8(3), uint8(7), uint8(3), 2)
	testFn(float32(3), float32(9.1), float32(3), 3)
	testFn(int32(10), int32(4), int32(-2), 3)
	testFn(int32(10), int32(5), int32(-2), 3)
	testFn(float64(10), float64(3.9), float64(-2), 4)
}
