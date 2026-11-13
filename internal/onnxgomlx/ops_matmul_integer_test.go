package onnxgomlx

import (
	"testing"

	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
)

func TestMatMulIntegerSingleElementZeroPoints(t *testing.T) {
	graphtest.RunTestGraphFn(t, "single-element zero points", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]uint8{{1, 2, 3}, {4, 5, 6}})
		b := Const(g, [][]uint8{{1, 2}, {3, 4}, {5, 6}})
		inputs = []*Node{a, b}
		outputs = []*Node{onnxMatMulInteger(a, b, Const(g, []uint8{1}), Const(g, []uint8{1}))}
		return
	}, []any{[][]int32{{10, 13}, {28, 40}}}, -1)
}
