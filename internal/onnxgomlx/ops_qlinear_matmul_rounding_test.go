package onnxgomlx

import (
	"testing"

	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
)

func TestQLinearMatMulRoundsBeforeZeroPoint(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ties with odd output zero point", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]uint8{{123}, {125}, {127}, {129}, {131}, {133}})
		b := Const(g, [][]int8{{1}})
		inputs = []*Node{a, b}
		outputs = []*Node{
			onnxQLinearMatMul(a, Const(g, float32(0.5)), Const(g, uint8(128)), b, Const(g, float32(1)), Const(g, int8(0)), Const(g, float32(1)), Const(g, int8(1))),
			onnxQLinearMatMul(a, Const(g, float32(0.5)), Const(g, uint8(128)), b, Const(g, float32(1)), Const(g, int8(0)), Const(g, float32(1)), Const(g, uint8(1))),
		}
		return
	}, []any{[][]int8{{-1}, {-1}, {1}, {1}, {3}, {3}}, [][]uint8{{0}, {0}, {1}, {1}, {3}, {3}}}, -1)
}
