package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"

	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestPad(t *testing.T) {
	t.Run("ReflectNegativePads", func(t *testing.T) {
		backend, err := gobackend.New("")
		require.NoError(t, err)
		defer backend.Finalize()
		tests := []struct {
			name string
			data any
			pads []int
			want any
		}{
			{
				name: "crop left", data: []float32{0, 1, 2, 3, 4}, pads: []int{-1, 0},
				want: []float32{1, 2, 3, 4},
			},
			{
				name: "crop right", data: []float32{0, 1, 2, 3, 4}, pads: []int{0, -2},
				want: []float32{0, 1, 2},
			},
			{
				name: "crop left and reflect right", data: []float32{0, 1, 2, 3, 4}, pads: []int{-1, 2},
				want: []float32{1, 2, 3, 4, 3, 2},
			},
			{
				name: "reflect left and crop right", data: []float32{0, 1, 2, 3, 4}, pads: []int{2, -1},
				want: []float32{2, 1, 0, 1, 2, 3},
			},
			{
				name: "crop both sides", data: []float32{0, 1, 2, 3, 4}, pads: []int{-1, -1},
				want: []float32{1, 2, 3},
			},
			{
				name: "positive padding control", data: []float32{0, 1, 2, 3, 4}, pads: []int{2, 1},
				want: []float32{2, 1, 0, 1, 2, 3, 4, 3},
			},
			{
				name: "multiple axes", data: [][]float32{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
				pads: []int{-1, 1, 0, -1},
				want: [][]float32{{5, 4, 5, 6}, {9, 8, 9, 10}},
			},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				exec := MustNewExec(backend, func(x *Node) *Node {
					return convertPadReflect(x, test.pads, x.Rank())
				})
				defer exec.Finalize()
				got := exec.MustExec1(test.data)
				defer got.FinalizeAll()
				require.Equal(t, test.want, got.Value())
			})
		}
	})
	t.Run("Axes", func(t *testing.T) {
		backend, err := gobackend.New("")
		require.NoError(t, err)
		defer backend.Finalize()
		constant := [][]float32{{0, 1, 2, 3}, {0, 4, 5, 6}}
		reflected := [][]float32{{2, 1, 2, 3}, {5, 4, 5, 6}}
		tests := []struct {
			name         string
			mode         string
			axes         []int64
			pads         []int64
			explicitFill bool
			want         [][]float32
		}{
			{"reordered axes", "constant", []int64{1, 0}, []int64{1, 0, 0, 0}, true, constant},
			{"omitted fill", "constant", []int64{1, 0}, []int64{1, 0, 0, 0}, false, constant},
			{"negative axes", "constant", []int64{-1, -2}, []int64{1, 0, 0, 0}, false, constant},
			{"subset of axes", "constant", []int64{-1}, []int64{1, 0}, false, constant},
			{"reflect reordered axes", "reflect", []int64{1, 0}, []int64{1, 0, 0, 0}, false, reflected},
			{"reflect negative axis", "reflect", []int64{-1}, []int64{1, 0}, false, reflected},
			{"default axes", "constant", nil, []int64{0, 1, 0, 0}, false, constant},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				exec := MustNewExec(backend, func(x *Node) *Node {
					pads := Const(x.Graph(), test.pads)
					inputs := []*Node{x, pads}
					node := &protos.NodeProto{Input: []string{"x", "pads"}, Attribute: []*protos.AttributeProto{{Name: "mode", Type: protos.AttributeProto_STRING, S: []byte(test.mode)}}}
					converted := map[string]*Node{"pads": pads}
					if test.axes != nil {
						var fill *Node
						fillName := ""
						if test.explicitFill {
							fill = Const(x.Graph(), float32(0))
							fillName = "fill"
						}
						axes := Const(x.Graph(), test.axes)
						inputs = append(inputs, fill, axes)
						node.Input = append(node.Input, fillName, "axes")
						converted["axes"] = axes
					}
					return convertPad(&Model{}, converted, node, inputs)
				})
				defer exec.Finalize()
				got := exec.MustExec1([][]float32{{1, 2, 3}, {4, 5, 6}})
				defer got.FinalizeAll()
				require.Equal(t, test.want, got.Value())
			})
		}
	})
}
