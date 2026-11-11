package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestPadAxes(t *testing.T) {
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
}
