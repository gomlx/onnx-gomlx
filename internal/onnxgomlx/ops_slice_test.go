package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/stretchr/testify/require"
)

func TestSliceOptionalPlaceholders(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name                      string
		inputs                    []string
		starts, ends, axes, steps []int64
		want                      [][]float32
	}{
		{
			name: "omitted axes with steps", inputs: []string{"x", "starts", "ends", "", "steps"},
			starts: []int64{0, 1}, ends: []int64{2, 3}, steps: []int64{1, 1},
			want: [][]float32{{1, 2}, {4, 5}},
		},
		{
			name: "omitted steps", inputs: []string{"x", "starts", "ends", "axes", ""},
			starts: []int64{0}, ends: []int64{2}, axes: []int64{1},
			want: [][]float32{{0, 1}, {3, 4}},
		},
		{
			name: "both omitted", inputs: []string{"x", "starts", "ends", "", ""},
			starts: []int64{0, 1}, ends: []int64{2, 3},
			want: [][]float32{{1, 2}, {4, 5}},
		},
		{
			name: "negative steps without axes", inputs: []string{"x", "starts", "ends", "", "steps"},
			starts: []int64{1, 2}, ends: []int64{-3, -4}, steps: []int64{-1, -1},
			want: [][]float32{{5, 4, 3}, {2, 1, 0}},
		},
		{
			name: "trailing inputs absent control", inputs: []string{"x", "starts", "ends"},
			starts: []int64{0, 1}, ends: []int64{2, 3},
			want: [][]float32{{1, 2}, {4, 5}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				g := x.Graph()
				converted := map[string]*Node{"x": x, "starts": Const(g, test.starts), "ends": Const(g, test.ends)}
				if test.axes != nil {
					converted["axes"] = Const(g, test.axes)
				}
				if test.steps != nil {
					converted["steps"] = Const(g, test.steps)
				}
				inputs := make([]*Node, len(test.inputs))
				for i, name := range test.inputs {
					inputs[i] = converted[name]
				}
				m := &Model{InputsNameSet: sets.Make[string]()}
				m.InputsNameSet.Insert("x")
				node := &protos.NodeProto{Input: test.inputs}
				return convertSlice(m, converted, node, inputs)
			})
			defer exec.Finalize()
			got := exec.MustExec1([][]float32{{0, 1, 2}, {3, 4, 5}})
			defer got.FinalizeAll()
			require.Equal(t, test.want, got.Value())
		})
	}
}
