package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestScatterNDNegativeIndices(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	data := [][]float32{{1, 2, 3}, {4, 5, 6}}
	tests := []struct {
		name, reduction  string
		indices, updates any
		want             [][]float32
	}{
		{
			name: "point update", reduction: "none",
			indices: [][]int64{{-1, -1}, {-2, -2}}, updates: []float32{-10, 20},
			want: [][]float32{{1, 20, 3}, {4, 5, -10}},
		},
		{
			name: "point sum", reduction: "add",
			indices: [][]int64{{-1, -1}, {-2, -2}}, updates: []float32{-10, 20},
			want: [][]float32{{1, 22, 3}, {4, 5, -4}},
		},
		{
			name: "point min", reduction: "min",
			indices: [][]int64{{-1, -1}, {-2, -2}}, updates: []float32{-10, 20},
			want: [][]float32{{1, 2, 3}, {4, 5, -10}},
		},
		{
			name: "point max", reduction: "max",
			indices: [][]int64{{-1, -1}, {-2, -2}}, updates: []float32{-10, 20},
			want: [][]float32{{1, 20, 3}, {4, 5, 6}},
		},
		{
			name: "slice update", reduction: "none",
			indices: [][]int64{{-1}, {0}}, updates: [][]float32{{10, 20, 30}, {-1, -2, -3}},
			want: [][]float32{{-1, -2, -3}, {10, 20, 30}},
		},
		{
			name: "repeated slice sum", reduction: "add",
			indices: [][]int64{{-1}, {1}}, updates: [][]float32{{10, 20, 30}, {-1, -2, -3}},
			want: [][]float32{{1, 2, 3}, {13, 23, 33}},
		},
		{
			name: "higher rank indices", reduction: "none",
			indices: [][][]int64{{{-1, -1}}, {{-2, -3}}}, updates: [][]float32{{10}, {20}},
			want: [][]float32{{20, 2, 3}, {4, 5, 10}},
		},
		{
			name: "single tuple", reduction: "none",
			indices: []int64{-1, -1}, updates: float32(10),
			want: [][]float32{{1, 2, 3}, {4, 5, 10}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(data, indices, updates *Node) *Node {
				node := &protos.NodeProto{Attribute: []*protos.AttributeProto{
					{Name: "reduction", Type: protos.AttributeProto_STRING, S: []byte(test.reduction)},
				}}
				return convertScatterND(nil, nil, node, []*Node{data, indices, updates})
			})
			defer exec.Finalize()
			got := exec.MustExec1(data, test.indices, test.updates)
			defer got.FinalizeAll()
			require.Equal(t, test.want, got.Value())
		})
	}
}
