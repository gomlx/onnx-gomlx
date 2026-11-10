package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestFlattenAxisBoundary(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name string
		data any
		axis int64
		want [][]float32
	}{
		{name: "scalar", data: float32(3), axis: 0, want: [][]float32{{3}}},
		{name: "vector at rank", data: []float32{1, 2, 3}, axis: 1, want: [][]float32{{1}, {2}, {3}}},
		{name: "matrix at rank", data: [][]float32{{1, 2}, {3, 4}}, axis: 2, want: [][]float32{{1}, {2}, {3}, {4}}},
		{name: "matrix at zero", data: [][]float32{{1, 2}, {3, 4}}, axis: 0, want: [][]float32{{1, 2, 3, 4}}},
		{name: "negative rank", data: [][]float32{{1, 2}, {3, 4}}, axis: -2, want: [][]float32{{1, 2, 3, 4}}},
		{name: "negative axis", data: [][]float32{{1, 2}, {3, 4}}, axis: -1, want: [][]float32{{1, 2}, {3, 4}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				node := &protos.NodeProto{Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: test.axis}}}
				return convertFlatten(node, []*Node{x})
			})
			defer exec.Finalize()
			got := exec.MustExec1(test.data)
			defer got.FinalizeAll()
			require.Equal(t, test.want, got.Value())
		})
	}
}

func TestFlattenInvalidAxes(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	for _, axis := range []int64{-3, 3} {
		exec := MustNewExec(backend, func(x *Node) *Node {
			node := &protos.NodeProto{Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: axis}}}
			return convertFlatten(node, []*Node{x})
		})
		require.Panics(t, func() { exec.MustExec1([][]float32{{1, 2}, {3, 4}}) })
		exec.Finalize()
	}
}
