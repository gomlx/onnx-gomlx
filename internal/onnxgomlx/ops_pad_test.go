package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestPadReflectNegativePads(t *testing.T) {
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
}
