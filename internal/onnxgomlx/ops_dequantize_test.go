package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestDequantizeLinearAxisMetadata(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name              string
		axis              int
		data, scale, zero any
		want              any
	}{
		{
			name: "row scales", axis: 0,
			data: [][]uint8{{10, 20, 30}, {40, 50, 60}}, scale: []float32{.5, 2},
			want: [][]float32{{5, 10, 15}, {80, 100, 120}},
		},
		{
			name: "negative row axis", axis: -2,
			data: [][]uint8{{10, 20, 30}, {40, 50, 60}}, scale: []float32{.5, 2},
			want: [][]float32{{5, 10, 15}, {80, 100, 120}},
		},
		{
			name: "row zero points", axis: 0,
			data: [][]uint8{{10, 20, 30}, {40, 50, 60}}, scale: []float32{.5, 2}, zero: []uint8{1, 10},
			want: [][]float32{{4.5, 9.5, 14.5}, {60, 80, 100}},
		},
		{
			name: "negative row axis and zero points", axis: -2,
			data: [][]uint8{{10, 20, 30}, {40, 50, 60}}, scale: []float32{.5, 2}, zero: []uint8{1, 10},
			want: [][]float32{{4.5, 9.5, 14.5}, {60, 80, 100}},
		},
		{
			name: "negative column axis and zero points", axis: -1,
			data: [][]uint8{{10, 20, 30}, {40, 50, 60}}, scale: []float32{.5, 1, 2}, zero: []uint8{1, 2, 3},
			want: [][]float32{{4.5, 18, 54}, {19.5, 48, 114}},
		},
		{
			name: "signed vector", axis: -1,
			data: []int8{-4, 4, 10}, scale: []float32{.5, 1, 2}, zero: []int8{-2, 2, 3},
			want: []float32{-1, 2, 14},
		},
		{
			name: "scalar quantization ignores axis", axis: 1,
			data: int8(10), scale: float32(.5), zero: int8(2), want: float32(4),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(xs []*Node) *Node {
				var zero *Node
				if len(xs) > 2 {
					zero = xs[2]
				}
				return onnxDequantizeLinear(xs[0], xs[1], zero, test.axis, dtypes.Float32)
			})
			defer exec.Finalize()
			inputs := []any{test.data, test.scale}
			if test.zero != nil {
				inputs = append(inputs, test.zero)
			}
			got := exec.MustExec1(inputs...)
			defer got.FinalizeAll()
			require.Equal(t, test.want, got.Value())
		})
	}
}
