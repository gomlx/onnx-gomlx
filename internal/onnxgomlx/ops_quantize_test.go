package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestQuantizeLinearDefaultMetadata(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name              string
		data, scale, zero any
		attrs             []*protos.AttributeProto
		want              any
	}{
		{
			name: "default unsigned output",
			data: [][]float32{{-10, 0, 100, 400, 600}}, scale: float32(2),
			want: [][]uint8{{0, 0, 50, 200, 255}},
		},
		{
			name: "scalar input",
			data: float32(8), scale: float32(2), want: uint8(4),
		},
		{
			name: "vector input ignores default axis",
			data: []float32{-4, 0, 256, 512}, scale: float32(2),
			want: []uint8{0, 0, 128, 255},
		},
		{
			name: "signed zero point determines dtype",
			data: []float32{-4, 0, 256, 512}, scale: float32(2), zero: int8(1),
			want: []int8{-1, 1, 127, 127},
		},
		{
			name: "unsigned zero point determines dtype",
			data: []float32{-4, 0, 256, 512}, scale: float32(2), zero: uint8(1),
			want: []uint8{0, 1, 129, 255},
		},
		{
			name: "explicit signed output dtype",
			data: []float32{-4, 0, 256, 512}, scale: float32(2),
			attrs: []*protos.AttributeProto{{Name: "output_dtype", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_INT8)}},
			want:  []int8{-2, 0, 127, 127},
		},
		{
			name: "per-axis scales control",
			data: [][]float32{{-4, 0, 256}, {1, 4, 8}}, scale: []float32{1, 2, 4},
			attrs: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: -1}},
			want:  [][]uint8{{0, 0, 64}, {1, 2, 2}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(inputs []*Node) *Node {
				node := &protos.NodeProto{Attribute: test.attrs}
				return convertQuantizeLinear(node, inputs)
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
