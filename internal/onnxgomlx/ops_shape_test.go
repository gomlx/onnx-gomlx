package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestShapeAttributeBounds(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name  string
		data  any
		attrs map[string]int64
		want  []int64
	}{
		{name: "default end", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, want: []int64{2, 3}},
		{name: "explicit zero end", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"end": 0}, want: nil},
		{name: "clamp both bounds", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"start": -100, "end": 100}, want: []int64{2, 3}},
		{name: "negative bounds", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"start": -1}, want: []int64{3}},
		{name: "end before start", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"start": 1, "end": 0}, want: nil},
		{name: "start beyond rank", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"start": 100}, want: nil},
		{name: "negative end beyond rank", data: [][]float32{{1, 2, 3}, {4, 5, 6}}, attrs: map[string]int64{"end": -100}, want: nil},
		{name: "scalar shape", data: float32(1), want: nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				node := &protos.NodeProto{}
				for name, value := range test.attrs {
					node.Attribute = append(node.Attribute, &protos.AttributeProto{Name: name, Type: protos.AttributeProto_INT, I: value})
				}
				return convertShape(node, []*Node{x})
			})
			defer exec.Finalize()
			got := exec.MustExec1(test.data)
			defer got.FinalizeAll()
			require.Equal(t, []int{len(test.want)}, got.Shape().Dimensions)
			require.Equal(t, test.want, tensors.MustCopyFlatData[int64](got))
		})
	}
}
