package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestReshapeAllowZero(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name      string
		shape     []int
		target    []int64
		allowzero int64
		want      []int
	}{
		{name: "move zero axis", shape: []int{0, 3}, target: []int64{3, 0}, allowzero: 1, want: []int{3, 0}},
		{name: "insert zero axis", shape: []int{0, 3}, target: []int64{1, 0, 3}, allowzero: 1, want: []int{1, 0, 3}},
		{name: "preserve empty shape", shape: []int{0, 3}, target: []int64{0, 3}, allowzero: 1, want: []int{0, 3}},
		{name: "copy dimension control", shape: []int{2, 3}, target: []int64{0, 3}, allowzero: 0, want: []int{2, 3}},
		{name: "inferred dimension control", shape: []int{0, 3}, target: []int64{-1, 3}, allowzero: 0, want: []int{0, 3}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				target := Const(x.Graph(), test.target)
				node := &protos.NodeProto{Input: []string{"x", "shape"}, Attribute: []*protos.AttributeProto{
					{Name: "allowzero", Type: protos.AttributeProto_INT, I: test.allowzero},
				}}
				return convertReshape(nil, map[string]*Node{"shape": target}, node, []*Node{x, target})
			})
			defer exec.Finalize()
			size := 1
			for _, dim := range test.shape {
				size *= dim
			}
			data := tensors.FromFlatDataAndDimensions(make([]float32, size), test.shape...)
			defer data.FinalizeAll()
			got := exec.MustExec1(data)
			defer got.FinalizeAll()
			require.Equal(t, test.want, got.Shape().Dimensions)
		})
	}
}
