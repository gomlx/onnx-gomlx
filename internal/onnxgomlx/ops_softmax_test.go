package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestSoftmaxOpsetSemantics(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	flat := []float32{0.032058604, 0.08714432, 0.23688282, 0.6439143}
	columns := []float32{0.11920292, 0.11920292, 0.8807971, 0.8807971}
	tests := []struct {
		name  string
		opset int64
		data  any
		axis  *int64
		shape []int
		want  []float32
	}{
		{"legacy axis zero", 11, [][]float32{{0, 1}, {2, 3}}, int64Ptr(0), []int{2, 2}, flat},
		{"legacy negative axis", 12, [][][]float32{{{0, 1}, {2, 3}}}, int64Ptr(-2), []int{1, 2, 2}, flat},
		{"legacy default axis", 11, [][][]float32{{{0, 1}, {2, 3}}}, nil, []int{1, 2, 2}, flat},
		{"modern axis zero", 13, [][]float32{{0, 1}, {2, 3}}, int64Ptr(0), []int{2, 2}, columns},
		{"modern negative axis", 18, [][][]float32{{{0, 1}, {2, 3}}}, int64Ptr(-2), []int{1, 2, 2}, columns},
		{"legacy vector", 11, []float32{0, 1, 2, 3}, int64Ptr(0), []int{4}, flat},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := &Model{Proto: protos.ModelProto{OpsetImport: []*protos.OperatorSetIdProto{{Version: test.opset}}}}
			exec := MustNewExec(backend, func(x *Node) *Node {
				node := &protos.NodeProto{}
				if test.axis != nil {
					node.Attribute = []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: *test.axis}}
				}
				return convertSoftmax(model, node, []*Node{x})
			})
			defer exec.Finalize()
			got := exec.MustExec1(test.data)
			defer got.FinalizeAll()
			require.Equal(t, test.shape, got.Shape().Dimensions)
			require.InDeltaSlice(t, test.want, tensors.MustCopyFlatData[float32](got), 1e-6)
		})
	}
}

func int64Ptr(value int64) *int64 { return &value }
