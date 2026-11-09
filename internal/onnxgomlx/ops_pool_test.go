package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestPoolDefaultStrides(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name    string
		convert func(*Model, map[string]*Node, *protos.NodeProto, []*Node) *Node
		stride  []int64
		want    []float32
	}{
		{name: "average omitted", convert: convertAveragePool, want: []float32{1.5, 2.5, 3.5, 4.5}},
		{name: "max omitted", convert: convertMaxPool, want: []float32{2, 3, 4, 5}},
		{name: "average explicit one", convert: convertAveragePool, stride: []int64{1}, want: []float32{1.5, 2.5, 3.5, 4.5}},
		{name: "max explicit one", convert: convertMaxPool, stride: []int64{1}, want: []float32{2, 3, 4, 5}},
		{name: "average explicit two", convert: convertAveragePool, stride: []int64{2}, want: []float32{1.5, 3.5}},
		{name: "max explicit two", convert: convertMaxPool, stride: []int64{2}, want: []float32{2, 4}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				node := &protos.NodeProto{Attribute: []*protos.AttributeProto{{Name: "kernel_shape", Type: protos.AttributeProto_INTS, Ints: []int64{2}}}}
				if test.stride != nil {
					node.Attribute = append(node.Attribute, &protos.AttributeProto{Name: "strides", Type: protos.AttributeProto_INTS, Ints: test.stride})
				}
				return test.convert(nil, nil, node, []*Node{x})
			})
			defer exec.Finalize()
			got := exec.MustExec1([][][]float32{{{1, 2, 3, 4, 5}}})
			defer got.FinalizeAll()
			require.Equal(t, [][][]float32{{test.want}}, got.Value())
		})
	}
}

func TestAveragePoolRejectsNonUnitDilation(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	exec := MustNewExec(backend, func(x *Node) *Node {
		node := &protos.NodeProto{Attribute: []*protos.AttributeProto{
			{Name: "kernel_shape", Type: protos.AttributeProto_INTS, Ints: []int64{2}},
			{Name: "strides", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
			{Name: "dilations", Type: protos.AttributeProto_INTS, Ints: []int64{2}},
		}}
		return convertAveragePool(nil, nil, node, []*Node{x})
	})
	defer exec.Finalize()
	err = exceptions.TryCatch[error](func() {
		got := exec.MustExec1([][][]float32{{{1, 2, 3, 4, 5}}})
		defer got.FinalizeAll()
	})
	require.ErrorContains(t, err, "dilations")
}
