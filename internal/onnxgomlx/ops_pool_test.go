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

func TestMaxPoolWindowDilation(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	exec := MustNewExec(backend, func(x *Node) *Node {
		node := &protos.NodeProto{Attribute: []*protos.AttributeProto{
			{Name: "kernel_shape", Type: protos.AttributeProto_INTS, Ints: []int64{2, 2}},
			{Name: "strides", Type: protos.AttributeProto_INTS, Ints: []int64{1, 1}},
			{Name: "dilations", Type: protos.AttributeProto_INTS, Ints: []int64{2, 2}},
		}}
		return convertMaxPool(nil, nil, node, []*Node{x})
	}).WithDynamicAxes([]string{"batch", "", "", ""})
	defer exec.Finalize()

	input := [][][][]float32{{{{
		1, 2, 3, 4,
	}, {
		5, 6, 7, 8,
	}, {
		9, 10, 11, 12,
	}, {
		13, 14, 15, 16,
	}}}}
	got := exec.MustExec1(input)
	defer got.FinalizeAll()
	require.Equal(t, [][][][]float32{{{{11, 12}, {15, 16}}}}, got.Value())
}

func TestGlobalAveragePoolDynamic(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	exec := MustNewExec(backend, func(x *Node) *Node {
		return convertGlobalAveragePool(nil, nil, &protos.NodeProto{OpType: "GlobalAveragePool"}, []*Node{x})
	}).WithDynamicAxes([]string{"batch", "", "h", "w"})
	defer exec.Finalize()

	got1 := exec.MustExec1([][][][]float32{{{{1, 3}, {5, 7}}}})
	defer got1.FinalizeAll()
	require.InDeltaSlice(t, []float32{4}, got1.Value().([][][][]float32)[0][0][0], 1e-6)

	got2 := exec.MustExec1([][][][]float32{{{{2, 4, 6}}}, {{{10, 20, 30}}}})
	defer got2.FinalizeAll()
	vals := got2.Value().([][][][]float32)
	require.InDelta(t, 4.0, vals[0][0][0][0], 1e-6)
	require.InDelta(t, 20.0, vals[1][0][0][0], 1e-6)
}
