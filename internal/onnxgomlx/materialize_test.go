// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func makeSymbolicValueInfo(name string, elemType protos.TensorProto_DataType, dimParams ...string) *protos.ValueInfoProto {
	dims := make([]*protos.TensorShapeProto_Dimension, len(dimParams))
	for i, p := range dimParams {
		dims[i] = &protos.TensorShapeProto_Dimension{
			Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: p},
		}
	}
	return &protos.ValueInfoProto{
		Name: name,
		Type: &protos.TypeProto{
			Value: &protos.TypeProto_TensorType{
				TensorType: &protos.TypeProto_Tensor{
					ElemType: int32(elemType),
					Shape:    &protos.TensorShapeProto{Dim: dims},
				},
			},
		},
	}
}

func makeInt64ConstNode(outputName string, dims []int64, values []int64) *protos.NodeProto {
	return &protos.NodeProto{
		OpType: "Constant",
		Output: []string{outputName},
		Attribute: []*protos.AttributeProto{
			{
				Name: "value",
				Type: protos.AttributeProto_TENSOR,
				T: &protos.TensorProto{
					Dims:      dims,
					DataType:  int32(protos.TensorProto_INT64),
					Int64Data: values,
				},
			},
		},
	}
}

func parseTestGraphProto(t *testing.T, graphProto *protos.GraphProto) *Model {
	t.Helper()
	modelProto := &protos.ModelProto{
		IrVersion:   9,
		OpsetImport: []*protos.OperatorSetIdProto{{Version: 19}},
		Graph:       graphProto,
	}
	raw, err := proto.Marshal(modelProto)
	require.NoError(t, err)
	m, err := Parse(raw)
	require.NoError(t, err)
	return m
}

// TestMaterializeFromConcreteGraphInputs verifies that Shape and Size expressions on ONNX inputs
// declared with symbolic dimensions ("batch", "seq") materialize as static constants when CallGraph
// is invoked with concrete GoMLX input shapes, while preserving runtime dynamic fallbacks when
// CallGraph is invoked with genuinely dynamic GoMLX inputs.
func TestMaterializeFromConcreteGraphInputs(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	// 1. Tile requires static materialization of 'repeats' (panics otherwise).
	tileModel := parseTestGraphProto(t, &protos.GraphProto{
		Name:   "symbolic_tile",
		Input:  []*protos.ValueInfoProto{makeSymbolicValueInfo("x", protos.TensorProto_FLOAT, "batch", "seq")},
		Output: []*protos.ValueInfoProto{makeSymbolicValueInfo("tiled", protos.TensorProto_FLOAT, "batch", "tiled_seq")},
		Node: []*protos.NodeProto{
			{OpType: "Shape", Input: []string{"x"}, Output: []string{"x_shape"}},
			makeInt64ConstNode("zero_idx", []int64{1}, []int64{0}),
			makeInt64ConstNode("one_vec", []int64{1}, []int64{1}),
			{OpType: "Gather", Input: []string{"x_shape", "zero_idx"}, Output: []string{"batch_1d"}},
			{
				OpType: "Concat", Input: []string{"one_vec", "batch_1d"}, Output: []string{"repeats"},
				Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: 0}},
			},
			{OpType: "Tile", Input: []string{"x", "repeats"}, Output: []string{"tiled"}},
		},
	})
	defer func() { _ = tileModel.Close() }()

	// At Parse time (before CallGraph), Shape(x) depends on symbolic ONNX dimensions.
	require.False(t, tileModel.IsConstantExpression("repeats"))

	// With concrete [2, 3] input in CallGraph, repeats = [1, 2] materializes statically and Tile succeeds.
	tileExec := MustNewExec(backend, func(x *Node) *Node {
		outs := tileModel.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.False(t, outs[0].Shape().IsDynamic(), "expected static output shape for concrete input")
		require.Equal(t, []int{2, 6}, outs[0].Shape().Dimensions)
		return outs[0]
	})
	defer tileExec.Finalize()

	gotTile := tileExec.MustExec1([][]float32{{1, 2, 3}, {4, 5, 6}})
	defer gotTile.FinalizeAll()
	require.Equal(t, [][]float32{{1, 2, 3, 1, 2, 3}, {4, 5, 6, 4, 5, 6}}, gotTile.Value())

	// 2. Reshape + Size -> ConstantOfShape: static when concrete, runtime dynamic when WithDynamicAxes is used.
	dynOrStaticModel := parseTestGraphProto(t, &protos.GraphProto{
		Name:  "symbolic_reshape_and_size",
		Input: []*protos.ValueInfoProto{makeSymbolicValueInfo("x", protos.TensorProto_FLOAT, "batch", "seq")},
		Output: []*protos.ValueInfoProto{
			makeSymbolicValueInfo("reshaped", protos.TensorProto_FLOAT, "batch", "seq"),
			makeSymbolicValueInfo("filled", protos.TensorProto_INT64, "total"),
		},
		Node: []*protos.NodeProto{
			{OpType: "Shape", Input: []string{"x"}, Output: []string{"x_shape"}},
			makeInt64ConstNode("ones_2", []int64{2}, []int64{1, 1}),
			{OpType: "Mul", Input: []string{"x_shape", "ones_2"}, Output: []string{"target_shape"}},
			{OpType: "Reshape", Input: []string{"x", "target_shape"}, Output: []string{"reshaped"}},
			{OpType: "Size", Input: []string{"x"}, Output: []string{"x_size"}},
			makeInt64ConstNode("shape_1", []int64{1}, []int64{1}),
			{OpType: "Reshape", Input: []string{"x_size", "shape_1"}, Output: []string{"size_1d"}},
			{
				OpType: "ConstantOfShape", Input: []string{"size_1d"}, Output: []string{"filled"},
				Attribute: []*protos.AttributeProto{{
					Name: "value",
					Type: protos.AttributeProto_TENSOR,
					T: &protos.TensorProto{
						Dims:      []int64{1},
						DataType:  int32(protos.TensorProto_INT64),
						Int64Data: []int64{7},
					},
				}},
			},
		},
	})
	defer func() { _ = dynOrStaticModel.Close() }()

	staticExec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
		outs := dynOrStaticModel.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.False(t, outs[0].Shape().IsDynamic())
		require.Equal(t, []int{2, 3}, outs[0].Shape().Dimensions)
		require.False(t, outs[1].Shape().IsDynamic())
		require.Equal(t, []int{6}, outs[1].Shape().Dimensions)
		return outs[0], outs[1]
	})
	defer staticExec.Finalize()

	staticOut := staticExec.MustExec([][]float32{{1, 2, 3}, {4, 5, 6}})
	require.Equal(t, [][]float32{{1, 2, 3}, {4, 5, 6}}, staticOut[0].Value())
	require.Equal(t, []int64{7, 7, 7, 7, 7, 7}, staticOut[1].Value())
	staticOut[0].FinalizeAll()
	staticOut[1].FinalizeAll()

	dynExec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
		outs := dynOrStaticModel.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.True(t, outs[0].Shape().IsDynamic())
		require.True(t, outs[1].Shape().IsDynamic())
		return outs[0], outs[1]
	}).WithDynamicAxes([]string{"batch", "seq"})
	defer dynExec.Finalize()

	dynOut1 := dynExec.MustExec([][]float32{{1, 2, 3}, {4, 5, 6}})
	require.Equal(t, [][]float32{{1, 2, 3}, {4, 5, 6}}, dynOut1[0].Value())
	require.Equal(t, []int64{7, 7, 7, 7, 7, 7}, dynOut1[1].Value())
	dynOut1[0].FinalizeAll()
	dynOut1[1].FinalizeAll()

	dynOut2 := dynExec.MustExec([][]float32{{10, 20, 30, 40}})
	require.Equal(t, [][]float32{{10, 20, 30, 40}}, dynOut2[0].Value())
	require.Equal(t, []int64{7, 7, 7, 7}, dynOut2[1].Value())
	dynOut2[0].FinalizeAll()
	dynOut2[1].FinalizeAll()
}

// TestNonZeroMaterializeFromConcreteGraphInput verifies that NonZero folds constant expressions
// derived from concrete CallGraph input shapes (including nested ConstantOfShape(Concat(Shape(x), ...))
// and tensors larger than MinConstValueSizeToKeep), while delegating to runtime NonZero when dynamic.
func TestNonZeroMaterializeFromConcreteGraphInput(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	// Graph: Shape(x) -> Concat([..., 2]) -> ConstantOfShape(value=1) -> NonZero
	// Also feeds ReduceMax(nz, axis=1) + 1 into Reshape(ones, ...) to verify materialization
	// walks through large Constant nodes (>= MinConstValueSizeToKeep=32, where ConstantValue() == nil).
	m := parseTestGraphProto(t, &protos.GraphProto{
		Name:  "nonzero_concrete_input",
		Input: []*protos.ValueInfoProto{makeSymbolicValueInfo("x", protos.TensorProto_FLOAT, "batch", "seq")},
		Output: []*protos.ValueInfoProto{
			makeSymbolicValueInfo("nz", protos.TensorProto_INT64, "rank", "nnz"),
			makeSymbolicValueInfo("ones_reshaped", protos.TensorProto_INT64, "b", "s", "c"),
		},
		Node: []*protos.NodeProto{
			{OpType: "Shape", Input: []string{"x"}, Output: []string{"x_shape"}},
			makeInt64ConstNode("two_vec", []int64{1}, []int64{2}),
			{
				OpType: "Concat", Input: []string{"x_shape", "two_vec"}, Output: []string{"mask_shape"},
				Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: 0}},
			},
			{
				OpType: "ConstantOfShape", Input: []string{"mask_shape"}, Output: []string{"ones"},
				Attribute: []*protos.AttributeProto{{
					Name: "value",
					Type: protos.AttributeProto_TENSOR,
					T: &protos.TensorProto{
						Dims:      []int64{1},
						DataType:  int32(protos.TensorProto_INT64),
						Int64Data: []int64{1},
					},
				}},
			},
			{OpType: "NonZero", Input: []string{"ones"}, Output: []string{"nz"}},
			makeInt64ConstNode("axis1", []int64{1}, []int64{1}),
			makeInt64ConstNode("one_scalar", []int64{1}, []int64{1}),
			{
				OpType: "ReduceMax", Input: []string{"nz", "axis1"}, Output: []string{"max_idx"},
				Attribute: []*protos.AttributeProto{{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0}},
			},
			{OpType: "Add", Input: []string{"max_idx", "one_scalar"}, Output: []string{"recovered_shape"}},
			{OpType: "Reshape", Input: []string{"ones", "recovered_shape"}, Output: []string{"ones_reshaped"}},
		},
	})
	defer func() { _ = m.Close() }()

	// 1) Concrete [5, 4] input (>= MinConstValueSizeToKeep elements):
	//    nz folds to NodeTypeConstant [3, 40] (ConstantValue() == nil since 120 >= 32),
	//    and recovered_shape materializes to [5, 4, 2] by traversing through nz's ONNX subgraph.
	staticExec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
		outs := m.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.Equal(t, NodeTypeConstant, outs[0].Type(), "expected NonZero to fold to Constant for concrete input")
		require.Nil(t, outs[0].ConstantValue(), "120-element constant should exceed MinConstValueSizeToKeep")
		require.False(t, outs[0].Shape().IsDynamic())
		require.Equal(t, []int{3, 40}, outs[0].Shape().Dimensions)
		require.False(t, outs[1].Shape().IsDynamic())
		require.Equal(t, []int{5, 4, 2}, outs[1].Shape().Dimensions)
		return outs[0], outs[1]
	})
	defer staticExec.Finalize()

	in5x4 := make([][]float32, 5)
	for i := range in5x4 {
		in5x4[i] = []float32{1, 2, 3, 4}
	}
	gotStatic := staticExec.MustExec(in5x4)
	defer gotStatic[0].FinalizeAll()
	defer gotStatic[1].FinalizeAll()

	// 2) Dynamic input on the same Model: delegates to runtime NonZero and matches the folded output.
	dynExec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
		outs := m.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.True(t, outs[0].Shape().IsDynamic())
		require.True(t, outs[1].Shape().IsDynamic())
		return outs[0], outs[1]
	}).WithDynamicAxes([]string{"batch", "seq"})
	defer dynExec.Finalize()

	gotDyn := dynExec.MustExec(in5x4)
	defer gotDyn[0].FinalizeAll()
	defer gotDyn[1].FinalizeAll()
	require.Equal(t, gotStatic[0].Value(), gotDyn[0].Value())
	require.Equal(t, gotStatic[1].Value(), gotDyn[1].Value())
}

// TestNonConstantDependenciesNilAndLargeConstInput verifies nil-safety when convertedOutputs is nil
// or contains nil *Node entries (as used for zero-size constant inputs), and materialization of
// InputsAsConstants tensors with >= MinConstValueSizeToKeep elements.
func TestNonConstantDependenciesNilAndLargeConstInput(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()

	m := parseTestGraphProto(t, &protos.GraphProto{
		Name: "nil_and_large_const_input",
		Input: []*protos.ValueInfoProto{
			makeSymbolicValueInfo("x", protos.TensorProto_FLOAT, "batch", "seq"),
			makeSymbolicValueInfo("const_in", protos.TensorProto_INT64, "len"),
		},
		Output: []*protos.ValueInfoProto{makeSymbolicValueInfo("out", protos.TensorProto_FLOAT, "d")},
		Node: []*protos.NodeProto{
			{OpType: "Shape", Input: []string{"x"}, Output: []string{"x_shape"}},
			makeInt64ConstNode("axis0", []int64{1}, []int64{0}),
			{
				OpType: "ReduceSum", Input: []string{"const_in", "axis0"}, Output: []string{"target_shape"},
				Attribute: []*protos.AttributeProto{{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1}},
			},
			{OpType: "Reshape", Input: []string{"x", "target_shape"}, Output: []string{"out"}},
		},
	})
	defer func() { _ = m.Close() }()

	// Nil convertedOutputs map and nil *Node map entry must both be safe and fall back to ONNX metadata.
	inputs1, _, _ := m.nonConstantDependencies("x_shape", nil)
	require.Equal(t, []string{"x"}, inputs1)
	inputs2, _, _ := m.nonConstantDependencies("x_shape", map[string]*Node{"x": nil})
	require.Equal(t, []string{"x"}, inputs2)

	// Configure const_in with 40 ones (>= MinConstValueSizeToKeep=32, so ConstantValue() == nil on its Const node).
	ones40 := make([]int64, 40)
	for i := range ones40 {
		ones40[i] = 1
	}
	m.WithInputsAsConstants(map[string]any{"const_in": ones40})

	exec := MustNewExec(backend, func(x *Node) *Node {
		outs := m.CallGraph(nil, x.Graph(), map[string]*Node{"x": x})
		require.False(t, outs[0].Shape().IsDynamic())
		require.Equal(t, []int{40}, outs[0].Shape().Dimensions)
		return outs[0]
	})
	defer exec.Finalize()

	in5x8 := make([][]float32, 5)
	for i := range in5x8 {
		in5x8[i] = make([]float32, 8)
		for j := range in5x8[i] {
			in5x8[i][j] = float32(i*8 + j)
		}
	}
	got := exec.MustExec1(in5x8)
	defer got.FinalizeAll()
	require.Equal(t, []int{40}, got.Shape().Dimensions)
}
