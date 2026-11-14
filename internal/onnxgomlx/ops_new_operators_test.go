// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"math"
	"strings"
	"testing"

	"github.com/gomlx/compute-onnx"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func floatAttr(name string, value float32) *protos.AttributeProto {
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_FLOAT, F: value}
}

func intsAttr(name string, values ...int64) *protos.AttributeProto {
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_INTS, Ints: values}
}

func TestEluSoftmaxLogSoftmaxReduceLogSumExpOneHot(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Verify Softmax (opset >= 13) emits FusedSoftmax even on dynamic shapes.
	var softmaxGraphStr string
	smExec := MustNewExec(backend, func(x *Node) []*Node {
		g := x.Graph()
		sm := convertTestNode(m, g, &protos.NodeProto{OpType: "Softmax", Input: []string{"x"}, Output: []string{"sm"}}, map[string]*Node{"x": x})[0]
		softmaxGraphStr = g.String()
		lsm := convertTestNode(m, g, &protos.NodeProto{OpType: "LogSoftmax", Input: []string{"x"}, Output: []string{"lsm"}}, map[string]*Node{"x": x})[0]
		elu := convertTestNode(m, g, &protos.NodeProto{
			OpType: "Elu", Input: []string{"x"}, Output: []string{"elu"},
			Attribute: []*protos.AttributeProto{floatAttr("alpha", 2.0)},
		}, map[string]*Node{"x": x})[0]
		rlse := convertTestNode(m, g, &protos.NodeProto{
			OpType: "ReduceLogSumExp", Input: []string{"x"}, Output: []string{"rlse"},
			Attribute: []*protos.AttributeProto{intsAttr("axes", -1), intAttr("keepdims", 0)},
		}, map[string]*Node{"x": x})[0]
		return []*Node{sm, lsm, elu, rlse}
	}).WithDynamicAxes([]string{"batch", ""})
	defer smExec.Finalize()

	outs := smExec.MustExec([][]float32{{0, 0}, {-1, 2}})
	for _, o := range outs {
		defer o.FinalizeAll()
	}
	require.Contains(t, softmaxGraphStr, "FusedSoftmax")
	require.InDeltaSlice(t, []float32{0.5, 0.5}, outs[0].Value().([][]float32)[0], 1e-5)
	require.InDeltaSlice(t, []float32{float32(-math.Ln2), float32(-math.Ln2)}, outs[1].Value().([][]float32)[0], 1e-5)
	require.InDeltaSlice(t, []float32{float32(2 * math.Expm1(-1)), 2}, outs[2].Value().([][]float32)[1], 1e-5)
	require.InDeltaSlice(t, []float32{float32(math.Ln2), float32(2 + math.Log(1+math.Exp(-3)))}, outs[3].Value().([]float32), 1e-5)

	// OneHot with dynamic batch, negative index, and axis=1.
	ohModel := &Model{
		Backend: backend,
		Proto: protos.ModelProto{
			Graph: &protos.GraphProto{
				Initializer: []*protos.TensorProto{{
					Name:      "depth",
					DataType:  int32(protos.TensorProto_INT64),
					Int64Data: []int64{3},
				}},
			},
		},
		VariableNameToValue: map[string]*protos.TensorProto{
			"depth": {Name: "depth", DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{3}},
		},
		NodeOutputToNode: map[string]*protos.NodeProto{},
	}
	ohExec := MustNewExec(backend, func(indices *Node) *Node {
		g := indices.Graph()
		depth := Const(g, int64(3))
		values := Const(g, []float32{0, 5})
		node := &protos.NodeProto{
			OpType:    "OneHot",
			Input:     []string{"indices", "depth", "values"},
			Output:    []string{"y"},
			Attribute: []*protos.AttributeProto{intAttr("axis", 1)},
		}
		return convertTestNode(ohModel, g, node, map[string]*Node{"indices": indices, "depth": depth, "values": values})[0]
	}).WithDynamicAxes([]string{"batch", ""})
	defer ohExec.Finalize()

	ohGot := ohExec.MustExec1([][]int64{{0, -1}, {1, 99}})
	defer ohGot.FinalizeAll()
	require.Equal(t, [][][]float32{{{5, 0}, {0, 0}, {0, 5}}, {{0, 0}, {5, 0}, {0, 0}}}, ohGot.Value())
}

func TestInstanceNormConvIntegerConvTransposeGridSampleEinsumArgMinMax(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// InstanceNormalization: verify FusedLayerNorm is used and dynamic spatial dims work.
	var inGraphStr string
	inExec := MustNewExec(backend, func(x *Node) *Node {
		g := x.Graph()
		scale := Const(g, []float32{2, 0.5})
		bias := Const(g, []float32{1, -1})
		node := &protos.NodeProto{
			OpType: "InstanceNormalization", Input: []string{"x", "s", "b"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{floatAttr("epsilon", 0)},
		}
		out := convertTestNode(m, g, node, map[string]*Node{"x": x, "s": scale, "b": bias})[0]
		inGraphStr = g.String()
		return out
	}).WithDynamicAxes([]string{"batch", "", "h", "w"})
	defer inExec.Finalize()

	inGot := inExec.MustExec1([][][][]float32{{{{1, 3}, {1, 3}}, {{2, 6}, {2, 6}}}})
	defer inGot.FinalizeAll()
	require.True(t, strings.Contains(inGraphStr, "FusedLayerNorm"))
	require.InDeltaSlice(t, []float32{-1, 3}, inGot.Value().([][][][]float32)[0][0][0], 1e-5)
	require.InDeltaSlice(t, []float32{-1.5, -0.5}, inGot.Value().([][][][]float32)[0][1][0], 1e-5)

	// ConvInteger with uint8 X, int8 W, and zero points.
	ciExec := MustNewExec(backend, func(x *Node) *Node {
		g := x.Graph()
		w := Const(g, [][][][]int8{{{{1, -1}, {2, 0}}}})
		xZp := Const(g, uint8(10))
		wZp := Const(g, []int8{0})
		node := &protos.NodeProto{OpType: "ConvInteger", Input: []string{"x", "w", "xz", "wz"}, Output: []string{"y"}}
		return convertTestNode(m, g, node, map[string]*Node{"x": x, "w": w, "xz": xZp, "wz": wZp})[0]
	}).WithDynamicAxes([]string{"batch", "", "", ""})
	defer ciExec.Finalize()

	ciGot := ciExec.MustExec1([][][][]uint8{{{{12, 11}, {13, 10}}}})
	defer ciGot.FinalizeAll()
	// Centered X = [[2, 1], [3, 0]]; dot with [[1, -1], [2, 0]] = 2 - 1 + 6 + 0 = 7.
	require.Equal(t, [][][][]int32{{{{7}}}}, ciGot.Value())

	// ConvTranspose 1D with stride=2, output_padding=1, bias, and dynamic batch.
	ctExec := MustNewExec(backend, func(x *Node) *Node {
		g := x.Graph()
		w := Const(g, [][][]float32{{{1, 2}}}) // [C_in=1, C_out/group=1, K=2]
		b := Const(g, []float32{0.5})
		node := &protos.NodeProto{
			OpType: "ConvTranspose", Input: []string{"x", "w", "b"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{intsAttr("strides", 2), intsAttr("output_padding", 1)},
		}
		return convertTestNode(m, g, node, map[string]*Node{"x": x, "w": w, "b": b})[0]
	}).WithDynamicAxes([]string{"batch", "", ""})
	defer ctExec.Finalize()

	ctGot := ctExec.MustExec1([][][]float32{{{3, 4}}})
	defer ctGot.FinalizeAll()
	require.Equal(t, [][][]float32{{{3.5, 6.5, 4.5, 8.5, 0.5}}}, ctGot.Value())

	// GridSample bilinear with align_corners=1 and dynamic batch.
	gsExec := MustNewExec(backend, func(x, grid *Node) *Node {
		node := &protos.NodeProto{
			OpType: "GridSample", Input: []string{"x", "grid"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{intAttr("align_corners", 1)},
		}
		return convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x, "grid": grid})[0]
	}).WithDynamicAxes([]string{"b", "", "", ""}, []string{"b", "oh", "ow", ""})
	defer gsExec.Finalize()

	gsGot := gsExec.MustExec1(
		[][][][]float32{{{{0, 2}, {4, 6}}}},
		[][][][]float32{{{{0, 0}, {-1, 1}}}}, // center (mean=3) and bottom-left (4)
	)
	defer gsGot.FinalizeAll()
	require.InDeltaSlice(t, []float32{3, 4}, gsGot.Value().([][][][]float32)[0][0][0], 1e-5)

	// Einsum with ellipsis and dynamic batch + ArgMax/ArgMin with select_last_index=1.
	eaExec := MustNewExec(backend, func(a, b, ties *Node) (*Node, *Node, *Node) {
		g := a.Graph()
		es := convertTestNode(m, g, &protos.NodeProto{
			OpType: "Einsum", Input: []string{"a", "b"}, Output: []string{"es"},
			Attribute: []*protos.AttributeProto{stringAttr("equation", "...ij,...jk->...ik")},
		}, map[string]*Node{"a": a, "b": b})[0]
		amax := convertTestNode(m, g, &protos.NodeProto{
			OpType: "ArgMax", Input: []string{"t"}, Output: []string{"amax"},
			Attribute: []*protos.AttributeProto{intAttr("axis", 1), intAttr("keepdims", 0), intAttr("select_last_index", 1)},
		}, map[string]*Node{"t": ties})[0]
		amin := convertTestNode(m, g, &protos.NodeProto{
			OpType: "ArgMin", Input: []string{"t"}, Output: []string{"amin"},
			Attribute: []*protos.AttributeProto{intAttr("axis", 1), intAttr("keepdims", 0), intAttr("select_last_index", 1)},
		}, map[string]*Node{"t": ties})[0]
		return es, amax, amin
	}).WithDynamicAxes([]string{"b", "", ""}, []string{"b", "", ""}, []string{"b", "len"})
	defer eaExec.Finalize()

	eaOuts := eaExec.MustExec(
		[][][]float32{{{1, 2}, {3, 4}}},
		[][][]float32{{{2, 0}, {0, 3}}},
		[][]float32{{5, 1, 5, 1, 2}},
	)
	for _, o := range eaOuts {
		defer o.FinalizeAll()
	}
	require.Equal(t, [][][]float32{{{2, 6}, {6, 12}}}, eaOuts[0].Value())
	require.Equal(t, []int64{2}, eaOuts[1].Value())
	require.Equal(t, []int64{3}, eaOuts[2].Value())
}

func TestONNXBackendCrossCheck(t *testing.T) {
	ortBackend, err := onnxbackend.New("cpu")
	if err != nil {
		t.Skipf("skipping ONNX Runtime backend cross-check: %v", err)
	}
	defer ortBackend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Dynamic spatial InstanceNormalization (exercises compute-onnx ErrNotImplemented -> decomposed fallback)
	// and dynamic RandomNormalLike + ConvInteger + MatMulNBits.
	inExec := MustNewExec(ortBackend, func(x *Node) *Node {
		g := x.Graph()
		scale := Const(g, []float32{2, 0.5})
		bias := Const(g, []float32{1, -1})
		node := &protos.NodeProto{
			OpType: "InstanceNormalization", Input: []string{"x", "s", "b"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{floatAttr("epsilon", 0)},
		}
		return convertTestNode(m, g, node, map[string]*Node{"x": x, "s": scale, "b": bias})[0]
	}).WithDynamicAxes([]string{"batch", "", "h", "w"})
	defer inExec.Finalize()

	inGot := inExec.MustExec1([][][][]float32{{{{1, 3}, {1, 3}}, {{2, 6}, {2, 6}}}})
	defer inGot.FinalizeAll()
	require.InDeltaSlice(t, []float32{-1, 3}, inGot.Value().([][][][]float32)[0][0][0], 1e-4)

	ciExec := MustNewExec(ortBackend, func(x *Node) *Node {
		g := x.Graph()
		w := Const(g, [][][][]int8{{{{1, -1}, {2, 0}}}})
		xZp := Const(g, uint8(10))
		node := &protos.NodeProto{OpType: "ConvInteger", Input: []string{"x", "w", "xz"}, Output: []string{"y"}}
		return convertTestNode(m, g, node, map[string]*Node{"x": x, "w": w, "xz": xZp})[0]
	}).WithDynamicAxes([]string{"batch", "", "", ""})
	defer ciExec.Finalize()

	ciGot := ciExec.MustExec1([][][][]uint8{{{{12, 11}, {13, 10}}}})
	defer ciGot.FinalizeAll()
	require.Equal(t, [][][][]int32{{{{7}}}}, ciGot.Value())

	rngExec := MustNewExec(ortBackend, func(x *Node) *Node {
		node := &protos.NodeProto{
			OpType: "RandomNormalLike", Input: []string{"x"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{floatAttr("seed", 7)},
		}
		return convertRandomNormalLike(nil, node, []*Node{x})
	}).WithDynamicAxes([]string{"batch", ""})
	defer rngExec.Finalize()

	rn := rngExec.MustExec1([][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}})
	defer rn.FinalizeAll()
	require.Equal(t, dtypes.Float32, rn.DType())
	require.Equal(t, []int{3, 3}, rn.Shape().Dimensions)

	nzExec := MustNewExec(ortBackend, func(x *Node) *Node {
		node := &protos.NodeProto{OpType: "NonZero", Input: []string{"x"}, Output: []string{"y"}}
		return convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x})[0]
	}).WithDynamicAxes([]string{"r", "c"})
	defer nzExec.Finalize()

	nz := nzExec.MustExec1([][]float32{{0, 2}, {5, 0}})
	defer nz.FinalizeAll()
	require.Equal(t, [][]int64{{0, 1}, {1, 0}}, nz.Value())
}
