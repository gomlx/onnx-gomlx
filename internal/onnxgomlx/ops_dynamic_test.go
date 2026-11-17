// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"strings"
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"
)

func TestDynamicRuntimeBroadcastAndMatMul(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Unrelated dynamic axes where one operand has runtime size 1 and expands to the other.
	addExec := MustNewExec(backend, func(a, b *Node) *Node {
		node := &protos.NodeProto{OpType: "Add", Input: []string{"a", "b"}, Output: []string{"y"}}
		return convertTestNode(m, a.Graph(), node, map[string]*Node{"a": a, "b": b})[0]
	}).WithDynamicAxes([]string{"a_rows", ""}, []string{"b_rows", ""})
	defer addExec.Finalize()

	got := addExec.MustExec1([][]float32{{10, 20}}, [][]float32{{1, 2}, {3, 4}, {5, 6}})
	defer got.FinalizeAll()
	require.Equal(t, [][]float32{{11, 22}, {13, 24}, {15, 26}}, got.Value())

	// Incompatible runtime dimensions (> 1 and unequal) must fail cleanly instead of truncating.
	err = exceptions.TryCatch[error](func() {
		bad := addExec.MustExec1([][]float32{{1, 2}, {3, 4}}, [][]float32{{1, 2}, {3, 4}, {5, 6}})
		defer bad.FinalizeAll()
	})
	require.Error(t, err)

	// Dynamic MatMul with runtime batch broadcast (1 vs 2) and dynamic K.
	mmExec := MustNewExec(backend, func(a, b *Node) *Node {
		node := &protos.NodeProto{OpType: "MatMul", Input: []string{"a", "b"}, Output: []string{"y"}}
		return convertTestNode(m, a.Graph(), node, map[string]*Node{"a": a, "b": b})[0]
	}).WithDynamicAxes([]string{"ba", "", "ka"}, []string{"bb", "kb", ""})
	defer mmExec.Finalize()

	mmGot := mmExec.MustExec1(
		[][][]float32{{{1, 2, 3}, {4, 5, 6}}},
		[][][]float32{{{1, 0}, {0, 1}, {1, 1}}, {{2, 0}, {0, 2}, {0, 0}}},
	)
	defer mmGot.FinalizeAll()
	require.Equal(t, [][][]float32{{{4, 5}, {10, 11}}, {{2, 4}, {8, 10}}}, mmGot.Value())

	// Dynamic axis broadcast against a non-unit static dimension narrows output axis to static 3,
	// while validating runtime 1-or-match compatibility.
	var narrowedShape shapes.Shape
	dynVsStaticExec := MustNewExec(backend, func(a *Node) *Node {
		b := Const(a.Graph(), [][]float32{{1, 2}, {3, 4}, {5, 6}})
		node := &protos.NodeProto{OpType: "Add", Input: []string{"a", "b"}, Output: []string{"y"}}
		out := convertTestNode(m, a.Graph(), node, map[string]*Node{"a": a, "b": b})[0]
		narrowedShape = out.Shape()
		return out
	}).WithDynamicAxes([]string{"a_rows", ""})
	defer dynVsStaticExec.Finalize()

	dsBroadcast := dynVsStaticExec.MustExec1([][]float32{{10, 20}})
	defer dsBroadcast.FinalizeAll()
	require.Equal(t, shapes.Make(dtypes.Float32, 3, 2), narrowedShape)
	require.Equal(t, [][]float32{{11, 22}, {13, 24}, {15, 26}}, dsBroadcast.Value())

	dsMatch := dynVsStaticExec.MustExec1([][]float32{{10, 20}, {30, 40}, {50, 60}})
	defer dsMatch.FinalizeAll()
	require.Equal(t, [][]float32{{11, 22}, {33, 44}, {55, 66}}, dsMatch.Value())

	err = exceptions.TryCatch[error](func() {
		bad := dynVsStaticExec.MustExec1([][]float32{{10, 20}, {30, 40}})
		defer bad.FinalizeAll()
	})
	require.Error(t, err)

	// Shared dynamic axis ("batch") is preserved on axis 0 when axis 1 performs runtime broadcast against static 3,
	// including for Mod (fmod=0).
	var modOutShape shapes.Shape
	modExec := MustNewExec(backend, func(a, b *Node) *Node {
		node := &protos.NodeProto{OpType: "Mod", Input: []string{"a", "b"}, Output: []string{"y"}}
		out := convertTestNode(m, a.Graph(), node, map[string]*Node{"a": a, "b": b})[0]
		modOutShape = out.Shape()
		return out
	}).WithDynamicAxes([]string{"batch", "a_cols"}, []string{"batch", ""})
	defer modExec.Finalize()

	modGot := modExec.MustExec1([][]int64{{-7}, {8}}, [][]int64{{3, 4, 5}, {3, 4, 5}})
	defer modGot.FinalizeAll()
	require.Equal(t, "batch", modOutShape.AxisName(0))
	require.Equal(t, 3, modOutShape.Dim(1))
	require.Equal(t, [][]int64{{2, 1, 3}, {2, 0, 3}}, modGot.Value())

	// Runtime 1 -> 0 zero-size broadcasting produces an empty [0, 2] tensor, while 2 vs 0 fails.
	empty0x2 := tensors.FromFlatDataAndDimensions([]float32{}, 0, 2)
	defer empty0x2.FinalizeAll()
	zeroGot := addExec.MustExec1([][]float32{{10, 20}}, empty0x2)
	defer zeroGot.FinalizeAll()
	require.Equal(t, shapes.Make(dtypes.Float32, 0, 2), zeroGot.Shape())

	err = exceptions.TryCatch[error](func() {
		bad := addExec.MustExec1([][]float32{{10, 20}, {30, 40}}, empty0x2)
		defer bad.FinalizeAll()
	})
	require.Error(t, err)
}

func TestDynamicReshapeSliceRangePadFlattenSize(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Runtime Reshape with 0 (copy input dim) and -1 (infer dim) on a dynamic input.
	reshapeExec := MustNewExec(backend, func(x, targetShape *Node) *Node {
		node := &protos.NodeProto{OpType: "Reshape", Input: []string{"x", "s"}, Output: []string{"y"}}
		return convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x, "s": targetShape})[0]
	}).WithDynamicAxes([]string{"batch", "seq"}, []string{""})
	defer reshapeExec.Finalize()

	rGot := reshapeExec.MustExec1([][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}, []int64{0, 2, -1})
	defer rGot.FinalizeAll()
	require.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, rGot.Value())

	// Runtime Slice with dynamic bounds and negative start/step.
	sliceExec := MustNewExec(backend, func(x, starts, ends, steps *Node) *Node {
		axes := Const(x.Graph(), []int64{1})
		node := &protos.NodeProto{OpType: "Slice", Input: []string{"x", "starts", "ends", "axes", "steps"}, Output: []string{"y"}}
		return convertTestNode(m, x.Graph(), node, map[string]*Node{
			"x": x, "starts": starts, "ends": ends, "axes": axes, "steps": steps,
		})[0]
	}).WithDynamicAxes([]string{"batch", "len"}, []string{""}, []string{""}, []string{""})
	defer sliceExec.Finalize()

	sGot := sliceExec.MustExec1([][]float32{{10, 20, 30, 40}, {50, 60, 70, 80}}, []int64{-1}, []int64{-4}, []int64{-1})
	defer sGot.FinalizeAll()
	require.Equal(t, [][]float32{{40, 30, 20}, {80, 70, 60}}, sGot.Value())

	// Runtime Range with positive, negative, and empty ranges.
	rangeExec := MustNewExec(backend, func(start, limit, delta *Node) *Node {
		node := &protos.NodeProto{OpType: "Range", Input: []string{"s", "l", "d"}, Output: []string{"y"}}
		return convertTestNode(m, start.Graph(), node, map[string]*Node{"s": start, "l": limit, "d": delta})[0]
	})
	defer rangeExec.Finalize()

	rg1 := rangeExec.MustExec1(int64(1), int64(7), int64(2))
	defer rg1.FinalizeAll()
	require.Equal(t, []int64{1, 3, 5}, rg1.Value())

	rgEmpty := rangeExec.MustExec1(int64(5), int64(2), int64(1))
	defer rgEmpty.FinalizeAll()
	require.Equal(t, []int64{}, rgEmpty.Value())

	// Dynamic Pad, Flatten, and Size.
	pfsExec := MustNewExec(backend, func(x, pads *Node) (*Node, *Node, *Node) {
		g := x.Graph()
		padNode := &protos.NodeProto{OpType: "Pad", Input: []string{"x", "pads"}, Output: []string{"padded"}}
		padded := convertTestNode(m, g, padNode, map[string]*Node{"x": x, "pads": pads})[0]
		flatNode := &protos.NodeProto{OpType: "Flatten", Input: []string{"x"}, Output: []string{"flat"}, Attribute: []*protos.AttributeProto{intAttr("axis", 1)}}
		flat := convertTestNode(m, g, flatNode, map[string]*Node{"x": x})[0]
		sizeNode := &protos.NodeProto{OpType: "Size", Input: []string{"x"}, Output: []string{"sz"}}
		sz := convertTestNode(m, g, sizeNode, map[string]*Node{"x": x})[0]
		return padded, flat, sz
	}).WithDynamicAxes([]string{"batch", "", ""}, []string{""})
	defer pfsExec.Finalize()

	outs := pfsExec.MustExec([][][]float32{{{1, 2}, {3, 4}}}, []int64{0, 1, 0, 0, 0, 0})
	for _, o := range outs {
		defer o.FinalizeAll()
	}
	require.Equal(t, [][][]float32{{{0, 0}, {1, 2}, {3, 4}}}, outs[0].Value())
	require.Equal(t, [][]float32{{1, 2, 3, 4}}, outs[1].Value())
	require.Equal(t, int64(4), outs[2].Value())
}

func TestDynamicGatherScatterExpandNonZeroRandom(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Dynamic Gather (with negative index on dynamic axis), GatherElements, and ScatterElements.
	gsExec := MustNewExec(backend, func(data, idx1d, elemIdx, updates *Node) (*Node, *Node, *Node) {
		g := data.Graph()
		gatherNode := &protos.NodeProto{OpType: "Gather", Input: []string{"d", "i"}, Output: []string{"g"}, Attribute: []*protos.AttributeProto{intAttr("axis", 1)}}
		gathered := convertTestNode(m, g, gatherNode, map[string]*Node{"d": data, "i": idx1d})[0]

		geNode := &protos.NodeProto{OpType: "GatherElements", Input: []string{"d", "ei"}, Output: []string{"ge"}, Attribute: []*protos.AttributeProto{intAttr("axis", 1)}}
		ge := convertTestNode(m, g, geNode, map[string]*Node{"d": data, "ei": elemIdx})[0]

		seNode := &protos.NodeProto{
			OpType: "ScatterElements", Input: []string{"d", "ei", "u"}, Output: []string{"se"},
			Attribute: []*protos.AttributeProto{intAttr("axis", 1), stringAttr("reduction", "add")},
		}
		se := convertTestNode(m, g, seNode, map[string]*Node{"d": data, "ei": elemIdx, "u": updates})[0]
		return gathered, ge, se
	}).WithDynamicAxes([]string{"b", "seq"}, []string{""}, []string{"b", "k"}, []string{"b", "k"})
	defer gsExec.Finalize()

	gsOuts := gsExec.MustExec(
		[][]float32{{10, 20, 30}, {40, 50, 60}},
		[]int64{-1, 0},
		[][]int64{{-1, 0}, {1, -3}},
		[][]float32{{5, 1}, {2, 3}},
	)
	for _, o := range gsOuts {
		defer o.FinalizeAll()
	}
	require.Equal(t, [][]float32{{30, 10}, {60, 40}}, gsOuts[0].Value())
	require.Equal(t, [][]float32{{30, 10}, {50, 40}}, gsOuts[1].Value())
	require.Equal(t, [][]float32{{11, 20, 35}, {43, 52, 60}}, gsOuts[2].Value())

	// Runtime NonZero (both non-empty and all-zero inputs on the same compiled executable).
	nzExec := MustNewExec(backend, func(x *Node) *Node {
		node := &protos.NodeProto{OpType: "NonZero", Input: []string{"x"}, Output: []string{"y"}}
		return convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x})[0]
	}).WithDynamicAxes([]string{"r", "c"})
	defer nzExec.Finalize()

	nz1 := nzExec.MustExec1([][]float32{{0, 3, 0}, {4, 0, -1}})
	defer nz1.FinalizeAll()
	require.Equal(t, [][]int64{{0, 1, 1}, {1, 0, 2}}, nz1.Value())

	nz0 := nzExec.MustExec1([][]float32{{0, 0}, {0, 0}})
	defer nz0.FinalizeAll()
	require.Equal(t, shapes.Make(dtypes.Int64, 2, 0), nz0.Shape())

	// Dynamic RandomNormalLike: verify nil-scope path does not create variables and dynamic batch works.
	rngExec := MustNewExec(backend, func(x *Node) *Node {
		node := &protos.NodeProto{
			OpType:    "RandomNormalLike",
			Input:     []string{"x"},
			Output:    []string{"y"},
			Attribute: []*protos.AttributeProto{{Name: "seed", Type: protos.AttributeProto_FLOAT, F: 42}},
		}
		return convertRandomNormalLike(nil, node, []*Node{x})
	}).WithDynamicAxes([]string{"batch", ""})
	defer rngExec.Finalize()

	rn1 := rngExec.MustExec1([][]float32{{0, 0, 0}, {0, 0, 0}})
	defer rn1.FinalizeAll()
	require.Equal(t, shapes.Make(dtypes.Float32, 2, 3), rn1.Shape())

	rn2 := rngExec.MustExec1([][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}})
	defer rn2.FinalizeAll()
	require.Equal(t, shapes.Make(dtypes.Float32, 4, 3), rn2.Shape())

	// Also verify scope-backed RandomNormalLike advances RNG state without leaking unexpected non-RNG vars.
	store := model.NewStore()
	scopeExec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		node := &protos.NodeProto{OpType: "RandomNormalLike", Input: []string{"x"}, Output: []string{"y"}}
		return convertRandomNormalLike(scope, node, []*Node{x})
	}).WithDynamicAxes([]string{"batch", ""})
	defer scopeExec.Finalize()
	sr1 := scopeExec.MustExec1([][]float32{{0, 0}, {0, 0}})
	defer sr1.FinalizeAll()
	sr2 := scopeExec.MustExec1([][]float32{{0, 0}, {0, 0}})
	defer sr2.FinalizeAll()
	require.NotEqual(t, sr1.Value(), sr2.Value())
	for v := range store.IterVariables() {
		require.True(t, strings.Contains(v.Name(), "rngState"), "unexpected variable: %s", v.Name())
	}

	// Expand and ConstantOfShape against Shape(ref) where ref is ["batch", 3] preserve static axis 1 == 3.
	var expShape, cosShape shapes.Shape
	expCosExec := MustNewExec(backend, func(x, ref *Node) (*Node, *Node) {
		shapeNode := &protos.NodeProto{OpType: "Shape", Input: []string{"ref"}, Output: []string{"sh"}}
		expNode := &protos.NodeProto{OpType: "Expand", Input: []string{"x", "sh"}, Output: []string{"exp"}}
		cosNode := &protos.NodeProto{OpType: "ConstantOfShape", Input: []string{"sh"}, Output: []string{"cos"}}
		mShape := createTestModelWithDTypePromoConfig(false, false)
		mShape.NodeOutputToNode = map[string]*protos.NodeProto{"sh": shapeNode}
		outputs := map[string]*Node{"x": x, "ref": ref}
		outputs["sh"] = convertShape(shapeNode, []*Node{ref})
		outputs["exp"] = convertExpand(mShape, outputs, expNode, []*Node{x, outputs["sh"]})
		outputs["cos"] = convertConstantOfShape(mShape, outputs, cosNode, []*Node{outputs["sh"]})
		expShape = outputs["exp"].Shape()
		cosShape = outputs["cos"].Shape()
		return outputs["exp"], outputs["cos"]
	}).WithDynamicAxes([]string{"", ""}, []string{"batch", ""})
	defer expCosExec.Finalize()

	expOuts := expCosExec.MustExec([][]float32{{7}}, [][]float32{{0, 0, 0}, {0, 0, 0}})
	for _, o := range expOuts {
		defer o.FinalizeAll()
	}
	require.Equal(t, 3, expShape.Dim(1))
	require.Equal(t, 3, cosShape.Dim(1))
	require.Equal(t, [][]float32{{7, 7, 7}, {7, 7, 7}}, expOuts[0].Value())
	require.Equal(t, [][]float32{{0, 0, 0}, {0, 0, 0}}, expOuts[1].Value())
}

func TestDynamicGridSampleBatchAlignment(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	// Static X batch (N=2) with dynamic grid batch (via Reshape with -1) and dynamic grid spatial dims.
	staticXDynGridExec := MustNewExec(backend, func(x, flatGrid, gridShape *Node) *Node {
		g := x.Graph()
		grid := convertReshape(m, map[string]*Node{}, &protos.NodeProto{
			OpType: "Reshape", Input: []string{"flat", "shape"}, Output: []string{"grid"},
		}, []*Node{flatGrid, gridShape})
		return convertTestNode(m, g, &protos.NodeProto{
			OpType: "GridSample", Input: []string{"x", "grid"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{intAttr("align_corners", 1)},
		}, map[string]*Node{"x": x, "grid": grid})[0]
	}).WithDynamicAxes([]string{"", "", "", ""}, []string{"flat_n", ""}, []string{""})
	defer staticXDynGridExec.Finalize()

	xStatic := [][][][]float32{
		{{{0, 2}, {4, 6}}},
		{{{10, 12}, {14, 16}}},
	}
	flatGrid := [][]float32{{0, 0}, {-1, 1}, {-1, -1}, {1, 1}}
	got1 := staticXDynGridExec.MustExec1(xStatic, flatGrid, []int64{-1, 1, 2, 2})
	defer got1.FinalizeAll()
	require.Equal(t, [][][][]float32{
		{{{3, 4}}},
		{{{10, 16}}},
	}, got1.Value())

	// Runtime mismatch on inferred grid batch (3 vs static X batch 2) must fail.
	err = exceptions.TryCatch[error](func() {
		bad := staticXDynGridExec.MustExec1(
			xStatic,
			[][]float32{{0, 0}, {0, 0}, {0, 0}},
			[]int64{-1, 1, 1, 2},
		)
		defer bad.FinalizeAll()
	})
	require.Error(t, err)

	// Dynamic X batch with static grid batch (N=2).
	dynXStaticGridExec := MustNewExec(backend, func(x, grid *Node) *Node {
		return convertTestNode(m, x.Graph(), &protos.NodeProto{
			OpType: "GridSample", Input: []string{"x", "grid"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{intAttr("align_corners", 1)},
		}, map[string]*Node{"x": x, "grid": grid})[0]
	}).WithDynamicAxes([]string{"batch", "", "", ""}, []string{"", "", "", ""})
	defer dynXStaticGridExec.Finalize()

	got2 := dynXStaticGridExec.MustExec1(xStatic, [][][][]float32{
		{{{0, 0}, {-1, 1}}},
		{{{-1, -1}, {1, 1}}},
	})
	defer got2.FinalizeAll()
	require.Equal(t, [][][][]float32{
		{{{3, 4}}},
		{{{10, 16}}},
	}, got2.Value())

	// Both dynamic with distinct axis names: matching runtime batch works, mismatching fails.
	bothDynExec := MustNewExec(backend, func(x, grid *Node) *Node {
		return convertTestNode(m, x.Graph(), &protos.NodeProto{
			OpType: "GridSample", Input: []string{"x", "grid"}, Output: []string{"y"},
			Attribute: []*protos.AttributeProto{intAttr("align_corners", 1)},
		}, map[string]*Node{"x": x, "grid": grid})[0]
	}).WithDynamicAxes([]string{"bx", "", "", ""}, []string{"bg", "oh", "ow", ""})
	defer bothDynExec.Finalize()

	got3 := bothDynExec.MustExec1(xStatic, [][][][]float32{
		{{{0, 0}}},
		{{{1, 1}}},
	})
	defer got3.FinalizeAll()
	require.Equal(t, [][][][]float32{{{{3}}}, {{{16}}}}, got3.Value())

	err = exceptions.TryCatch[error](func() {
		bad := bothDynExec.MustExec1(xStatic, [][][][]float32{{{{0, 0}}}})
		defer bad.FinalizeAll()
	})
	require.Error(t, err)
}
