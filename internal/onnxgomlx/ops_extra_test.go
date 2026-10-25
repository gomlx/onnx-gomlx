package onnxgomlx

// Tests for the ops in ops_extra.go and for Slice with negative steps.
//
// They use the pure Go backend directly (instead of graphtest.RunTestGraphFn, which may try to auto-install XLA).
// Expected values were computed by hand or with numpy / the ONNX operator documentation examples.

import (
	"math"
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/stretchr/testify/require"
)

// runGoBackendTest is graphtest.RunTestGraphFn using the pure Go backend.
func runGoBackendTest(t *testing.T, testName string, graphFn graphtest.TestGraphFn, want []any, delta float64) {
	t.Helper()
	graphtest.RunTestGraphFnWithBackend(t, testName, gobackend.GetBackend(), graphFn, want, delta)
}

// convertTestNode converts a single ONNX node with the given inputs, and returns its outputs.
func convertTestNode(m *Model, g *Graph, node *protos.NodeProto, inputs map[string]*Node) []*Node {
	converted := make(map[string]*Node, len(inputs))
	for name, n := range inputs {
		converted[name] = n
	}
	m.convertNode(nil, g, node, converted)
	outputs := make([]*Node, len(node.Output))
	for ii, name := range node.Output {
		outputs[ii] = converted[name]
	}
	return outputs
}

func intAttr(name string, value int) *protos.AttributeProto {
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_INT, I: int64(value)}
}

func stringAttr(name, value string) *protos.AttributeProto {
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_STRING, S: []byte(value)}
}

func TestExtraUnaryMathOps(t *testing.T) {
	m := createTestModelWithDTypePromoConfig(false, false)
	unary := func(g *Graph, opType string, x *Node) *Node {
		node := &protos.NodeProto{OpType: opType, Input: []string{"x"}, Output: []string{"y"}}
		return convertTestNode(m, g, node, map[string]*Node{"x": x})[0]
	}

	runGoBackendTest(t, "Asinh", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-1e30, -100, -1, -0.5, 0, 0.5, 1, 3, 5000, 1e20})
		tiny := Const(g, []float32{1e-6, -1e-10, 1e-30})
		x64 := Const(g, []float64{-1e300, -0.5, 1e-200, 3})
		inputs = []*Node{x}
		outputs = []*Node{
			unary(g, "Asinh", x),
			// asinh(x)/x -> 1 for x -> 0: checks there is no precision loss for small values.
			Div(unary(g, "Asinh", tiny), tiny),
			unary(g, "Asinh", x64),
			// NaN and +/-Inf.
			IsNaN(unary(g, "Asinh", Const(g, []float32{float32(math.NaN())}))),
			unary(g, "Asinh", Const(g, []float32{float32(math.Inf(1)), float32(math.Inf(-1))})),
		}
		return
	}, []any{
		[]float32{-69.7706985, -5.29834223, -0.88137358, -0.48121181, 0, 0.48121181, 0.88137358, 1.81844652, 9.21034050, 46.7448502},
		[]float32{1, 1, 1},
		[]float64{-691.4686750787736, -0.48121182505960347, 1e-200, 1.8184464592320668},
		[]bool{true},
		[]float32{float32(math.Inf(1)), float32(math.Inf(-1))},
	}, 1e-4)

	runGoBackendTest(t, "Sinh", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-10, -1, -0.25, 0, 0.25, 1, 10})
		tiny := Const(g, []float32{1e-6, -1e-10, 1e-30})
		inputs = []*Node{x}
		outputs = []*Node{
			unary(g, "Sinh", x),
			// sinh(x)/x -> 1 for x -> 0: checks there is no cancellation for small values.
			Div(unary(g, "Sinh", tiny), tiny),
			// sinh(89) ~ 2.2e38 fits in float32, but exp(89) doesn't: log(sinh(89)) = 89 - log(2).
			Log(Abs(unary(g, "Sinh", Const(g, []float32{89, -89})))),
			unary(g, "Sinh", Const(g, []float64{-2, 0.5})),
			unary(g, "Sinh", Const(g, []float32{float32(math.Inf(1)), -100})),
		}
		return
	}, []any{
		[]float32{-11013.2324, -1.17520118, -0.252612323, 0, 0.252612323, 1.17520118, 11013.2324},
		[]float32{1, 1, 1},
		[]float32{88.3068528, 88.3068528},
		[]float64{-3.626860407847019, 0.5210953054937474},
		[]float32{float32(math.Inf(1)), float32(math.Inf(-1))},
	}, 1e-2)

	runGoBackendTest(t, "Softplus", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-100, -20, -1, 0, 1, 20, 100})
		inputs = []*Node{x}
		outputs = []*Node{
			unary(g, "Softplus", x),
			// Values for which log(1+exp(x)) would overflow/underflow if computed naively.
			unary(g, "Softplus", Const(g, []float64{-800, 800})),
			// Integer inputs are promoted to Float32.
			unary(g, "Softplus", Const(g, []int32{0})),
		}
		return
	}, []any{
		[]float32{0, 2.06115369e-09, 0.313261688, 0.693147182, 1.31326163, 20, 100},
		[]float64{0, 800},
		[]float32{0.693147182},
	}, 1e-5)
}

func TestExtraIsInf(t *testing.T) {
	m := createTestModelWithDTypePromoConfig(false, false)
	isInf := func(g *Graph, x *Node, attrs ...*protos.AttributeProto) *Node {
		node := &protos.NodeProto{OpType: "IsInf", Input: []string{"x"}, Output: []string{"y"}, Attribute: attrs}
		return convertTestNode(m, g, node, map[string]*Node{"x": x})[0]
	}
	runGoBackendTest(t, "IsInf", func(g *Graph) (inputs, outputs []*Node) {
		inf, nan := math.Inf(1), math.NaN()
		x := Const(g, []float32{float32(-inf), -1.2, 0, 1.2, float32(inf), float32(nan)})
		inputs = []*Node{x}
		outputs = []*Node{
			isInf(g, x),
			isInf(g, x, intAttr("detect_negative", 0)),
			isInf(g, x, intAttr("detect_positive", 0)),
			isInf(g, x, intAttr("detect_positive", 0), intAttr("detect_negative", 0)),
			isInf(g, Const(g, [][]float64{{inf, 1}, {-inf, nan}})),
			isInf(g, Const(g, []int32{math.MaxInt32, 0})),
		}
		return
	}, []any{
		[]bool{true, false, false, false, true, false},
		[]bool{false, false, false, false, true, false},
		[]bool{true, false, false, false, false, false},
		[]bool{false, false, false, false, false, false},
		[][]bool{{true, false}, {true, false}},
		[]bool{false, false},
	}, -1)
}

func TestExtraScatterElements(t *testing.T) {
	m := createTestModelWithDTypePromoConfig(false, false)
	scatterElements := func(g *Graph, data, indices, updates *Node, attrs ...*protos.AttributeProto) *Node {
		node := &protos.NodeProto{
			OpType:    "ScatterElements",
			Input:     []string{"data", "indices", "updates"},
			Output:    []string{"y"},
			Attribute: attrs,
		}
		return convertTestNode(m, g, node, map[string]*Node{"data": data, "indices": indices, "updates": updates})[0]
	}

	runGoBackendTest(t, "ScatterElements: ONNX examples", func(g *Graph) (inputs, outputs []*Node) {
		zeros := Zeros(g, shapes.Make(dtypes.Float32, 3, 3))
		indices := Const(g, [][]int64{{1, 0, 2}, {0, 2, 1}})
		updates := Const(g, [][]float32{{1.0, 1.1, 1.2}, {2.0, 2.1, 2.2}})
		row := Const(g, [][]float32{{1, 2, 3, 4, 5}})
		outputs = []*Node{
			// Example 1: default axis=0.
			scatterElements(g, zeros, indices, updates),
			// Example 2: axis=1.
			scatterElements(g, row, Const(g, [][]int64{{1, 3}}), Const(g, [][]float32{{1.1, 2.1}}), intAttr("axis", 1)),
			// Negative indices and negative axis.
			scatterElements(g, row, Const(g, [][]int32{{1, -3}}), Const(g, [][]float32{{1.1, 2.1}}), intAttr("axis", -1)),
		}
		return
	}, []any{
		[][]float32{{2.0, 1.1, 0.0}, {1.0, 0.0, 2.2}, {0.0, 2.1, 1.2}},
		[][]float32{{1.0, 1.1, 3.0, 2.1, 5.0}},
		[][]float32{{1.0, 1.1, 2.1, 4.0, 5.0}},
	}, 1e-6)

	runGoBackendTest(t, "ScatterElements: reductions with duplicate indices", func(g *Graph) (inputs, outputs []*Node) {
		row := Const(g, [][]float32{{1, 2, 3, 4, 5}})
		indices := Const(g, [][]int64{{1, 1}})
		updates := Const(g, [][]float32{{1.1, 2.1}})
		outputs = []*Node{
			scatterElements(g, row, indices, updates, intAttr("axis", 1), stringAttr("reduction", "add")),
			scatterElements(g, row, indices, updates, intAttr("axis", 1), stringAttr("reduction", "mul")),
			scatterElements(g, row, indices, updates, intAttr("axis", 1), stringAttr("reduction", "max")),
			scatterElements(g, row, indices, updates, intAttr("axis", 1), stringAttr("reduction", "min")),
		}
		return
	}, []any{
		[][]float32{{1, 5.2, 3, 4, 5}},
		[][]float32{{1, 4.62, 3, 4, 5}},
		[][]float32{{1, 2.1, 3, 4, 5}},
		[][]float32{{1, 1.1, 3, 4, 5}},
	}, 1e-5)

	runGoBackendTest(t, "ScatterElements: indices smaller than data, 3D, ints and bools", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]int32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
		// indices only cover the first 2 rows.
		indices := Const(g, [][]int64{{0, 0}, {2, 1}})
		updates := Const(g, [][]int32{{2, 3}, {4, 5}})
		data3D := IotaFull(g, shapes.Make(dtypes.Float32, 2, 3, 2))
		outputs = []*Node{
			scatterElements(g, data, indices, updates, intAttr("axis", 1), stringAttr("reduction", "mul")),
			scatterElements(g, data, indices, updates, intAttr("axis", 1), stringAttr("reduction", "add")),
			scatterElements(g, data, Const(g, [][]int64{{2, 0}}), Const(g, [][]int32{{-1, -2}})),
			// 3D, axis=1: output[i][indices[i][j][k]][k] = updates[i][j][k].
			scatterElements(g, data3D, Const(g, [][][]int64{{{2, 0}}, {{1, 1}}}), Const(g, [][][]float32{{{-1, -2}}, {{-3, -4}}}),
				intAttr("axis", 1)),
			// Booleans: max is a logical OR, min a logical AND.
			scatterElements(g, Const(g, []bool{false, false, false, false}), Const(g, []int64{0, 0, 3}),
				Const(g, []bool{false, true, true}), stringAttr("reduction", "max")),
			scatterElements(g, Const(g, []bool{true, true, true}), Const(g, []int64{1, 1}),
				Const(g, []bool{true, false}), stringAttr("reduction", "min")),
			scatterElements(g, Const(g, []bool{true, false}), Const(g, []int64{1}), Const(g, []bool{true})),
		}
		return
	}, []any{
		[][]int32{{6, 2, 3}, {4, 25, 24}, {7, 8, 9}},
		[][]int32{{6, 2, 3}, {4, 10, 10}, {7, 8, 9}},
		[][]int32{{1, -2, 3}, {4, 5, 6}, {-1, 8, 9}},
		[][][]float32{{{0, -2}, {2, 3}, {-1, 5}}, {{6, 7}, {-3, -4}, {10, 11}}},
		[]bool{true, false, false, true},
		[]bool{true, false, true},
		[]bool{true, true},
	}, -1)
}

func TestExtraUnique(t *testing.T) {
	m := createTestModelWithDTypePromoConfig(false, false)
	unique := func(g *Graph, x *Node, attrs ...*protos.AttributeProto) []*Node {
		node := &protos.NodeProto{
			OpType:    "Unique",
			Input:     []string{"x"},
			Output:    []string{"y", "indices", "inverse_indices", "counts"},
			Attribute: attrs,
		}
		return convertTestNode(m, g, node, map[string]*Node{"x": x})
	}

	runGoBackendTest(t, "Unique: ONNX examples without axis", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{2, 1, 1, 3, 4, 3})
		outputs = append(outputs, unique(g, x, intAttr("sorted", 0))...)
		outputs = append(outputs, unique(g, x)...) // sorted=1 is the default.
		// Inputs are flattened if no axis is given.
		outputs = append(outputs, unique(g, Const(g, [][]int64{{1, 3}, {2, 1}}))...)
		return
	}, []any{
		// Not sorted.
		[]float32{2, 1, 3, 4}, []int64{0, 1, 3, 4}, []int64{0, 1, 1, 2, 3, 2}, []int64{1, 2, 2, 1},
		// Sorted.
		[]float32{1, 2, 3, 4}, []int64{1, 0, 3, 4}, []int64{1, 0, 0, 2, 3, 2}, []int64{2, 1, 2, 1},
		// Flattened 2D.
		[]int64{1, 2, 3}, []int64{0, 2, 1}, []int64{0, 2, 1, 0}, []int64{2, 1, 1},
	}, -1)

	runGoBackendTest(t, "Unique: ONNX examples with axis", func(g *Graph) (inputs, outputs []*Node) {
		x0 := Const(g, [][]int32{{1, 0, 0}, {1, 0, 0}, {2, 3, 4}})
		x1 := Const(g, [][][]float32{{{1, 1}, {0, 1}, {2, 1}, {0, 1}}, {{1, 1}, {0, 1}, {2, 1}, {0, 1}}})
		outputs = append(outputs, unique(g, x0, intAttr("axis", 0))...)
		outputs = append(outputs, unique(g, x1, intAttr("axis", 1))...)
		outputs = append(outputs, unique(g, x1, intAttr("axis", -2), intAttr("sorted", 0))...)
		return
	}, []any{
		[][]int32{{1, 0, 0}, {2, 3, 4}}, []int64{0, 2}, []int64{0, 0, 1}, []int64{2, 1},
		[][][]float32{{{0, 1}, {1, 1}, {2, 1}}, {{0, 1}, {1, 1}, {2, 1}}}, []int64{1, 0, 2}, []int64{1, 0, 2, 0}, []int64{2, 1, 1},
		[][][]float32{{{1, 1}, {0, 1}, {2, 1}}, {{1, 1}, {0, 1}, {2, 1}}}, []int64{0, 1, 2}, []int64{0, 1, 2, 1}, []int64{1, 2, 1},
	}, -1)

	t.Run("Unique: only some outputs", func(t *testing.T) {
		g := NewGraph(gobackend.GetBackend(), "unique_partial")
		node := &protos.NodeProto{
			OpType: "Unique",
			Input:  []string{"x"},
			Output: []string{"y", "", "inverse_indices"},
		}
		outputs := convertTestNode(m, g, node, map[string]*Node{"x": Const(g, []int32{5, 5, 7})})
		require.Equal(t, []int{2}, outputs[0].Shape().Dimensions)
		require.Nil(t, outputs[1])
		require.Equal(t, shapes.Make(dtypes.Int64, 3), outputs[2].Shape())
	})

	t.Run("Unique: non-constant input fails with a clear error", func(t *testing.T) {
		g := NewGraph(gobackend.GetBackend(), "unique_dynamic")
		x := Parameter(g, "x", shapes.Make(dtypes.Float32, 3))
		mDynamic := &Model{InputsNameSet: sets.MakeWith("x")}
		node := &protos.NodeProto{OpType: "Unique", Input: []string{"x"}, Output: []string{"y"}}
		err := exceptions.TryCatch[error](func() { convertTestNode(mDynamic, g, node, map[string]*Node{"x": x}) })
		require.Error(t, err)
		require.ErrorContains(t, err, "Unique is only supported for inputs that can be evaluated at graph-build time")
	})
}

func TestComputeUniqueDTypes(t *testing.T) {
	y, indices, inverse, counts, err := computeUnique(tensors.FromValue([]bool{true, false, true}), -1, true)
	require.NoError(t, err)
	require.Equal(t, []bool{false, true}, tensors.MustCopyFlatData[bool](y))
	require.Equal(t, []int64{1, 0}, tensors.MustCopyFlatData[int64](indices))
	require.Equal(t, []int64{1, 0, 1}, tensors.MustCopyFlatData[int64](inverse))
	require.Equal(t, []int64{1, 2}, tensors.MustCopyFlatData[int64](counts))

	// -0 and +0 are the same value; NaNs are all equal and sorted first.
	nan := math.NaN()
	y, _, inverse, counts, err = computeUnique(tensors.FromValue([]float64{0, nan, math.Copysign(0, -1), nan, -1}), -1, true)
	require.NoError(t, err)
	yValues := tensors.MustCopyFlatData[float64](y)
	require.Len(t, yValues, 3)
	require.True(t, math.IsNaN(yValues[0]))
	require.Equal(t, []float64{-1, 0}, yValues[1:])
	require.Equal(t, []int64{2, 0, 2, 0, 1}, tensors.MustCopyFlatData[int64](inverse))
	require.Equal(t, []int64{2, 1, 2}, tensors.MustCopyFlatData[int64](counts))
}

func TestSliceNegativeSteps(t *testing.T) {
	m := createTestModelWithDTypePromoConfig(false, false)
	slice := func(g *Graph, x *Node, starts, ends, axes, steps []int64) *Node {
		node := &protos.NodeProto{
			OpType: "Slice",
			Input:  []string{"x", "starts", "ends", "axes", "steps"},
			Output: []string{"y"},
		}
		return convertTestNode(m, g, node, map[string]*Node{
			"x":      x,
			"starts": Const(g, starts),
			"ends":   Const(g, ends),
			"axes":   Const(g, axes),
			"steps":  Const(g, steps),
		})[0]
	}
	runGoBackendTest(t, "Slice with negative steps", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
		x2D := Const(g, [][]int32{{1, 2, 3}, {4, 5, 6}})
		outputs = []*Node{
			// torch.flip as exported by PyTorch: start=-1, end=INT64_MIN, step=-1.
			slice(g, x, []int64{-1}, []int64{math.MinInt64}, []int64{-1}, []int64{-1}),
			// x[8:2:-2]
			slice(g, x, []int64{8}, []int64{2}, []int64{0}, []int64{-2}),
			// x[100:-100:-3] -> start clamped to the last element.
			slice(g, x, []int64{100}, []int64{-100}, []int64{0}, []int64{-3}),
			// x[-2:0:-1]
			slice(g, x, []int64{-2}, []int64{0}, []int64{0}, []int64{-1}),
			// 2D: reverse rows, and columns 2:0:-1.
			slice(g, x2D, []int64{-1, 2}, []int64{math.MinInt64, 0}, []int64{0, 1}, []int64{-1, -1}),
			// Mixed positive and negative steps.
			slice(g, x2D, []int64{0, -1}, []int64{2, math.MinInt64}, []int64{0, 1}, []int64{1, -2}),
		}
		return
	}, []any{
		[]float32{9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
		[]float32{8, 6, 4},
		[]float32{9, 6, 3, 0},
		[]float32{8, 7, 6, 5, 4, 3, 2, 1},
		[][]int32{{6, 5}, {3, 2}},
		[][]int32{{3, 1}, {6, 4}},
	}, -1)
}
