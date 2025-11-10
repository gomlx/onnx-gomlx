package onnx

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
)

func TestONNXWhere(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Where(): Dense", func(g *Graph) (inputs, outputs []*Node) {
		cond := ConvertDType(Iota(g, shapes.Make(dtypes.Int32, 3, 2), -1), dtypes.Bool)
		onTrue := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3, 2)))
		onFalse := Neg(onTrue)
		inputs = []*Node{cond, onTrue, onFalse}
		outputs = []*Node{
			onnxWhere([]*Node{cond, onTrue, onFalse}),
			onnxWhere([]*Node{Const(g, true), onTrue, onFalse}),
			onnxWhere([]*Node{Const(g, false), onTrue, onFalse}),
			onnxWhere([]*Node{cond, Const(g, float32(100)), onFalse}),
			onnxWhere([]*Node{cond, onTrue, Const(g, []float32{100, 1000})}),
		}
		return
	}, []any{
		[][]float32{{-1, 2}, {-3, 4}, {-5, 6}},
		[][]float32{{1, 2}, {3, 4}, {5, 6}},
		[][]float32{{-1, -2}, {-3, -4}, {-5, -6}},
		[][]float32{{-1, 100}, {-3, 100}, {-5, 100}},
		[][]float32{{100, 2}, {100, 4}, {100, 6}},
	}, -1)
}

func TestONNXGather(t *testing.T) {
	graphtest.RunTestGraphFn(t, "onnxGather(axis=0)", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]float32{{1.0, 1.2}, {2.3, 3.4}, {4.5, 5.7}})
		indices := Const(g, [][]int32{{0, 1}, {1, 2}})
		inputs = []*Node{data, indices}
		outputs = []*Node{onnxGather(data, indices, 0)}
		return
	}, []any{
		[][][]float32{
			{
				{1.0, 1.2},
				{2.3, 3.4},
			},
			{
				{2.3, 3.4},
				{4.5, 5.7},
			},
		},
	}, -1)

	graphtest.RunTestGraphFn(t, "onnxGather(axis=1)", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]float32{
			{1.0, 1.2, 1.9},
			{2.3, 3.4, 3.9},
			{4.5, 5.7, 5.9},
		})
		indices := Const(g, [][]int32{{0, 2}})
		inputs = []*Node{data, indices}
		outputs = []*Node{onnxGather(data, indices, 1)}
		return
	}, []any{
		[][][]float32{
			{{1.0, 1.9}},
			{{2.3, 3.9}},
			{{4.5, 5.9}},
		},
	}, -1)
}

func TestTile(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Tile 1D", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, []float32{1, 2})
		inputs = []*Node{operand}
		outputs = []*Node{onnxTile(operand, []int{2})}
		return
	}, []any{
		[]float32{1, 2, 1, 2},
	}, -1)

	graphtest.RunTestGraphFn(t, "Tile 2D", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, [][]float32{{1.0, 1.2}, {2.3, 3.4}, {4.5, 5.7}})
		inputs = []*Node{operand}
		outputs = []*Node{onnxTile(operand, []int{1, 2})}
		return
	}, []any{
		[][]float32{
			{1.0, 1.2, 1.0, 1.2},
			{2.3, 3.4, 2.3, 3.4},
			{4.5, 5.7, 4.5, 5.7},
		},
	}, -1)
}

func TestRangeCount(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	testFn := func(start, limit, delta any, want int) {
		startT := tensors.FromAnyValue(start)
		limitT := tensors.FromAnyValue(limit)
		deltaT := tensors.FromAnyValue(delta)
		got := rangeCount(backend, startT, limitT, deltaT)
		fmt.Printf("\trangeCount(start=%s, limit=%s, delta=%s) = %d (want %d)\n", startT, limitT, deltaT, got, want)
		assert.Equal(t, want, got)
	}

	testFn(uint8(3), uint8(9), uint8(3), 2)
	testFn(uint8(3), uint8(8), uint8(3), 2)
	testFn(uint8(3), uint8(7), uint8(3), 2)
	testFn(float32(3), float32(9.1), float32(3), 3)
	testFn(int32(10), int32(4), int32(-2), 3)
	testFn(int32(10), int32(5), int32(-2), 3)
	testFn(float64(10), float64(3.9), float64(-2), 4)
}

func TestOnnxGatherElements(t *testing.T) {
	graphtest.RunTestGraphFn(t, "GatherElements", func(g *Graph) (inputs, outputs []*Node) {
		data := Const(g, [][]float32{{1, 2}, {3, 4}})
		indices := Const(g, [][]int32{{0, 0}, {1, 0}})
		inputs = []*Node{data, indices}
		outputs = []*Node{
			onnxGatherElements(data, indices, 0),
			onnxGatherElements(data, indices, 1),
		}
		return
	}, []any{
		[][]float32{{1, 2}, {3, 2}},
		[][]float32{{1, 1}, {4, 3}},
	}, -1)

	graphtest.RunTestGraphFn(t, "GatherElements w/ incomplete indices", func(g *Graph) (inputs, outputs []*Node) {
		data := OnePlus(IotaFull(g, shapes.Make(dtypes.Float64, 3, 2)))
		indices0 := Const(g, [][]int8{{1, 2}})
		indices1 := Const(g, [][]int8{{0}, {0}, {1}})
		outputs = []*Node{
			onnxGatherElements(data, indices0, 0),
			onnxGatherElements(data, indices1, 1),
		}
		return
	}, []any{
		[][]float64{{3, 6}},
		[][]float64{{1}, {3}, {6}},
	}, -1)

	graphtest.RunTestGraphFn(t, "GatherElements: shape test with larger shapes", func(g *Graph) (inputs, outputs []*Node) {
		data := IotaFull(g, shapes.Make(dtypes.Float64, 3, 2, 512))
		indices := Iota(g, shapes.Make(dtypes.Int64, 3, 2, 7), 0)
		outputs = []*Node{
			Const(g, onnxGatherElements(data, indices, 2).Shape().Dimensions),
		}
		return
	}, []any{
		[]int64{3, 2, 7},
	}, -1)
}

func TestONNXCumSum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "CumSum", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, []float32{1, 2, 3})
		inputs = []*Node{operand}
		outputs = []*Node{
			onnxCumSum(operand, 0, false, false),
			onnxCumSum(operand, 0, true, false),
			onnxCumSum(operand, 0, false, true),
			onnxCumSum(operand, 0, true, true),
		}
		return
	}, []any{
		[]float32{1, 3, 6},
		[]float32{0, 1, 3},
		[]float32{6, 5, 3},
		[]float32{5, 3, 0},
	}, -1)
}

func TestONNXFlatten(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	testIdx := 0
	flattenFn := func(shape shapes.Shape, splitAxis int) shapes.Shape {
		g := NewGraph(backend, fmt.Sprintf("Flatten #%d", testIdx))
		testIdx++
		operand := IotaFull(g, shape)
		newShape := onnxFlatten(operand, splitAxis).Shape()
		g.Finalize()
		return newShape
	}

	// Scalar becomes a 1x1 matrix.
	flattenFn(shapes.Make(dtypes.Float32), 0).Assert(dtypes.Float32, 1, 1)

	// Vector can be split in 2 different ways.
	flattenFn(shapes.Make(dtypes.Int32, 7), 0).Assert(dtypes.Int32, 1, 7)
	flattenFn(shapes.Make(dtypes.Int32, 7), 1).AssertDims(7, 1)

	// Higher-dimensional tensor.
	flattenFn(shapes.Make(dtypes.Float32, 7, 2, 3, 4), 2).AssertDims(14, 12)
}

func TestONNXDequantizeLinear(t *testing.T) {
	graphtest.RunTestGraphFn(t, "DequantizeLinear-scalar", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int8{{-1, 0, 1}, {-2, 3, 4}})
		scale := Const(g, float32(3))
		inputs = []*Node{x, scale}
		outputs = []*Node{
			onnxDequantizeLinear(x, scale, nil, 1, scale.DType()),
		}
		return
	}, []any{
		[][]float32{{-3, 0, 3}, {-6, 9, 12}},
	}, -1)

	graphtest.RunTestGraphFn(t, "DequantizeLinear-outputDType", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int8{{-1, 0, 1}, {-2, 3, 4}})
		scale := Const(g, float32(3))
		inputs = []*Node{x, scale}
		outputs = []*Node{
			onnxDequantizeLinear(x, scale, nil, 1, dtypes.Float64),
		}
		return
	}, []any{
		[][]float64{{-3, 0, 3}, {-6, 9, 12}},
	}, -1)

	graphtest.RunTestGraphFn(t, "DequantizeLinear-zero-point", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int8{{-1, 0, 1}, {-2, 3, 4}})
		scale := Const(g, float32(3))
		zeroPoint := Const(g, int8(1))
		inputs = []*Node{x, scale}
		outputs = []*Node{
			onnxDequantizeLinear(x, scale, zeroPoint, 1, scale.DType()),
		}
		return
	}, []any{
		[][]float32{{-6, -3, 0}, {-9, 6, 9}},
	}, -1)

	graphtest.RunTestGraphFn(t, "DequantizeLinear-axis", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int8{{-1, 0, 1}, {-2, 3, 4}})
		scale := Const(g, []float32{3, 30, 300})
		inputs = []*Node{x, scale}
		outputs = []*Node{
			onnxDequantizeLinear(x, scale, nil, 1, scale.DType()),
		}
		return
	}, []any{
		[][]float32{{-3, 0, 300}, {-6, 90, 1200}},
	}, -1)
}

func TestONNX_DynamicQuantizeLinear(t *testing.T) {
	graphtest.RunTestGraphFn(t, "DequantizeLinear-scalar", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-3, -0, 3}, {-6, 9, 12}})
		inputs = []*Node{x}
		y, yScale, yZeroPoint := onnxDynamicQuantizeLinear(x)
		outputs = []*Node{y, yScale, yZeroPoint}
		return
	}, []any{
		[][]uint8{{43, 85, 127}, {0, 212, 255}},
		float32(0.07058824),
		uint8(85),
	}, 1e-3)
}

func TestONNXAveragePool(t *testing.T) {
	graphtest.RunTestGraphFn(t, "AveragePool", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][][][]float32{
			{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}},
		})
		inputs = []*Node{x}

		// Simple case: 2x2 kernel, 1x1 stride.
		pool := MeanPool(x).ChannelsAxis(timage.ChannelsFirst).WindowPerAxis(2, 2).StridePerAxis(1, 1)
		outputs = append(outputs, pool.Done())
		return
	}, []any{
		[][][][]float32{
			{{{3, 4}, {6, 7}}},
		},
	}, -1)
}

func TestONNXPad(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Pad", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{
			{1, 2, 3},
			{4, 5, 6},
		})
		inputs = []*Node{x}

		// Simple case: 1 pad on each side.
		paddings := []backends.PadAxis{{Low: 1, High: 1}, {Low: 1, High: 1}}
		padded := Pad(x, Scalar(g, x.DType(), 0), paddings)
		outputs = append(outputs, padded)
		return
	}, []any{
		[][]float32{
			{0, 0, 0, 0, 0},
			{0, 1, 2, 3, 0},
			{0, 4, 5, 6, 0},
			{0, 0, 0, 0, 0},
		},
	}, -1)
}

func TestONNXPad(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Pad", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{
			{1, 2, 3},
			{4, 5, 6},
		})
		inputs = []*Node{x}

		// Simple case: 1 pad on each side.
		paddings := [][2]int{{1, 1}, {1, 1}}
		padded := Pad(x, Scalar(g, x.DType(), 0), paddings)
		outputs = append(outputs, padded)
		return
	}, []any{
		[][]float32{
			{0, 0, 0, 0, 0},
			{0, 1, 2, 3, 0},
			{0, 4, 5, 6, 0},
			{0, 0, 0, 0, 0},
		},
	}, -1)
}
