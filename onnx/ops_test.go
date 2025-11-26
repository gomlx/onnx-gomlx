package onnx

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
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

func TestONNX_MatMulInteger(t *testing.T) {
	// Test basic MatMulInteger without zero points
	graphtest.RunTestGraphFn(t, "MatMulInteger-no-zero-points", func(g *Graph) (inputs, outputs []*Node) {
		// A: [2, 3] int8 matrix
		a := Const(g, [][]int8{{1, 2, 3}, {4, 5, 6}})
		// B: [3, 2] int8 matrix
		b := Const(g, [][]int8{{1, 2}, {3, 4}, {5, 6}})
		inputs = []*Node{a, b}
		outputs = []*Node{onnxMatMulInteger(a, b, nil, nil)}
		return
	}, []any{
		// Expected: A @ B
		// [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
		// [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
		[][]int32{{22, 28}, {49, 64}},
	}, -1)

	// Test MatMulInteger with scalar zero points
	graphtest.RunTestGraphFn(t, "MatMulInteger-scalar-zero-points", func(g *Graph) (inputs, outputs []*Node) {
		// A: [2, 3] int8 matrix
		a := Const(g, [][]int8{{1, 2, 3}, {4, 5, 6}})
		// B: [3, 2] int8 matrix
		b := Const(g, [][]int8{{1, 2}, {3, 4}, {5, 6}})
		// Zero points
		aZeroPoint := Const(g, int8(1))
		bZeroPoint := Const(g, int8(1))
		inputs = []*Node{a, b, aZeroPoint, bZeroPoint}
		outputs = []*Node{onnxMatMulInteger(a, b, aZeroPoint, bZeroPoint)}
		return
	}, []any{
		// Expected: (A - 1) @ (B - 1)
		// A-1 = [[0, 1, 2], [3, 4, 5]]
		// B-1 = [[0, 1], [2, 3], [4, 5]]
		// [0*0+1*2+2*4, 0*1+1*3+2*5] = [10, 13]
		// [3*0+4*2+5*4, 3*1+4*3+5*5] = [28, 40]
		[][]int32{{10, 13}, {28, 40}},
	}, -1)

	// Test MatMulInteger with uint8 inputs
	graphtest.RunTestGraphFn(t, "MatMulInteger-uint8", func(g *Graph) (inputs, outputs []*Node) {
		// A: [2, 2] uint8 matrix
		a := Const(g, [][]uint8{{10, 20}, {30, 40}})
		// B: [2, 2] uint8 matrix
		b := Const(g, [][]uint8{{1, 2}, {3, 4}})
		// Zero points
		aZeroPoint := Const(g, uint8(5))
		bZeroPoint := Const(g, uint8(1))
		inputs = []*Node{a, b, aZeroPoint, bZeroPoint}
		outputs = []*Node{onnxMatMulInteger(a, b, aZeroPoint, bZeroPoint)}
		return
	}, []any{
		// Expected: (A - 5) @ (B - 1)
		// A-5 = [[5, 15], [25, 35]]
		// B-1 = [[0, 1], [2, 3]]
		// [5*0+15*2, 5*1+15*3] = [30, 50]
		// [25*0+35*2, 25*1+35*3] = [70, 130]
		[][]int32{{30, 50}, {70, 130}},
	}, -1)

	// Test MatMulInteger with only A zero point
	graphtest.RunTestGraphFn(t, "MatMulInteger-a-zero-point-only", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]int8{{2, 3}, {4, 5}})
		b := Const(g, [][]int8{{1, 2}, {3, 4}})
		aZeroPoint := Const(g, int8(1))
		inputs = []*Node{a, b, aZeroPoint}
		outputs = []*Node{onnxMatMulInteger(a, b, aZeroPoint, nil)}
		return
	}, []any{
		// Expected: (A - 1) @ B
		// A-1 = [[1, 2], [3, 4]]
		// [1*1+2*3, 1*2+2*4] = [7, 10]
		// [3*1+4*3, 3*2+4*4] = [15, 22]
		[][]int32{{7, 10}, {15, 22}},
	}, -1)

	// Test MatMulInteger with only B zero point
	graphtest.RunTestGraphFn(t, "MatMulInteger-b-zero-point-only", func(g *Graph) (inputs, outputs []*Node) {
		a := Const(g, [][]int8{{1, 2}, {3, 4}})
		b := Const(g, [][]int8{{2, 3}, {4, 5}})
		bZeroPoint := Const(g, int8(1))
		inputs = []*Node{a, b}
		outputs = []*Node{onnxMatMulInteger(a, b, nil, bZeroPoint)}
		return
	}, []any{
		// Expected: A @ (B - 1)
		// B-1 = [[1, 2], [3, 4]]
		// [1*1+2*3, 1*2+2*4] = [7, 10]
		// [3*1+4*3, 3*2+4*4] = [15, 22]
		[][]int32{{7, 10}, {15, 22}},
	}, -1)

	// Test 3D batch matrix multiplication
	graphtest.RunTestGraphFn(t, "MatMulInteger-batch", func(g *Graph) (inputs, outputs []*Node) {
		// A: [2, 2, 3] - batch of 2 matrices
		a := Const(g, [][][]int8{
			{{1, 2, 3}, {4, 5, 6}},
			{{7, 8, 9}, {10, 11, 12}},
		})
		// B: [2, 3, 2] - batch of 2 matrices
		b := Const(g, [][][]int8{
			{{1, 2}, {3, 4}, {5, 6}},
			{{1, 0}, {0, 1}, {1, 0}},
		})
		inputs = []*Node{a, b}
		outputs = []*Node{onnxMatMulInteger(a, b, nil, nil)}
		return
	}, []any{
		// Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
		//   = [[22, 28], [49, 64]]
		// Batch 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[1,0]]
		//   = [[7+0+9, 0+8+0], [10+0+12, 0+11+0]] = [[16, 8], [22, 11]]
		[][][]int32{
			{{22, 28}, {49, 64}},
			{{16, 8}, {22, 11}},
		},
	}, -1)
}

////////////////////////////////////////////////////////////////////
//
// Tests for new operators added in this branch
//
////////////////////////////////////////////////////////////////////

func TestLayerNormalization(t *testing.T) {
	// Test basic layer normalization with default axis (-1)
	graphtest.RunTestGraphFn(t, "LayerNormalization-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Input tensor [2, 3]: normalize over last axis (axis=-1, which is axis 1)
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})
		scale := Const(g, []float32{1.0, 1.0, 1.0})
		bias := Const(g, []float32{0.0, 0.0, 0.0})

		// Create a mock node to pass attributes
		node := &protos.NodeProto{
			OpType: "LayerNormalization",
		}
		inputs = []*Node{x, scale, bias}
		outputs = []*Node{
			convertLayerNormalization(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Expected: normalized values with mean=0, variance=1 for each row
		// For [1,2,3]: mean=2, std≈0.8165, normalized ≈ [-1.224, 0, 1.224]
		// For [4,5,6]: mean=5, std≈0.8165, normalized ≈ [-1.224, 0, 1.224]
		[][]float32{{-1.2247, 0.0, 1.2247}, {-1.2247, 0.0, 1.2247}},
	}, 1e-3)

	// Test with scale and bias
	graphtest.RunTestGraphFn(t, "LayerNormalization-scale-bias", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})
		scale := Const(g, []float32{2.0, 2.0, 2.0})
		bias := Const(g, []float32{1.0, 1.0, 1.0})

		node := &protos.NodeProto{
			OpType: "LayerNormalization",
		}
		inputs = []*Node{x, scale, bias}
		outputs = []*Node{
			convertLayerNormalization(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Expected: (normalized * 2) + 1
		[][]float32{{-1.4494, 1.0, 3.4494}, {-1.4494, 1.0, 3.4494}},
	}, 1e-3)

	// Test with custom axis
	graphtest.RunTestGraphFn(t, "LayerNormalization-axis0", func(g *Graph) (inputs, outputs []*Node) {
		// Normalize over axis 0 and axis 1 (from axis 0 to end)
		x := Const(g, [][]float32{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}})
		scale := Const(g, [][]float32{{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}})
		bias := Const(g, [][]float32{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}})

		node := &protos.NodeProto{
			OpType: "LayerNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
			},
		}
		inputs = []*Node{x, scale, bias}
		outputs = []*Node{
			convertLayerNormalization(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Normalize over all elements (axis 0 to end means entire tensor)
		// Mean = 3.5, values normalized around that
		[][]float32{{-1.4638, 0.2928}, {-0.8783, 0.8783}, {-0.2928, 1.4638}},
	}, 1e-3)

	// Test 3D tensor (common in transformers: batch, sequence, features)
	graphtest.RunTestGraphFn(t, "LayerNormalization-3D", func(g *Graph) (inputs, outputs []*Node) {
		// Shape [2, 2, 3]: batch=2, seq_len=2, features=3
		x := Const(g, [][][]float32{
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			{{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}},
		})
		scale := Const(g, []float32{1.0, 1.0, 1.0})
		bias := Const(g, []float32{0.0, 0.0, 0.0})

		node := &protos.NodeProto{
			OpType: "LayerNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1}, // Last axis
			},
		}
		inputs = []*Node{x, scale, bias}
		outputs = []*Node{
			convertLayerNormalization(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Each feature vector should be normalized independently
		[][][]float32{
			{{-1.2247, 0.0, 1.2247}, {-1.2247, 0.0, 1.2247}},
			{{-1.2247, 0.0, 1.2247}, {-1.2247, 0.0, 1.2247}},
		},
	}, 1e-3)
}

func TestSplit(t *testing.T) {
	// Test equal splits
	graphtest.RunTestGraphFn(t, "Split-equal", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}})

		node := &protos.NodeProto{
			OpType: "Split",
			Output: []string{"out1", "out2"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{x}
		convertSplit(nil, convertedOutputs, node, inputs)

		outputs = []*Node{
			convertedOutputs["out1"],
			convertedOutputs["out2"],
		}
		return
	}, []any{
		[][]float32{{1.0, 2.0}, {5.0, 6.0}},
		[][]float32{{3.0, 4.0}, {7.0, 8.0}},
	}, -1)

	// Test Split on different axis (axis=1)
	graphtest.RunTestGraphFn(t, "Split-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0}})

		node := &protos.NodeProto{
			OpType: "Split",
			Output: []string{"out1", "out2", "out3"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{x}
		convertSplit(nil, convertedOutputs, node, inputs)

		outputs = []*Node{
			convertedOutputs["out1"],
			convertedOutputs["out2"],
			convertedOutputs["out3"],
		}
		return
	}, []any{
		[][]float32{{1.0, 2.0}, {7.0, 8.0}},
		[][]float32{{3.0, 4.0}, {9.0, 10.0}},
		[][]float32{{5.0, 6.0}, {11.0, 12.0}},
	}, -1)

	// Test 3-way equal split on axis 0
	graphtest.RunTestGraphFn(t, "Split-3way", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{
			{1.0, 2.0},
			{3.0, 4.0},
			{5.0, 6.0},
			{7.0, 8.0},
			{9.0, 10.0},
			{11.0, 12.0},
		})

		node := &protos.NodeProto{
			OpType: "Split",
			Output: []string{"out1", "out2", "out3"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{x}
		convertSplit(nil, convertedOutputs, node, inputs)

		outputs = []*Node{
			convertedOutputs["out1"],
			convertedOutputs["out2"],
			convertedOutputs["out3"],
		}
		return
	}, []any{
		[][]float32{{1.0, 2.0}, {3.0, 4.0}},
		[][]float32{{5.0, 6.0}, {7.0, 8.0}},
		[][]float32{{9.0, 10.0}, {11.0, 12.0}},
	}, -1)
}

func TestIf(t *testing.T) {
	// Test basic If with scalar condition - true case
	graphtest.RunTestGraphFn(t, "If-true-branch", func(g *Graph) (inputs, outputs []*Node) {
		cond := Const(g, true)

		// Create simple then/else branches
		thenGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2, 2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{1.0, 2.0, 3.0, 4.0},
							},
						},
					},
				},
			},
		}

		elseGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2, 2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{10.0, 20.0, 30.0, 40.0},
							},
						},
					},
				},
			},
		}

		node := &protos.NodeProto{
			OpType: "If",
			Input:  []string{"cond"},
			Output: []string{"output"},
			Attribute: []*protos.AttributeProto{
				{Name: "then_branch", Type: protos.AttributeProto_GRAPH, G: thenGraph},
				{Name: "else_branch", Type: protos.AttributeProto_GRAPH, G: elseGraph},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{cond}
		result := convertIf(model, convertedOutputs, node, inputs)
		outputs = []*Node{result}
		return
	}, []any{
		[][]float32{{1.0, 2.0}, {3.0, 4.0}},
	}, -1)

	// Test basic If with scalar condition - false case
	graphtest.RunTestGraphFn(t, "If-false-branch", func(g *Graph) (inputs, outputs []*Node) {
		cond := Const(g, false)

		thenGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2, 2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{1.0, 2.0, 3.0, 4.0},
							},
						},
					},
				},
			},
		}

		elseGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2, 2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{10.0, 20.0, 30.0, 40.0},
							},
						},
					},
				},
			},
		}

		node := &protos.NodeProto{
			OpType: "If",
			Input:  []string{"cond"},
			Output: []string{"output"},
			Attribute: []*protos.AttributeProto{
				{Name: "then_branch", Type: protos.AttributeProto_GRAPH, G: thenGraph},
				{Name: "else_branch", Type: protos.AttributeProto_GRAPH, G: elseGraph},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{cond}
		result := convertIf(model, convertedOutputs, node, inputs)
		outputs = []*Node{result}
		return
	}, []any{
		[][]float32{{10.0, 20.0}, {30.0, 40.0}},
	}, -1)

	// Test If with multiple outputs
	graphtest.RunTestGraphFn(t, "If-multiple-outputs", func(g *Graph) (inputs, outputs []*Node) {
		cond := Const(g, true)

		thenGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result1"}, {Name: "result2"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result1"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{1.0, 2.0},
							},
						},
					},
				},
				{
					OpType: "Constant",
					Output: []string{"result2"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{3.0, 4.0},
							},
						},
					},
				},
			},
		}

		elseGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result1"}, {Name: "result2"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Constant",
					Output: []string{"result1"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{10.0, 20.0},
							},
						},
					},
				},
				{
					OpType: "Constant",
					Output: []string{"result2"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{30.0, 40.0},
							},
						},
					},
				},
			},
		}

		node := &protos.NodeProto{
			OpType: "If",
			Input:  []string{"cond"},
			Output: []string{"output1", "output2"},
			Attribute: []*protos.AttributeProto{
				{Name: "then_branch", Type: protos.AttributeProto_GRAPH, G: thenGraph},
				{Name: "else_branch", Type: protos.AttributeProto_GRAPH, G: elseGraph},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{cond}
		convertIf(model, convertedOutputs, node, inputs)

		outputs = []*Node{
			convertedOutputs["output1"],
			convertedOutputs["output2"],
		}
		return
	}, []any{
		[]float32{1.0, 2.0},
		[]float32{3.0, 4.0},
	}, -1)

	// Test If with sub-graph that references parent outputs
	graphtest.RunTestGraphFn(t, "If-subgraph-parent-reference", func(g *Graph) (inputs, outputs []*Node) {
		cond := Const(g, true)
		parentValue := Const(g, []float32{100.0, 200.0})

		// Then branch: Add parent value
		thenGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Add",
					Input:  []string{"parent_val", "const_val"},
					Output: []string{"result"},
				},
				{
					OpType: "Constant",
					Output: []string{"const_val"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{1.0, 2.0},
							},
						},
					},
				},
			},
		}

		// Else branch: Subtract from parent value
		elseGraph := &protos.GraphProto{
			Output: []*protos.ValueInfoProto{{Name: "result"}},
			Node: []*protos.NodeProto{
				{
					OpType: "Sub",
					Input:  []string{"parent_val", "const_val"},
					Output: []string{"result"},
				},
				{
					OpType: "Constant",
					Output: []string{"const_val"},
					Attribute: []*protos.AttributeProto{
						{
							Name: "value",
							Type: protos.AttributeProto_TENSOR,
							T: &protos.TensorProto{
								Dims:      []int64{2},
								DataType:  int32(protos.TensorProto_FLOAT),
								FloatData: []float32{10.0, 20.0},
							},
						},
					},
				},
			},
		}

		node := &protos.NodeProto{
			OpType: "If",
			Input:  []string{"cond"},
			Output: []string{"output"},
			Attribute: []*protos.AttributeProto{
				{Name: "then_branch", Type: protos.AttributeProto_GRAPH, G: thenGraph},
				{Name: "else_branch", Type: protos.AttributeProto_GRAPH, G: elseGraph},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		convertedOutputs["parent_val"] = parentValue
		inputs = []*Node{cond}
		result := convertIf(model, convertedOutputs, node, inputs)
		outputs = []*Node{result}
		return
	}, []any{
		// Then branch should execute: 100+1=101, 200+2=202
		[]float32{101.0, 202.0},
	}, -1)
}
