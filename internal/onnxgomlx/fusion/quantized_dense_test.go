package fusion

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/gomlx/gomlx/core/graph" //nolint
)

// makeInt8TensorProto creates a TensorProto with the given shape and int8 data.
func makeInt8TensorProto(name string, dims []int64, data []int8) *protos.TensorProto {
	raw := make([]byte, len(data))
	for i, v := range data {
		raw[i] = byte(v)
	}
	return &protos.TensorProto{
		Name:     name,
		Dims:     dims,
		DataType: int32(protos.TensorProto_INT8),
		RawData:  raw,
	}
}

// makeZeroUint8TensorProtoN creates a zero-valued uint8 TensorProto with N elements (used as zero point).
func makeZeroUint8TensorProtoN(name string, n int) *protos.TensorProto {
	return &protos.TensorProto{
		Name:     name,
		Dims:     []int64{int64(n)},
		DataType: int32(protos.TensorProto_UINT8),
		RawData:  make([]byte, n),
	}
}

// makeValueInfoWithType creates a ValueInfoProto with the given dtype.
func makeValueInfoWithType(name string, dims []int64, elemType protos.TensorProto_DataType) *protos.ValueInfoProto {
	shapeDims := make([]*protos.TensorShapeProto_Dimension, len(dims))
	for i, d := range dims {
		shapeDims[i] = &protos.TensorShapeProto_Dimension{
			Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: d},
		}
	}
	return &protos.ValueInfoProto{
		Name: name,
		Type: &protos.TypeProto{
			Value: &protos.TypeProto_TensorType{
				TensorType: &protos.TypeProto_Tensor{
					ElemType: int32(elemType),
					Shape:    &protos.TensorShapeProto{Dim: shapeDims},
				},
			},
		},
	}
}

// makeQuantizedDenseGraph builds the DynamicQuantizeLinear → MatMulInteger → Cast → Mul(a_scale*B_scale) → Mul pattern.
// If perChannelScale is true, bScale is [N] (per output channel) instead of scalar.
func makeQuantizedDenseGraph(K, N int, perChannelScale bool) *protos.GraphProto {
	// B weights: [K, N] int8
	bData := make([]int8, K*N)
	for i := range bData {
		bData[i] = int8(i%5 - 2)
	}

	// B scale: scalar or [N] (per output channel)
	var bScaleInit *protos.TensorProto
	if perChannelScale {
		bScaleData := make([]float32, N)
		for i := range bScaleData {
			bScaleData[i] = 0.01 * float32(i+1)
		}
		bScaleInit = makeFloatTensorProto("b_scale", []int64{int64(N)}, bScaleData)
	} else {
		bScaleInit = makeScalarFloatTensorProto("b_scale", 0.05)
	}

	return &protos.GraphProto{
		Input: []*protos.ValueInfoProto{
			makeValueInfo("float_input", []int64{2, int64(K)}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("output", []int64{2, int64(N)}),
		},
		Initializer: []*protos.TensorProto{
			makeInt8TensorProto("B", []int64{int64(K), int64(N)}, bData),
			bScaleInit,
			makeZeroUint8TensorProtoN("b_zp", N),
		},
		ValueInfo: []*protos.ValueInfoProto{
			makeValueInfoWithType("quantized_uint8", []int64{2, int64(K)}, protos.TensorProto_UINT8),
			makeValueInfoWithType("a_scale", []int64{}, protos.TensorProto_FLOAT),
			makeValueInfoWithType("a_zp", []int64{}, protos.TensorProto_UINT8),
			makeValueInfoWithType("matmul_out", []int64{2, int64(N)}, protos.TensorProto_INT32),
			makeValueInfo("cast_out", []int64{2, int64(N)}),
			makeValueInfo("combined_scale", []int64{}),
			makeValueInfo("output", []int64{2, int64(N)}),
		},
		Node: []*protos.NodeProto{
			{
				OpType: "DynamicQuantizeLinear",
				Input:  []string{"float_input"},
				Output: []string{"quantized_uint8", "a_scale", "a_zp"},
			},
			{
				OpType: "MatMulInteger",
				Input:  []string{"quantized_uint8", "B", "a_zp", "b_zp"},
				Output: []string{"matmul_out"},
			},
			{
				OpType: "Cast",
				Input:  []string{"matmul_out"},
				Output: []string{"cast_out"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)},
				},
			},
			{
				OpType: "Mul",
				Input:  []string{"a_scale", "b_scale"},
				Output: []string{"combined_scale"},
			},
			{
				OpType: "Mul",
				Input:  []string{"cast_out", "combined_scale"},
				Output: []string{"output"},
			},
		},
	}
}

// TestQuantizedDensePerChannelScale tests that QuantizedDense fusion works with per-channel
// (1D [K]) scales, not just scalar scales. This reproduces the bug from
// knights-analytics/hugot#72 where ExpandAndBroadcast panicked on 1D scales.
func TestQuantizedDensePerChannelScale(t *testing.T) {
	K, N := 8, 4
	graphProto := makeQuantizedDenseGraph(K, N, true) // per-channel
	graphProto.Name = "quantized_dense_per_channel"

	// Build model — should detect QuantizedDense fusion without panicking.
	m := buildTestModel(t, graphProto)
	require.Contains(t, m.DetectedFusions, "output")
	assert.Equal(t, "QuantizedDense", m.DetectedFusions["output"].Name())

	// VariablesToScope should succeed.
	backend, err := gobackend.New("")
	require.NoError(t, err)
	store := model.NewStore()
	require.NoError(t, m.VariablesToScope(store.RootScope()))

	// Build and execute graph — should not panic during ExpandAndBroadcast.
	xData := make([]float32, 2*K)
	for i := range xData {
		xData[i] = float32(i%7)*0.1 - 0.3
	}

	results := model.MustCallOnceN(backend, store, func(scope *model.Scope, g *Graph) []*Node {
		xNode := Const(g, tensors.FromFlatDataAndDimensions(xData, 2, K))
		return m.CallGraph(scope, g, map[string]*Node{"float_input": xNode})
	})

	require.Len(t, results, 1)
	result := results[0]
	assert.Equal(t, dtypes.Float32, result.DType())
	assert.Equal(t, []int{2, N}, result.Shape().Dimensions)
}

// TestQuantizedDenseScalarScale tests that QuantizedDense fusion still works with scalar scales.
func TestQuantizedDenseScalarScale(t *testing.T) {
	K, N := 8, 4
	graphProto := makeQuantizedDenseGraph(K, N, false) // scalar
	graphProto.Name = "quantized_dense_scalar"

	m := buildTestModel(t, graphProto)

	backend, err := gobackend.New("")
	require.NoError(t, err)
	store := model.NewStore()
	require.NoError(t, m.VariablesToScope(store.RootScope()))

	xData := make([]float32, 2*K)
	for i := range xData {
		xData[i] = float32(i%7)*0.1 - 0.3
	}

	results := model.MustCallOnceN(backend, store, func(scope *model.Scope, g *Graph) []*Node {
		xNode := Const(g, tensors.FromFlatDataAndDimensions(xData, 2, K))
		return m.CallGraph(scope, g, map[string]*Node{"float_input": xNode})
	})

	require.Len(t, results, 1)
	result := results[0]
	assert.Equal(t, dtypes.Float32, result.DType())
	assert.Equal(t, []int{2, N}, result.Shape().Dimensions)
}

// TestQuantizedDenseFusionParity verifies that DynamicQuantizeLinear + MatMulInteger graphs
// produce identical outputs whether fusion is enabled or disabled, preserving the uint8
// quantization rounding of DynamicQuantizeLinear.
func TestQuantizedDenseFusionParity(t *testing.T) {
	graphProto := &protos.GraphProto{
		Name: "mlp_0_perchannel_0",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("x", []int64{1, 3}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("result", []int64{1, 1}),
		},
		Initializer: []*protos.TensorProto{
			makeInt8TensorProto("w0_quantized", []int64{3, 1}, []int8{0, 127, 0}),
			makeScalarFloatTensorProto("w0_scale", 1.0/127.0),
			makeInt8TensorProto("w0_zero_point", nil, []int8{0}),
			makeFloatTensorProto("b0", []int64{1}, []float32{0}),
		},
		Node: []*protos.NodeProto{
			{
				OpType: "DynamicQuantizeLinear",
				Input:  []string{"x"},
				Output: []string{"x_quantized", "x_scale", "x_zero_point"},
			},
			{
				OpType: "Mul",
				Input:  []string{"x_scale", "w0_scale"},
				Output: []string{"x_scale_w0_scale_mul"},
			},
			{
				OpType: "MatMulInteger",
				Input:  []string{"x_quantized", "w0_quantized", "x_zero_point", "w0_zero_point"},
				Output: []string{"mm0_output_quantized"},
			},
			{
				OpType: "Cast",
				Input:  []string{"mm0_output_quantized"},
				Output: []string{"mm0_output_quantized_cast_output"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)},
				},
			},
			{
				OpType: "Mul",
				Input:  []string{"mm0_output_quantized_cast_output", "x_scale_w0_scale_mul"},
				Output: []string{"mm0"},
			},
			{
				OpType: "Add",
				Input:  []string{"mm0", "b0"},
				Output: []string{"result"},
			},
		},
	}

	backend, err := gobackend.New("")
	require.NoError(t, err)

	runModel := func(disableFusion bool) []float32 {
		m := buildTestModel(t, graphProto)
		if disableFusion {
			m.DisableFusion()
		} else {
			require.Contains(t, m.DetectedFusions, "result")
			assert.Equal(t, "QuantizedDense", m.DetectedFusions["result"].Name())
		}
		store := model.NewStore()
		require.NoError(t, m.VariablesToScope(store.RootScope()))

		results := model.MustCallOnceN(backend, store, func(scope *model.Scope, g *Graph) []*Node {
			xNode := Const(g, tensors.FromFlatDataAndDimensions([]float32{-1, 0.1, 1}, 1, 3))
			return m.CallGraph(scope, g, map[string]*Node{"x": xNode})
		})
		require.Len(t, results, 1)
		vals, err := tensors.CopyFlatData[float32](results[0])
		require.NoError(t, err)
		return vals
	}

	fusedOut := runModel(false)
	unfusedOut := runModel(true)
	require.Equal(t, unfusedOut, fusedOut)
	assert.InDelta(t, float32(26.0/255.0), fusedOut[0], 1e-7)
}

// TestQuantizedQKVDenseFusionParity verifies that three MatMulInteger Q/K/V projections sharing
// one DynamicQuantizeLinear(x) fuse into a single QuantizedQKVDense op (including GQA QDim != KVDim,
// mixed scalar/per-channel scales, non-zero weight zero point, and biases) with exact numerical parity.
func TestQuantizedQKVDenseFusionParity(t *testing.T) {
	const K, QDim, KVDim = 6, 4, 2
	makeWeight := func(seed int, rows, cols int) []int8 {
		out := make([]int8, rows*cols)
		for i := range out {
			out[i] = int8(((i*3 + seed) % 11) - 5)
		}
		return out
	}

	graphProto := &protos.GraphProto{
		Name: "quantized_qkv_gqa_parity",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("x", []int64{2, K}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("q_out", []int64{2, QDim}),
			makeValueInfo("k_out", []int64{2, KVDim}),
			makeValueInfo("v_out", []int64{2, KVDim}),
		},
		Initializer: []*protos.TensorProto{
			makeInt8TensorProto("w_q", []int64{K, QDim}, makeWeight(1, K, QDim)),
			makeInt8TensorProto("w_k", []int64{K, KVDim}, makeWeight(2, K, KVDim)),
			makeInt8TensorProto("w_v", []int64{K, KVDim}, makeWeight(4, K, KVDim)),
			makeFloatTensorProto("s_q", []int64{QDim}, []float32{0.02, 0.03, 0.04, 0.05}),
			makeScalarFloatTensorProto("s_k", 0.025),
			makeFloatTensorProto("s_v", []int64{KVDim}, []float32{0.03, 0.015}),
			makeInt8TensorProto("zp_q", []int64{QDim}, []int8{1, -1, 0, 2}),
			makeInt8TensorProto("zp_zero_kv", []int64{KVDim}, []int8{0, 0}),
			makeFloatTensorProto("b_q", []int64{QDim}, []float32{0.1, -0.2, 0.3, -0.4}),
			makeFloatTensorProto("b_k", []int64{KVDim}, []float32{0.05, -0.05}),
			makeFloatTensorProto("b_v", []int64{KVDim}, []float32{-0.1, 0.2}),
		},
		Node: []*protos.NodeProto{
			{
				OpType: "DynamicQuantizeLinear",
				Input:  []string{"x"},
				Output: []string{"x_q", "x_scale", "x_zp"},
			},
			// Q branch
			{OpType: "MatMulInteger", Input: []string{"x_q", "w_q", "x_zp", "zp_q"}, Output: []string{"q_i32"}},
			{OpType: "Cast", Input: []string{"q_i32"}, Output: []string{"q_f32"}, Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}}},
			{OpType: "Mul", Input: []string{"x_scale", "s_q"}, Output: []string{"cs_q"}},
			{OpType: "Mul", Input: []string{"q_f32", "cs_q"}, Output: []string{"q_scaled"}},
			{OpType: "Add", Input: []string{"q_scaled", "b_q"}, Output: []string{"q_out"}},
			// K branch
			{OpType: "MatMulInteger", Input: []string{"x_q", "w_k", "x_zp", "zp_zero_kv"}, Output: []string{"k_i32"}},
			{OpType: "Cast", Input: []string{"k_i32"}, Output: []string{"k_f32"}, Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}}},
			{OpType: "Mul", Input: []string{"x_scale", "s_k"}, Output: []string{"cs_k"}},
			{OpType: "Mul", Input: []string{"k_f32", "cs_k"}, Output: []string{"k_scaled"}},
			{OpType: "Add", Input: []string{"k_scaled", "b_k"}, Output: []string{"k_out"}},
			// V branch
			{OpType: "MatMulInteger", Input: []string{"x_q", "w_v", "x_zp", "zp_zero_kv"}, Output: []string{"v_i32"}},
			{OpType: "Cast", Input: []string{"v_i32"}, Output: []string{"v_f32"}, Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}}},
			{OpType: "Mul", Input: []string{"x_scale", "s_v"}, Output: []string{"cs_v"}},
			{OpType: "Mul", Input: []string{"v_f32", "cs_v"}, Output: []string{"v_scaled"}},
			{OpType: "Add", Input: []string{"v_scaled", "b_v"}, Output: []string{"v_out"}},
		},
	}

	backend, err := gobackend.New("")
	require.NoError(t, err)
	xData := []float32{-0.8, 0.2, 0.5, -0.3, 1.1, -0.4, 0.6, -1.2, 0.0, 0.9, -0.5, 0.3}

	run := func(disableFusion bool) [][]float32 {
		m := buildTestModel(t, graphProto)
		if disableFusion {
			m.DisableFusion()
		} else {
			require.Contains(t, m.DetectedFusions, "q_out")
			assert.Equal(t, "QuantizedQKVDense", m.DetectedFusions["q_out"].Name())
		}
		store := model.NewStore()
		require.NoError(t, m.VariablesToScope(store.RootScope()))

		results := model.MustCallOnceN(backend, store, func(scope *model.Scope, g *Graph) []*Node {
			xNode := Const(g, tensors.FromFlatDataAndDimensions(xData, 2, K))
			return m.CallGraph(scope, g, map[string]*Node{"x": xNode})
		})
		require.Len(t, results, 3)
		outs := make([][]float32, 3)
		for i, r := range results {
			vals, err := tensors.CopyFlatData[float32](r)
			require.NoError(t, err)
			outs[i] = vals
		}
		return outs
	}

	fused := run(false)
	unfused := run(true)
	for i, name := range []string{"Q", "K", "V"} {
		assert.InDeltaSlice(t, unfused[i], fused[i], 1e-6, "%s projection mismatch", name)
	}
}
