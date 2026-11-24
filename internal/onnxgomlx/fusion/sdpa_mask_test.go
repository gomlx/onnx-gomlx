// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusion

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func makeTypedValueInfo(name string, dt protos.TensorProto_DataType, dims []int64, dimParams ...string) *protos.ValueInfoProto {
	shapeDims := make([]*protos.TensorShapeProto_Dimension, len(dims))
	for i, d := range dims {
		if d < 0 {
			param := "dyn"
			if i < len(dimParams) && dimParams[i] != "" {
				param = dimParams[i]
			}
			shapeDims[i] = &protos.TensorShapeProto_Dimension{
				Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: param},
			}
		} else {
			shapeDims[i] = &protos.TensorShapeProto_Dimension{
				Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: d},
			}
		}
	}
	return &protos.ValueInfoProto{
		Name: name,
		Type: &protos.TypeProto{
			Value: &protos.TypeProto_TensorType{
				TensorType: &protos.TypeProto_Tensor{
					ElemType: int32(dt),
					Shape:    &protos.TensorShapeProto{Dim: shapeDims},
				},
			},
		},
	}
}

func makeSDPABaseNodes(maskOutputName string, scale float32) (inits []*protos.TensorProto, nodes []*protos.NodeProto) {
	inits = []*protos.TensorProto{
		makeScalarFloatTensorProto("sdpa_scale", scale),
	}
	nodes = []*protos.NodeProto{
		{
			Name: "kt_tr", OpType: "Transpose", Input: []string{"K"}, Output: []string{"K_T"},
			Attribute: []*protos.AttributeProto{
				{Name: "perm", Type: protos.AttributeProto_INTS, Ints: []int64{0, 1, 3, 2}},
			},
		},
		{Name: "mm1", OpType: "MatMul", Input: []string{"Q", "K_T"}, Output: []string{"qk"}},
		{Name: "div", OpType: "Div", Input: []string{"qk", "sdpa_scale"}, Output: []string{"qk_scaled"}},
		{Name: "add_mask", OpType: "Add", Input: []string{"qk_scaled", maskOutputName}, Output: []string{"qk_masked"}},
		{
			Name: "softmax", OpType: "Softmax", Input: []string{"qk_masked"}, Output: []string{"attn_weights"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1},
			},
		},
		{Name: "mm2", OpType: "MatMul", Input: []string{"attn_weights", "V"}, Output: []string{"output"}},
	}
	return inits, nodes
}

func makeSampleQKV(batch, heads, seqLen, headDim int) (q, k, v *tensors.Tensor) {
	n := batch * heads * seqLen * headDim
	qData := make([]float32, n)
	kData := make([]float32, n)
	vData := make([]float32, n)
	for i := range n {
		qData[i] = float32(i%7)*0.15 - 0.4
		kData[i] = float32((i+3)%11)*0.12 - 0.5
		vData[i] = float32((i+5)%13)*0.1 - 0.3
	}
	return tensors.FromFlatDataAndDimensions(qData, batch, heads, seqLen, headDim),
		tensors.FromFlatDataAndDimensions(kData, batch, heads, seqLen, headDim),
		tensors.FromFlatDataAndDimensions(vData, batch, heads, seqLen, headDim)
}

func TestSDPABoolMask_ClassicBERTMulSub(t *testing.T) {
	// Classic HuggingFace BERT / RoBERTa pattern:
	//   attention_mask [B, S] (Int64) -> Unsqueeze(1) -> Unsqueeze(2) -> Cast(f32)
	//   -> Sub(1.0, ·) -> Mul(·, sentinel)
	for _, tc := range []struct {
		name     string
		sentinel float32
	}{
		{"Neg10000", -10000.0},
		{"NegFloat32Min", -3.4028235e+38},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const batch, heads, seqLen, headDim = 2, 2, 4, 8
			baseInits, baseNodes := makeSDPABaseNodes("mask_4d", float32(math.Sqrt(headDim)))
			graph := &protos.GraphProto{
				Name: "bert_mul_sub_" + tc.name,
				Input: []*protos.ValueInfoProto{
					makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
					makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
					makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
					makeTypedValueInfo("attention_mask", protos.TensorProto_INT64, []int64{batch, seqLen}),
				},
				Output: []*protos.ValueInfoProto{
					makeValueInfo("output", []int64{batch, heads, seqLen, headDim}),
				},
				Initializer: append([]*protos.TensorProto{
					{Name: "axis1", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{1}},
					{Name: "axis2", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2}},
					makeScalarFloatTensorProto("one_f32", 1.0),
					makeScalarFloatTensorProto("sentinel", tc.sentinel),
				}, baseInits...),
				Node: append([]*protos.NodeProto{
					{Name: "u1", OpType: "Unsqueeze", Input: []string{"attention_mask", "axis1"}, Output: []string{"m3d"}},
					{Name: "u2", OpType: "Unsqueeze", Input: []string{"m3d", "axis2"}, Output: []string{"m4d_i64"}},
					{
						Name: "cast_f32", OpType: "Cast", Input: []string{"m4d_i64"}, Output: []string{"m4d_f32"},
						Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
					},
					{Name: "sub", OpType: "Sub", Input: []string{"one_f32", "m4d_f32"}, Output: []string{"m4d_inv"}},
					{Name: "mul", OpType: "Mul", Input: []string{"m4d_inv", "sentinel"}, Output: []string{"mask_4d"}},
				}, baseNodes...),
			}

			m := buildTestModel(t, graph)
			sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
			require.NotNil(t, sdpa.params.BoolMask)
			assert.Equal(t, sdpaBoolMaskLeafEqualOne, sdpa.params.BoolMask.Op)
			assert.Equal(t, "m4d_i64", sdpa.params.BoolMask.SourceName)
			assert.Contains(t, sdpa.ExternalInputs(), "m4d_i64")
			assert.NotContains(t, sdpa.ExternalInputs(), "mask_4d")

			q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
			maskData := []int64{1, 1, 1, 0, 1, 1, 0, 0}
			runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
				"Q": q, "K": k, "V": v,
				"attention_mask": tensors.FromFlatDataAndDimensions(maskData, batch, seqLen),
			})
		})
	}
}

func TestSDPABoolMask_PositiveSentinelSubOne(t *testing.T) {
	// (mask - 1.0) * +10000.0 variant.
	const batch, heads, seqLen, headDim = 2, 2, 4, 8
	baseInits, baseNodes := makeSDPABaseNodes("mask_4d", float32(math.Sqrt(headDim)))
	graph := &protos.GraphProto{
		Name: "mul_sub_pos_sentinel",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
			makeTypedValueInfo("attention_mask", protos.TensorProto_INT64, []int64{batch, seqLen}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("output", []int64{batch, heads, seqLen, headDim}),
		},
		Initializer: append([]*protos.TensorProto{
			{Name: "axis1", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{1}},
			{Name: "axis2", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2}},
			makeScalarFloatTensorProto("one_f32", 1.0),
			makeScalarFloatTensorProto("pos_sentinel", 10000.0),
		}, baseInits...),
		Node: append([]*protos.NodeProto{
			{Name: "u1", OpType: "Unsqueeze", Input: []string{"attention_mask", "axis1"}, Output: []string{"m3d"}},
			{Name: "u2", OpType: "Unsqueeze", Input: []string{"m3d", "axis2"}, Output: []string{"m4d_i64"}},
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"m4d_i64"}, Output: []string{"m4d_f32"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"m4d_f32", "one_f32"}, Output: []string{"m4d_neg"}},
			{Name: "mul", OpType: "Mul", Input: []string{"m4d_neg", "pos_sentinel"}, Output: []string{"mask_4d"}},
		}, baseNodes...),
	}

	m := buildTestModel(t, graph)
	sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
	require.NotNil(t, sdpa.params.BoolMask)
	assert.Equal(t, sdpaBoolMaskLeafEqualOne, sdpa.params.BoolMask.Op)
	assert.Equal(t, "m4d_i64", sdpa.params.BoolMask.SourceName)

	q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
	runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
		"Q": q, "K": k, "V": v,
		"attention_mask": tensors.FromFlatDataAndDimensions([]int64{1, 1, 0, 0, 1, 1, 1, 0}, batch, seqLen),
	})
}

func TestSDPABoolMask_WhereDoubleBoolCast(t *testing.T) {
	// Modern HuggingFace ONNX export (e.g. all-MiniLM-L6-v2, roberta-base-go_emotions):
	//   Where(Cast(bool, Cast(bool, Sub(1.0, Cast(f32, m4d)))), -3.4028235e+38, Sub(1.0, Cast(f32, m4d)))
	const batch, heads, seqLen, headDim = 2, 2, 4, 8
	baseInits, baseNodes := makeSDPABaseNodes("mask_4d", float32(math.Sqrt(headDim)))
	graph := &protos.GraphProto{
		Name: "where_double_bool_cast",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
			makeTypedValueInfo("attention_mask", protos.TensorProto_INT64, []int64{batch, 1, 1, seqLen}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("output", []int64{batch, heads, seqLen, headDim}),
		},
		Initializer: append([]*protos.TensorProto{
			makeScalarFloatTensorProto("one_f32", 1.0),
			makeScalarFloatTensorProto("f32_min", -3.4028235e+38),
		}, baseInits...),
		Node: append([]*protos.NodeProto{
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"attention_mask"}, Output: []string{"m_f32"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"one_f32", "m_f32"}, Output: []string{"m_inv"}},
			{
				Name: "cast_b1", OpType: "Cast", Input: []string{"m_inv"}, Output: []string{"m_b1"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_BOOL)}},
			},
			{
				Name: "cast_b2", OpType: "Cast", Input: []string{"m_b1"}, Output: []string{"m_b2"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_BOOL)}},
			},
			{Name: "where", OpType: "Where", Input: []string{"m_b2", "f32_min", "m_inv"}, Output: []string{"mask_4d"}},
		}, baseNodes...),
	}

	m := buildTestModel(t, graph)
	sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
	require.NotNil(t, sdpa.params.BoolMask)
	assert.Equal(t, sdpaBoolMaskLeafEqualOne, sdpa.params.BoolMask.Op)
	assert.Equal(t, "attention_mask", sdpa.params.BoolMask.SourceName)

	q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
	runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
		"Q": q, "K": k, "V": v,
		"attention_mask": tensors.FromFlatDataAndDimensions([]int64{1, 1, 1, 0, 1, 0, 0, 0}, batch, 1, 1, seqLen),
	})
}

func TestSDPABoolMask_WhereDirectAndExpandedZero(t *testing.T) {
	// Tests:
	//   1) Where(bool_mask, 0.0, -Inf) -> direct Bool mask (grounding-dino pattern)
	//   2) Where(bool_mask, Expand(0.0, shape), -3.4028235e+38) -> direct Bool mask (torch.export pattern)
	//   3) Where(Not(bool_mask), -1e9, 0.0) -> direct Bool mask (cancels double negation)
	const batch, heads, seqLen, headDim = 2, 2, 4, 8

	for _, tc := range []struct {
		name      string
		maskNodes []*protos.NodeProto
		inits     []*protos.TensorProto
	}{
		{
			name: "ZeroAndNegInf",
			inits: []*protos.TensorProto{
				makeScalarFloatTensorProto("zero_f32", 0.0),
				makeScalarFloatTensorProto("neg_inf", float32(math.Inf(-1))),
			},
			maskNodes: []*protos.NodeProto{
				{Name: "where", OpType: "Where", Input: []string{"bool_mask", "zero_f32", "neg_inf"}, Output: []string{"mask_4d"}},
			},
		},
		{
			name: "ExpandedZeroAndF32Min",
			inits: []*protos.TensorProto{
				makeScalarFloatTensorProto("zero_f32", 0.0),
				makeScalarFloatTensorProto("f32_min", -3.4028235e+38),
				{Name: "zshape", Dims: []int64{4}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{batch, 1, 1, seqLen}},
			},
			maskNodes: []*protos.NodeProto{
				{Name: "exp_zero", OpType: "Expand", Input: []string{"zero_f32", "zshape"}, Output: []string{"zeros_4d"}},
				{Name: "where", OpType: "Where", Input: []string{"bool_mask", "zeros_4d", "f32_min"}, Output: []string{"mask_4d"}},
			},
		},
		{
			name: "NotCondAndSentinel",
			inits: []*protos.TensorProto{
				makeScalarFloatTensorProto("zero_f32", 0.0),
				makeScalarFloatTensorProto("neg_1e9", -1e9),
			},
			maskNodes: []*protos.NodeProto{
				{Name: "not_m", OpType: "Not", Input: []string{"bool_mask"}, Output: []string{"inv_mask"}},
				{Name: "where", OpType: "Where", Input: []string{"inv_mask", "neg_1e9", "zero_f32"}, Output: []string{"mask_4d"}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			baseInits, baseNodes := makeSDPABaseNodes("mask_4d", float32(math.Sqrt(headDim)))
			graph := &protos.GraphProto{
				Name: "where_" + tc.name,
				Input: []*protos.ValueInfoProto{
					makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
					makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
					makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
					makeTypedValueInfo("bool_mask", protos.TensorProto_BOOL, []int64{batch, 1, 1, seqLen}),
				},
				Output: []*protos.ValueInfoProto{
					makeValueInfo("output", []int64{batch, heads, seqLen, headDim}),
				},
				Initializer: append(tc.inits, baseInits...),
				Node:        append(tc.maskNodes, baseNodes...),
			}

			m := buildTestModel(t, graph)
			sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
			require.NotNil(t, sdpa.params.BoolMask)
			assert.Equal(t, sdpaBoolMaskLeafDirect, sdpa.params.BoolMask.Op)
			assert.Equal(t, "bool_mask", sdpa.params.BoolMask.SourceName)

			q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
			maskData := []bool{true, true, true, false, true, true, false, false}
			runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
				"Q": q, "K": k, "V": v,
				"bool_mask": tensors.FromFlatDataAndDimensions(maskData, batch, 1, 1, seqLen),
			})
		})
	}
}

func TestSDPABoolMask_CombinedAddMasks(t *testing.T) {
	// Add(causal_sentinel_mask [1,1,S,S], padding_sentinel_mask [B,1,1,S]) -> LogicalAnd(causal_bool, pad_bool).
	const batch, heads, seqLen, headDim = 2, 2, 4, 8
	baseInits, baseNodes := makeSDPABaseNodes("combined_mask", float32(math.Sqrt(headDim)))
	graph := &protos.GraphProto{
		Name: "add_combined_masks",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
			makeTypedValueInfo("causal_bool", protos.TensorProto_BOOL, []int64{1, 1, seqLen, seqLen}),
			makeTypedValueInfo("pad_mask", protos.TensorProto_INT64, []int64{batch, 1, 1, seqLen}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("output", []int64{batch, heads, seqLen, headDim}),
		},
		Initializer: append([]*protos.TensorProto{
			makeScalarFloatTensorProto("zero_f32", 0.0),
			makeScalarFloatTensorProto("one_f32", 1.0),
			makeScalarFloatTensorProto("neg_1e9", -1e9),
		}, baseInits...),
		Node: append([]*protos.NodeProto{
			{Name: "w_causal", OpType: "Where", Input: []string{"causal_bool", "zero_f32", "neg_1e9"}, Output: []string{"causal_add"}},
			{
				Name: "cast_pad", OpType: "Cast", Input: []string{"pad_mask"}, Output: []string{"pad_f32"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
			},
			{Name: "sub_pad", OpType: "Sub", Input: []string{"one_f32", "pad_f32"}, Output: []string{"pad_inv"}},
			{Name: "mul_pad", OpType: "Mul", Input: []string{"pad_inv", "neg_1e9"}, Output: []string{"pad_add"}},
			{Name: "add_masks", OpType: "Add", Input: []string{"causal_add", "pad_add"}, Output: []string{"combined_mask"}},
		}, baseNodes...),
	}

	m := buildTestModel(t, graph)
	sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
	require.NotNil(t, sdpa.params.BoolMask)
	assert.Equal(t, sdpaBoolMaskAnd, sdpa.params.BoolMask.Op)
	assert.Equal(t, sdpaBoolMaskLeafDirect, sdpa.params.BoolMask.Left.Op)
	assert.Equal(t, "causal_bool", sdpa.params.BoolMask.Left.SourceName)
	assert.Equal(t, sdpaBoolMaskLeafEqualOne, sdpa.params.BoolMask.Right.Op)
	assert.Equal(t, "pad_mask", sdpa.params.BoolMask.Right.SourceName)

	q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
	causalData := make([]bool, seqLen*seqLen)
	for i := range seqLen {
		for j := range seqLen {
			causalData[i*seqLen+j] = j <= i
		}
	}
	runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
		"Q": q, "K": k, "V": v,
		"causal_bool": tensors.FromFlatDataAndDimensions(causalData, 1, 1, seqLen, seqLen),
		"pad_mask":    tensors.FromFlatDataAndDimensions([]int64{1, 1, 1, 0, 1, 1, 0, 0}, batch, 1, 1, seqLen),
	})
}

func TestSDPABoolMask_HalfPrecisionSentinelAndCastWrapper(t *testing.T) {
	// Verify tensorProtoToScalarOk decodes Float16 (-65504) and BFloat16 (-1e9) raw/int32 constants,
	// and Cast(to=FLOAT16) wrapping a float32 sentinel mask is unwrapped cleanly.
	f16Min := float16.FromFloat32(-65504.0)
	f16Raw := make([]byte, 2)
	binary.LittleEndian.PutUint16(f16Raw, uint16(f16Min))

	bf16Min := bfloat16.FromFloat32(-1e9)
	bf16Raw := make([]byte, 2)
	binary.LittleEndian.PutUint16(bf16Raw, uint16(bf16Min))

	v, ok := tensorProtoToScalarOk(&protos.TensorProto{DataType: int32(protos.TensorProto_FLOAT16), RawData: f16Raw})
	require.True(t, ok)
	assert.Equal(t, -65504.0, v)

	v, ok = tensorProtoToScalarOk(&protos.TensorProto{DataType: int32(protos.TensorProto_FLOAT16), Int32Data: []int32{int32(uint16(f16Min))}})
	require.True(t, ok)
	assert.Equal(t, -65504.0, v)

	v, ok = tensorProtoToScalarOk(&protos.TensorProto{DataType: int32(protos.TensorProto_BFLOAT16), RawData: bf16Raw})
	require.True(t, ok)
	assert.LessOrEqual(t, v, sdpaMaskSentinelThreshold)

	// Cast(to=FLOAT16) wrapping Where(bool_mask, 0.0, -3.4e38).
	baseInits, baseNodes := makeSDPABaseNodes("mask_f16", 1.0)
	graph := &protos.GraphProto{
		Name: "cast_wrapped_sentinel_mask",
		Input: []*protos.ValueInfoProto{
			makeTypedValueInfo("Q", protos.TensorProto_FLOAT16, []int64{1, 1, 2, 4}),
			makeTypedValueInfo("K", protos.TensorProto_FLOAT16, []int64{1, 1, 2, 4}),
			makeTypedValueInfo("V", protos.TensorProto_FLOAT16, []int64{1, 1, 2, 4}),
			makeTypedValueInfo("bool_mask", protos.TensorProto_BOOL, []int64{1, 1, 1, 2}),
		},
		Output: []*protos.ValueInfoProto{
			makeTypedValueInfo("output", protos.TensorProto_FLOAT16, []int64{1, 1, 2, 4}),
		},
		Initializer: append([]*protos.TensorProto{
			makeScalarFloatTensorProto("zero_f32", 0.0),
			makeScalarFloatTensorProto("f32_min", -3.4028235e+38),
		}, baseInits...),
		Node: append([]*protos.NodeProto{
			{Name: "where", OpType: "Where", Input: []string{"bool_mask", "zero_f32", "f32_min"}, Output: []string{"mask_f32"}},
			{
				Name: "cast_f16", OpType: "Cast", Input: []string{"mask_f32"}, Output: []string{"mask_f16"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT16)}},
			},
		}, baseNodes...),
	}
	m := buildTestModel(t, graph)
	sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
	require.NotNil(t, sdpa.params.BoolMask)
	assert.Equal(t, sdpaBoolMaskLeafDirect, sdpa.params.BoolMask.Op)
	assert.Equal(t, "bool_mask", sdpa.params.BoolMask.SourceName)
}

func TestSDPABoolMask_NegativeCasesNotConverted(t *testing.T) {
	const batch, heads, seqLen, headDim = 1, 1, 2, 1

	t.Run("RawFiniteAdditiveMaskPreservesExactSemantics", func(t *testing.T) {
		// Mirrors compute PR #84: Q=[1,1], K=[0,10000], V=[2,9], raw float mask=[0,-10000].
		// Must NOT be converted to a Bool mask; softmax([0, 0]) @ [2, 9] = 5.5.
		baseInits, baseNodes := makeSDPABaseNodes("raw_mask", 1.0)
		graph := &protos.GraphProto{
			Name: "raw_finite_additive_mask",
			Input: []*protos.ValueInfoProto{
				makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
				makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
				makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
				makeValueInfo("raw_mask", []int64{1, 1, 1, seqLen}),
			},
			Output:      []*protos.ValueInfoProto{makeValueInfo("output", []int64{batch, heads, seqLen, headDim})},
			Initializer: baseInits,
			Node:        baseNodes,
		}

		m := buildTestModel(t, graph)
		sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
		assert.Nil(t, sdpa.params.BoolMask, "raw float additive mask input must not be converted to Bool")

		backend, err := gobackend.New("")
		require.NoError(t, err)
		ctx := model.NewStore()
		require.NoError(t, m.VariablesToScope(ctx.RootScope()))
		out := model.MustCallOnce(backend, ctx, func(scope *model.Scope, g *Graph) *Node {
			return m.CallGraph(scope, g, map[string]*Node{
				"Q":        Const(g, [][][][]float32{{{{1}, {1}}}}),
				"K":        Const(g, [][][][]float32{{{{0}, {10000}}}}),
				"V":        Const(g, [][][][]float32{{{{2}, {9}}}}),
				"raw_mask": Const(g, [][][][]float32{{{{0, -10000}}}}),
			})[0]
		})
		got := out.Value().([][][][]float32)
		assert.InDelta(t, 5.5, got[0][0][0][0], 1e-4)
		assert.InDelta(t, 5.5, got[0][0][1][0], 1e-4)
	})

	t.Run("SmallNegativeBiasNotConverted", func(t *testing.T) {
		// Where(bool_mask, -2.5, 0.0) is a soft bias, not a sentinel mask.
		baseInits, baseNodes := makeSDPABaseNodes("bias_4d", 1.0)
		graph := &protos.GraphProto{
			Name: "small_negative_bias",
			Input: []*protos.ValueInfoProto{
				makeValueInfo("Q", []int64{1, 1, 2, 4}),
				makeValueInfo("K", []int64{1, 1, 2, 4}),
				makeValueInfo("V", []int64{1, 1, 2, 4}),
				makeTypedValueInfo("bool_mask", protos.TensorProto_BOOL, []int64{1, 1, 1, 2}),
			},
			Output: []*protos.ValueInfoProto{makeValueInfo("output", []int64{1, 1, 2, 4})},
			Initializer: append([]*protos.TensorProto{
				makeScalarFloatTensorProto("small_bias", -2.5),
				makeScalarFloatTensorProto("zero_f32", 0.0),
			}, baseInits...),
			Node: append([]*protos.NodeProto{
				{Name: "where", OpType: "Where", Input: []string{"bool_mask", "small_bias", "zero_f32"}, Output: []string{"bias_4d"}},
			}, baseNodes...),
		}

		m := buildTestModel(t, graph)
		sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
		assert.Nil(t, sdpa.params.BoolMask, "small negative bias (-2.5) must not be converted to Bool")
	})

	t.Run("AddWithNonSentinelBiasNotConverted", func(t *testing.T) {
		// Add(sentinel_mask, rel_pos_bias) must stay float so rel_pos_bias is preserved.
		baseInits, baseNodes := makeSDPABaseNodes("mask_plus_bias", 1.0)
		graph := &protos.GraphProto{
			Name: "mask_plus_rel_bias",
			Input: []*protos.ValueInfoProto{
				makeValueInfo("Q", []int64{1, 1, 2, 4}),
				makeValueInfo("K", []int64{1, 1, 2, 4}),
				makeValueInfo("V", []int64{1, 1, 2, 4}),
				makeTypedValueInfo("bool_mask", protos.TensorProto_BOOL, []int64{1, 1, 1, 2}),
				makeValueInfo("rel_bias", []int64{1, 1, 2, 2}),
			},
			Output: []*protos.ValueInfoProto{makeValueInfo("output", []int64{1, 1, 2, 4})},
			Initializer: append([]*protos.TensorProto{
				makeScalarFloatTensorProto("zero_f32", 0.0),
				makeScalarFloatTensorProto("neg_1e9", -1e9),
			}, baseInits...),
			Node: append([]*protos.NodeProto{
				{Name: "where", OpType: "Where", Input: []string{"bool_mask", "zero_f32", "neg_1e9"}, Output: []string{"pad_mask"}},
				{Name: "add_bias", OpType: "Add", Input: []string{"pad_mask", "rel_bias"}, Output: []string{"mask_plus_bias"}},
			}, baseNodes...),
		}

		m := buildTestModel(t, graph)
		sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
		assert.Nil(t, sdpa.params.BoolMask, "Add with non-sentinel bias must not be converted to Bool")
	})

	t.Run("WhereWithNonZeroFalseBranchNotConverted", func(t *testing.T) {
		// Where(bool_mask, -1e9, other_bias) must stay float.
		baseInits, baseNodes := makeSDPABaseNodes("mask_4d", 1.0)
		graph := &protos.GraphProto{
			Name: "where_nonzero_false",
			Input: []*protos.ValueInfoProto{
				makeValueInfo("Q", []int64{1, 1, 2, 4}),
				makeValueInfo("K", []int64{1, 1, 2, 4}),
				makeValueInfo("V", []int64{1, 1, 2, 4}),
				makeTypedValueInfo("bool_mask", protos.TensorProto_BOOL, []int64{1, 1, 1, 2}),
				makeValueInfo("other_bias", []int64{1, 1, 2, 2}),
			},
			Output: []*protos.ValueInfoProto{makeValueInfo("output", []int64{1, 1, 2, 4})},
			Initializer: append([]*protos.TensorProto{
				makeScalarFloatTensorProto("neg_1e9", -1e9),
			}, baseInits...),
			Node: append([]*protos.NodeProto{
				{Name: "where", OpType: "Where", Input: []string{"bool_mask", "neg_1e9", "other_bias"}, Output: []string{"mask_4d"}},
			}, baseNodes...),
		}

		m := buildTestModel(t, graph)
		sdpa := m.DetectedFusions["output"].(*sdpaCandidate)
		assert.Nil(t, sdpa.params.BoolMask, "Where with non-zero fallback bias must not be converted to Bool")
	})
}

func TestSDPABoolMask_DynamicShapes(t *testing.T) {
	// Verify dynamic [batch, heads, seq, dim] SDPA with a classic BERT sentinel mask
	// converted to Bool executes cleanly across different batch/seq shapes without recompilation.
	const heads, headDim = 2, 8
	baseInits, baseNodes := makeSDPABaseNodes("mask_4d", float32(math.Sqrt(headDim)))
	graph := &protos.GraphProto{
		Name: "dynamic_bool_mask",
		Input: []*protos.ValueInfoProto{
			makeTypedValueInfo("Q", protos.TensorProto_FLOAT, []int64{-1, heads, -1, headDim}, "batch", "", "seq", ""),
			makeTypedValueInfo("K", protos.TensorProto_FLOAT, []int64{-1, heads, -1, headDim}, "batch", "", "seq", ""),
			makeTypedValueInfo("V", protos.TensorProto_FLOAT, []int64{-1, heads, -1, headDim}, "batch", "", "seq", ""),
			makeTypedValueInfo("attention_mask", protos.TensorProto_INT64, []int64{-1, -1}, "batch", "seq"),
		},
		Output: []*protos.ValueInfoProto{
			makeTypedValueInfo("output", protos.TensorProto_FLOAT, []int64{-1, heads, -1, headDim}, "batch", "", "seq", ""),
		},
		Initializer: append([]*protos.TensorProto{
			{Name: "axis1", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{1}},
			{Name: "axis2", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2}},
			makeScalarFloatTensorProto("one_f32", 1.0),
			makeScalarFloatTensorProto("sentinel", -10000.0),
		}, baseInits...),
		Node: append([]*protos.NodeProto{
			{Name: "u1", OpType: "Unsqueeze", Input: []string{"attention_mask", "axis1"}, Output: []string{"m3d"}},
			{Name: "u2", OpType: "Unsqueeze", Input: []string{"m3d", "axis2"}, Output: []string{"m4d_i64"}},
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"m4d_i64"}, Output: []string{"m4d_f32"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"one_f32", "m4d_f32"}, Output: []string{"m4d_inv"}},
			{Name: "mul", OpType: "Mul", Input: []string{"m4d_inv", "sentinel"}, Output: []string{"mask_4d"}},
		}, baseNodes...),
	}

	raw, err := proto.Marshal(&protos.ModelProto{Graph: graph})
	require.NoError(t, err)
	mFused, err := onnxgomlx.Parse(raw)
	require.NoError(t, err)
	mUnfused, err := onnxgomlx.Parse(raw)
	require.NoError(t, err)
	mUnfused.DisableFusion()

	sdpa := mFused.DetectedFusions["output"].(*sdpaCandidate)
	require.NotNil(t, sdpa.params.BoolMask)

	backend, err := gobackend.New("")
	require.NoError(t, err)
	ctxFused := model.NewStore()
	require.NoError(t, mFused.VariablesToScope(ctxFused.RootScope()))
	execFused := model.MustNewExec(backend, ctxFused, func(scope *model.Scope, q, k, v, mask *Node) *Node {
		return mFused.CallGraph(scope, q.Graph(), map[string]*Node{
			"Q": q, "K": k, "V": v, "attention_mask": mask,
		})[0]
	})
	execFused.WithDynamicAxes(
		[]string{"batch", "", "seq", ""},
		[]string{"batch", "", "seq", ""},
		[]string{"batch", "", "seq", ""},
		[]string{"batch", "seq"},
	)
	defer execFused.Finalize()

	ctxUnfused := model.NewStore()
	require.NoError(t, mUnfused.VariablesToScope(ctxUnfused.RootScope()))

	for _, tc := range []struct {
		batch, seqLen int
		mask          []int64
	}{
		{batch: 2, seqLen: 4, mask: []int64{1, 1, 1, 0, 1, 1, 0, 0}},
		{batch: 1, seqLen: 6, mask: []int64{1, 1, 1, 1, 0, 0}},
		{batch: 3, seqLen: 3, mask: []int64{1, 1, 1, 1, 0, 0, 1, 1, 0}},
	} {
		q, k, v := makeSampleQKV(tc.batch, heads, tc.seqLen, headDim)
		maskT := tensors.FromFlatDataAndDimensions(tc.mask, tc.batch, tc.seqLen)
		gotFused := execFused.MustCall1(q, k, v, maskT)
		gotUnfused := model.MustCallOnce(backend, ctxUnfused, func(scope *model.Scope, g *Graph) *Node {
			return mUnfused.CallGraph(scope, g, map[string]*Node{
				"Q": Const(g, q), "K": Const(g, k), "V": Const(g, v), "attention_mask": Const(g, maskT),
			})[0]
		})
		fData, err := tensors.CopyFlatData[float32](gotFused)
		require.NoError(t, err)
		uData, err := tensors.CopyFlatData[float32](gotUnfused)
		require.NoError(t, err)
		require.Equal(t, len(uData), len(fData))
		for i := range uData {
			assert.InDelta(t, uData[i], fData[i], 1e-4, "mismatch at batch=%d seq=%d idx=%d", tc.batch, tc.seqLen, i)
		}
	}
}

func TestSDPABoolMask_SharedAcrossLayers(t *testing.T) {
	// Verify two SDPA layers sharing the same sentinel mask share a single converted Bool mask node.
	const batch, heads, seqLen, headDim = 1, 2, 4, 8
	scale := float32(math.Sqrt(headDim))
	graph := &protos.GraphProto{
		Name: "two_layer_shared_mask",
		Input: []*protos.ValueInfoProto{
			makeValueInfo("Q", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("K", []int64{batch, heads, seqLen, headDim}),
			makeValueInfo("V", []int64{batch, heads, seqLen, headDim}),
			makeTypedValueInfo("attention_mask", protos.TensorProto_INT64, []int64{batch, 1, 1, seqLen}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfo("out2", []int64{batch, heads, seqLen, headDim}),
		},
		Initializer: []*protos.TensorProto{
			makeScalarFloatTensorProto("sdpa_scale", scale),
			makeScalarFloatTensorProto("one_f32", 1.0),
			makeScalarFloatTensorProto("sentinel", -1e9),
		},
		Node: []*protos.NodeProto{
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"attention_mask"}, Output: []string{"m_f32"},
				Attribute: []*protos.AttributeProto{{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)}},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"one_f32", "m_f32"}, Output: []string{"m_inv"}},
			{Name: "mul", OpType: "Mul", Input: []string{"m_inv", "sentinel"}, Output: []string{"mask_4d"}},
			// Layer 1
			{Name: "kt1", OpType: "Transpose", Input: []string{"K"}, Output: []string{"K_T1"}, Attribute: []*protos.AttributeProto{{Name: "perm", Type: protos.AttributeProto_INTS, Ints: []int64{0, 1, 3, 2}}}},
			{Name: "mm1_1", OpType: "MatMul", Input: []string{"Q", "K_T1"}, Output: []string{"qk1"}},
			{Name: "div1", OpType: "Div", Input: []string{"qk1", "sdpa_scale"}, Output: []string{"qks1"}},
			{Name: "add1", OpType: "Add", Input: []string{"qks1", "mask_4d"}, Output: []string{"qkm1"}},
			{Name: "sm1", OpType: "Softmax", Input: []string{"qkm1"}, Output: []string{"w1"}, Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: -1}}},
			{Name: "mm2_1", OpType: "MatMul", Input: []string{"w1", "V"}, Output: []string{"out1"}},
			// Layer 2 (uses out1 as Q, shares mask_4d)
			{Name: "kt2", OpType: "Transpose", Input: []string{"K"}, Output: []string{"K_T2"}, Attribute: []*protos.AttributeProto{{Name: "perm", Type: protos.AttributeProto_INTS, Ints: []int64{0, 1, 3, 2}}}},
			{Name: "mm1_2", OpType: "MatMul", Input: []string{"out1", "K_T2"}, Output: []string{"qk2"}},
			{Name: "div2", OpType: "Div", Input: []string{"qk2", "sdpa_scale"}, Output: []string{"qks2"}},
			{Name: "add2", OpType: "Add", Input: []string{"qks2", "mask_4d"}, Output: []string{"qkm2"}},
			{Name: "sm2", OpType: "Softmax", Input: []string{"qkm2"}, Output: []string{"w2"}, Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: -1}}},
			{Name: "mm2_2", OpType: "MatMul", Input: []string{"w2", "V"}, Output: []string{"out2"}},
		},
	}

	m := buildTestModel(t, graph)
	require.Len(t, m.DetectedFusions, 2)
	s1 := m.DetectedFusions["out1"].(*sdpaCandidate)
	s2 := m.DetectedFusions["out2"].(*sdpaCandidate)
	require.NotNil(t, s1.params.BoolMask)
	require.NotNil(t, s2.params.BoolMask)
	assert.Equal(t, s1.params.BoolMask.cacheKey(), s2.params.BoolMask.cacheKey())

	q, k, v := makeSampleQKV(batch, heads, seqLen, headDim)
	runFusedVsUnfused(t, graph, map[string]*tensors.Tensor{
		"Q": q, "K": k, "V": v,
		"attention_mask": tensors.FromFlatDataAndDimensions([]int64{1, 1, 1, 0}, batch, 1, 1, seqLen),
	})
}

func TestSDPABoolMask_CachedBoolMaskDTypeIsBool(t *testing.T) {
	// Directly verify emit produces a Bool node and caches it in convertedOutputs.
	backend, err := gobackend.New("")
	require.NoError(t, err)
	g := NewGraph(backend, "test_bool_mask_emit")
	src := Parameter(g, "mask", shapes.Make(dtypes.Int64, 2, 1, 1, 4))
	converted := map[string]*Node{"mask": src}

	expr := &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafEqualOne, SourceName: "mask"}
	n1 := expr.emit(g, converted)
	n2 := expr.emit(g, converted)
	assert.Equal(t, dtypes.Bool, n1.DType())
	assert.Equal(t, []int{2, 1, 1, 4}, n1.Shape().Dimensions)
	assert.Same(t, n1, n2, "expected repeated emit to reuse cached Bool node")
}
