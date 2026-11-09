package passes

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func makeValueInfoTyped(name string, elemType protos.TensorProto_DataType, dims []int64) *protos.ValueInfoProto {
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

func TestImplicitExpandPushdownReconvergentMaskDAG(t *testing.T) {
	// Models the standard HuggingFace ONNX export pattern for 2D attention_mask -> 4D additive mask:
	//   attention_mask [B, S]
	//   -> Unsqueeze(1) -> [B, 1, S]
	//   -> Unsqueeze(2) -> [B, 1, 1, S]
	//   -> Expand(target_shape=[B, 1, S, S]) -> [B, 1, S, S]
	//   -> Cast(float32) -> Sub(1.0, ·) -> Cast(bool) -> Where(·, -1e9, sub_out) -> mask_4d
	graphProto := &protos.GraphProto{
		Name: "reconvergent_expand_dag",
		Input: []*protos.ValueInfoProto{
			makeValueInfoTyped("attention_mask", protos.TensorProto_INT64, []int64{2, 4}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfoTyped("mask_4d", protos.TensorProto_FLOAT, []int64{2, 1, 4, 4}),
		},
		Initializer: []*protos.TensorProto{
			{Name: "axis1", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{1}},
			{Name: "axis2", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2}},
			{Name: "target_shape", Dims: []int64{4}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2, 1, 4, 4}},
			{Name: "one", Dims: []int64{}, DataType: int32(protos.TensorProto_FLOAT), FloatData: []float32{1.0}},
			{Name: "neg_inf", Dims: []int64{}, DataType: int32(protos.TensorProto_FLOAT), FloatData: []float32{-1e9}},
		},
		Node: []*protos.NodeProto{
			{Name: "u1", OpType: "Unsqueeze", Input: []string{"attention_mask", "axis1"}, Output: []string{"u1_out"}},
			{Name: "u2", OpType: "Unsqueeze", Input: []string{"u1_out", "axis2"}, Output: []string{"u2_out"}},
			{Name: "exp", OpType: "Expand", Input: []string{"u2_out", "target_shape"}, Output: []string{"exp_out"}},
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"exp_out"}, Output: []string{"cast_f32_out"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)},
				},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"one", "cast_f32_out"}, Output: []string{"sub_out"}},
			{
				Name: "cast_bool", OpType: "Cast", Input: []string{"sub_out"}, Output: []string{"cast_bool_out"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_BOOL)},
				},
			},
			{Name: "where", OpType: "Where", Input: []string{"cast_bool_out", "neg_inf", "sub_out"}, Output: []string{"mask_4d"}},
		},
	}

	content, err := proto.Marshal(&protos.ModelProto{Graph: graphProto})
	require.NoError(t, err)

	mOpt, err := onnxgomlx.Parse(content)
	require.NoError(t, err)

	// Verify all intermediate Expand nodes were pruned and only 1 Expand remains at the end.
	var expandNodes []*protos.NodeProto
	for _, n := range mOpt.Proto.Graph.Node {
		if n.OpType == "Expand" {
			expandNodes = append(expandNodes, n)
		}
	}
	require.Len(t, expandNodes, 1, "expected exactly 1 Expand node at the exit of the DAG")
	assert.Equal(t, "mask_4d#unexpanded", expandNodes[0].Input[0])
	assert.Equal(t, "target_shape", expandNodes[0].Input[1])
	assert.Equal(t, "mask_4d", expandNodes[0].Output[0])

	// Verify inferred shape of the unexpanded Where output is [2, 1, 1, 4].
	unexpShape := mOpt.InferShapeForName("mask_4d#unexpanded")
	assert.Equal(t, []int{2, 1, 1, 4}, unexpShape.Dimensions)

	// Verify numerical equivalence between optimized and unoptimized execution.
	mUnopt, err := onnxgomlx.ParseWithOptions(content, onnxgomlx.WithoutOptimizationPasses())
	require.NoError(t, err)

	backend, err := gobackend.New("")
	require.NoError(t, err)

	maskInput := tensors.FromFlatDataAndDimensions([]int64{
		1, 1, 1, 0,
		1, 1, 0, 0,
	}, 2, 4)

	runModel := func(m *onnxgomlx.Model) []float32 {
		ctx := model.NewStore()
		require.NoError(t, m.VariablesToScope(ctx.RootScope()))
		res := model.MustCallOnceN(backend, ctx, func(scope *model.Scope, g *Graph) []*Node {
			return m.CallGraph(scope, g, map[string]*Node{
				"attention_mask": Const(g, maskInput),
			})
		})
		require.Len(t, res, 1)
		return tensors.MustCopyFlatData[float32](res[0])
	}

	assert.InDeltaSlice(t, runModel(mUnopt), runModel(mOpt), 1e-6)
}

func TestImplicitExpandPushdownSkipsIncompatibleBinaryOperand(t *testing.T) {
	// Adding an expanded tensor [2, 4] (from [2, 1]) to a non-constant tensor of shape [2, 4]
	// must NOT push Expand past Add because the other operand is not [2, 1] or a scalar constant.
	graphProto := &protos.GraphProto{
		Name: "incompatible_binary_operand",
		Input: []*protos.ValueInfoProto{
			makeValueInfoTyped("x", protos.TensorProto_FLOAT, []int64{2, 1}),
			makeValueInfoTyped("y", protos.TensorProto_FLOAT, []int64{2, 4}),
		},
		Output: []*protos.ValueInfoProto{
			makeValueInfoTyped("out", protos.TensorProto_FLOAT, []int64{2, 4}),
		},
		Initializer: []*protos.TensorProto{
			{Name: "target_shape", Dims: []int64{2}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2, 4}},
		},
		Node: []*protos.NodeProto{
			{Name: "exp", OpType: "Expand", Input: []string{"x", "target_shape"}, Output: []string{"x_exp"}},
			{Name: "add", OpType: "Add", Input: []string{"x_exp", "y"}, Output: []string{"out"}},
		},
	}

	content, err := proto.Marshal(&protos.ModelProto{Graph: graphProto})
	require.NoError(t, err)

	m, err := onnxgomlx.Parse(content)
	require.NoError(t, err)

	require.Len(t, m.Proto.Graph.Node, 2)
	assert.Equal(t, "Expand", m.Proto.Graph.Node[0].OpType)
	assert.Equal(t, "Add", m.Proto.Graph.Node[1].OpType)
	assert.Equal(t, []string{"x_exp", "y"}, m.Proto.Graph.Node[1].Input)
}

func TestImplicitExpandPushdownDynamicDims(t *testing.T) {
	// Verifies that ImplicitExpandPushdown and InferShapeForName work when graph inputs
	// have symbolic (dynamic) dimensions and Graph.ValueInfo is empty, matching PyTorch ONNX exports.
	dynInput := &protos.ValueInfoProto{
		Name: "attention_mask",
		Type: &protos.TypeProto{
			Value: &protos.TypeProto_TensorType{
				TensorType: &protos.TypeProto_Tensor{
					ElemType: int32(protos.TensorProto_INT64),
					Shape: &protos.TensorShapeProto{
						Dim: []*protos.TensorShapeProto_Dimension{
							{Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: "batch_size"}},
							{Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: "sequence_length"}},
						},
					},
				},
			},
		},
	}

	graphProto := &protos.GraphProto{
		Name:   "dynamic_expand_dag",
		Input:  []*protos.ValueInfoProto{dynInput},
		Output: []*protos.ValueInfoProto{makeValueInfoTyped("mask_4d", protos.TensorProto_FLOAT, []int64{-1, 1, -1, -1})},
		Initializer: []*protos.TensorProto{
			{Name: "axis1", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), RawData: []byte{1, 0, 0, 0, 0, 0, 0, 0}},
			{Name: "axis2", Dims: []int64{1}, DataType: int32(protos.TensorProto_INT64), RawData: []byte{2, 0, 0, 0, 0, 0, 0, 0}},
			{Name: "target_shape", Dims: []int64{4}, DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{2, 1, 4, 4}},
			{Name: "one", Dims: []int64{}, DataType: int32(protos.TensorProto_FLOAT), FloatData: []float32{1.0}},
			{Name: "neg_inf", Dims: []int64{}, DataType: int32(protos.TensorProto_FLOAT), FloatData: []float32{-1e9}},
		},
		Node: []*protos.NodeProto{
			{Name: "u1", OpType: "Unsqueeze", Input: []string{"attention_mask", "axis1"}, Output: []string{"u1_out"}},
			{Name: "u2", OpType: "Unsqueeze", Input: []string{"u1_out", "axis2"}, Output: []string{"u2_out"}},
			{Name: "exp", OpType: "Expand", Input: []string{"u2_out", "target_shape"}, Output: []string{"exp_out"}},
			{
				Name: "cast_f32", OpType: "Cast", Input: []string{"exp_out"}, Output: []string{"cast_f32_out"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)},
				},
			},
			{Name: "sub", OpType: "Sub", Input: []string{"one", "cast_f32_out"}, Output: []string{"sub_out"}},
			{
				Name: "cast_bool", OpType: "Cast", Input: []string{"sub_out"}, Output: []string{"cast_bool_out"},
				Attribute: []*protos.AttributeProto{
					{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_BOOL)},
				},
			},
			{Name: "where", OpType: "Where", Input: []string{"cast_bool_out", "neg_inf", "sub_out"}, Output: []string{"mask_4d"}},
		},
	}

	content, err := proto.Marshal(&protos.ModelProto{Graph: graphProto})
	require.NoError(t, err)

	m, err := onnxgomlx.Parse(content)
	require.NoError(t, err)

	unexpShape := m.InferShapeForName("mask_4d#unexpanded")
	assert.Equal(t, []int{-1, 1, 1, -1}, unexpShape.Dimensions)
}

