package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

func TestValidateInputs(t *testing.T) {
	m := &Model{
		InputsNames: []string{"i0", "i1"},
		InputsShapes: []shapes.Shape{
			shapes.MakeDynamic(dtypes.Float32, []int{-1, -1}, []string{"batch_size", "feature_dim"}),
			shapes.MakeDynamic(dtypes.Int32, []int{-1, 3}, []string{"batch_size", "other"}),
		},
	}

	// Example valid input, batch_size=5
	require.NoError(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32, 5, 3)))

	// Wrong dtype:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7, 1),
		shapes.Make( /**/ dtypes.Int64, 5, 3)))

	// Wrong rank:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7 /**/, 1),
		shapes.Make(dtypes.Int32, 5, 3)))

	// Fixed dimension not matching:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32, 5 /**/, 4)))

	// Dynamic dimension not matching:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32 /**/, 6, 3)))

	// Unnamed ONNX dynamic dimensions get distinct generated "#onnx_<n>" names so different inputs/axes don't collide.
	protoTensor := &protos.TypeProto_Tensor{
		ElemType: int32(protos.TensorProto_FLOAT),
		Shape: &protos.TensorShapeProto{
			Dim: []*protos.TensorShapeProto_Dimension{
				{}, // unset
				{Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: ""}}, // empty DimParam
			},
		},
	}
	s0, err := makeShapeFromProto(protoTensor)
	require.NoError(t, err)
	s1, err := makeShapeFromProto(protoTensor)
	require.NoError(t, err)
	require.True(t, shapes.IsGeneratedAxisName(s0.AxisName(0)))
	require.True(t, shapes.IsGeneratedAxisName(s0.AxisName(1)))
	require.NotEqual(t, s0.AxisName(0), s0.AxisName(1))
	require.NotEqual(t, s0.AxisName(0), s1.AxisName(0))

	mUnnamed := &Model{
		InputsNames:  []string{"a", "b"},
		InputsShapes: []shapes.Shape{s0, s1},
	}
	require.NoError(t, mUnnamed.ValidateInputs(
		shapes.Make(dtypes.Float32, 2, 7),
		shapes.Make(dtypes.Float32, 5, 3)))
}
