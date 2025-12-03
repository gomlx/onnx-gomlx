package onnx

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// DynamicShape represents a shape for which some of the axes have unknown dimensions.
//
// Similar to GoMLX Shape but some of the dimensions may be -1, denoting an undefined dimension.
//
// Dimensions may also be named, in which case shapes of inputs and outputs with the same name should match.
type DynamicShape struct {
	dtypes.DType
	Dimensions []int
	Names      []string
}

// UnnamedDynamicDimension is a placeholder name for an unnamed dynamic dimension, that doesn't necessarily match any other (in inputs/outputs).
const UnnamedDynamicDimension = "?"

// makeDynamicShapeFromProto converts from a tensor proto type to a DynamicShape.
func makeDynamicShapeFromProto(proto *protos.TypeProto_Tensor) (dshape DynamicShape, err error) {
	dshape.DType, err = dtypeForONNX(protos.TensorProto_DataType(proto.GetElemType()))
	if err != nil {
		return
	}
	dshape.Names = make([]string, len(proto.Shape.Dim))
	dshape.Dimensions = make([]int, len(proto.Shape.Dim))
	for ii, dProto := range proto.Shape.Dim {
		if dim, ok := dProto.GetValue().(*protos.TensorShapeProto_Dimension_DimValue); ok {
			dshape.Names[ii] = strconv.Itoa(int(dim.DimValue))
			dshape.Dimensions[ii] = int(dim.DimValue)
		} else if dimParam, ok := dProto.GetValue().(*protos.TensorShapeProto_Dimension_DimParam); ok {
			dshape.Names[ii] = dimParam.DimParam
			dshape.Dimensions[ii] = -1
		} else {
			dshape.Names[ii] = "?" // Un-named dynamic dimension.
			dshape.Dimensions[ii] = -1
		}
	}
	return
}

// Rank returns the DynamicShape's rank.
func (dshape DynamicShape) Rank() int {
	return len(dshape.Dimensions)
}

// String implements fmt.Stringer.
func (dshape DynamicShape) String() string {
	if len(dshape.Dimensions) == 0 {
		return fmt.Sprintf("(%s)", dshape.DType)
	}
	return fmt.Sprintf("(%s) [%s]", dshape.DType, strings.Join(dshape.Names, ", "))
}

// ValidateInputs checks the inputs has a shape that is compatible with the DynamicShapes of the inputs for the model.
func (m *Model) ValidateInputs(inputsShapes ...shapes.Shape) error {
	if len(inputsShapes) != len(m.InputsNames) {
		return errors.Errorf("model takes %d inputs, but %d inputs provided",
			len(m.InputsNames), len(inputsShapes))
	}
	dimValues := make(map[string]int)
	for idx, input := range inputsShapes {
		name := m.InputsNames[idx]
		givenShape := input.Shape()
		wantShape := m.InputsShapes[idx]
		if givenShape.Rank() != wantShape.Rank() {
			return errors.Errorf("model input #%d (%q) should be rank %d, got rank %d instead",
				idx, name, wantShape.Rank(), givenShape.Rank())
		}
		if givenShape.DType != wantShape.DType {
			return errors.Errorf("model input #%d (%q) should have dtype %s, got dtype %s instead",
				idx, name, wantShape.DType, givenShape.DType)
		}
		for axis, wantDim := range wantShape.Dimensions {
			gotDim := givenShape.Dim(axis)
			if wantDim > 0 {
				if wantDim != gotDim {
					return errors.Errorf("model input #%d (%q) has invalid shape: want %s, got %s",
						idx, name, wantShape, givenShape)
				}
			} else {
				dimName := wantShape.Names[axis]
				var found bool
				wantDim, found = dimValues[dimName]
				if !found {
					// Define dynamic shape based on input.
					dimValues[dimName] = gotDim
				} else if wantDim != gotDim {
					return errors.Errorf("model input #%d (%q) shaped %s got unmatching invalid shape %s for axis %q (wanted dim %d)",
						idx, name, wantShape, givenShape, dimName, wantDim)
				}
			}
		}
	}
	return nil
}
