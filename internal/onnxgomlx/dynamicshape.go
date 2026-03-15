package onnxgomlx

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

// GoMLX returns a shapes.Shape representation of the DynamicShape.
// Dimensions that are dynamic (-1) are set to shapes.DynamicDim and the
// corresponding axis names from the ONNX model are preserved via
// shapes.MakeDynamic so that the specialization system can resolve them.
func (dshape DynamicShape) GoMLX() shapes.Shape {
	if dshape.Rank() == 0 {
		return shapes.Make(dshape.DType)
	}
	hasDynamic := false
	for _, d := range dshape.Dimensions {
		if d == -1 {
			hasDynamic = true
			break
		}
	}
	if !hasDynamic {
		return shapes.Make(dshape.DType, dshape.Dimensions...)
	}
	axisNames := make([]string, len(dshape.Names))
	for i, name := range dshape.Names {
		if dshape.Dimensions[i] == -1 {
			axisNames[i] = dynamicAxisName(name, i)
		}
	}
	return shapes.MakeDynamic(dshape.DType, slices.Clone(dshape.Dimensions), axisNames)
}

// UnnamedDynamicDimension is a placeholder name for an unnamed dynamic dimension, that doesn't necessarily match any other (in inputs/outputs).
const UnnamedDynamicDimension = "?"

// dynamicAxisName returns the axis name for a dynamic dimension at the given
// index. Unnamed dimensions ("?") get a positional name like "dyn_0".
// This is the single source of truth for naming unnamed dynamic axes.
func dynamicAxisName(name string, index int) string {
	if name == UnnamedDynamicDimension {
		return fmt.Sprintf("dyn_%d", index)
	}
	return name
}

// ForceStaticShapes disables dynamic shape propagation even when the backend
// supports DynamicAxes. All shapes will be resolved to concrete values at
// graph build time.
func (m *Model) ForceStaticShapes() {
	m.forceStaticShapes = true
}

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
			dshape.Names[ii] = UnnamedDynamicDimension // Un-named dynamic dimension.
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

// DynamicAxesConfig returns the dynamic axes configuration for each model input,
// suitable for passing to Exec.WithDynamicAxes(). Each element corresponds to
// an input in InputsNames order; axis names are non-empty for dynamic axes and
// empty ("") for static axes.
//
// Returns nil if the backend does not support dynamic axes or ForceStaticShapes
// has been called. The caller should skip WithDynamicAxes when nil is returned.
func (m *Model) DynamicAxesConfig(backend backends.Backend) [][]string {
	if m.forceStaticShapes {
		return nil
	}
	if backend == nil || !backend.Capabilities().DynamicAxes {
		return nil
	}

	// Single pass: build axis names and detect whether any dynamic axis exists.
	hasAnyDynamic := false
	result := make([][]string, len(m.InputsShapes))
	for i, dshape := range m.InputsShapes {
		axisNames := make([]string, len(dshape.Dimensions))
		for j, dim := range dshape.Dimensions {
			if dim == -1 {
				axisNames[j] = dynamicAxisName(dshape.Names[j], j)
				hasAnyDynamic = true
			}
		}
		result[i] = axisNames
	}
	if !hasAnyDynamic {
		return nil
	}
	return result
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
				dimName := dynamicAxisName(wantShape.Names[axis], axis)
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
