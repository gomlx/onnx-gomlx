package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertQuantizeLinear converts the corresponding ONNX node to a GoMLX node.
//
// Not yet supporting block quantization.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
func convertQuantizeLinear(nodeProto *protos.NodeProto, inputs []*Node) *Node {
	// Attributes:
	// - Axis (optional) on which to apply the multi-valued scale.
	// - blockSize: optional, only active if != 0. Not yet implemented.
	// - output_dtype: optional, specifies the output dtype.
	// - saturate: optional, for float8 types only.
	targetAxis := GetIntAttrOr(nodeProto, "axis", 1)
	blockSize := GetIntAttrOr(nodeProto, "blockSize", 0)
	if blockSize != 0 {
		exceptions.Panicf("QuantizeLinear: support for attribute 'block_size' is not yet implemented")
	}

	x := inputs[0]
	yScale := inputs[1]
	var yZeroPoint *Node
	if len(inputs) > 2 {
		yZeroPoint = inputs[2]
	}

	// Determine output dtype
	var outputDType dtypes.DType
	if yZeroPoint != nil {
		outputDType = yZeroPoint.DType()
	} else {
		// Default to uint8 if no zero point is provided.
		outputDType = GetDTypeAttrOr(nodeProto, "output_dtype", dtypes.Uint8)
	}

	return onnxQuantizeLinear(x, yScale, yZeroPoint, targetAxis, outputDType)
}

// onnxQuantizeLinear implements the ONNX QuantizeLinear operation.
// Formula: y = saturate((x / y_scale) + y_zero_point)
func onnxQuantizeLinear(x, yScale, yZeroPoint *Node, targetAxis int, outputDType dtypes.DType) *Node {
	g := x.Graph()

	// Reshape scale to match input rank if it's 1-D
	if !yScale.IsScalar() {
		targetAxis = MustAdjustAxis(targetAxis, x)
		if yScale.Rank() != 1 {
			exceptions.Panicf("QuantizeLinear: y_scale must be a scalar or 1D, got %s instead", yScale.Shape())
		}
		scaleLen := yScale.Shape().Dim(0)
		if xDim := x.Shape().Dim(targetAxis); xDim >= 0 && xDim != scaleLen {
			exceptions.Panicf("QuantizeLinear: y_scale must have same dimension as the input axis %d (input shape=%s), got %s instead", targetAxis, x.Shape(), yScale.Shape())
		} else if xDim < 0 && scaleLen >= 0 {
			specs := DimensionSpecsFor(x)
			specs[targetAxis] = StaticDim(scaleLen)
			x = DynamicReshape(x, specs...)
		}
		scaleDims := make([]int, x.Rank())
		for axis := range scaleDims {
			scaleDims[axis] = 1
		}
		scaleDims[targetAxis] = scaleLen
		yScale = Reshape(yScale, scaleDims...)
	}

	// Similarly reshape zero point if provided
	if yZeroPoint != nil && !yZeroPoint.IsScalar() {
		targetAxis = MustAdjustAxis(targetAxis, x)
		if yZeroPoint.Rank() != 1 {
			exceptions.Panicf("QuantizeLinear: y_zero_point must be a scalar or 1D, got %s instead", yZeroPoint.Shape())
		}
		zpLen := yZeroPoint.Shape().Dim(0)
		if xDim := x.Shape().Dim(targetAxis); xDim >= 0 && xDim != zpLen {
			exceptions.Panicf("QuantizeLinear: y_zero_point must have same dimension as the input axis %d (input shape=%s), got %s instead", targetAxis, x.Shape(), yZeroPoint.Shape())
		} else if xDim < 0 && zpLen >= 0 {
			specs := DimensionSpecsFor(x)
			specs[targetAxis] = StaticDim(zpLen)
			x = DynamicReshape(x, specs...)
		}
		zpDims := make([]int, x.Rank())
		for axis := range zpDims {
			zpDims[axis] = 1
		}
		zpDims[targetAxis] = zpLen
		yZeroPoint = Reshape(yZeroPoint, zpDims...)
	}

	// Convert input to scale's dtype for division
	x = ConvertDType(x, yScale.DType())

	// Quantize: y = Round(Div(x, yScale))
	y := Round(Div(x, yScale))

	// Add zero point if provided
	if yZeroPoint != nil {
		y = Add(y, ConvertDType(yZeroPoint, y.DType()))
	}

	// Saturate to output dtype range
	var minVal, maxVal *Node
	switch outputDType {
	case dtypes.Int8:
		minVal = Scalar(g, y.DType(), -128)
		maxVal = Scalar(g, y.DType(), 127)
	case dtypes.Uint8:
		minVal = Scalar(g, y.DType(), 0)
		maxVal = Scalar(g, y.DType(), 255)
	case dtypes.Int16:
		minVal = Scalar(g, y.DType(), -32768)
		maxVal = Scalar(g, y.DType(), 32767)
	case dtypes.Uint16:
		minVal = Scalar(g, y.DType(), 0)
		maxVal = Scalar(g, y.DType(), 65535)
	case dtypes.Int32:
		minVal = Scalar(g, y.DType(), -2147483648)
		maxVal = Scalar(g, y.DType(), 2147483647)
	default:
		// For other types (float8, etc.), no saturation needed
	}

	if minVal != nil && maxVal != nil {
		y = Clip(y, minVal, maxVal)
	}

	// Convert to output dtype
	y = ConvertDType(y, outputDType)
	return y
}
