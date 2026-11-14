package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertDequantizeLinear converts the corresponding ONNX node to a GoMLX node.
//
// Not yet supporting block dequantization.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html
func (m *Model) convertDequantizeLinear(convertedOutputs map[string]*Node, nodeProto *protos.NodeProto, inputs []*Node) *Node {
	// Attributes:
	// - Axis (optional) on which to apply the multi-valued scale.
	// - blockSize: optional, only active if != 0. Not yet implemented.
	targetAxis := GetIntAttrOr(nodeProto, "axis", 1)
	blockSize := GetIntAttrOr(nodeProto, "blockSize", 0)
	if blockSize != 0 {
		exceptions.Panicf("DequantizeLinear: support for attribute 'block_size' is not yet implemented")
	}
	outputDType := GetDTypeAttrOr(nodeProto, "output_dtype", dtypes.Float32)

	x := inputs[0]
	scale := inputs[1]
	var xZeroPoint *Node
	if len(inputs) > 2 {
		xZeroPoint = inputs[2]
	}
	return onnxDequantizeLinear(x, scale, xZeroPoint, targetAxis, outputDType)
}

func onnxDequantizeLinear(x, scale, xZeroPoint *Node, targetAxis int, outputDType dtypes.DType) *Node {
	if !scale.IsScalar() {
		// Add extra axes of dim=1 in scale to match x's rank.
		if scale.Rank() != 1 {
			exceptions.Panicf("DequantizeLinear: scale must be a scalar or 1D, got %s instead", scale.Shape())
		}
		targetAxis = MustAdjustAxis(targetAxis, x)
		scaleLen := scale.Shape().Dim(0)
		if xDim := x.Shape().Dim(targetAxis); xDim >= 0 && xDim != scaleLen {
			exceptions.Panicf("DequantizeLinear: scale must have same dimension as the input axis %d (input shape=%s), got %s instead", targetAxis, x.Shape(), scale.Shape())
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
		scale = Reshape(scale, scaleDims...)
		if xZeroPoint != nil && !xZeroPoint.IsScalar() {
			xZeroPoint = Reshape(xZeroPoint, scaleDims...)
		}
	}
	if xZeroPoint != nil {
		x = Sub(ConvertDType(x, dtypes.Int32), ConvertDType(xZeroPoint, dtypes.Int32))
	}
	x = Mul(ConvertDType(x, scale.DType()), scale)
	if x.DType() != outputDType {
		x = ConvertDType(x, outputDType)
	}
	return x
}
