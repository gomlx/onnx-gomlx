// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertConvInteger converts an ONNX ConvInteger node to an exact Int32 GoMLX Convolve operation.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ConvInteger.html
func convertConvInteger(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 2 {
		exceptions.Panicf("ConvInteger requires at least 2 inputs (x, w), got %d in %s", len(inputs), NodeToString(node))
	}
	x, w := inputs[0], inputs[1]
	for _, value := range []*Node{x, w} {
		if value.DType() != dtypes.Int8 && value.DType() != dtypes.Uint8 {
			exceptions.Panicf("ConvInteger: expected int8 or uint8, got %s", value.Shape())
		}
	}
	if x.Rank() < 3 || w.Rank() != x.Rank() {
		exceptions.Panicf("ConvInteger: invalid ranks for x=%s and w=%s", x.Shape(), w.Shape())
	}

	xI32 := ConvertDType(x, dtypes.Int32)
	wI32 := ConvertDType(w, dtypes.Int32)

	if len(inputs) > 2 && inputs[2] != nil {
		if inputs[2].Shape().Size() != 1 {
			exceptions.Panicf("ConvInteger: x_zero_point must be scalar, got %s", inputs[2].Shape())
		}
		xI32 = Sub(xI32, Reshape(ConvertDType(inputs[2], dtypes.Int32)))
	}
	if len(inputs) > 3 && inputs[3] != nil {
		zero := ConvertDType(inputs[3], dtypes.Int32)
		if zero.Shape().Size() == 1 {
			zero = Reshape(zero)
		} else {
			if zero.Rank() != 1 || zero.Shape().Dim(0) != wI32.Shape().Dim(0) {
				exceptions.Panicf("ConvInteger: invalid per-channel w_zero_point %s for weights %s", zero.Shape(), w.Shape())
			}
			dims := make([]int, wI32.Rank())
			for i := range dims {
				dims[i] = 1
			}
			dims[0] = zero.Shape().Dim(0)
			zero = Reshape(zero, dims...)
		}
		wI32 = Sub(wI32, zero)
	}

	autoPad := GetStringAttrOr(node, "auto_pad", "NOTSET")
	strides := GetIntsAttrOr(node, "strides", nil)
	pads := GetIntsAttrOr(node, "pads", nil)
	dilations := GetIntsAttrOr(node, "dilations", nil)
	groups := GetIntAttrOr(node, "group", 1)

	numSpatialDims := xI32.Rank() - 2
	spatialAxes := make([]int, numSpatialDims)
	for i := range spatialAxes {
		spatialAxes[i] = i + 2
	}

	var paddings [][2]int
	switch autoPad {
	case "NOTSET":
		if pads != nil {
			if len(pads) != 2*numSpatialDims {
				exceptions.Panicf("ConvInteger: expected %d padding values, got %d", 2*numSpatialDims, len(pads))
			}
			paddings = make([][2]int, numSpatialDims)
			for i := range numSpatialDims {
				paddings[i] = [2]int{pads[i], pads[i+numSpatialDims]}
			}
		}
	case "VALID":
		// Zero padding (default).
	case "SAME_UPPER", "SAME_LOWER":
		paddings = make([][2]int, numSpatialDims)
		for i := range numSpatialDims {
			inDim := xI32.Shape().Dim(i + 2)
			if inDim < 0 {
				exceptions.Panicf("ConvInteger: auto_pad=%q with dynamic spatial axis %d is not supported", autoPad, i+2)
			}
			stride, dilation := 1, 1
			if i < len(strides) {
				stride = strides[i]
			}
			if i < len(dilations) {
				dilation = dilations[i]
			}
			outDim := (inDim + stride - 1) / stride
			effectiveKernel := (wI32.Shape().Dim(i+2)-1)*dilation + 1
			totalPad := max((outDim-1)*stride+effectiveKernel-inDim, 0)
			padBefore := totalPad / 2
			padAfter := totalPad - padBefore
			if autoPad == "SAME_LOWER" {
				padBefore, padAfter = padAfter, padBefore
			}
			paddings[i] = [2]int{padBefore, padAfter}
		}
	default:
		exceptions.Panicf("ConvInteger: unsupported auto_pad %q", autoPad)
	}

	axes := compute.ConvolveAxesConfig{
		InputBatch:           0,
		InputChannels:        1,
		InputSpatial:         spatialAxes,
		KernelOutputChannels: 0,
		KernelInputChannels:  1,
		KernelSpatial:        spatialAxes,
		OutputBatch:          0,
		OutputChannels:       1,
		OutputSpatial:        spatialAxes,
	}
	conv := Convolve(xI32, wI32).AxesConfig(axes)
	if len(strides) > 0 {
		conv = conv.StridePerAxis(strides...)
	}
	if len(dilations) > 0 {
		conv = conv.DilationPerAxis(dilations...)
	}
	if len(paddings) > 0 {
		conv = conv.PaddingPerDim(paddings)
	}
	if groups > 1 {
		conv = conv.ChannelGroupCount(groups)
	}
	return conv.Done()
}
