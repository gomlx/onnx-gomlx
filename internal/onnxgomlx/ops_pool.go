package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	timage "github.com/gomlx/gomlx/core/tensors/images"
)

// convertAveragePool converts an ONNX AveragePool node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__AveragePool.html
func convertAveragePool(_ *Model, _ map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	autoPad := GetStringAttrOr(node, "auto_pad", "NOTSET")
	if autoPad != "NOTSET" {
		exceptions.Panicf("AveragePool: support for attribute 'auto_pad' (%s) is not yet implemented", autoPad)
	}
	ceilMode := GetIntAttrOr(node, "ceil_mode", 0)
	if ceilMode != 0 {
		exceptions.Panicf("AveragePool: support for attribute 'ceil_mode' is not yet implemented")
	}
	countIncludePad := GetIntAttrOr(node, "count_include_pad", 0)
	if countIncludePad != 0 {
		// GoMLX MeanPool doesn't support including padding in the count.
		exceptions.Panicf("AveragePool: support for attribute 'count_include_pad' is not yet implemented")
	}
	kernelShape := GetIntsAttrOr(node, "kernel_shape", nil)
	strides := GetIntsAttrOr(node, "strides", nil)
	pads := GetIntsAttrOr(node, "pads", nil)

	x := inputs[0]

	var paddings [][2]int
	numSpatialDims := x.Rank() - 2
	if pads != nil {
		if len(pads) != 2*numSpatialDims {
			exceptions.Panicf("invalid number of padding values: %d spatial axes, got %d padding values -- expected 2 pads per axis", numSpatialDims, len(pads))
		}
		for i := range numSpatialDims {
			paddings = append(paddings, [2]int{pads[i], pads[i+numSpatialDims]})
		}
	}

	pool := MeanPool(x).ChannelsAxis(timage.ChannelsFirst).Strides(1)
	if kernelShape != nil {
		pool = pool.WindowPerAxis(kernelShape...)
	}
	if strides != nil {
		pool = pool.StridePerAxis(strides...)
	}
	if paddings != nil {
		pool = pool.PaddingPerDim(paddings)
	}
	out := pool.Done()
	return out
}

// convertMaxPool converts an ONNX MaxPool node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__MaxPool.html
func convertMaxPool(_ *Model, _ map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	autoPad := GetStringAttrOr(node, "auto_pad", "NOTSET")
	if autoPad != "NOTSET" {
		exceptions.Panicf("MaxPool: support for attribute 'auto_pad' (%s) is not yet implemented", autoPad)
	}
	ceilMode := GetIntAttrOr(node, "ceil_mode", 0)
	if ceilMode != 0 {
		exceptions.Panicf("MaxPool: support for attribute 'ceil_mode' is not yet implemented")
	}
	dilations := GetIntsAttrOr(node, "dilations", nil)
	if dilations != nil {
		exceptions.Panicf("MaxPool: support for attribute 'dilations' is not yet implemented")
	}
	storageOrder := GetIntAttrOr(node, "storage_order", 0)
	if storageOrder != 0 {
		exceptions.Panicf("MaxPool: support for attribute 'storage_order' is not yet implemented")
	}
	kernelShape := GetIntsAttrOr(node, "kernel_shape", nil)
	strides := GetIntsAttrOr(node, "strides", nil)
	pads := GetIntsAttrOr(node, "pads", nil)

	x := inputs[0]

	var paddings [][2]int
	numSpatialDims := x.Rank() - 2
	if pads != nil {
		if len(pads) != 2*numSpatialDims {
			exceptions.Panicf("invalid number of padding values: %d spatial axes, got %d padding values -- "+
				"expected 2 pads per axis", numSpatialDims, len(pads))
		}
		for i := range numSpatialDims {
			paddings = append(paddings, [2]int{pads[i], pads[i+numSpatialDims]})
		}
	}

	pool := MaxPool(x).ChannelsAxis(timage.ChannelsFirst).Strides(1)
	if kernelShape != nil {
		pool = pool.WindowPerAxis(kernelShape...)
	}
	if strides != nil {
		pool = pool.StridePerAxis(strides...)
	}
	if paddings != nil {
		pool = pool.PaddingPerDim(paddings)
	}
	out := pool.Done()
	return out
}

// convertGlobalAveragePool converts an ONNX GlobalAveragePool node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html
func convertGlobalAveragePool(_ *Model, _ map[string]*Node, _ *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	spatialDims := x.Rank() - 2
	window := make([]int, spatialDims)
	for i := range window {
		window[i] = x.Shape().Dim(i + 2)
	}
	pool := MeanPool(x).ChannelsAxis(timage.ChannelsFirst).Strides(1).WindowPerAxis(window...)
	return pool.Done()
}
