// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/images"
)

// convertConvTranspose lowers ONNX ConvTranspose to a convolution over an input-dilated
// operand with spatially reversed weights and transposed input/output channel axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
func convertConvTranspose(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 2 {
		exceptions.Panicf("ConvTranspose requires at least 2 inputs (X, W), got %d in %s", len(inputs), NodeToString(node))
	}
	x, w := inputs[0], inputs[1]
	spatial := x.Rank() - 2
	if spatial < 1 || w.Rank() != x.Rank() {
		exceptions.Panicf("ConvTranspose: invalid input %s or weights %s", x.Shape(), w.Shape())
	}
	groups := GetIntAttrOr(node, "group", 1)
	strides := GetIntsAttrOr(node, "strides", nil)
	dilations := GetIntsAttrOr(node, "dilations", nil)
	pads := GetIntsAttrOr(node, "pads", nil)
	outputPads := GetIntsAttrOr(node, "output_padding", nil)
	outputShape := GetIntsAttrOr(node, "output_shape", nil)
	for name, values := range map[string][]int{"strides": strides, "dilations": dilations, "output_padding": outputPads, "output_shape": outputShape} {
		if values != nil && len(values) != spatial {
			exceptions.Panicf("ConvTranspose: invalid %s length %d for %d spatial axes", name, len(values), spatial)
		}
	}
	if pads != nil && len(pads) != 2*spatial {
		exceptions.Panicf("ConvTranspose: invalid pads length %d for %d spatial axes", len(pads), spatial)
	}
	if groups < 1 || w.Shape().Dim(0) < 0 || w.Shape().Dim(0)%groups != 0 {
		exceptions.Panicf("ConvTranspose: invalid group %d for weights %s", groups, w.Shape())
	}

	steps, rates := make([]int, spatial), make([]int, spatial)
	padding := make([][2]int, spatial)
	axes := make([]int, spatial)
	autoPad := GetStringAttrOr(node, "auto_pad", "NOTSET")
	for i := range spatial {
		axes[i] = i + 2
		steps[i], rates[i] = 1, 1
		if strides != nil {
			steps[i] = strides[i]
		}
		if dilations != nil {
			rates[i] = dilations[i]
		}
		if steps[i] < 1 || rates[i] < 1 {
			exceptions.Panicf("ConvTranspose: strides and dilations must be positive")
		}
		extra := 0
		if outputPads != nil {
			extra = outputPads[i]
		}
		if extra < 0 || (extra >= steps[i] && extra >= rates[i]) {
			exceptions.Panicf("ConvTranspose: invalid output_padding %d for stride=%d, dilation=%d", extra, steps[i], rates[i])
		}
		effective := (w.Shape().Dim(i+2)-1)*rates[i] + 1
		before, after := 0, 0
		if pads != nil {
			before, after = pads[i], pads[i+spatial]
		}
		switch autoPad {
		case "NOTSET", "VALID":
			if autoPad == "VALID" {
				before, after = 0, 0
			}
		case "SAME_UPPER", "SAME_LOWER":
			total := effective + extra - steps[i]
			before, after = total/2, total-total/2
			if autoPad == "SAME_LOWER" {
				before, after = after, before
			}
		default:
			exceptions.Panicf("ConvTranspose: unsupported auto_pad %q", autoPad)
		}
		if outputShape != nil {
			inSpatialDim := x.Shape().Dim(i + 2)
			if inSpatialDim < 0 {
				exceptions.Panicf("ConvTranspose: output_shape requires static spatial input dimensions, got %s", x.Shape())
			}
			total := steps[i]*(inSpatialDim-1) + extra + effective - outputShape[i]
			before, after = total/2, total-total/2
			if autoPad == "SAME_LOWER" {
				before, after = after, before
			}
		}
		padding[i] = [2]int{effective - 1 - before, effective - 1 - after + extra}
	}

	w = Reverse(w, axes...)
	// Keep each group's output channels adjacent after swapping the channel axes.
	if groups == 1 {
		w = Transpose(w, 0, 1)
	} else {
		dims := append([]int{groups, w.Shape().Dim(0) / groups, w.Shape().Dim(1)}, w.Shape().Dimensions[2:]...)
		w = Transpose(Reshape(w, dims...), 1, 2)
		w = Reshape(w, append([]int{dims[0] * dims[2], dims[1]}, dims[3:]...)...)
	}

	result := Convolve(x, w).ChannelsAxis(images.ChannelsFirst).
		InputDilationPerAxis(steps...).DilationPerAxis(rates...).PaddingPerDim(padding).ChannelGroupCount(groups).Done()
	if len(inputs) > 2 && inputs[2] != nil {
		dims := make([]int, x.Rank())
		for i := range dims {
			dims[i] = 1
		}
		dims[1] = inputs[2].Shape().Dim(0)
		result = Add(result, Reshape(inputs[2], dims...))
	}
	return result
}
