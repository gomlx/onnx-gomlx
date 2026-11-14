// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertGridSample implements 2D nearest and bilinear GridSample, including grids
// whose batch or sample dimensions are dynamic at runtime.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GridSample.html
func convertGridSample(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 2 {
		exceptions.Panicf("GridSample requires 2 inputs (X, grid), got %d in %s", len(inputs), NodeToString(node))
	}
	x, grid := inputs[0], inputs[1]
	if x.Rank() != 4 || grid.Rank() != 4 || (grid.Shape().Dim(3) != shapes.DynamicDim && grid.Shape().Dim(3) != 2) {
		exceptions.Panicf("GridSample: expected NCHW input and NHW2 grid, got %s and %s", x.Shape(), grid.Shape())
	}
	n, c, h, w := x.Shape().Dim(0), x.Shape().Dim(1), x.Shape().Dim(2), x.Shape().Dim(3)
	if (n != shapes.DynamicDim && n < 1) || c < 1 || h < 1 || w < 1 {
		exceptions.Panicf("GridSample: channel and spatial input dimensions must be positive static sizes, got %s", x.Shape())
	}
	align := GetBoolAttrOr(node, "align_corners", false)
	mode, padding := GetStringAttrOr(node, "mode", "bilinear"), GetStringAttrOr(node, "padding_mode", "zeros")
	if padding != "zeros" && padding != "border" && padding != "reflection" {
		exceptions.Panicf("GridSample: unsupported padding_mode %q", padding)
	}

	grid = ConvertDType(grid, x.DType())
	// Align batch axis name if both x and grid have dynamic batch dimensions with distinct names.
	if x.Shape().Dim(0) == shapes.DynamicDim && grid.Shape().Dim(0) == shapes.DynamicDim &&
		x.Shape().AxisName(0) != grid.Shape().AxisName(0) {
		gSpecs := DimensionSpecsFor(grid)
		gSpecs[0] = NamedDynamicDim(x.Shape().AxisName(0), DimensionSize(grid, 0))
		grid = DynamicReshape(grid, gSpecs...)
	}

	xg := Squeeze(SliceAxis(grid, 3, AxisRange(0, 1)), 3)
	yg := Squeeze(SliceAxis(grid, 3, AxisRange(1, 2)), 3)
	coordinate := func(v *Node, size int) *Node {
		if align {
			v = MulScalar(OnePlus(v), float64(size-1)/2)
		} else {
			v = AddScalar(MulScalar(OnePlus(v), float64(size)/2), -0.5)
		}
		if padding == "reflection" {
			low, span := 0.0, float64(size-1)
			if !align {
				low, span = -0.5, float64(size)
			}
			if span == 0 {
				return ZerosLike(v)
			}
			distance := Abs(AddScalar(v, -low))
			period := 2 * span
			remainder := Sub(distance, MulScalar(Floor(DivScalar(distance, period)), period))
			v = AddScalar(Min(remainder, Sub(Scalar(v.Graph(), v.DType(), period), remainder)), low)
		}
		if padding != "zeros" {
			v = MinScalar(MaxScalar(v, 0), size-1)
		}
		return v
	}
	xg, yg = coordinate(xg, w), coordinate(yg, h)
	flatInput := DynamicReshape(TransposeAllAxes(x, 0, 2, 3, 1), InferredDim(), StaticDim(c))
	batchOffset := MulScalar(DynamicIota(x.Graph(), dtypes.Int64, 0, DimensionSpecFor(x, 0), StaticDim(1), StaticDim(1)), h*w)
	sample := func(xi, yi *Node) *Node {
		valid := LogicalAnd(
			LogicalAnd(GreaterOrEqual(xi, ScalarZero(x.Graph(), xi.DType())), LessThan(xi, Scalar(x.Graph(), xi.DType(), w))),
			LogicalAnd(GreaterOrEqual(yi, ScalarZero(x.Graph(), yi.DType())), LessThan(yi, Scalar(x.Graph(), yi.DType(), h))),
		)
		xi = ConvertDType(MinScalar(MaxScalar(xi, 0), w-1), dtypes.Int64)
		yi = ConvertDType(MinScalar(MaxScalar(yi, 0), h-1), dtypes.Int64)
		indices := Add(Add(MulScalar(yi, w), xi), batchOffset)
		values := Gather(flatInput, DynamicReshape(indices, InferredDim(), StaticDim(1)))
		values = DynamicReshape(values, DimensionSpecFor(grid, 0), DimensionSpecFor(grid, 1), DimensionSpecFor(grid, 2), StaticDim(c))
		if padding == "zeros" {
			values = Where(InsertAxes(valid, -1), values, ScalarZero(x.Graph(), x.DType()))
		}
		return values
	}

	var result *Node
	switch mode {
	case "nearest":
		result = sample(Round(xg), Round(yg))
	case "bilinear", "linear":
		x0, y0 := Floor(xg), Floor(yg)
		x1, y1 := OnePlus(x0), OnePlus(y0)
		dx, dy := InsertAxes(Sub(xg, x0), -1), InsertAxes(Sub(yg, y0), -1)
		top := Add(Mul(sample(x0, y0), OneMinus(dx)), Mul(sample(x1, y0), dx))
		bottom := Add(Mul(sample(x0, y1), OneMinus(dx)), Mul(sample(x1, y1), dx))
		result = Add(Mul(top, OneMinus(dy)), Mul(bottom, dy))
	default:
		exceptions.Panicf("GridSample: unsupported interpolation mode %q", mode)
	}
	return TransposeAllAxes(result, 0, 3, 1, 2)
}
