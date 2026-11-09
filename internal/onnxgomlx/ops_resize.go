package onnxgomlx

import (
	"math"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// resizeWithCoordinates preserves ONNX coordinate and rounding rules that are
// not expressible through Interpolate's output-size-only configuration.
// Applying one axis at a time gives separable nearest or linear interpolation.
func resizeWithCoordinates(x *Node, sizes []int, scales []float64, mode, coordinateMode, nearestMode string) *Node {
	for axis, outSize := range sizes {
		inSize := x.Shape().Dimensions[axis]
		scale := float64(outSize) / float64(inSize)
		if scales != nil {
			scale = scales[axis]
		}
		if inSize == outSize {
			continue
		}
		lower := make([]int64, outSize)
		upper := make([]int64, outSize)
		weights := make([]float64, outSize)
		for i := range outSize {
			coordinate := resizeCoordinate(i, inSize, outSize, scale, coordinateMode)
			if mode == "nearest" {
				switch nearestMode {
				case "floor":
					coordinate = math.Floor(coordinate)
				case "ceil":
					coordinate = math.Ceil(coordinate)
				case "round_prefer_floor":
					// Treat float32 coordinates within 1e-6 of a half-integer as ties.
					coordinate = math.Ceil(coordinate - 0.5 - 1e-6)
				case "round_prefer_ceil":
					coordinate = math.Floor(coordinate + 0.5 + 1e-6)
				default:
					exceptions.Panicf("Resize: unsupported nearest_mode %q", nearestMode)
				}
			}
			coordinate = max(0, min(float64(inSize-1), coordinate))
			lower[i] = int64(math.Floor(coordinate))
			upper[i] = min(lower[i]+1, int64(inSize-1))
			weights[i] = coordinate - float64(lower[i])
		}
		g := x.Graph()
		lo := onnxGather(x, Const(g, lower), axis)
		if mode == "nearest" {
			x = lo
			continue
		}
		hi := onnxGather(x, Const(g, upper), axis)
		weight := ConvertDType(Const(g, weights), x.DType())
		weight = BroadcastLike(weight, lo, axis)
		x = Add(Mul(lo, OneMinus(weight)), Mul(hi, weight))
	}
	return x
}

// ONNX scales and the reference runtime's sampling coordinates use float32.
// Keep this precision before rounding so coordinates on integer boundaries agree.
func resizeCoordinate(index, inSize, outSize int, scale float64, mode string) float64 {
	position, factor := float32(index), float32(scale)
	switch mode {
	case "half_pixel":
		return float64((position+0.5)/factor - 0.5)
	case "pytorch_half_pixel":
		if outSize == 1 {
			return 0
		}
		return float64((position+0.5)/factor - 0.5)
	case "half_pixel_symmetric":
		adjustment := float32(outSize) / (float32(inSize) * factor)
		offset := float32(inSize) / 2 * (1 - adjustment)
		return float64(float32(float64(offset) + (float64(position)+0.5)/float64(factor) - 0.5))
	case "align_corners":
		if outSize == 1 {
			return 0
		}
		return float64(position * float32(inSize-1) / float32(outSize-1))
	case "tf_crop_and_resize":
		exceptions.Panicf("Resize: coordinate_transformation_mode %q is not supported", mode)
	}
	return float64(position / factor)
}
