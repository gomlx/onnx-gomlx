package onnxgomlx

import (
	"math"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/pkg/errors"
)

// convertResize converts an ONNX Resize node to GoMLX.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Resize.html
//
// Inputs: X, roi (unused), scales (optional), sizes (optional).
// Either scales or sizes must be provided. If both are present, sizes takes priority.
func convertResize(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	inputDims := x.Shape().Dimensions

	mode := GetStringAttrOr(node, "mode", "nearest")
	coordTransformMode := GetStringAttrOr(node, "coordinate_transformation_mode", "half_pixel")

	// Resolve target output sizes from either the "sizes" or "scales" input.
	outputSizes, scales := resizeOutputSizes(m, convertedOutputs, node, inputDims)
	if mode == "nearest" || (mode == "linear" && (scales != nil || coordTransformMode == "pytorch_half_pixel")) {
		return resizeWithCoordinates(x, outputSizes, scales, mode, coordTransformMode, GetStringAttrOr(node, "nearest_mode", "round_prefer_floor"))
	}

	// Use NoInterpolation for dimensions that don't change.
	for i := range outputSizes {
		if outputSizes[i] == inputDims[i] {
			outputSizes[i] = NoInterpolation
		}
	}

	// Check if any dimension actually needs resizing.
	needsWork := false
	for _, s := range outputSizes {
		if s != NoInterpolation {
			needsWork = true
			break
		}
	}
	if !needsWork {
		return x
	}

	config := Interpolate(x, outputSizes...)

	switch mode {
	case "nearest":
		config.Nearest()
	case "linear":
		config.Bilinear()
	case "cubic":
		// GoMLX doesn't support cubic; bilinear is the closest available.
		config.Bilinear()
	default:
		exceptions.Panicf("Resize: unsupported mode %q in %s", mode, NodeToString(node))
	}

	switch coordTransformMode {
	case "half_pixel", "half_pixel_symmetric", "pytorch_half_pixel":
		// Default is already halfPixelCenters=true, alignCorner=false.
	case "align_corners":
		config.HalfPixelCenters(false).AlignCorner(true)
	case "asymmetric":
		config.HalfPixelCenters(false)
	case "tf_crop_and_resize":
		exceptions.Panicf("Resize: coordinate_transformation_mode %q is not supported in %s",
			coordTransformMode, NodeToString(node))
	default:
		config.HalfPixelCenters(false)
	}

	return config.Done()
}

// resizeOutputSizes resolves the target dimensions for a Resize op from either
// the "sizes" input (index 3) or the "scales" input (index 2).
func resizeOutputSizes(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputDims []int) ([]int, []float64) {
	rank := len(inputDims)

	// Try sizes first (takes priority over scales per ONNX spec).
	if len(node.Input) > 3 && node.Input[3] != "" {
		sizesT, err := m.materializeConstantExpression(node.Input[3], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'sizes' for node %s", NodeToString(node)))
		}
		if sizesT.Size() > 0 {
			sizes := tensorToInts(sizesT)
			if len(sizes) != rank {
				exceptions.Panicf("Resize: sizes length (%d) != input rank (%d) in %s",
					len(sizes), rank, NodeToString(node))
			}
			return sizes, nil
		}
	}

	// Fall back to scales.
	if len(node.Input) > 2 && node.Input[2] != "" {
		scalesT, err := m.materializeConstantExpression(node.Input[2], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'scales' for node %s", NodeToString(node)))
		}
		if scalesT.Size() > 0 {
			scales := tensorToFloat64s(scalesT)
			if len(scales) != rank {
				exceptions.Panicf("Resize: scales length (%d) != input rank (%d) in %s",
					len(scales), rank, NodeToString(node))
			}
			out := make([]int, rank)
			for i, s := range scales {
				out[i] = max(int(float64(inputDims[i])*s), 1)
			}
			return out, scales
		}
	}

	// Neither provided — return input dimensions unchanged.
	out := make([]int, rank)
	copy(out, inputDims)
	return out, nil
}

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
