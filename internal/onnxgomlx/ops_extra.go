package onnxgomlx

// This file implements ONNX operators that were missing from onnx-gomlx v0.5.14:
// Asinh, Sinh, Softplus, IsInf, ScatterElements and Unique (for constant inputs).

import (
	"cmp"
	"math"
	"reflect"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/pkg/errors"
)

////////////////////////////////////////////////////////////////////
//
// Element-wise math ops.
//
////////////////////////////////////////////////////////////////////

// withFloat32ForHalfPrecision computes fn in Float32 if x is a half-precision float (Float16/BFloat16),
// and converts the result back to the original dtype. Otherwise, it simply calls fn(x).
//
// This keeps the thresholds used by the numerically safe formulas below simple.
func withFloat32ForHalfPrecision(x *Node, fn func(x *Node) *Node) *Node {
	dtype := x.DType()
	if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
		return ConvertDType(fn(ConvertDType(x, dtypes.Float32)), dtype)
	}
	return fn(x)
}

// applyOddSymmetry returns -r where x < 0, and r elsewhere (NaNs in r are preserved).
// It is used by odd functions (f(-x) = -f(x)) computed on |x|.
func applyOddSymmetry(x, r *Node) *Node {
	return Where(LessThan(x, ScalarZero(x.Graph(), x.DType())), Neg(r), r)
}

// onnxAsinh implements the ONNX Asinh operator in a numerically safe way:
//
//	asinh(x) = sign(x) * log(|x| + sqrt(x²+1))
//
// For small |x| it uses the equivalent log1p(a + a²/(1+sqrt(1+a²))) (with a=|x|), which doesn't lose precision
// for |x| close to 0. For large |x|, where x² would lose precision or overflow, it uses log(|x|) + log(2).
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Asinh.html
func onnxAsinh(x *Node) *Node {
	return withFloat32ForHalfPrecision(x, func(x *Node) *Node {
		g := x.Graph()
		dtype := x.DType()
		// Above the threshold sqrt(1+a²) == a in the dtype's precision (threshold = 1/sqrt(epsilon)).
		threshold := 4096.0 // 2^12 for Float32.
		if dtype == dtypes.Float64 {
			threshold = 67108864.0 // 2^26
		}
		thresholdNode := Scalar(g, dtype, threshold)
		one := ScalarOne(g, dtype)
		a := Abs(x)
		isLarge := GreaterThan(a, thresholdNode)
		// Clamp a for the small branch, so a² never overflows. NaNs are preserved, since NaN > threshold is false.
		aSmall := Where(isLarge, thresholdNode, a)
		a2 := Square(aSmall)
		small := Log1p(Add(aSmall, Div(a2, Add(one, Sqrt(Add(one, a2))))))
		large := Add(Log(a), Scalar(g, dtype, math.Ln2))
		return applyOddSymmetry(x, Where(isLarge, large, small))
	})
}

// onnxSinh implements the ONNX Sinh operator in a numerically safe way:
//
//	sinh(x) = sign(x) * (t + t/(t+1)) / 2, with t = expm1(|x|)
//
// which is exactly (e^|x| - e^-|x|)/2 but without the cancellation for |x| close to 0.
// For large |x| (> 20), where e^-|x| is negligible, it uses exp(|x| - log(2)), which only overflows
// when sinh(x) itself overflows.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Sinh.html
func onnxSinh(x *Node) *Node {
	return withFloat32ForHalfPrecision(x, func(x *Node) *Node {
		g := x.Graph()
		dtype := x.DType()
		thresholdNode := Scalar(g, dtype, 20.0)
		one := ScalarOne(g, dtype)
		a := Abs(x)
		isLarge := GreaterThan(a, thresholdNode)
		aSmall := Where(isLarge, thresholdNode, a) // NaNs are preserved.
		t := Expm1(aSmall)
		small := Mul(Scalar(g, dtype, 0.5), Add(t, Div(t, Add(t, one))))
		large := Exp(Sub(a, Scalar(g, dtype, math.Ln2)))
		return applyOddSymmetry(x, Where(isLarge, large, small))
	})
}

// onnxSoftplus implements the ONNX Softplus operator (log(exp(x) + 1)) in a numerically safe way:
//
//	softplus(x) = max(x, 0) + log1p(exp(-|x|))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Softplus.html
func onnxSoftplus(x *Node) *Node {
	zero := ScalarZero(x.Graph(), x.DType())
	return Add(Max(x, zero), Log1p(Exp(Neg(Abs(x)))))
}

// convertIsInf converts an ONNX IsInf node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__IsInf.html
func convertIsInf(node *protos.NodeProto, inputs []*Node) *Node {
	detectPositive := GetIntAttrOr(node, "detect_positive", 1) != 0
	detectNegative := GetIntAttrOr(node, "detect_negative", 1) != 0
	return onnxIsInf(inputs[0], detectPositive, detectNegative)
}

// onnxIsInf returns a boolean tensor that is true where x is +Inf (if detectPositive) or -Inf (if detectNegative).
// Non-float inputs can't hold infinities, so the result is all false for them.
func onnxIsInf(x *Node, detectPositive, detectNegative bool) *Node {
	g := x.Graph()
	if !x.DType().IsFloat() || (!detectPositive && !detectNegative) {
		// All false.
		return BroadcastToShape(Const(g, false), shapes.Make(dtypes.Bool, x.Shape().Dimensions...))
	}
	// Infinite is neither finite nor NaN.
	isInf := LogicalNot(LogicalOr(IsFinite(x), IsNaN(x)))
	zero := ScalarZero(g, x.DType())
	switch {
	case detectPositive && detectNegative:
		return isInf
	case detectPositive:
		return LogicalAnd(isInf, GreaterThan(x, zero))
	default:
		return LogicalAnd(isInf, LessThan(x, zero))
	}
}

////////////////////////////////////////////////////////////////////
//
// ScatterElements
//
////////////////////////////////////////////////////////////////////

// convertScatterElements converts an ONNX ScatterElements node to a GoMLX node.
//
// Attributes: axis (default 0, negative values count from the end) and reduction ("none" (default), "add", "mul",
// "max" or "min").
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ScatterElements.html
func convertScatterElements(node *protos.NodeProto, inputs []*Node) *Node {
	data, indices, updates := inputs[0], inputs[1], inputs[2]
	axis := GetIntAttrOr(node, "axis", 0)
	reduction := GetStringAttrOr(node, "reduction", "none")
	if data.Rank() < 1 {
		exceptions.Panicf("ScatterElements: data must have rank >= 1, got shape %s in %s", data.Shape(), NodeToString(node))
	}
	scatterAxis := MustAdjustAxis(axis, data)
	var output *Node
	err := exceptions.TryCatch[error](func() { output = onnxScatterElements(data, indices, updates, scatterAxis, reduction) })
	if err != nil {
		panic(errors.WithMessagef(err, "converting node %s", NodeToString(node)))
	}
	return output
}

// onnxScatterElements implements ONNX ScatterElements. scatterAxis must already be non-negative.
//
// For a 3D tensor and axis=1, it does:
//
//	output = copy(data)
//	output[i][indices[i][j][k]][k] = updates[i][j][k]  // or reduced with the previous value, see reduction.
//
// It mirrors onnxGatherElements: the full coordinates of each element of indices are built by concatenating
// the given indices (on scatterAxis) with Iota (on all other axes), and then a GoMLX scatter op is used.
func onnxScatterElements(data, indices, updates *Node, scatterAxis int, reduction string) *Node {
	rank := data.Rank()
	if indices.Rank() != rank {
		exceptions.Panicf("ScatterElements(data=%s, indices=%s, updates=%s): data and indices must have the same rank",
			data.Shape(), indices.Shape(), updates.Shape())
	}
	if !slices.Equal(indices.Shape().Dimensions, updates.Shape().Dimensions) {
		exceptions.Panicf("ScatterElements(data=%s, indices=%s, updates=%s): indices and updates must have the same dimensions",
			data.Shape(), indices.Shape(), updates.Shape())
	}
	if !indices.DType().IsInt() {
		exceptions.Panicf("ScatterElements: indices must be integers, got %s", indices.Shape())
	}
	if data.DType() != updates.DType() {
		exceptions.Panicf("ScatterElements: data (%s) and updates (%s) must have the same dtype", data.Shape(), updates.Shape())
	}
	for axis, dim := range indices.Shape().Dimensions {
		if axis != scatterAxis && dim > data.Shape().Dim(axis) {
			exceptions.Panicf("ScatterElements(data=%s, indices=%s, axis=%d): indices dimension on axis #%d is larger than data's",
				data.Shape(), indices.Shape(), scatterAxis, axis)
		}
	}
	if indices.Shape().Size() == 0 {
		// Nothing to scatter.
		return Identity(data)
	}
	if data.DType() == dtypes.Bool {
		// Backends don't support scatter on booleans: use Int32 (0/1) and convert back, which yields the
		// logical equivalents: max/add -> OR, min/mul -> AND.
		output := onnxScatterElements(ConvertDType(data, dtypes.Int32), indices, ConvertDType(updates, dtypes.Int32),
			scatterAxis, reduction)
		return ConvertDType(output, dtypes.Bool)
	}

	// Normalize negative indices: ONNX allows -1 for last element, -2 for second-to-last, etc.
	normalizedIndices := normalizeNegativeIndices(indices, data.Shape().Dim(scatterAxis))
	if reduction == "mul" {
		return onnxScatterElementsMul(data, normalizedIndices, updates, scatterAxis)
	}

	// fullIndices is shaped [indicesSize, rank]: the full coordinates in data for each element of indices.
	indicesSize := indices.Shape().Size()
	g := data.Graph()
	iotaShape := shapes.Make(indices.DType(), indices.Shape().Dimensions...)
	fullIndicesParts := make([]*Node, 0, rank)
	for axis := range rank {
		var part *Node
		if axis == scatterAxis {
			part = normalizedIndices
		} else {
			// On all other axes, the coordinates are the same in indices/updates and output.
			part = Iota(g, iotaShape, axis)
		}
		fullIndicesParts = append(fullIndicesParts, Reshape(part, indicesSize, 1))
	}
	fullIndices := Concatenate(fullIndicesParts, -1)
	flatUpdates := Reshape(updates, indicesSize)

	switch reduction {
	case "none", "":
		// ONNX leaves the result undefined for duplicate indices with reduction "none".
		return ScatterUpdate(data, fullIndices, flatUpdates, false, true)
	case "add":
		return ScatterSum(data, fullIndices, flatUpdates, false, false)
	case "max":
		return ScatterMax(data, fullIndices, flatUpdates, false, false)
	case "min":
		return ScatterMin(data, fullIndices, flatUpdates, false, false)
	default:
		exceptions.Panicf("ScatterElements: unknown reduction %q", reduction)
		return nil
	}
}

// onnxScatterElementsMul implements ScatterElements with reduction "mul".
//
// GoMLX has no ScatterMul, so the products are computed densely along the scatter axis: since ScatterElements
// only moves values along scatterAxis, for each position of the other axes the product over indices[..., k, ...]
// equal to j is computed with a one-hot mask of shape [<indices dims...>, data.Dim(scatterAxis)].
// This is exact (also for integers), and costs O(indices.Size() * data.Dim(scatterAxis)).
func onnxScatterElementsMul(data, normalizedIndices, updates *Node, scatterAxis int) *Node {
	g := data.Graph()
	rank := data.Rank()
	dtype := data.DType()
	axisDim := data.Shape().Dim(scatterAxis)
	indicesDims := normalizedIndices.Shape().Dimensions

	// One-hot along a new trailing axis: maskDims = [<indices dims...>, axisDim].
	maskDims := append(slices.Clone(indicesDims), axisDim)
	targets := BroadcastToDims(ExpandAxes(normalizedIndices, -1), maskDims...)
	positions := Iota(g, shapes.Make(normalizedIndices.DType(), maskDims...), rank)
	oneHot := Equal(targets, positions)
	factors := Where(oneHot, BroadcastToDims(ExpandAxes(updates, -1), maskDims...), ScalarOne(g, dtype))

	// Multiply over the indices' scatter axis: the result is [<indices dims without scatterAxis...>, axisDim].
	products := ReduceMultiply(factors, scatterAxis)

	// Move the trailing axis (axisDim) back to scatterAxis.
	permutation := make([]int, rank)
	for axis := range rank {
		switch {
		case axis < scatterAxis:
			permutation[axis] = axis
		case axis == scatterAxis:
			permutation[axis] = rank - 1
		default:
			permutation[axis] = axis - 1
		}
	}
	products = TransposeAllAxes(products, permutation...)

	// indices may be smaller than data on the non-scatter axes: pad with ones (no-op for the multiplication).
	padding := make([]compute.PadAxis, rank)
	needsPadding := false
	for axis := range rank {
		if diff := data.Shape().Dim(axis) - products.Shape().Dim(axis); diff > 0 {
			padding[axis].End = diff
			needsPadding = true
		}
	}
	if needsPadding {
		products = Pad(products, ScalarOne(g, dtype), padding...)
	}
	return Mul(data, products)
}

////////////////////////////////////////////////////////////////////
//
// Unique
//
////////////////////////////////////////////////////////////////////

// convertUnique converts an ONNX Unique node to GoMLX constants.
//
// The shape of Unique's outputs depends on the values of its input, and GoMLX graphs are statically shaped.
// So Unique is only supported if its input is a constant expression (it can be evaluated at graph-build time),
// e.g. if it depends only on constants and on the shapes of the inputs. This is the case, for instance,
// of group ids derived from the batch size.
//
// Outputs (all optional): Y, indices, inverse_indices and counts. The last 3 are Int64.
// Attributes: sorted (default 1) and axis (optional: if not given, the input is flattened).
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Unique.html
func convertUnique(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	g := inputs[0].Graph()
	x, err := m.materializeConstantExpression(node.Input[0], convertedOutputs)
	if err != nil {
		exceptions.Panicf("Unique is only supported for inputs that can be evaluated at graph-build time "+
			"(its output shape depends on the input values, and GoMLX graphs are statically shaped); "+
			"failed to materialize the input of %s: %v", NodeToString(node), err)
	}
	sorted := GetIntAttrOr(node, "sorted", 1) != 0
	axis := -1 // Flatten.
	if axisAttr := GetNodeAttr(node, "axis", false); axisAttr != nil {
		assertNodeAttrType(node, axisAttr, protos.AttributeProto_INT)
		axis = int(axisAttr.I)
		rank := x.Shape().Rank()
		if axis < 0 {
			axis += rank
		}
		if axis < 0 || axis >= rank {
			exceptions.Panicf("Unique: axis=%d out of range for input shape %s in %s", axisAttr.I, x.Shape(), NodeToString(node))
		}
	}
	y, indices, inverseIndices, counts, err := computeUnique(x, axis, sorted)
	if err != nil {
		panic(errors.WithMessagef(err, "converting node %s", NodeToString(node)))
	}
	outputs := []*Node{
		Const(g, y),
		Const(g, indices),
		Const(g, inverseIndices),
		Const(g, counts),
	}
	for ii, name := range node.Output {
		if name == "" || ii >= len(outputs) {
			continue
		}
		convertedOutputs[name] = outputs[ii]
	}
	return outputs[0]
}

// computeUnique implements ONNX Unique on a materialized tensor.
//
// If axis < 0, the tensor is flattened and unique elements are returned. Otherwise, it returns the unique
// sub-tensors (slices) along axis, compared lexicographically in row-major order (same as numpy.unique(axis=...)).
//
// If sorted is true, the unique values are sorted in ascending order. Otherwise, they are returned in the order
// of their first occurrence. Floats are compared with cmp.Compare: -0 == +0, and all NaNs are equal.
//
// It returns the unique values (y) and the Int64 tensors indices (first occurrence of each unique value),
// inverseIndices (for each element/slice of x, the index of its value in y) and counts.
func computeUnique(x *tensors.Tensor, axis int, sorted bool) (y, indices, inverseIndices, counts *tensors.Tensor, err error) {
	xShape := x.Shape()
	var pre, n, post int
	var yDims []int
	if axis < 0 {
		pre, n, post = 1, xShape.Size(), 1
	} else {
		pre, n, post = 1, xShape.Dim(axis), 1
		for ii, dim := range xShape.Dimensions {
			if ii < axis {
				pre *= dim
			} else if ii > axis {
				post *= dim
			}
		}
		yDims = slices.Clone(xShape.Dimensions)
	}

	var firstOccurrence, groupCounts, inverse []int64
	var innerErr error
	err = x.ConstFlatData(func(flat any) {
		compareElements, cmpErr := flatElementsComparator(flat)
		if cmpErr != nil {
			innerErr = cmpErr
			return
		}
		compareSlices := func(k0, k1 int) int {
			for p := range pre {
				base0, base1 := (p*n+k0)*post, (p*n+k1)*post
				for q := range post {
					if c := compareElements(base0+q, base1+q); c != 0 {
						return c
					}
				}
			}
			return 0
		}

		// A stable sort keeps the first occurrence first among equal slices.
		order := make([]int, n)
		for ii := range order {
			order[ii] = ii
		}
		slices.SortStableFunc(order, compareSlices)

		// Group equal consecutive slices: groupMembers[i] lists the original positions of the i-th unique slice.
		var groupMembers [][]int
		for ii, k := range order {
			if ii == 0 || compareSlices(order[ii-1], k) != 0 {
				groupMembers = append(groupMembers, []int{k})
			} else {
				groupMembers[len(groupMembers)-1] = append(groupMembers[len(groupMembers)-1], k)
			}
		}
		if !sorted {
			// Order of first occurrence.
			slices.SortFunc(groupMembers, func(a, b []int) int { return cmp.Compare(a[0], b[0]) })
		}

		numUnique := len(groupMembers)
		firstOccurrence = make([]int64, numUnique)
		groupCounts = make([]int64, numUnique)
		inverse = make([]int64, n)
		for groupIdx, members := range groupMembers {
			firstOccurrence[groupIdx] = int64(members[0])
			groupCounts[groupIdx] = int64(len(members))
			for _, k := range members {
				inverse[k] = int64(groupIdx)
			}
		}

		// Build y by copying the first occurrence of each unique slice.
		if axis < 0 {
			yDims = []int{numUnique}
		} else {
			yDims[axis] = numUnique
		}
		y = tensors.FromShape(shapes.Make(xShape.DType, yDims...))
		src := reflect.ValueOf(flat)
		innerErr = y.MutableFlatData(func(yFlat any) {
			dst := reflect.ValueOf(yFlat)
			for p := range pre {
				for groupIdx, first := range firstOccurrence {
					srcBase := (p*n + int(first)) * post
					dstBase := (p*numUnique + groupIdx) * post
					for q := range post {
						dst.Index(dstBase + q).Set(src.Index(srcBase + q))
					}
				}
			}
		})
	})
	if err == nil {
		err = innerErr
	}
	if err != nil {
		return nil, nil, nil, nil, errors.WithMessagef(err, "Unique of tensor shaped %s", xShape)
	}
	indices = tensors.FromFlatDataAndDimensions(firstOccurrence, len(firstOccurrence))
	inverseIndices = tensors.FromFlatDataAndDimensions(inverse, len(inverse))
	counts = tensors.FromFlatDataAndDimensions(groupCounts, len(groupCounts))
	return
}

// flatElementsComparator returns a function that compares two elements (given by their flat indices) of
// the flat slice of values of a tensor.
func flatElementsComparator(flat any) (func(i, j int) int, error) {
	switch values := flat.(type) {
	case []float32:
		return orderedComparator(values), nil
	case []float64:
		return orderedComparator(values), nil
	case []int8:
		return orderedComparator(values), nil
	case []int16:
		return orderedComparator(values), nil
	case []int32:
		return orderedComparator(values), nil
	case []int64:
		return orderedComparator(values), nil
	case []uint8:
		return orderedComparator(values), nil
	case []uint16:
		return orderedComparator(values), nil
	case []uint32:
		return orderedComparator(values), nil
	case []uint64:
		return orderedComparator(values), nil
	case []bool:
		return func(i, j int) int {
			if values[i] == values[j] {
				return 0
			}
			if !values[i] {
				return -1
			}
			return 1
		}, nil
	case []float16.Float16:
		return func(i, j int) int { return cmp.Compare(values[i].Float32(), values[j].Float32()) }, nil
	case []bfloat16.BFloat16:
		return func(i, j int) int { return cmp.Compare(values[i].Float32(), values[j].Float32()) }, nil
	default:
		return nil, errors.Errorf("Unique: unsupported tensor values type %T", flat)
	}
}

func orderedComparator[T cmp.Ordered](values []T) func(i, j int) int {
	return func(i, j int) int { return cmp.Compare(values[i], values[j]) }
}
