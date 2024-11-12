package onnx

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"github.com/pkg/errors"
	"reflect"
	"slices"
)

// This file implements the ONNX operators that don't have a direct corresponding GoMLX operator.

// gomlxBinaryOp is a GoMLX binary op. Used by convertBinaryOp.
type gomlxBinaryOp func(lhs, rhs *Node) *Node

// onnxImplicitExpansion expands operands to the largest rank, expanding to the left.
// This is part of ONNX implicit broadcasting rule.
// Scalars are left untouched, because generally, XLA will broadcast them.
//
// Returns the list of broadcast operands.
func onnxImplicitBroadcast(operands []*Node) []*Node {
	ranks := sliceMap(operands, func(n *Node) int { return n.Rank() })
	maxRank := slices.Max(ranks)
	return sliceMap(operands, func(n *Node) *Node {
		if n.IsScalar() || n.Rank() == maxRank {
			return n
		}
		return ExpandLeftToRank(n, maxRank)
	})
}

// convertBinaryOp applies ONNX broadcasting rule before calling the fn.
//
// It differs from GoMLX and XLA in that it automatically prepend 1-dimensional axes to
// any of the operands, if they differ in rank.
func convertBinaryOp(fn gomlxBinaryOp, lhs, rhs *Node) *Node {
	operands := onnxImplicitBroadcast([]*Node{lhs, rhs})
	return fn(operands[0], operands[1])
}

// convertClip converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Clip.html
//
// Notice max/min values are optional, hence the special conversion code.
func convertClip(_ *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) == 1 {
		return inputs[0]
	}
	if len(inputs) == 2 {
		return Max(inputs[0], inputs[1])
	}
	return Min(inputs[2], Max(inputs[0], inputs[1]))
}

// convertWhere converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Where.html
//
// Notice broadcast rules for ONNX are difference, hence the special conversion code.
func convertWhere(node *protos.NodeProto, inputs []*Node) *Node {
	var output *Node
	err := exceptions.TryCatch[error](func() { output = onnxWhere(inputs) })
	if err != nil {
		panic(errors.WithMessagef(err, "converting node %s", node))
	}
	return output
}

// onnxWhere implements ONNX implicit broadcasting rules.
// inputs is a tuple with (cond, onTrue, onFalse) values.
func onnxWhere(inputs []*Node) *Node {
	// Broadcast according to ONNX rules.
	inputs = onnxImplicitBroadcast(inputs)
	ranks := sliceMap(inputs, func(n *Node) int { return n.Rank() })
	maxRank := slices.Max(ranks)
	maxDims := make([]int, maxRank)
	for axis := range maxRank {
		allDims := sliceMap(inputs, func(n *Node) int {
			if n.IsScalar() {
				return 1
			}
			return n.Shape().Dim(axis)
		})
		maxDims[axis] = slices.Max(allDims)
	}
	for ii, input := range inputs {
		if !input.IsScalar() {
			inputs[ii] = BroadcastToDims(input, maxDims...)
		}
	}

	// Now we can use GoMLX Where:
	cond, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	return Where(cond, onTrue, onFalse)
}

////////////////////////////////////////////////////////////////////
//
// Ops that take attributes as static inputs.
//
////////////////////////////////////////////////////////////////////

// getNodeAttr returns the given node attribute. If required is true, it will panic with a message about
// the missing attribute.
func getNodeAttr(node *protos.NodeProto, name string, required bool) *protos.AttributeProto {
	for _, attr := range node.Attribute {
		if attr.Name == name {
			return attr
		}
	}
	if required {
		exceptions.Panicf("ONNX %s is missing required attribute %q", nodeToString(node), name)
	}
	return nil
}

func assertNodeAttrType(node *protos.NodeProto, attr *protos.AttributeProto, attributeType protos.AttributeProto_AttributeType) {
	if attr.Type != attributeType {
		exceptions.Panicf("unsupported ONNX attribute %q of type %q in %s", attr.Name, attr.Type, nodeToString(node))
	}
}

// mustGetIntAttr get the attribute as an integer.
// It panics with an exception if attribute is not set or if it is of the wrong type.
func mustGetIntAttr(node *protos.NodeProto, attrName string) int {
	attr := getNodeAttr(node, attrName, true)
	assertNodeAttrType(node, attr, protos.AttributeProto_INT)
	return int(attr.I)
}

// mustGetIntsAttr gets a list of integers attribute for node.
// It panics with an error message if the attribute is not present of if it is of the wrong type.
func mustGetIntsAttr(node *protos.NodeProto, attrName string) []int {
	attr := getNodeAttr(node, attrName, true)
	if attr.Type == protos.AttributeProto_INT {
		return []int{int(attr.I)}
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_INTS)
	return sliceMap(attr.Ints, func(i int64) int { return int(i) })
}

// getIntAttrOr gets an integer attribute for node if present or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getIntAttrOr(node *protos.NodeProto, attrName string, defaultValue int) int {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValue
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_INT)
	return int(attr.I)
}

// getBoolAttrOr gets a boolean attribute (ONNX uses an int value of 0 or 1) for node if present or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getBoolAttrOr(node *protos.NodeProto, attrName string, defaultValue bool) bool {
	defaultInt := 0
	if defaultValue {
		defaultInt = 1
	}
	intValue := getIntAttrOr(node, attrName, defaultInt)
	return intValue != 0
}

// getFloatAttrOr gets an integer attribute for node if present or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getFloatAttrOr(node *protos.NodeProto, attrName string, defaultValue float32) float32 {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValue
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_FLOAT)
	return attr.F
}

// getIntsAttrOr gets an integer list attribute for node if present or return the given defaultValues.
// It panics with an error message if the attribute is present but is of the wrong type.
func getIntsAttrOr(node *protos.NodeProto, attrName string, defaultValues []int) []int {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValues
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_INTS)
	return sliceMap(attr.Ints, func(i int64) int { return int(i) })
}

// convertConstant converts a ONNX node to a GoMLX node.
func convertConstant(node *protos.NodeProto, g *Graph) *Node {
	valueAttr := getNodeAttr(node, "value", true)
	assertNodeAttrType(node, valueAttr, protos.AttributeProto_TENSOR)
	tensor, err := togomlx.Tensor(valueAttr.T)
	if err != nil {
		err = errors.WithMessagef(err, "while converting ONNX %s", nodeToString(node))
		panic(err)
	}
	return Const(g, tensor)
}

// convertGather converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Gather.html
func convertGather(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", 0)
	gatherAxis := AdjustAxisToOperandRank(inputs[0], axis)
	if gatherAxis >= inputs[0].Rank() || gatherAxis < 0 {
		exceptions.Panicf("Gather(data, indices, axis=%d), axis within d.Rank()=%d range", axis, inputs[0].Rank())
	}
	return onnxGather(inputs[0], inputs[1], gatherAxis)
}

func onnxGather(data, indices *Node, gatherAxis int) *Node {
	expandedIndices := ExpandAxes(indices, -1)
	if gatherAxis == 0 {
		// Trivial case, like GoMLX version.
		return Gather(data, expandedIndices)
	}

	// We want to transpose data, such that we can gather on the first axis.
	axesPermutation := make([]int, data.Rank())
	for axis := range axesPermutation {
		if axis == 0 {
			// The first axis will be the one we are gathering on.
			axesPermutation[axis] = gatherAxis
		} else if axis <= gatherAxis {
			// These axes have been shifted to the right, to give space for the gatherAxis
			axesPermutation[axis] = axis - 1
		} else {
			// The tail axes remain the same.
			axesPermutation[axis] = axis
		}
	}
	transposedData := TransposeAllDims(data, axesPermutation...)
	transposed := Gather(transposedData, expandedIndices)

	// Now we have to transpose back the result.
	// transposed is shaped [<indices_dims...>, <data_dims...>] and we want to transpose to
	// [<data_prefix_dims...>, <indices_dims...>, <data_suffix_dims...>], where data_prefix_dims and
	// data_suffix_dims is divided by the gatherAxis.
	axesPermutation = make([]int, transposed.Rank())
	for axis := range axesPermutation {
		if axis < gatherAxis {
			// data_prefix_dims:
			axesPermutation[axis] = indices.Rank() + axis
		} else if axis < gatherAxis+indices.Rank() {
			// indices_dims
			axesPermutation[axis] = axis - gatherAxis
		} else {
			// data_suffix_dims, which don't change from the transposed results.
			axesPermutation[axis] = axis
		}
	}
	return TransposeAllDims(transposed, axesPermutation...)
}

// convertGatherElements converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GatherElements.html
func convertGatherElements(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", 0)
	gatherAxis := AdjustAxisToOperandRank(inputs[0], axis)
	if gatherAxis >= inputs[0].Rank() || gatherAxis < 0 {
		exceptions.Panicf("Gather(data, indices, axis=%d), axis within d.Rank()=%d range", axis, inputs[0].Rank())
	}
	if inputs[0].Rank() != inputs[1].Rank() {
		exceptions.Panicf("Gather(data=%s, indices=%s, axis=%d): data and indices must have the same rank", inputs[0].Shape(), inputs[1].Shape(), axis)
	}
	var output *Node
	err := exceptions.TryCatch[error](func() { output = onnxGatherElements(inputs[0], inputs[1], gatherAxis) })
	if err != nil {
		panic(errors.WithMessagef(err, "converting node %s", node))
	}
	return output
}

func onnxGatherElements(data *Node, indices *Node, gatherAxis int) *Node {
	indicesDims := indices.Shape().Dimensions
	indicesSize := indices.Shape().Size()
	for axis, dim := range indicesDims {
		if axis != gatherAxis && dim != data.Shape().Dim(axis) {
			exceptions.Panicf("Gather(data=%s, indices=%s, gatherAxis=%d): data and indices must have the same shape except on the gather axis, but axis #%d are different", data.Shape(), indices.Shape(), gatherAxis, axis)
		}
	}

	// fullIndicesParts is a slice with one value per axis of the data to gather.
	// Each part will be shaped [indicesSize, 1], and it will eventually be concatenated
	// to shape [indicesSize, <data.Rank()>].
	fullIndicesParts := make([]*Node, 0, data.Rank())
	iotaShape := indices.Shape().Clone()
	iotaShape.Dimensions = append(iotaShape.Dimensions, 1)
	g := data.Graph()
	for axis := range data.Rank() {
		var part *Node
		if axis == gatherAxis {
			// On the gatherAxis, the index is the one given by the caller.
			part = Reshape(indices, indicesSize, 1)
		} else {
			// On all axes that we are not gathering, the indices are the same in input and output.
			part = Iota(g, iotaShape, axis)
			part = Reshape(part, indicesSize, 1)
		}
		fullIndicesParts = append(fullIndicesParts, part)
	}
	fullIndices := Concatenate(fullIndicesParts, -1)
	output := Reshape(Gather(data, fullIndices), indicesDims...)
	return output
}

// convertShape converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Shape.html
func convertShape(node *protos.NodeProto, inputs []*Node) *Node {
	shape := inputs[0].Shape()
	start := getIntAttrOr(node, "start", 0)
	if start < 0 {
		start = shape.Rank() + start
	}
	end := getIntAttrOr(node, "end", 0)
	if end == 0 {
		end = shape.Rank()
	} else if end < 0 {
		end = shape.Rank() + end
	}
	dims := sliceMap(shape.Dimensions[start:end], func(dim int) int64 { return int64(dim) })
	g := inputs[0].Graph()
	return Const(g, dims)
}

// convertConcat converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Concat.html
func convertConcat(node *protos.NodeProto, inputs []*Node) *Node {
	axis := mustGetIntAttr(node, "axis")
	return Concatenate(inputs, axis)
}

// convertSoftmax converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Softmax.html
func convertSoftmax(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", -1)
	return Softmax(inputs[0], axis)
}

// convertCast converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Cast.html
func convertCast(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]

	saturate := getIntAttrOr(node, "saturate", 1) > 0
	_ = saturate // Not implemented.
	toDtype, err := togomlx.DType(
		protos.TensorProto_DataType(
			mustGetIntAttr(node, "to")))
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'to' attribute for node %s", nodeToString(node)))
	}

	return ConvertDType(operand, toDtype)
}

// convertTranspose converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Transpose.html
func convertTranspose(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	permutations := getIntsAttrOr(node, "perm", nil)
	if permutations == nil {
		// Reverse axes.
		permutations = make([]int, operand.Rank())
		for axis := range permutations {
			permutations[axis] = operand.Rank() - axis - 1
		}
	}
	if len(permutations) != operand.Rank() {
		exceptions.Panicf("Tranpose(data=%s, perm=%v) must have one permutation value per axis of the data: %s", operand.Shape(), permutations, nodeToString(node))
	}
	return TransposeAllDims(operand, permutations...)
}

// convertGemm converts a ONNX node to a GoMLX node.
// Gemm stands for general matrix multiplication.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Gemm.html
func convertGemm(node *protos.NodeProto, inputs []*Node) *Node {
	operandA := inputs[0]
	operandB := inputs[1]

	transposeA := getBoolAttrOr(node, "transA", false)
	transposeB := getBoolAttrOr(node, "transB", false)
	alpha := getFloatAttrOr(node, "alpha", 1.0)
	beta := getFloatAttrOr(node, "alpha", 1.0)

	aAxes, bAxes := "ij", "jk"
	if transposeA {
		aAxes = "ji"
	}
	if transposeB {
		bAxes = "kj"
	}
	equation := fmt.Sprintf("%s,%s->ik", aAxes, bAxes)
	result := Einsum(equation, operandA, operandB)
	if alpha != 1.0 {
		result = MulScalar(result, alpha)
	}

	// Include the C term if given.
	if len(inputs) > 2 {
		operandC := inputs[2]
		if beta != 1.0 {
			operandC = MulScalar(operandC, beta)
		}
		// Add with ONNX broadcast semantics.
		result = convertBinaryOp(Add, result, operandC)
	}
	return result
}

////////////////////////////////////////////////////////////////////
//
// Ops that require materialization of constant sub-expressions
//
////////////////////////////////////////////////////////////////////

func tensorToInts(t *tensors.Tensor) []int {
	res := make([]int, t.Size())
	intType := reflect.TypeOf(int(0))
	t.ConstFlatData(func(flat any) {
		valueOf := reflect.ValueOf(flat)
		for ii := range valueOf.Len() {
			elemV := valueOf.Index(ii)
			res[ii] = elemV.Convert(intType).Interface().(int)
		}
	})
	return res
}

// convertSqueeze converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Squeeze.html
func convertSqueeze(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]

	// Version 11 and earlier take the axes from the attribute:
	axes := getIntsAttrOr(node, "axes", nil)
	if len(axes) == 0 && len(inputs) >= 2 {
		// Instead take axes from inputs[1].
		if !inputs[1].DType().IsInt() {
			exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
		}
		axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}
	if len(axes) == 0 {
		// If axes is not given, pick all axes that have dimension == 1.
		for axis, dim := range operand.Shape().Dimensions {
			if dim == 1 {
				axes = append(axes, axis)
			}
		}
	}
	return Squeeze(operand, axes...)
}

// convertUnsqueeze converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
func convertUnsqueeze(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// Version 11 and earlier take the axes from the attribute:
	axes := getIntsAttrOr(node, "axes", nil)
	if len(axes) == 0 {
		// Instead take axes from inputs[1].
		if !inputs[1].DType().IsInt() {
			exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
		}
		axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}
	return ExpandAxes(inputs[0], axes...)
}

// convertSlice converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Slice.html
func convertSlice(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 {
		exceptions.Panicf("Slice requires at least 3 inputs, got %d in node %s", len(inputs), nodeToString(node))
	}

	operand := inputs[0]

	startsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'starts' for node %s", nodeToString(node)))
	}
	starts := tensorToInts(startsT)

	endsT, err := m.materializeConstantExpression(node.Input[2], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'ends' for node %s", nodeToString(node)))
	}
	ends := tensorToInts(endsT)

	var axes []int
	if len(inputs) > 3 {
		axesT, err := m.materializeConstantExpression(node.Input[3], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}

	var strides []int
	if len(inputs) > 4 {
		stridesT, err := m.materializeConstantExpression(node.Input[4], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'strides' for node %s", nodeToString(node)))
		}
		strides = tensorToInts(stridesT)
	}

	specs := make([]SliceAxisSpec, operand.Rank())
	numAxesToDefine := operand.Rank()
	if len(axes) != 0 {
		// Define only given axes, and slice the other axes as full range.
		for ii := range specs {
			specs[ii] = AxisRange() // Full range.
		}
		numAxesToDefine = len(axes)
	}
	for ii := range numAxesToDefine {
		axis := ii
		if len(axes) != 0 {
			axis = axes[ii]
		}
		// ONNX often uses INT64_MAX as end value.
		endValue := min(ends[ii], operand.Shape().Dim(axis))
		specs[axis] = AxisRange(starts[ii], endValue)
		if len(strides) != 0 {
			specs[axis] = specs[axis].Stride(strides[ii])
		}
	}
	return Slice(operand, specs...)
}

// convertReshape converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Reshape.html
func convertReshape(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	if !inputs[1].DType().IsInt() {
		exceptions.Panicf("shape must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
	}
	allowZero := getIntAttrOr(node, "allowZero", 0)

	dimsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'shape' for node %s", nodeToString(node)))
	}
	dims := tensorToInts(dimsT)
	if allowZero == 0 {
		// If new shape dim is 0, copy over from previous shape.
		for newAxis, dim := range dims {
			if dim == 0 && newAxis < operand.Rank() {
				dims[newAxis] = operand.Shape().Dim(newAxis) // Copy over dimension from previous shape.
			}
		}
	}
	return Reshape(inputs[0], dims...)
}

// convertReduceMean converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
func convertReduceMean(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	var axes []int
	if len(inputs) > 1 {
		if !inputs[1].DType().IsInt() {
			exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
		}

		axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}

	axesFromAttr := getIntsAttrOr(node, "axes", nil)
	if len(axesFromAttr) > 0 {
		if len(axes) > 0 {
			exceptions.Panicf("ReduceMean(operand, [axes]): axes and axes attribute cannot be used together for node %s", nodeToString(node))
		}
		axes = axesFromAttr
	}

	// If there are no axes to reduce, this is a no-op.
	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := ReduceAllMean(operand)
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	if !keepDims {
		return ReduceMean(operand, axes...)
	} else {
		return ReduceAndKeep(operand, ReduceMean, axes...)
	}
}

// convertConstantOfShape converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
func convertConstantOfShape(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	g := inputs[0].Graph()

	valueAttr := getNodeAttr(node, "value", true)
	assertNodeAttrType(node, valueAttr, protos.AttributeProto_TENSOR)
	tensor, err := togomlx.Tensor(valueAttr.T)
	if err != nil {
		err = errors.WithMessagef(err, "while converting ONNX %s", nodeToString(node))
		panic(err)
	}
	valueN := Const(g, tensor)

	dimsN := inputs[0]
	if !dimsN.DType().IsInt() {
		exceptions.Panicf("input (shape) must be integer, got %s for node %s", dimsN.DType(), nodeToString(node))
	}

	var dims []int // Default is a scalar.
	if dimsN.Shape().Size() > 0 {
		dimsT, err := m.materializeConstantExpression(node.Input[0], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'shape' to a static value for node %s", nodeToString(node)))
		}
		dims = tensorToInts(dimsT)
	}

	return BroadcastToDims(valueN, dims...)
}

// convertExpand converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Expand.html
func convertExpand(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	dimsN := inputs[1]
	if !dimsN.DType().IsInt() {
		exceptions.Panicf("input (shape) must be integer, got %s for node %s", dimsN.DType(), nodeToString(node))
	}
	var dims []int // Default is a scalar.
	if dimsN.Shape().Size() > 0 {
		dimsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'shape' to a static value for node %s", nodeToString(node)))
		}
		dims = tensorToInts(dimsT)
	}

	// Trivial cases first:
	if len(dims) == 0 {
		return operand
	}
	if operand.IsScalar() {
		return BroadcastToDims(operand, dims...)
	}

	// Reproduce multi-dimension broadcasting rule:
	if len(dims) > operand.Rank() {
		// Prepend 1-dimensional axes to match the target dims.
		operand = ExpandLeftToRank(operand, len(dims))
	} else if len(dims) < operand.Rank() {
		// Prepend 1-dimensional axes to match original operand rank.
		newDims := make([]int, 0, operand.Rank())
		for range operand.Rank() - len(dims) {
			newDims = append(newDims, 1)
		}
		newDims = append(newDims, dims...)
		dims = newDims
	}
	// Convert dimensions equal to 1 to whatever the original operand has.
	for ii, dim := range dims {
		if dim == 1 {
			dims[ii] = operand.Shape().Dim(ii)
		}
	}
	return BroadcastToDims(operand, dims...)
}

// convertTile converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Tile.html
func convertTile(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	repeatsN := inputs[1]
	if !repeatsN.DType().IsInt() {
		exceptions.Panicf("Tile(input, repeats): repeats (shape) must be integer, got %s for node %s", repeatsN.DType(), nodeToString(node))
	}
	repeatsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'repeats' to a static value for node %s", nodeToString(node)))
	}
	repeats := tensorToInts(repeatsT)
	return onnxTile(operand, repeats)
}

func onnxTile(operand *Node, repeats []int) *Node {
	if len(repeats) != operand.Rank() {
		exceptions.Panicf("Tile(input, repeats) must have len(repeats) == input.Rank(), but input.Rank()=%d, and len(repeats)=%d", operand.Rank(), len(repeats))
	}
	for _, r := range repeats {
		if r < 1 {
			exceptions.Panicf("Tile(input, repeats) must have repeats >= 1, got %v instead", repeats)
		}
	}

	// Insert new axes to be broadcast (repeated).
	insertAxes := make([]int, len(repeats))
	for ii := range insertAxes {
		insertAxes[ii] = ii
	}
	output := InsertAxes(operand, insertAxes...)

	// Broadcast with repeats in interleaved inserted dimensions.
	newShape := output.Shape().Clone()
	for ii := 0; ii < newShape.Rank(); ii += 2 {
		newShape.Dimensions[ii] = repeats[ii/2]
	}
	output = BroadcastToDims(output, newShape.Dimensions...)

	// Merge inserted dimensions to get he tiling.
	newShape = operand.Shape().Clone()
	for axis := range newShape.Dimensions {
		newShape.Dimensions[axis] *= repeats[axis]
	}
	output = Reshape(output, newShape.Dimensions...)
	return output
}

// convertTile converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Range.html
func convertRange(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	startN, limitN, deltaN := inputs[0], inputs[1], inputs[2]
	if startN.DType() != limitN.DType() || deltaN.DType() != limitN.DType() ||
		!startN.IsScalar() || !limitN.IsScalar() || !deltaN.IsScalar() {
		exceptions.Panicf("Range(scalar, limit, delta) all operands must have same scalar dtypes, got %s, %s, %s instead",
			startN.Shape(), limitN.Shape(), deltaN.Shape())
	}
	startT, err := m.materializeConstantExpression(node.Input[0], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'start' to a static value for node %s", nodeToString(node)))
	}
	limitT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'limit' to a static value for node %s", nodeToString(node)))
	}
	deltaT, err := m.materializeConstantExpression(node.Input[2], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'delta' to a static value for node %s", nodeToString(node)))
	}

	// Find the number of elements:
	count := rangeCount(startN.Graph().Backend(), startT, limitT, deltaT)
	g := startN.Graph()
	dtype := startN.DType()

	// Range is the iota, scaled by delta and shifted by start.
	output := Iota(g, shapes.Make(dtype, count), 0)
	output = Add(Mul(output, deltaN), startN)
	return output
}

// isUnsigned returns whether the dtype is unsigned.
// TODO: after gopjrt > v.0.4.5 is released, used DType.IsUnsigned instead.
func isUnsigned(dtype dtypes.DType) bool {
	return dtype == dtypes.Uint8 || dtype == dtypes.Uint16 || dtype == dtypes.Uint32 || dtype == dtypes.Uint64
}

func rangeCount(backend backends.Backend, start, limit, delta *tensors.Tensor) int {
	count := ExecOnce(backend, func(start, limit, delta *Node) *Node {
		amount := Sub(limit, start)
		var count *Node
		if start.DType().IsFloat() {
			// Float rounding up.
			count = Ceil(Div(amount, delta))
		} else {
			// Int rounding up.
			var roundUp *Node
			if isUnsigned(delta.DType()) {
				roundUp = AddScalar(delta, -1)
			} else {
				roundUp = Add(delta, Neg(Sign(delta))) // -1 if delta is positive, +1 if delta is negative.
			}
			amount = Add(amount, roundUp)
			count = Div(amount, delta)
		}
		return ConvertDType(count, dtypes.Int64)
	}, start, limit, delta)
	return int(tensors.ToScalar[int64](count))
}

// convertCumSum converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__CumSum.html
func convertCumSum(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	exclusiveAttr := getBoolAttrOr(node, "exclusive", false)
	reverseAttr := getBoolAttrOr(node, "reverse", false)

	axisN := inputs[1]
	if !axisN.DType().IsInt() || !axisN.IsScalar() {
		exceptions.Panicf("axis (shape) must be a scalar integer, got %s for node %s", axisN.Shape(), nodeToString(node))
	}
	axisT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'axis' to a static value for node %s", nodeToString(node)))
	}
	axis := tensorToInts(axisT)[0]
	return onnxCumSum(operand, axis, exclusiveAttr, reverseAttr)
}

// onnxCumSum adds "exclusive" and "reverse" options to the normal CumSum.
// TODO: reimplement exclusive/reverse by changing original CumSum implementation: it will be much more efficient.
func onnxCumSum(operand *Node, axis int, exclusive, reverse bool) *Node {
	adjustedAxis := AdjustAxisToOperandRank(operand, axis)
	if reverse {
		operand = Reverse(operand, adjustedAxis)
	}
	output := CumSum(operand, axis)
	if exclusive {
		output = ShiftWithScalar(output, adjustedAxis, ShiftDirRight, 1, 0)
	}
	if reverse {
		output = Reverse(output, adjustedAxis)
	}
	return output
}
