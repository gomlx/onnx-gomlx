package onnx

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/layers/lstm"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// This file implements the ONNX operators that don't have a direct corresponding GoMLX operator.

// gomlxBinaryOp is a GoMLX binary op. Used by convertBinaryOp.
type gomlxBinaryOp func(lhs, rhs *Node) *Node

// onnxImplicitExpansion expands operands to the largest rank, expanding to the left.
// This is part of ONNX implicit broadcasting rule.
// Scalars are left untouched, because generally, XLA will broadcast them.
//
// Returns the list of broadcast operands.
func onnxImplicitExpansion(operands []*Node) []*Node {
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
	operands := onnxImplicitExpansion([]*Node{lhs, rhs})
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
	inputs = onnxImplicitExpansion(inputs)
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
		if !input.IsScalar() && !slices.Equal(input.Shape().Dimensions, maxDims) {
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

// getDTypeAttrOr gets a int attribute for node if present and convert to a GoMLX dtype, or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getDTypeAttrOr(node *protos.NodeProto, attrName string, defaultValue dtypes.DType) dtypes.DType {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValue
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_INT)
	onnxDType := protos.TensorProto_DataType(int32(attr.I))
	dtype, err := dtypeForONNX(onnxDType)
	if err != nil {
		exceptions.Panicf("unsupported ONNX data type %q for attribute %q in %s", onnxDType, attrName, nodeToString(node))
	}
	return dtype
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

// getFloatAttrOr gets a float attribute for node if present or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getFloatAttrOr(node *protos.NodeProto, attrName string, defaultValue float32) float32 {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValue
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_FLOAT)
	return attr.F
}

// getStringAttrOr gets a string attribute for node if present or return the given defaultValue.
// It panics with an error message if the attribute is present but is of the wrong type.
func getStringAttrOr(node *protos.NodeProto, attrName string, defaultValue string) string {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValue
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_STRING)
	return string(attr.S)
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

// getFloatsAttrOr gets a float list attribute for node if present or return the given defaultValues.
// It panics with an error message if the attribute is present but is of the wrong type.
func getFloatsAttrOr(node *protos.NodeProto, attrName string, defaultValues []float32) []float32 {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValues
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_FLOATS)
	return attr.Floats
}

// getStringsAttrOr gets a string list attribute for node if present or return the given defaultValues.
// It panics with an error message if the attribute is present but is of the wrong type.
func getStringsAttrOr(node *protos.NodeProto, attrName string, defaultValues []string) []string {
	attr := getNodeAttr(node, attrName, false)
	if attr == nil {
		return defaultValues
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_STRINGS)
	return sliceMap(attr.Strings, func(v []byte) string { return string(v) })
}

// convertConstant converts a ONNX node to a GoMLX node.
func convertConstant(m *Model, node *protos.NodeProto, g *Graph) *Node {
	valueAttr := getNodeAttr(node, "value", true)
	if valueAttr == nil {
		panic(errors.Errorf("'value' attribute for ONNX node %s is nil!?", nodeToString(node)))
	}
	assertNodeAttrType(node, valueAttr, protos.AttributeProto_TENSOR)
	if valueAttr.T == nil {
		panic(errors.Errorf("TENSOR attribute for ONNX node %s is nil!?", nodeToString(node)))
	}
	tensor, err := tensorToGoMLX(m.backend, valueAttr.T)
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

// convertFlatten converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Flatten.html
func convertFlatten(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	splitAxis := getIntAttrOr(node, "axis", 1)
	splitAxis = AdjustAxisToOperandRank(operand, splitAxis)
	return onnxFlatten(operand, splitAxis)
}

// onnxFlatten implements the corresponding ONNX operation.
func onnxFlatten(operand *Node, splitAxis int) *Node {
	outerDim, innerDim := 1, 1
	for axis, dim := range operand.Shape().Dimensions {
		if axis < splitAxis {
			outerDim *= dim
		} else {
			innerDim *= dim
		}
	}
	return Reshape(operand, outerDim, innerDim)
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
	toDtype, err := dtypeForONNX(
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

// tensorToInts converts elements of the tensor to a slice of ints.
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

// convertPow, with special casing if the exponential is a known constant.
func convertPow(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// defaultPow returns the generic Pow function:
	defaultPow := func() *Node {
		operands := onnxImplicitExpansion([]*Node{inputs[0], inputs[1]})
		return Pow(operands[0], operands[1])
	}
	exponentNode := node.Input[1]
	exponentT, err := m.materializeConstantExpression(exponentNode, convertedOutputs)
	if err != nil || !exponentT.IsScalar() {
		// Assume exponent is not a constant expression, hence we use proper Pow operand.
		return defaultPow()
	}

	exponentV := reflect.ValueOf(exponentT.Value())
	var exponent float64
	float64T := reflect.TypeOf(exponent)
	if !exponentV.CanConvert(float64T) {
		// Complex number exponent ?
		return defaultPow()
	}
	exponent = exponentV.Convert(float64T).Float()
	switch exponent {
	case 2:
		return Square(inputs[0])
	case 1:
		return inputs[0]
	case 0.5:
		return Sqrt(inputs[0])
	case -0.5:
		return Inverse(Sqrt(inputs[0]))
	case -1:
		return Inverse(inputs[0])
	case -2:
		return Inverse(Square(inputs[0]))
	default:
		return defaultPow()
	}
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
	rank := operand.Rank()

	startsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'starts' for node %s", nodeToString(node)))
	}
	inputStarts := tensorToInts(startsT)

	endsT, err := m.materializeConstantExpression(node.Input[2], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'ends' for node %s", nodeToString(node)))
	}
	inputEnds := tensorToInts(endsT)

	// optional axes param
	var inputAxes []int
	if len(inputs) > 3 {
		axesT, err := m.materializeConstantExpression(node.Input[3], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		inputAxes = tensorToInts(axesT)
	} else {
		// default values according to spec
		inputAxes = make([]int, rank)
		for i := 0; i < rank; i++ {
			inputAxes[i] = i
		}
	}

	// optional steps param
	var inputSteps []int
	if len(inputs) > 4 {
		stepsT, err := m.materializeConstantExpression(node.Input[4], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'steps' for node %s", nodeToString(node)))
		}
		inputSteps = tensorToInts(stepsT)
	} else {
		// default steps according to spec
		inputSteps = make([]int, len(inputStarts))
		for i := range inputSteps {
			inputSteps[i] = 1
		}
	}

	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}

	effectiveStarts := make([]int, rank)
	effectiveEnds := make([]int, rank)
	effectiveSteps := make([]int, rank)

	for i := 0; i < rank; i++ {
		effectiveStarts[i] = 0
		effectiveEnds[i] = operand.Shape().Dim(i)
		effectiveSteps[i] = 1
	}

	normalizedAxes := make([]int, len(inputAxes))
	for i, axis := range inputAxes {
		if axis < 0 {
			normalizedAxes[i] = axis + rank
		} else {
			normalizedAxes[i] = axis
		}

		if normalizedAxes[i] < 0 || normalizedAxes[i] >= rank {
			exceptions.Panicf("axis %d is out of bounds for tensor of rank %d in node %s",
				inputAxes[i], rank, nodeToString(node))
		}
	}

	// Process each specified axis to override the effective values
	for i := range normalizedAxes {
		axis := normalizedAxes[i]
		start := inputStarts[i]
		end := inputEnds[i]
		step := inputSteps[i]
		dimSize := operand.Shape().Dim(axis)

		// Validate step is not zero
		if step == 0 {
			panic(errors.Errorf("step cannot be 0 for axis %d in node %s", axis, nodeToString(node)))
		}

		// Handle negative start and end indices by adding dimension size
		if start < 0 {
			start += dimSize
		}
		if end < 0 {
			end += dimSize
		}

		if step > 0 {
			// Positive stepping
			// start clamped to [0, dimSize]
			// end clamped to [0, dimSize]
			start = max(0, min(start, dimSize))
			end = max(0, min(end, dimSize))
		} else {
			// Negative stepping (step < 0)
			// start clamped to [0, dimSize-1]
			// end clamped to [-1, dimSize-1]
			start = max(0, min(start, dimSize-1))
			end = max(-1, min(end, dimSize-1))
		}

		effectiveStarts[axis] = start
		effectiveEnds[axis] = end
		effectiveSteps[axis] = step
	}

	specs := make([]SliceAxisSpec, rank)
	for i := 0; i < rank; i++ {
		specs[i] = AxisRange(effectiveStarts[i], effectiveEnds[i]).Stride(effectiveSteps[i])
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
// https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
func convertConstantOfShape(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	g := inputs[0].Graph()

	valueAttr := getNodeAttr(node, "value", true)
	assertNodeAttrType(node, valueAttr, protos.AttributeProto_TENSOR)

	tensor, err := tensorToGoMLX(m.backend, valueAttr.T)
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
	count := MustExecOnce(backend, func(start, limit, delta *Node) *Node {
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

	result := int(tensors.ToScalar[int64](count))
	count.FinalizeAll()
	return result
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

// convertMin operator. It's different from the GoMLX Min operator in that it can take a list of inputs.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Min.html
func convertMin(operands []*Node) *Node {
	output := operands[0]
	for _, operand := range operands[1:] {
		output = convertBinaryOp(Min, output, operand)
	}
	return output
}

// convertMax operator. It's different from the GoMLX Max operator in that it can take a list of inputs.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Max.html
func convertMax(operands []*Node) *Node {
	output := operands[0]
	for _, operand := range operands[1:] {
		output = convertBinaryOp(Max, output, operand)
	}
	return output
}

// convertTrilu operator: given one or batches of 2D-matrices, returns the upper or lower triangular  part.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Trilu.html
func convertTrilu(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	input := inputs[0]
	// get offset k, default is 0
	k := 0
	if len(inputs) > 1 {
		kT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'k' for node %s", nodeToString(node)))
		}
		kValues := tensorToInts(kT)
		if len(kValues) != 1 {
			exceptions.Panicf("Trilu 'k' must be scalar, got shape %v", kT.Shape())
		}
		k = kValues[0]
	}

	// Get upper attribute (default: true)
	upper := getIntAttrOr(node, "upper", 1)

	// Apply Trilu mask
	if upper == 1 {
		return TakeUpperTriangular(input, k)
	} else {
		return TakeLowerTriangular(input, k)
	}
}

// convertScatterND operator
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ScatterND.html
func convertScatterND(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// inputs
	data := inputs[0]
	indices := inputs[1]
	updates := inputs[2]

	// attributes
	reduction := getStringAttrOr(node, "reduction", "none")

	r := data.Rank()
	if !(r >= 1) {
		exceptions.Panicf("ScatterND: data must have rank >= 1, got %d", r)
	}

	q := indices.Rank()
	if !(q >= 1) {
		exceptions.Panicf("ScatterND: indices must have rank >= 1, got %d", r)
	}

	v := q + r - indices.Shape().Dimensions[len(indices.Shape().Dimensions)-1] - 1

	if updates.Rank() != v {
		exceptions.Panicf("ScatterND: updates has wrong rank")
	}

	operand := Identity(data)
	var output *Node
	switch reduction {
	case "add":
		output = ScatterSum(operand, indices, updates, false, false)
	case "mul":
		exceptions.Panicf("ScatterMul has not been implemented yet")
	case "max":
		output = ScatterMax(operand, indices, updates, false, false)
	case "min":
		output = ScatterMin(operand, indices, updates, false, false)
	case "none", "":
		output = ScatterUpdate(operand, indices, updates, false, true)
	default:
		exceptions.Panicf("ScatterND: unrecognized reduction mode %q", reduction)
	}

	if output.Rank() < 1 {
		exceptions.Panicf("ScatterND: output must have rank >= 1, got rank %d", output.Rank())
	}
	return output
}

////////////////////////////////////////////////////////////////////
//
// Ops that are full ML layers.
//
////////////////////////////////////////////////////////////////////

// convertLSTM converts an ONNX node to a GoMLX node.
//
// The GoMLX version used ONNX version as inspiration, so they have the same feature support.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__LSTM.html
func convertLSTM(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// Inputs
	{
		newInputs := make([]*Node, 8)
		copy(newInputs, inputs)
		inputs = newInputs
	}
	operand := inputs[0]
	inputsW := inputs[1]
	recurrentW := inputs[2]
	biasesW := inputs[3]
	operandLengths := inputs[4]
	initialHidden := inputs[5]
	initialCell := inputs[6]
	peepholeW := inputs[7]

	// Reshape compacted weights.
	numDirections := inputsW.Shape().Dim(0)
	featuresDim := inputsW.Shape().Dim(-1)
	inputsW = Reshape(inputsW, numDirections, 4, -1, featuresDim)
	hiddenDim := inputsW.Shape().Dim(2)
	recurrentW = Reshape(recurrentW, numDirections, 4, hiddenDim, hiddenDim)
	biasesW = Reshape(biasesW, numDirections, 8, hiddenDim)

	// Attributes:
	activationAlpha := getFloatAttrOr(node, "activation_alpha", 0.01)
	activationBeta := getFloatsAttrOr(node, "activation_alpha", nil)
	activations := getStringsAttrOr(node, "activations", nil)
	if activations != nil {
		exceptions.Panicf("LSTM custom activaitons is not supported yet -- pls open an issue on github.com/gomlx/onnx-gomlx")
	}
	_, _ = activationAlpha, activationBeta
	clip := getFloatAttrOr(node, "clip", 0)
	if clip != 0 {
		exceptions.Panicf("LSTM clip is not supported yet -- pls open an issue on github.com/gomlx/onnx-gomlx")
	}
	directionAttr := getStringAttrOr(node, "direction", "forward")
	var direction lstm.DirectionType
	switch directionAttr {
	case "forward":
		direction = lstm.DirForward
	case "reverse":
		direction = lstm.DirReverse
	case "bidirectional":
		direction = lstm.DirBidirectional
	default:
		exceptions.Panicf("LSTM direction must be 'forward', 'reverse' or 'bidirectional', got %s", directionAttr)
	}
	hiddenSize := getIntAttrOr(node, "hidden_size", 0)
	if hiddenSize != 0 && hiddenSize != inputsW.Shape().Dim(-2) {
		exceptions.Panicf("LSTM hidden_size (%d) must match inputsW one befere last axis dimension (%s)", hiddenSize, inputsW.Shape())
	}
	inputForget := getBoolAttrOr(node, "input_forget", false)
	if inputForget {
		exceptions.Panicf("LSTM input_forget is not supported yet -- pls open an issue on github.com/gomlx/onnx-gomlx")
	}
	layout := getIntAttrOr(node, "layout", 0)

	// Operand for ONNX has shape [sequenceLength, batchSize, inputSize], we need to transpose to [batchSize, sequenceLength, inputSize]
	// (Except if layout == 1).
	switch layout {
	case 0:
		operand = TransposeAllDims(operand, 1, 0, 2)
	case 1:
		// [batchSize, numDirections, hiddenDim] -> [numDirections, batchSize, hiddenDim]
		if initialHidden != nil {
			initialHidden = TransposeAllDims(initialHidden, 1, 0, 2)
		}
		if initialCell != nil {
			initialCell = TransposeAllDims(initialCell, 1, 0, 2)
		}
	default:
		exceptions.Panicf("unsupported layout %d for LSTM: only values 0 or 1 are supported", layout)
	}

	lstmLayer := lstm.NewWithWeights(operand, inputsW, recurrentW, biasesW, peepholeW).
		Ragged(operandLengths).Direction(direction)
	allHiddenStates, lastHiddenState, lastCellState := lstmLayer.Done()

	// Transpose according to requested layout.
	switch layout {
	case 0:
		lastHiddenState = TransposeAllDims(lastHiddenState, 1, 0, 2)
		lastCellState = TransposeAllDims(lastCellState, 1, 0, 2)
	case 1:
		allHiddenStates = TransposeAllDims(allHiddenStates, 2, 0, 1, 3)
	}

	if len(node.Output) >= 2 && node.Output[1] != "" {
		convertedOutputs[node.Output[1]] = lastHiddenState
	}
	if len(node.Output) >= 3 && node.Output[2] != "" {
		convertedOutputs[node.Output[2]] = lastCellState
	}

	return allHiddenStates
}

// convertConv converts an ONNX Conv node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Conv.html
func convertConv(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	autoPad := getStringAttrOr(node, "auto_pad", "NOTSET")
	if autoPad != "NOTSET" {
		exceptions.Panicf("Conv: support for attribute 'auto_pad' (%s) is not yet implemented", autoPad)
	}
	kernelShape := getIntsAttrOr(node, "kernel_shape", nil)
	if kernelShape == nil {
		exceptions.Panicf("Conv: support for inferring 'kernel_shape' is not yet implemented")
	}
	strides := getIntsAttrOr(node, "strides", nil)
	pads := getIntsAttrOr(node, "pads", nil)
	dilations := getIntsAttrOr(node, "dilations", nil)
	groups := getIntAttrOr(node, "group", 1)

	x := inputs[0]
	w := inputs[1]
	var b *Node
	if len(inputs) > 2 {
		b = inputs[2]
	}

	var paddings [][2]int
	numSpatialDims := x.Rank() - 2
	if pads != nil {
		if len(pads) != 2*numSpatialDims {
			exceptions.Panicf("invalid number of padding values: %d spatial axes, got %d padding values -- expected 2 pads per axis", numSpatialDims, len(pads))
		}
		paddings = make([][2]int, numSpatialDims)
		for i := range numSpatialDims {
			paddings[i][0] = pads[i]
			paddings[i][1] = pads[i+numSpatialDims]
		}
	}

	inputRank := x.Rank()
	spatialAxes := make([]int, inputRank-2)
	for i := range spatialAxes {
		spatialAxes[i] = i + 2
	}

	// why: cause onnx standard is [O, I, spatial...]
	// but gomlx Conv accepts different orders by default in channels first/last mode
	// e.g input as first kernel dim in channelsFirst mode. So we just specify the dimensions.
	axes := backends.ConvolveAxesConfig{
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
	conv := Convolve(x, w).AxesConfig(axes)
	if len(strides) > 0 {
		conv = conv.StridePerDim(strides...)
	}
	if len(dilations) > 0 {
		conv = conv.DilationPerDim(dilations...)
	}
	if len(paddings) > 0 {
		conv = conv.PaddingPerDim(paddings)
	}
	if groups > 1 {
		conv = conv.FeatureGroupCount(groups)
	}
	out := conv.Done()
	if b != nil {
		// the bias stuff
		if b.Rank() == 1 && out.Rank() >= 3 {
			shape := make([]int, out.Rank())
			shape[0] = 1
			shape[1] = b.Shape().Dim(0)
			for i := 2; i < out.Rank(); i++ {
				shape[i] = 1
			}
			b = Reshape(b, shape...)
		}
		out = Add(out, b)
	}
	return out
}

// convertMaxPool converts an ONNX MaxPool node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__MaxPool.html
func convertAveragePool(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	autoPad := getStringAttrOr(node, "auto_pad", "NOTSET")
	if autoPad != "NOTSET" {
		exceptions.Panicf("AveragePool: support for attribute 'auto_pad' (%s) is not yet implemented", autoPad)
	}
	ceilMode := getIntAttrOr(node, "ceil_mode", 0)
	if ceilMode != 0 {
		exceptions.Panicf("AveragePool: support for attribute 'ceil_mode' is not yet implemented")
	}
	countIncludePad := getIntAttrOr(node, "count_include_pad", 0)
	if countIncludePad != 0 {
		// GoMLX MeanPool doesn't support including padding in the count.
		exceptions.Panicf("AveragePool: support for attribute 'count_include_pad' is not yet implemented")
	}
	kernelShape := getIntsAttrOr(node, "kernel_shape", nil)
	strides := getIntsAttrOr(node, "strides", nil)
	pads := getIntsAttrOr(node, "pads", nil)

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

	pool := MeanPool(x).ChannelsAxis(timage.ChannelsFirst)
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

func convertPad(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	mode := getStringAttrOr(node, "mode", "constant")
	if mode != "constant" {
		exceptions.Panicf("Pad: support for mode '%s' is not yet implemented", mode)
	}
	padsT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'pads' for node %s", nodeToString(node)))
	}
	pads := tensorToInts(padsT)

	x := inputs[0]
	var constantValueNode *Node
	if len(inputs) > 2 {
		constantValueNode = inputs[2]
	} else {
		constantValueNode = Scalar(x.Graph(), x.DType(), 0)
	}

	rank := x.Rank()
	if len(pads) != 2*rank {
		exceptions.Panicf("invalid number of padding values: %d axes, got %d padding values -- expected 2 pads per axis", rank, len(pads))
	}
	paddings := make([]backends.PadAxis, rank)
	for i := range rank {
		paddings[i] = backends.PadAxis{Low: int64(pads[i]), High: int64(pads[i+rank])}
	}

	return Pad(x, constantValueNode, paddings)
}

// convertMaxPool converts an ONNX MaxPool node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__MaxPool.html
func convertMaxPool(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	autoPad := getStringAttrOr(node, "auto_pad", "NOTSET")
	if autoPad != "NOTSET" {
		exceptions.Panicf("MaxPool: support for attribute 'auto_pad' (%s) is not yet implemented", autoPad)
	}
	ceilMode := getIntAttrOr(node, "ceil_mode", 0)
	if ceilMode != 0 {
		exceptions.Panicf("MaxPool: support for attribute 'ceil_mode' is not yet implemented")
	}
	dilations := getIntsAttrOr(node, "dilations", nil)
	if dilations != nil {
		exceptions.Panicf("MaxPool: support for attribute 'dilations' is not yet implemented")
	}
	storageOrder := getIntAttrOr(node, "storage_order", 0)
	if storageOrder != 0 {
		exceptions.Panicf("MaxPool: support for attribute 'storage_order' is not yet implemented")
	}
	kernelShape := getIntsAttrOr(node, "kernel_shape", nil)
	strides := getIntsAttrOr(node, "strides", nil)
	pads := getIntsAttrOr(node, "pads", nil)

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

	pool := MaxPool(x).ChannelsAxis(timage.ChannelsFirst)
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
func convertGlobalAveragePool(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	spatialDims := x.Rank() - 2
	window := make([]int, spatialDims)
	for i := range window {
		window[i] = x.Shape().Dim(i + 2)
	}
	pool := MeanPool(x).ChannelsAxis(timage.ChannelsFirst).WindowPerAxis(window...)
	out := pool.Done()
	if out.Rank() > 2 {
		out = Reshape(out, out.Shape().Dim(0), out.Shape().Dim(1))
	}
	return out
}

// convertBatchNormalization converts an ONNX BatchNormalization node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
func convertBatchNormalization(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// Inputs: [input, scale, bias, mean, var]
	x := inputs[0]
	scale := inputs[1]
	bias := inputs[2]
	mean := inputs[3]
	variance := inputs[4]

	epsilon := getFloatAttrOr(node, "epsilon", 1e-5)
	momentum := getFloatAttrOr(node, "momentum", 0.9)
	if momentum != 0.9 {
		exceptions.Panicf("BatchNormalization: support for attribute 'momentum' is not yet implemented")
	}
	trainingMode := getIntAttrOr(node, "training_mode", 0)
	if trainingMode != 0 {
		exceptions.Panicf("BatchNormalization: support for attribute 'training_mode' is not yet implemented")
	}

	inputRank := x.Rank()
	if scale.Rank() == 1 && inputRank >= 2 {
		c := scale.Shape().Dim(0)
		shape := make([]int, inputRank)
		shape[0] = 1
		shape[1] = c
		for i := 2; i < inputRank; i++ {
			shape[i] = 1
		}
		scale = Reshape(scale, shape...)
		bias = Reshape(bias, shape...)
		mean = Reshape(mean, shape...)
		variance = Reshape(variance, shape...)
	}
	normed := Div(Sub(x, mean), Sqrt(Add(variance, Scalar(x.Graph(), variance.DType(), epsilon))))
	out := Add(Mul(normed, scale), bias)
	return out
}

////////////////////////////////////////////////////////////////////
//
// Quantization related ops.
//
////////////////////////////////////////////////////////////////////

// convertDequantizeLinear converts the corresponding ONNX node to a GoMLX node.
//
// Not yet supporting block dequantization.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html
func convertDequantizeLinear(nodeProto *protos.NodeProto, inputs []*Node) *Node {
	// Attributes:
	// - Axis (optional) on which to apply the multi-valued scale.
	// - blockSize: optional, only active if != 0. Not yet implemented.
	targetAxis := getIntAttrOr(nodeProto, "axis", 1)
	blockSize := getIntAttrOr(nodeProto, "blockSize", 0)
	if blockSize != 0 {
		exceptions.Panicf("DequantizeLinear: support for attribute 'block_size' is not yet implemented")
	}
	outputDType := getDTypeAttrOr(nodeProto, "output_dtype", dtypes.Float32)

	x := inputs[0]
	scale := inputs[1]
	var xZeroPoint *Node
	if len(inputs) > 3 {
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
		newScaleShape := x.Shape().Clone()
		for axis := range newScaleShape.Dimensions {
			if axis != targetAxis {
				newScaleShape.Dimensions[axis] = 1
			} else if newScaleShape.Dimensions[axis] != scale.Shape().Dimensions[0] {
				exceptions.Panicf("DequantizeLinear: scale must have same dimension as the input axis %d (input shape=%s), got %s instead", targetAxis, x.Shape(), scale.Shape())
			}
		}
		scale = Reshape(scale, newScaleShape.Dimensions...)
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

// convertDynamicQuantizeLinear converts the corresponding ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html
func convertDynamicQuantizeLinear(convertedOutputs map[string]*Node, nodeProto *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	if len(nodeProto.Output) != 3 {
		exceptions.Panicf("DynamicQuantizeLinear: expected 3 outputs (y, y_scale, y_zero_point), got %d instead (%q)", len(nodeProto.Output), nodeProto.Output)
	}
	y, yScale, yZeroPoint := onnxDynamicQuantizeLinear(x)
	convertedOutputs[nodeProto.Output[0]] = y
	convertedOutputs[nodeProto.Output[1]] = yScale
	convertedOutputs[nodeProto.Output[2]] = yZeroPoint
	return y
}

func onnxDynamicQuantizeLinear(x *Node) (y, yScale, yZeroPoint *Node) {
	g := x.Graph()
	dtype := x.DType()
	quantizedDType := dtypes.Uint8
	zero := ScalarZero(g, dtype)
	one := ScalarOne(g, dtype)

	qMax := Scalar(g, dtype, 255)
	xMin := Min(ReduceAllMin(x), zero)
	xMax := Max(ReduceAllMax(x), zero)
	xRange := Sub(xMax, xMin)
	yScale = Div(xRange, qMax)
	yScale = Where(Equal(yScale, zero), one, yScale)
	xMinScaled := Div(xMin, yScale)
	yZeroPoint = Round(Clip(Neg(xMinScaled), zero, qMax))

	// QuantizeLinear: important detail is that the rounding occurs **before** adding the yZeroPoint.
	y = Add(Round(Div(x, yScale)), yZeroPoint)
	y = Clip(y, zero, qMax)

	// Convert to quantize dtype.
	y = ConvertDType(y, quantizedDType)
	yZeroPoint = ConvertDType(yZeroPoint, quantizedDType)
	return
}
