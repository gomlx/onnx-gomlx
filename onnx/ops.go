package onnx

import (
	"fmt"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"github.com/pkg/errors"
	"reflect"
)

// This file implements the ONNX operators that don't have a direct corresponding GoMLX operator.

// GOMLXBinaryOp
type GOMLXBinaryOp func(lhs, rhs *Node) *Node

// convertBinaryOp applies ONNX broadcasting rule before calling the fn.
//
// It differs from GoMLX and XLA in that it automatically prepend 1-dimensional axes to
// any of the operands, if they differ in rank.
func convertBinaryOp(fn GOMLXBinaryOp, lhs, rhs *Node) *Node {
	if lhs.IsScalar() || rhs.IsScalar() {
		return fn(lhs, rhs)
	}
	if lhs.Rank() < rhs.Rank() {
		lhs = ExpandLeftToRank(lhs, rhs.Rank())
	} else if rhs.Rank() < lhs.Rank() {
		rhs = ExpandLeftToRank(rhs, lhs.Rank())
	}
	return fn(lhs, rhs)
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
	if axis == 0 {
		indices := ExpandAxes(inputs[1], -1)
		return Gather(inputs[0], indices)
	}
	exceptions.Panicf("conversion of Gather with gather axis_%d != 0 not implemented while converting ONNX %s", axis, nodeToString(node))
	return nil
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

// convertUnsqueeze converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
func convertUnsqueeze(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	if !inputs[1].DType().IsInt() {
		exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
	}

	axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
	}
	axes := tensorToInts(axesT)
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
	fmt.Printf("endsT=%s\n", endsT.GoStr())
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

	//fmt.Printf("Slice:\n")
	//fmt.Printf("\toperand.shape=%s\n", dataN.Shape())
	//fmt.Printf("\tstarts=%v\n", starts)
	//fmt.Printf("\tends=%v\n", ends)
	//fmt.Printf("\taxes=%v\n", axes)
	//fmt.Printf("\tstrides=%v\n", strides)
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
		specs[axis] = AxisRange(starts[ii], ends[ii])
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

	// If there are no axes to reduce, this is a no-op.
	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			return ReduceAllMean(operand)
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
			panic(errors.WithMessagef(err, "while converting 'shape' for node %s", nodeToString(node)))
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
			panic(errors.WithMessagef(err, "while converting 'shape' for node %s", nodeToString(node)))
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
		for _ = range operand.Rank() - len(dims) {
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
