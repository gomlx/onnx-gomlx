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

// ConvertConstant converts a ONNX node to a GoMLX node.
func ConvertConstant(node *protos.NodeProto, g *Graph) *Node {
	valueAttr := getNodeAttr(node, "value", true)
	assertNodeAttrType(node, valueAttr, protos.AttributeProto_TENSOR)
	tensor, err := togomlx.Tensor(valueAttr.T)
	if err != nil {
		err = errors.WithMessagef(err, "while converting ONNX %s", nodeToString(node))
		panic(err)
	}
	return Const(g, tensor)
}

// mustGetIntAttr get the attribute as an integer.
// It panics with an exception if attribute is not set or if it is of the wrong type.
func mustGetIntAttr(node *protos.NodeProto, attrName string) int {
	attr := getNodeAttr(node, attrName, true)
	assertNodeAttrType(node, attr, protos.AttributeProto_INT)
	return int(attr.I)
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

// getIntsAttrOr gets a list of integers attribute for node if present or return the given defaultValues.
// It panics with an error message if the attribute is not present of if it is of the wrong type.
func mustGetIntsAttrOr(node *protos.NodeProto, attrName string) []int {
	attr := getNodeAttr(node, attrName, true)
	if attr.Type == protos.AttributeProto_INT {
		return []int{int(attr.I)}
	}
	assertNodeAttrType(node, attr, protos.AttributeProto_INTS)
	return sliceMap(attr.Ints, func(i int64) int { return int(i) })
}

// ConvertGather converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Gather.html
func ConvertGather(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", 0)
	if axis == 0 {
		indices := ExpandDims(inputs[1], -1)
		return Gather(inputs[0], indices)
	}
	exceptions.Panicf("conversion of Gather with gather axis_%d != 0 not implemented while converting ONNX %s", axis, nodeToString(node))
	return nil
}

// ConvertShape converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Shape.html
func ConvertShape(node *protos.NodeProto, inputs []*Node) *Node {
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

func convertToInts(t *tensors.Tensor) []int {
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

// ConvertUnsqueeze converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
func ConvertUnsqueeze(node *protos.NodeProto, inputs []*Node) *Node {
	var axesT *tensors.Tensor
	err := exceptions.TryCatch[error](func() { axesT = materializeConstantExpression(inputs[1]) })
	if err != nil {
		panic(errors.WithMessagef(err, "while converting node %s", nodeToString(node)))
	}
	if !axesT.DType().IsInt() {
		exceptions.Panicf("axes must be integer, got dtype=%s for node %s", axesT.DType(), nodeToString(node))
	}
	axes := convertToInts(axesT)
	return ExpandAxes(inputs[0], axes...)
}

// ConvertConcat converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Concat.html
func ConvertConcat(node *protos.NodeProto, inputs []*Node) *Node {
	axis := mustGetIntAttr(node, "axis")
	return Concatenate(inputs, axis)
}

// ConvertSlice converts a ONNX node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Slice.html
func ConvertSlice(node *protos.NodeProto, inputs []*Node) *Node {
	if len(inputs) < 3 {
		exceptions.Panicf("Slice requires at least 3 inputs, got %d in node %s", len(inputs), nodeToString(node))
	}

	var dataN, startsN, endsN, axesN, stridesN *Node
	for ii, nodePtr := range []**Node{&dataN, &startsN, &endsN, &axesN, &stridesN} {
		if ii < len(inputs) {
			*nodePtr = inputs[ii]
		}
	}

	for ii, node := range []*Node{dataN, startsN, endsN, axesN, stridesN} {
		if node != nil {
			fmt.Printf("\tinput[%d].IsConstantExpression=%v\n", ii, node.IsConstantExpression())
		}
	}

	starts := convertToInts(materializeConstantExpression(startsN))
	ends := convertToInts(materializeConstantExpression(endsN))
	var axes []int
	if axesN != nil {
		axes = convertToInts(materializeConstantExpression(axesN))
	}
	var strides []int
	if stridesN != nil {
		strides = convertToInts(materializeConstantExpression(stridesN))
	}
	fmt.Printf("Slice:\n")
	fmt.Printf("\toperand.shape=%s\n", dataN.Shape())
	fmt.Printf("\tstarts=%v\n", starts)
	fmt.Printf("\tends=%v\n", ends)
	fmt.Printf("\taxes=%v\n", axes)
	fmt.Printf("\tstrides=%v\n", strides)
	return nil
}
