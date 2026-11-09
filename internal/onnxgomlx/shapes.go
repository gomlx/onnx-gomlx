package onnxgomlx

import (
	"encoding/binary"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// ShapeForName returns the shapes.Shape for a named output, input, or initializer in the ONNX graph.
// Dynamic dimensions are represented as onnx.DynamicDim.
// Returns a zero-value shapes.Shape (nil Dimensions, InvalidDType) if the name is not found or shape is unknown.
func (m *Model) ShapeForName(name string) shapes.Shape {
	graph := m.Proto.Graph
	sources := [][]*protos.ValueInfoProto{graph.ValueInfo, graph.Input, graph.Output}
	for _, vis := range sources {
		for _, vi := range vis {
			if vi.Name == name {
				return ShapeFromValueInfo(vi)
			}
		}
	}
	// Check initializers.
	if tp, ok := m.VariableNameToValue[name]; ok {
		return shapeFromTensorProto(tp)
	}
	return shapes.Shape{}
}

func shapeFromTensorProto(tp *protos.TensorProto) shapes.Shape {
	if tp == nil {
		return shapes.Shape{}
	}
	dims := make([]int, len(tp.Dims))
	for i, d := range tp.Dims {
		dims[i] = int(d)
	}
	dt, err := dtypeForONNX(protos.TensorProto_DataType(tp.DataType))
	if err != nil {
		dt = dtypes.InvalidDType
	}
	return shapes.Shape{DType: dt, Dimensions: dims}
}

// ShapeFromValueInfo extracts a shapes.Shape from a ValueInfoProto.
// Dynamic dimensions are returned as shapes.DynamicDim.
// Returns a zero-value shapes.Shape if the type is not a tensor or the shape is nil.
func ShapeFromValueInfo(vi *protos.ValueInfoProto) shapes.Shape {
	tt, ok := vi.Type.Value.(*protos.TypeProto_TensorType)
	if !ok || tt.TensorType.Shape == nil {
		return shapes.Shape{}
	}
	dims := make([]int, len(tt.TensorType.Shape.Dim))
	for i, d := range tt.TensorType.Shape.Dim {
		if dv, ok := d.Value.(*protos.TensorShapeProto_Dimension_DimValue); ok {
			dims[i] = int(dv.DimValue)
		} else {
			dims[i] = shapes.DynamicDim
		}
	}
	dt, err := dtypeForONNX(protos.TensorProto_DataType(tt.TensorType.ElemType))
	if err != nil {
		dt = dtypes.InvalidDType
	}
	return shapes.Shape{DType: dt, Dimensions: dims}
}

// InferShapeForName returns the shape of a named tensor in the ONNX graph, falling back to
// recursive shape inference through intermediate nodes when Graph.ValueInfo is not populated.
// Returns a zero-value shapes.Shape (Dimensions == nil) if the shape cannot be inferred.
func (m *Model) InferShapeForName(name string) shapes.Shape {
	return m.inferShapeForNameDepth(name, 0)
}

func (m *Model) inferShapeForNameDepth(name string, depth int) shapes.Shape {
	if name == "" || depth > 32 {
		return shapes.Shape{}
	}
	if s := m.ShapeForName(name); s.Dimensions != nil {
		return s
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok {
		return shapes.Shape{}
	}

	switch node.OpType {
	case "Constant":
		return shapeFromConstantNode(node)

	case "Unsqueeze":
		if len(node.Input) == 0 {
			return shapes.Shape{}
		}
		inShape := m.inferShapeForNameDepth(node.Input[0], depth+1)
		if inShape.Dimensions == nil {
			return shapes.Shape{}
		}
		axes := m.extractUnsqueezeAxes(node)
		if len(axes) == 0 {
			return shapes.Shape{}
		}
		newRank := len(inShape.Dimensions) + len(axes)
		inserted := make([]bool, newRank)
		for _, ax := range axes {
			if ax < 0 {
				ax += newRank
			}
			if ax < 0 || ax >= newRank || inserted[ax] {
				return shapes.Shape{}
			}
			inserted[ax] = true
		}
		outDims := make([]int, newRank)
		inIdx := 0
		for i := range newRank {
			if inserted[i] {
				outDims[i] = 1
			} else {
				outDims[i] = inShape.Dimensions[inIdx]
				inIdx++
			}
		}
		return shapes.Shape{DType: inShape.DType, Dimensions: outDims}

	case "Cast":
		if len(node.Input) == 0 {
			return shapes.Shape{}
		}
		inShape := m.inferShapeForNameDepth(node.Input[0], depth+1)
		if inShape.Dimensions == nil {
			return shapes.Shape{}
		}
		outDims := make([]int, len(inShape.Dimensions))
		copy(outDims, inShape.Dimensions)
		dt := inShape.DType
		if toAttr := GetIntAttrOr(node, "to", 0); toAttr != 0 {
			if castDT, err := dtypeForONNX(protos.TensorProto_DataType(toAttr)); err == nil {
				dt = castDT
			}
		}
		return shapes.Shape{DType: dt, Dimensions: outDims}

	case "Neg", "Abs", "Exp", "Log", "Sqrt", "Erf", "Relu", "LeakyRelu", "Elu", "Celu", "Selu",
		"Sigmoid", "HardSigmoid", "Tanh", "Softplus", "Softsign", "Not", "Floor", "Ceil", "Round",
		"Sign", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh",
		"Reciprocal", "Identity", "Clip", "Gelu", "Softmax", "LogSoftmax":
		if len(node.Input) == 0 {
			return shapes.Shape{}
		}
		inShape := m.inferShapeForNameDepth(node.Input[0], depth+1)
		if inShape.Dimensions == nil {
			return shapes.Shape{}
		}
		outDims := make([]int, len(inShape.Dimensions))
		copy(outDims, inShape.Dimensions)
		return shapes.Shape{DType: inShape.DType, Dimensions: outDims}

	case "Add", "Sub", "Mul", "Div", "Mod", "Pow", "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor",
		"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Xor",
		"Min", "Max", "Sum", "Mean", "PRelu":
		if len(node.Input) < 2 {
			return shapes.Shape{}
		}
		cur := m.inferShapeForNameDepth(node.Input[0], depth+1)
		if cur.Dimensions == nil {
			return shapes.Shape{}
		}
		for i := 1; i < len(node.Input); i++ {
			next := m.inferShapeForNameDepth(node.Input[i], depth+1)
			if next.Dimensions == nil {
				return shapes.Shape{}
			}
			bDims, ok := broadcastDimensions(cur.Dimensions, next.Dimensions)
			if !ok {
				return shapes.Shape{}
			}
			dt := cur.DType
			if dt == dtypes.InvalidDType {
				dt = next.DType
			}
			cur = shapes.Shape{DType: dt, Dimensions: bDims}
		}
		switch node.OpType {
		case "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Xor":
			cur.DType = dtypes.Bool
		}
		return cur

	case "Where":
		if len(node.Input) < 3 {
			return shapes.Shape{}
		}
		condShape := m.inferShapeForNameDepth(node.Input[0], depth+1)
		xShape := m.inferShapeForNameDepth(node.Input[1], depth+1)
		yShape := m.inferShapeForNameDepth(node.Input[2], depth+1)
		if condShape.Dimensions == nil || xShape.Dimensions == nil || yShape.Dimensions == nil {
			return shapes.Shape{}
		}
		xyDims, ok := broadcastDimensions(xShape.Dimensions, yShape.Dimensions)
		if !ok {
			return shapes.Shape{}
		}
		outDims, ok := broadcastDimensions(condShape.Dimensions, xyDims)
		if !ok {
			return shapes.Shape{}
		}
		dt := xShape.DType
		if dt == dtypes.InvalidDType {
			dt = yShape.DType
		}
		return shapes.Shape{DType: dt, Dimensions: outDims}

	case "Transpose":
		if len(node.Input) == 0 {
			return shapes.Shape{}
		}
		inShape := m.inferShapeForNameDepth(node.Input[0], depth+1)
		if inShape.Dimensions == nil {
			return shapes.Shape{}
		}
		rank := len(inShape.Dimensions)
		perm := GetIntsAttrOr(node, "perm", nil)
		outDims := make([]int, rank)
		if len(perm) == 0 {
			for i := range rank {
				outDims[i] = inShape.Dimensions[rank-1-i]
			}
		} else {
			if len(perm) != rank {
				return shapes.Shape{}
			}
			for i, p := range perm {
				if p < 0 || p >= rank {
					return shapes.Shape{}
				}
				outDims[i] = inShape.Dimensions[p]
			}
		}
		return shapes.Shape{DType: inShape.DType, Dimensions: outDims}
	}

	return shapes.Shape{}
}

func shapeFromConstantNode(node *protos.NodeProto) shapes.Shape {
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "value":
			if attr.T != nil {
				return shapeFromTensorProto(attr.T)
			}
		case "value_float":
			return shapes.Shape{DType: dtypes.Float32, Dimensions: []int{}}
		case "value_int":
			return shapes.Shape{DType: dtypes.Int64, Dimensions: []int{}}
		case "value_floats":
			return shapes.Shape{DType: dtypes.Float32, Dimensions: []int{len(attr.Floats)}}
		case "value_ints":
			return shapes.Shape{DType: dtypes.Int64, Dimensions: []int{len(attr.Ints)}}
		}
	}
	return shapes.Shape{}
}

// broadcastDimensions applies ONNX multidirectional broadcasting rules to two dimension slices.
func broadcastDimensions(dimsA, dimsB []int) ([]int, bool) {
	if dimsA == nil || dimsB == nil {
		return nil, false
	}
	rankA, rankB := len(dimsA), len(dimsB)
	outRank := max(rankA, rankB)
	out := make([]int, outRank)
	for i := range outRank {
		dA := 1
		if idxA := i - (outRank - rankA); idxA >= 0 {
			dA = dimsA[idxA]
		}
		dB := 1
		if idxB := i - (outRank - rankB); idxB >= 0 {
			dB = dimsB[idxB]
		}
		switch {
		case dA == dB:
			out[i] = dA
		case dA == 1:
			out[i] = dB
		case dB == 1:
			out[i] = dA
		default:
			return nil, false
		}
	}
	return out, true
}

func (m *Model) extractUnsqueezeAxes(node *protos.NodeProto) []int {
	if axes := GetIntsAttrOr(node, "axes", nil); len(axes) > 0 {
		return axes
	}
	if len(node.Input) < 2 {
		return nil
	}
	axesName := node.Input[1]
	if tp, ok := m.VariableNameToValue[axesName]; ok {
		return tensorProtoToInts(tp)
	}
	if cNode, ok := m.NodeOutputToNode[axesName]; ok && cNode.OpType == "Constant" {
		for _, attr := range cNode.Attribute {
			if attr.Name == "value" && attr.T != nil {
				return tensorProtoToInts(attr.T)
			}
			if attr.Name == "value_int" {
				return []int{int(attr.I)}
			}
			if attr.Name == "value_ints" {
				res := make([]int, len(attr.Ints))
				for i, v := range attr.Ints {
					res[i] = int(v)
				}
				return res
			}
		}
	}
	return nil
}

func tensorProtoToInts(tp *protos.TensorProto) []int {
	if tp == nil {
		return nil
	}
	if len(tp.Int64Data) > 0 {
		res := make([]int, len(tp.Int64Data))
		for i, v := range tp.Int64Data {
			res[i] = int(v)
		}
		return res
	}
	if len(tp.Int32Data) > 0 {
		res := make([]int, len(tp.Int32Data))
		for i, v := range tp.Int32Data {
			res[i] = int(v)
		}
		return res
	}
	if len(tp.RawData) > 0 {
		switch tp.DataType {
		case int32(protos.TensorProto_INT64):
			n := len(tp.RawData) / 8
			res := make([]int, n)
			for i := range n {
				res[i] = int(int64(binary.LittleEndian.Uint64(tp.RawData[i*8 : (i+1)*8])))
			}
			return res
		case int32(protos.TensorProto_INT32):
			n := len(tp.RawData) / 4
			res := make([]int, n)
			for i := range n {
				res[i] = int(int32(binary.LittleEndian.Uint32(tp.RawData[i*4 : (i+1)*4])))
			}
			return res
		}
	}
	return nil
}

// IsScalarOrBroadcast1Constant reports whether name refers to a constant (or a unary Cast/Identity
// of a constant) whose shape has all dimensions equal to 1 (including rank-0 scalars).
func (m *Model) IsScalarOrBroadcast1Constant(name string) bool {
	return m.isScalarOrBroadcast1ConstantDepth(name, 0)
}

func (m *Model) isScalarOrBroadcast1ConstantDepth(name string, depth int) bool {
	if name == "" || depth > 8 {
		return false
	}
	if m.IsConstant(name) {
		s := m.InferShapeForName(name)
		if s.Dimensions == nil {
			return false
		}
		for _, d := range s.Dimensions {
			if d != 1 {
				return false
			}
		}
		return true
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok {
		return false
	}
	if (node.OpType == "Cast" || node.OpType == "Identity") && len(node.Input) > 0 {
		return m.isScalarOrBroadcast1ConstantDepth(node.Input[0], depth+1)
	}
	return false
}

