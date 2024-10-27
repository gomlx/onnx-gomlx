package togomlx

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// Shape converts an ONNX data type and shape to GoMLX shapes.Shape (it includes the dtype).
func Shape(proto *protos.TensorProto) (shape shapes.Shape, err error) {
	if proto == nil {
		err = errors.New("ONNX TensorProto is nil")
		return
	}
	shape.DType, err = DType(protos.TensorProto_DataType(proto.DataType))
	if err != nil {
		return
	}
	shape.Dimensions = make([]int, len(proto.Dims))
	for axis, dim := range proto.Dims {
		shape.Dimensions[axis] = int(dim)
	}
	if proto.Segment != nil {
		err = errors.Errorf("segmented tensor not supported (%v)", proto.Segment)
		return
	}
	return
}

// SparseShape returns what would be the dense shape of an ONNX SparseTensor.
func SparseShape(proto *protos.SparseTensorProto) (shape shapes.Shape, err error) {
	if proto == nil || proto.Values == nil || proto.Indices == nil {
		err = errors.New("ONNX SparseTensorProto or its components are nil")
		return
	}
	shape.DType, err = DType(protos.TensorProto_DataType(proto.Values.DataType))
	if err != nil {
		return
	}
	shape.Dimensions = make([]int, len(proto.Dims))
	for axis, dim := range proto.Dims {
		shape.Dimensions[axis] = int(dim)
	}
	return
}
