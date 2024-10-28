package togomlx

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
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

// Tensor converts a protos.TensorProto object to a tensors.Tensor object, handling errors and different data types.
func Tensor(proto *protos.TensorProto) (t *tensors.Tensor, err error) {
	if proto.Name == "" {
		exceptions.Panicf("initializer has no name")
	}
	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		err = errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
	}
	if proto.RawData != nil {
		t = tensors.FromShape(shape)
		t.MutableBytes(func(data []byte) {
			if len(data) != len(proto.RawData) {
				err = errors.Errorf("tensor %q shaped %s uses %d bytes, but ONNX model provided %d bytes of raw-data!?",
					proto.Name, shape, len(data), len(proto.RawData))
			} else {
				copy(data, proto.RawData)
			}
		})
		if err != nil {
			t.FinalizeAll()
			t = nil
			return nil, err
		}
	} else if proto.FloatData != nil {
		if shape.DType != dtypes.Float32 {
			return nil, errors.Errorf("tensor %q shaped %s provided data as []float32!?", proto.Name, shape)
		}
		if len(proto.FloatData) != shape.Size() {
			return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
				proto.Name, shape, shape.Size(), len(proto.FloatData))
		}
		t = tensors.FromFlatDataAndDimensions(proto.FloatData, shape.Dimensions...)
	} else if proto.DoubleData != nil {
		if shape.DType != dtypes.Float64 {
			return nil, errors.Errorf("tensor %q shaped %s provided data as []Float64!?", proto.Name, shape)
		}
		if len(proto.DoubleData) != shape.Size() {
			return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
				proto.Name, shape, shape.Size(), len(proto.DoubleData))
		}
		t = tensors.FromFlatDataAndDimensions(proto.DoubleData, shape.Dimensions...)
	} else if proto.Int32Data != nil {
		if shape.DType != dtypes.Int32 {
			return nil, errors.Errorf("tensor %q shaped %s provided data as []Int32!?", proto.Name, shape)
		}
		if len(proto.Int32Data) != shape.Size() {
			return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
				proto.Name, shape, shape.Size(), len(proto.Int32Data))
		}
		t = tensors.FromFlatDataAndDimensions(proto.Int32Data, shape.Dimensions...)
	} else if proto.Int64Data != nil {
		if shape.DType != dtypes.Int64 {
			return nil, errors.Errorf("tensor %q shaped %s provided data as []Int64!?", proto.Name, shape)
		}
		if len(proto.Int64Data) != shape.Size() {
			return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
				proto.Name, shape, shape.Size(), len(proto.Int64Data))
		}
		t = tensors.FromFlatDataAndDimensions(proto.Int64Data, shape.Dimensions...)
	} else if proto.Uint64Data != nil {
		if shape.DType != dtypes.Int64 {
			return nil, errors.Errorf("tensor %q shaped %s provided data as []uint64!?", proto.Name, shape)
		}
		if len(proto.Uint64Data) != shape.Size() {
			return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
				proto.Name, shape, shape.Size(), len(proto.Uint64Data))
		}
		t = tensors.FromFlatDataAndDimensions(proto.Uint64Data, shape.Dimensions...)
	} else {
		return nil, errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
	}
	return
}
