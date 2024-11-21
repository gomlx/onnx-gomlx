package onnx

import (
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
	shape.DType, err = dtypeForONNX(protos.TensorProto_DataType(proto.DataType))
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
	shape.DType, err = dtypeForONNX(protos.TensorProto_DataType(proto.Values.DataType))
	if err != nil {
		return
	}
	shape.Dimensions = make([]int, len(proto.Dims))
	for axis, dim := range proto.Dims {
		shape.Dimensions[axis] = int(dim)
	}
	return
}

// checkAndCreateTensor implements the generic check and copy of the ONNX proto data to a tensor for the supported data type.
func checkAndCreateTensor[T interface {
	float32 | float64 | int32 | int64 | uint64
}](proto *protos.TensorProto, onnxData []T, shape shapes.Shape) (*tensors.Tensor, error) {
	if onnxData == nil {
		// Not this type of data.
		return nil, nil
	}
	if shape.DType != dtypes.FromGenericsType[T]() {
		return nil, errors.Errorf("tensor %q shaped %s provided data as %T!?", proto.Name, shape, onnxData)
	}
	if len(onnxData) != shape.Size() {
		return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
			proto.Name, shape, shape.Size(), len(onnxData))
	}
	return tensors.FromFlatDataAndDimensions[T](onnxData, shape.Dimensions...), nil
}

// tensorToGoMLX converts a protos.TensorProto object to a tensors.Tensor object, handling errors and different data types.
func tensorToGoMLX(proto *protos.TensorProto) (t *tensors.Tensor, err error) {
	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		err = errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
		return
	}

	// If data is provided as RawData: check that the size of the data is the same used in GoMLX.
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
		return
	}

	// Tries to convert to each data type.
	t, err = checkAndCreateTensor(proto, proto.FloatData, shape)
	if t != nil || err != nil {
		return
	}
	t, err = checkAndCreateTensor(proto, proto.DoubleData, shape)
	if t != nil || err != nil {
		return
	}
	t, err = checkAndCreateTensor(proto, proto.Int32Data, shape)
	if t != nil || err != nil {
		return
	}
	t, err = checkAndCreateTensor(proto, proto.Int64Data, shape)
	if t != nil || err != nil {
		return
	}
	t, err = checkAndCreateTensor(proto, proto.Uint64Data, shape)
	if t != nil || err != nil {
		return
	}
	// Unknown tensor data type!?
	return nil, errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
}

// checkAndCopyTensor implements the generic check and copy of the tensor to the ONNX proto data.
func checkAndCopyTensor[T interface {
	float32 | float64 | int32 | int64 | uint64
}](t *tensors.Tensor, proto *protos.TensorProto, onnxData []T) error {
	shape := t.Shape()
	if shape.DType != dtypes.FromGenericsType[T]() {
		return errors.Errorf("tensor %q shaped %s provided data as %T!?", proto.Name, shape, onnxData)
	}
	if len(onnxData) != shape.Size() {
		return errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
			proto.Name, shape, shape.Size(), len(onnxData))
	}
	tensors.ConstFlatData(t, func(tensorData []T) {
		copy(onnxData, tensorData) // Copy data to ONNX proto.
	})
	return nil
}

// TensorValueToONNX copies the value of a GoMLX tensors.Tensor to the ONNX protos.TensorProto object handling errors and different data types.
//
// Both tensors (GoMLX and ONNX) must already have the same shape.
func TensorValueToONNX(t *tensors.Tensor, proto *protos.TensorProto) (err error) {
	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		return errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
	}
	if !shape.Equal(t.Shape()) {
		return errors.Errorf("TensorValueToONNX: cannot copy value of GoMLX tensor shaped %s to ONNX tensor shaped %s",
			t.Shape(), shape)
	}

	// Raw data tensor.
	if proto.RawData != nil {
		t.ConstBytes(func(data []byte) {
			if len(data) != len(proto.RawData) {
				err = errors.Errorf("tensor %q shaped %s uses %d bytes, but ONNX model provided %d bytes of raw-data!?",
					proto.Name, shape, len(data), len(proto.RawData))
			}
			copy(proto.RawData, data) // Copy data to ONNX proto.
		})
		return err
	}

	// Float32
	if proto.FloatData != nil {
		return checkAndCopyTensor(t, proto, proto.FloatData)
	}
	if proto.DoubleData != nil {
		return checkAndCopyTensor(t, proto, proto.DoubleData)
	}
	if proto.Int32Data != nil {
		return checkAndCopyTensor(t, proto, proto.Int32Data)
	}
	if proto.Int64Data != nil {
		return checkAndCopyTensor(t, proto, proto.Int64Data)
	}
	if proto.Uint64Data != nil {
		return checkAndCopyTensor(t, proto, proto.Uint64Data)
	}
	return errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
}
