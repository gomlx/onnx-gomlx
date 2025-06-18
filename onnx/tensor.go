package onnx

import (
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/graph"
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

// checkAndCreateTensorFromProto implements the generic check and copy of the ONNX proto data to a tensor for the supported data type.
// TODO: It assumes it was saved in the same endian-ness and row-major order. Check/adjust if not.
func checkAndCreateTensorFromProto[T interface {
	float32 | float64 | int32 | int64 | uint64
}](backend backends.Backend, proto *protos.TensorProto, onnxData []T, shape shapes.Shape) (*tensors.Tensor, error) {
	if onnxData == nil {
		// Not this type of data.
		return nil, nil
	}
	if len(onnxData) != shape.Size() {
		return nil, errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
			proto.Name, shape, shape.Size(), len(onnxData))
	}

	onnxDataTensor := tensors.FromFlatDataAndDimensions[T](onnxData, shape.Dimensions...)
	if shape.DType == dtypes.FromGenericsType[T]() {
		// The provided ONNX tensor is exactly what we want:
		return onnxDataTensor, nil
	}
	defer onnxDataTensor.FinalizeAll() // Help the GC.

	// Convert from the ONNX proto data type to the target datatype.
	// It uses GoMLX SimpleGo backend.
	var converted *tensors.Tensor
	err := exceptions.TryCatch[error](func() {
		converted = graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
			return graph.ConvertDType(x, shape.DType)
		}, onnxDataTensor)
		converted.ToLocal() // Detach from the conversion backend.
	})
	return converted, err
}

// tensorToGoMLX converts a protos.TensorProto object to a tensors.Tensor object, handling errors and different data types.
func tensorToGoMLX(backend backends.Backend, proto *protos.TensorProto) (t *tensors.Tensor, err error) {
	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		err = errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
		return
	}

	// data is stored in external data
	if proto.ExternalData != nil {
		rawData, err := loadExternalData(proto)
		if err != nil {
			return nil, errors.WithMessagef(err, "loading external data for tensor %q", proto.Name)
		}
		proto.RawData = rawData
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
	if proto.DoubleData != nil {
		return checkAndCreateTensorFromProto(backend, proto, proto.DoubleData, shape)
	}
	if proto.FloatData != nil {
		return checkAndCreateTensorFromProto(backend, proto, proto.FloatData, shape)
	}
	if proto.Int64Data != nil {
		return checkAndCreateTensorFromProto(backend, proto, proto.Int64Data, shape)
	}
	if proto.Uint64Data != nil {
		return checkAndCreateTensorFromProto(backend, proto, proto.Uint64Data, shape)
	}
	if proto.Int32Data != nil {
		return checkAndCreateTensorFromProto(backend, proto, proto.Int32Data, shape)
	}
	if proto.StringData != nil {
		return nil, errors.Errorf("ONNX model tensor %q holds string data which is not supported in GoMLX models", proto.Name)
	}
	// Unknown tensor data type!?
	return nil, errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
}

func loadExternalData(proto *protos.TensorProto) ([]byte, error) {
	kv := make(map[string]string)
	for _, entry := range proto.ExternalData {
		kv[entry.Key] = entry.Value
	}
	fmt.Print("name: ", proto.Name)
	// fmt.Println("")
	location, ok := kv["location"]
	if !ok {
		return nil, errors.New("external_data missing required 'location' key")
	}

	// manage offset, if it exists
	offset := int64(0) // default is no offset
	if val, ok := kv["offset"]; ok {
		var err error
		offset, err = strconv.ParseInt(val, 10, 64)
		if err != nil {
			return nil, errors.WithMessagef(err, "invalid 'offset' in external_data")
		}
	}

	// how long to read, if limit exists
	length := int64(-1) // default is end of file
	if val, ok := kv["length"]; ok {
		var err error
		length, err = strconv.ParseInt(val, 10, 64)
		if err != nil {
			return nil, errors.WithMessagef(err, "invalid 'length' in external_data")
		}
	}
	fmt.Println("file location: " + location)
	file, err := os.Open("/home/rileyoh6/summer_2025/onnx_models/" + location)
	if err != nil {
		return nil, errors.WithMessagef(err, "opening external data file %q", location)
	}
	defer file.Close()

	_, err = file.Seek(offset, io.SeekStart)
	if err != nil {
		return nil, errors.Wrap(err, "seeking to offset in external file")
	}

	if length >= 0 {
		buf := make([]byte, length)
		_, err = io.ReadFull(file, buf)
		if err != nil {
			return nil, errors.Wrap(err, "reading external data of specified length")
		}
		return buf, nil
	} else {
		return io.ReadAll(file)
	}

}

// checkAndCopyTensorToProto implements the generic check and copy of the tensor to the ONNX proto data.
func checkAndCopyTensorToProto[T interface {
	float32 | float64 | int32 | int64 | uint64
}](t *tensors.Tensor, proto *protos.TensorProto, onnxData []T) error {
	shape := t.Shape()
	if len(onnxData) != shape.Size() {
		return errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
			proto.Name, shape, shape.Size(), len(onnxData))
	}

	// If dtype of the tensor doesn't match the dtype of the proto storing it:
	if shape.DType != dtypes.FromGenericsType[T]() {
		// Convert from GoMLX tensor the ONNX proto data type.
		// It uses GoMLX SimpleGo backend.
		var converted *tensors.Tensor
		backend, err := simplego.New("")
		if err != nil {
			return err
		}
		defer backend.Finalize()
		err = exceptions.TryCatch[error](func() {
			converted = graph.ExecOnce(backend, func(x *graph.Node) *graph.Node {
				return graph.ConvertDType(x, shape.DType)
			}, t.OnDeviceClone(backend))
			converted.ToLocal() // Detach from the temporarily created backend.
		})
		t = converted
	}

	// Copy GoMLX value (potentially converted) to the ONNX proto.
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
		return checkAndCopyTensorToProto(t, proto, proto.FloatData)
	}
	if proto.DoubleData != nil {
		return checkAndCopyTensorToProto(t, proto, proto.DoubleData)
	}
	if proto.Int32Data != nil {
		return checkAndCopyTensorToProto(t, proto, proto.Int32Data)
	}
	if proto.Int64Data != nil {
		return checkAndCopyTensorToProto(t, proto, proto.Int64Data)
	}
	if proto.Uint64Data != nil {
		return checkAndCopyTensorToProto(t, proto, proto.Uint64Data)
	}
	return errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
}
