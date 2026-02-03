package onnx

import (
	"strconv"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// DefaultDeviceNum is the device number used in local graph operations
// (like converting tensors for different types).
var DefaultDeviceNum = backends.DeviceNum(0)

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

	onnxDataTensor := tensors.FromFlatDataAndDimensions(onnxData, shape.Dimensions...)
	if shape.DType == dtypes.FromGenericsType[T]() {
		// The provided ONNX tensor is exactly what we want:
		return onnxDataTensor, nil
	}
	defer onnxDataTensor.FinalizeAll() // Help the GC.

	// Convert from the ONNX proto data type to the target datatype.
	// It uses GoMLX SimpleGo backend.
	var converted *tensors.Tensor
	err := exceptions.TryCatch[error](func() {
		converted = MustExecOnce(backend, func(x *Node) *Node {
			return ConvertDType(x, shape.DType)
		}, onnxDataTensor)
		converted.ToLocal() // Detach from the conversion backend.
	})
	return converted, err
}

// tensorToGoMLX converts a protos.TensorProto object to a tensors.Tensor object, handling errors and different data types.
func tensorToGoMLX(backend backends.Backend, proto *protos.TensorProto) (t *tensors.Tensor, err error) {
	if proto == nil {
		return nil, errors.New("ONNX TensorProto is nil")
	}

	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		err = errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
		return
	}

	// Handle zero-sized tensors (e.g., shape [0]): no data is expected.
	if shape.Size() == 0 {
		return tensors.FromShape(shape), nil
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
	if len(proto.ExternalData) > 0 {
		return nil, errors.Errorf("ONNX model tensor %q is stored as external data; use tensorToGoMLXWithBaseDir with the model's base directory", proto.Name)
	}
	// Unknown tensor data type!?
	return nil, errors.Errorf("tensor %q shaped %s has no supported format of data in the ONNX model!?", proto.Name, shape)
}

// externalDataInfo holds parsed external data parameters from ONNX TensorProto.
type externalDataInfo struct {
	location string // path to external file (relative to model directory)
	offset   int64  // byte offset in file, default 0
	length   int64  // number of bytes to read, -1 means read to EOF
}

// parseExternalData extracts external data parameters from the TensorProto.
// Returns nil if there is no external data.
func parseExternalData(proto *protos.TensorProto) (*externalDataInfo, error) {
	if len(proto.ExternalData) == 0 {
		return nil, nil
	}

	info := &externalDataInfo{
		offset: 0,
		length: -1, // -1 means read to EOF
	}

	for _, entry := range proto.ExternalData {
		switch entry.Key {
		case "location":
			info.location = entry.Value
		case "offset":
			offset, err := strconv.ParseInt(entry.Value, 10, 64)
			if err != nil {
				return nil, errors.Wrapf(err, "invalid offset value %q for tensor %q", entry.Value, proto.Name)
			}
			info.offset = offset
		case "length":
			length, err := strconv.ParseInt(entry.Value, 10, 64)
			if err != nil {
				return nil, errors.Wrapf(err, "invalid length value %q for tensor %q", entry.Value, proto.Name)
			}
			info.length = length
		case "checksum":
			// Checksum is optional and used for verification; we don't validate it currently
		default:
			// Ignore unknown keys for forward compatibility
		}
	}

	if info.location == "" {
		return nil, errors.Errorf("external data for tensor %q is missing required 'location' key", proto.Name)
	}

	return info, nil
}


// tensorToGoMLXWithBaseDir converts a protos.TensorProto object to a tensors.Tensor object,
// handling errors and different data types including external data.
// baseDir is the directory containing the ONNX model file, used to resolve external data paths.
// reader is an optional ExternalDataReader for memory-efficient external data loading.
// If reader is nil and external data is present, it falls back to direct file I/O.
// If baseDir is empty and external data is present, an error is returned.
func tensorToGoMLXWithBaseDir(backend backends.Backend, proto *protos.TensorProto, baseDir string, reader *ExternalDataReader) (t *tensors.Tensor, err error) {
	if proto == nil {
		return nil, errors.New("ONNX TensorProto is nil")
	}

	var shape shapes.Shape
	shape, err = Shape(proto)
	if err != nil {
		err = errors.WithMessagef(err, "while parsing tensor %q", proto.Name)
		return
	}

	// Handle zero-sized tensors (e.g., shape [0]): no data is expected.
	if shape.Size() == 0 {
		return tensors.FromShape(shape), nil
	}

	// Check for external data first
	if len(proto.ExternalData) > 0 {
		info, err := parseExternalData(proto)
		if err != nil {
			return nil, err
		}

		// Create tensor and read directly into its backing memory
		t = tensors.FromShape(shape)
		t.MutableBytes(func(data []byte) {
			// Use the reader if available (mmap path), otherwise fall back to direct I/O
			if reader != nil {
				err = reader.ReadInto(info, data)
			} else {
				err = readExternalDataDirect(baseDir, info, data)
			}
		})
		if err != nil {
			t.FinalizeAll()
			return nil, errors.WithMessagef(err, "while reading external data for tensor %q", proto.Name)
		}
		return t, nil
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

// checkAndCopyTensorToProto implements the generic check and copy of the tensor to the ONNX proto data.
func checkAndCopyTensorToProto[T interface {
	float32 | float64 | int32 | int64 | uint64
}](t *tensors.Tensor, proto *protos.TensorProto, onnxData []T) error {
	shape := t.Shape()
	if len(onnxData) != shape.Size() {
		return errors.Errorf("tensor %q shaped %s has size %d , but ONNX model provided a slice with %d values!?",
			proto.Name, shape, shape.Size(), len(onnxData))
	}

	// If the dtype of the tensor doesn't match the dtype of the proto storing it:
	var converted *tensors.Tensor
	if shape.DType != dtypes.FromGenericsType[T]() {
		// Convert from GoMLX tensor to the ONNX proto data type.
		// It uses GoMLX SimpleGo backend.
		backend, err := simplego.New("")
		if err != nil {
			return err
		}
		defer backend.Finalize()
		cloned, err := t.OnDeviceClone(backend, DefaultDeviceNum)
		if err != nil {
			return err
		}
		converted, err = ExecOnce(backend, func(x *Node) *Node {
			return ConvertDType(x, dtypes.FromGenericsType[T]())
		}, cloned)
		if err != nil {
			return err
		}
		err = converted.ToLocal() // Detach from the temporarily created backend.
		if err != nil {
			return err
		}
		t = converted
		defer func() {
			// Notice this only gets used if there was another error, so we ignore this one, if one
			// happens here.
			_ = converted.FinalizeAll()
		}()
	}

	// Copy GoMLX value (potentially converted) to the ONNX proto.
	err := tensors.ConstFlatData(t, func(tensorData []T) {
		copy(onnxData, tensorData) // Copy data to ONNX proto.
	})
	if err != nil {
		return err
	}
	if converted != nil {
		// Return the error of the FinalizeAll -- it will be called again by the deferred fucntion again,
		// but it is fine.
		return converted.FinalizeAll()
	}
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
