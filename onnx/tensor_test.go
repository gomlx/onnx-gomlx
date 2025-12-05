package onnx

import (
	"testing"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// TestShape tests the Shape() function that converts ONNX TensorProto to GoMLX shapes.Shape
func TestShape(t *testing.T) {
	t.Run("NilProto", func(t *testing.T) {
		_, err := Shape(nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "nil")
	})

	t.Run("Float32Scalar", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{},
			DataType: int32(protos.TensorProto_FLOAT),
		}
		shape, err := Shape(proto)
		require.NoError(t, err)
		require.Equal(t, dtypes.Float32, shape.DType)
		require.Equal(t, 0, shape.Rank())
	})

	t.Run("Float32_1D", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{5},
			DataType: int32(protos.TensorProto_FLOAT),
		}
		shape, err := Shape(proto)
		require.NoError(t, err)
		require.Equal(t, dtypes.Float32, shape.DType)
		require.Equal(t, []int{5}, shape.Dimensions)
	})

	t.Run("Int32_2D", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{3, 4},
			DataType: int32(protos.TensorProto_INT32),
		}
		shape, err := Shape(proto)
		require.NoError(t, err)
		require.Equal(t, dtypes.Int32, shape.DType)
		require.Equal(t, []int{3, 4}, shape.Dimensions)
	})

	t.Run("Int64_4D", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{2, 3, 4, 5},
			DataType: int32(protos.TensorProto_INT64),
		}
		shape, err := Shape(proto)
		require.NoError(t, err)
		require.Equal(t, dtypes.Int64, shape.DType)
		require.Equal(t, []int{2, 3, 4, 5}, shape.Dimensions)
	})

	t.Run("SegmentedTensorNotSupported", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{10},
			DataType: int32(protos.TensorProto_FLOAT),
			Segment:  &protos.TensorProto_Segment{Begin: 0, End: 5},
		}
		_, err := Shape(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "segmented tensor not supported")
	})
}

// TestSparseShape tests the SparseShape() function
func TestSparseShape(t *testing.T) {
	t.Run("NilProto", func(t *testing.T) {
		_, err := SparseShape(nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "nil")
	})

	t.Run("NilValues", func(t *testing.T) {
		proto := &protos.SparseTensorProto{
			Values:  nil,
			Indices: &protos.TensorProto{},
			Dims:    []int64{3, 3},
		}
		_, err := SparseShape(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "nil")
	})

	t.Run("NilIndices", func(t *testing.T) {
		proto := &protos.SparseTensorProto{
			Values:  &protos.TensorProto{DataType: int32(protos.TensorProto_FLOAT)},
			Indices: nil,
			Dims:    []int64{3, 3},
		}
		_, err := SparseShape(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "nil")
	})

	t.Run("ValidSparseFloat32", func(t *testing.T) {
		proto := &protos.SparseTensorProto{
			Values: &protos.TensorProto{
				DataType: int32(protos.TensorProto_FLOAT),
			},
			Indices: &protos.TensorProto{
				DataType: int32(protos.TensorProto_INT64),
			},
			Dims: []int64{10, 20},
		}
		shape, err := SparseShape(proto)
		require.NoError(t, err)
		require.Equal(t, dtypes.Float32, shape.DType)
		require.Equal(t, []int{10, 20}, shape.Dimensions)
	})
}

// TestTensorToGoMLX tests the tensorToGoMLX() function for ONNX→GoMLX conversion
func TestTensorToGoMLX(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("NilProto", func(t *testing.T) {
		_, err := tensorToGoMLX(backend, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "nil")
	})

	t.Run("FloatData_Float32", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:      []int64{2, 2},
			DataType:  int32(protos.TensorProto_FLOAT),
			FloatData: []float32{1.0, 2.0, 3.0, 4.0},
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll() // Memory leak detection

		require.Equal(t, dtypes.Float32, tensor.Shape().DType)
		require.Equal(t, []int{2, 2}, tensor.Shape().Dimensions)
		data := tensors.MustCopyFlatData[float32](tensor)
		require.Equal(t, []float32{1.0, 2.0, 3.0, 4.0}, data)
	})

	t.Run("Int32Data_Int32", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:      []int64{3},
			DataType:  int32(protos.TensorProto_INT32),
			Int32Data: []int32{10, 20, 30},
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Int32, tensor.Shape().DType)
		require.Equal(t, []int{3}, tensor.Shape().Dimensions)
		data := tensors.MustCopyFlatData[int32](tensor)
		require.Equal(t, []int32{10, 20, 30}, data)
	})

	t.Run("Int64Data_Int64", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:      []int64{2},
			DataType:  int32(protos.TensorProto_INT64),
			Int64Data: []int64{100, 200},
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Int64, tensor.Shape().DType)
		data := tensors.MustCopyFlatData[int64](tensor)
		require.Equal(t, []int64{100, 200}, data)
	})

	t.Run("DTypeConversion_Int64ToInt32", func(t *testing.T) {
		// ONNX proto has int64 data but requests int32 dtype
		proto := &protos.TensorProto{
			Dims:      []int64{2},
			DataType:  int32(protos.TensorProto_INT32),
			Int64Data: []int64{5, 10},
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Int32, tensor.Shape().DType)
		data := tensors.MustCopyFlatData[int32](tensor)
		require.Equal(t, []int32{5, 10}, data)
	})

	t.Run("RawData_Float32", func(t *testing.T) {
		// Create raw bytes for float32 data
		data := []float32{1.5, 2.5, 3.5, 4.5}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}

		proto := &protos.TensorProto{
			Dims:     []int64{4},
			DataType: int32(protos.TensorProto_FLOAT),
			RawData:  rawBytes,
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Float32, tensor.Shape().DType)
		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})

	t.Run("RawData_Int8_Quantized", func(t *testing.T) {
		// Quantized int8 data (common in quantized models)
		proto := &protos.TensorProto{
			Dims:     []int64{4},
			DataType: int32(protos.TensorProto_INT8),
			RawData:  []byte{128, 255, 0, 127}, // -128, -1, 0, 127 as unsigned bytes
		}
		tensor, err := tensorToGoMLX(backend, proto)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Int8, tensor.Shape().DType)
		result := tensors.MustCopyFlatData[int8](tensor)
		require.Equal(t, []int8{-128, -1, 0, 127}, result)
	})

	t.Run("StringDataNotSupported", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:       []int64{2},
			DataType:   int32(protos.TensorProto_STRING),
			StringData: [][]byte{[]byte("hello"), []byte("world")},
		}
		_, err := tensorToGoMLX(backend, proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unsupported")
	})

	t.Run("ExternalDataNotImplemented", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "external.bin"},
			},
		}
		_, err := tensorToGoMLX(backend, proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "external data, which is not implemented")
	})

	t.Run("SizeMismatch", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:      []int64{2, 2}, // Expects 4 elements
			DataType:  int32(protos.TensorProto_FLOAT),
			FloatData: []float32{1.0, 2.0}, // Only 2 elements
		}
		_, err := tensorToGoMLX(backend, proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "size")
	})
}

// TestTensorValueToONNX tests the TensorValueToONNX() function for GoMLX→ONNX conversion
func TestTensorValueToONNX(t *testing.T) {
	t.Run("Float32Copy", func(t *testing.T) {
		// Create GoMLX tensor
		gomlxTensor := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2)
		defer gomlxTensor.FinalizeAll()

		// Create ONNX proto with matching shape
		proto := &protos.TensorProto{
			Dims:      []int64{2, 2},
			DataType:  int32(protos.TensorProto_FLOAT),
			FloatData: make([]float32, 4),
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.Equal(t, []float32{1.0, 2.0, 3.0, 4.0}, proto.FloatData)
	})

	t.Run("Int32Copy", func(t *testing.T) {
		gomlxTensor := tensors.FromFlatDataAndDimensions([]int32{10, 20, 30}, 3)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{3},
			DataType:  int32(protos.TensorProto_INT32),
			Int32Data: make([]int32, 3),
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.Equal(t, []int32{10, 20, 30}, proto.Int32Data)
	})

	t.Run("RawDataCopy", func(t *testing.T) {
		gomlxTensor := tensors.FromFlatDataAndDimensions([]float32{1.5, 2.5}, 2)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			RawData:  make([]byte, 8), // 2 floats * 4 bytes
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.Len(t, proto.RawData, 8)
		// Verify non-zero data was copied
		hasNonZero := false
		for _, b := range proto.RawData {
			if b != 0 {
				hasNonZero = true
				break
			}
		}
		require.True(t, hasNonZero, "RawData should contain non-zero bytes")
	})

	t.Run("ShapeMismatch", func(t *testing.T) {
		gomlxTensor := tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0}, 2)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{3}, // Different shape
			DataType:  int32(protos.TensorProto_FLOAT),
			FloatData: make([]float32, 3),
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "cannot copy")
	})

	// Test dtype conversion in checkAndCopyTensorToProto (the critical path with simplego backend)
	t.Run("DTypeConversion_Int32ToFloat32", func(t *testing.T) {
		// GoMLX tensor is int32, but ONNX proto wants to store it as float32
		// This tests the conversion path with simplego backend
		gomlxTensor := tensors.FromFlatDataAndDimensions([]int32{1, 2, 3, 4}, 2, 2)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{2, 2},
			DataType:  int32(protos.TensorProto_INT32),
			FloatData: make([]float32, 4), // Storage type differs from proto dtype
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.Equal(t, []float32{1.0, 2.0, 3.0, 4.0}, proto.FloatData)
	})

	t.Run("DTypeConversion_Float64ToFloat32", func(t *testing.T) {
		// GoMLX tensor is float64, ONNX proto storage is float32
		gomlxTensor := tensors.FromFlatDataAndDimensions([]float64{1.5, 2.5, 3.5}, 3)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{3},
			DataType:  int32(protos.TensorProto_DOUBLE),
			FloatData: make([]float32, 3), // Storage type differs from proto dtype
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.InDeltaSlice(t, []float32{1.5, 2.5, 3.5}, proto.FloatData, 0.0001)
	})

	t.Run("DTypeConversion_Int64ToInt32", func(t *testing.T) {
		// GoMLX tensor is int64, ONNX proto storage is int32
		gomlxTensor := tensors.FromFlatDataAndDimensions([]int64{100, 200, 300}, 3)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{3},
			DataType:  int32(protos.TensorProto_INT64),
			Int32Data: make([]int32, 3), // Storage type differs from proto dtype
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		require.Equal(t, []int32{100, 200, 300}, proto.Int32Data)
	})

	t.Run("DTypeConversion_Float32ToInt32", func(t *testing.T) {
		// GoMLX tensor is float32, ONNX proto storage is int32 (with truncation)
		gomlxTensor := tensors.FromFlatDataAndDimensions([]float32{1.9, 2.1, 3.7, 4.2}, 2, 2)
		defer gomlxTensor.FinalizeAll()

		proto := &protos.TensorProto{
			Dims:      []int64{2, 2},
			DataType:  int32(protos.TensorProto_FLOAT),
			Int32Data: make([]int32, 4), // Storage type differs from proto dtype
		}

		err := TensorValueToONNX(gomlxTensor, proto)
		require.NoError(t, err)
		// Conversion truncates floats to ints
		require.Equal(t, []int32{1, 2, 3, 4}, proto.Int32Data)
	})
}

// TestRoundTripConversion tests GoMLX→ONNX→GoMLX conversion preserves data
func TestRoundTripConversion(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	tests := []struct {
		name      string
		original  *tensors.Tensor
		onnxDType protos.TensorProto_DataType
		makeProto func(dims []int64, size int) *protos.TensorProto
	}{
		{
			name:      "Float32_2D",
			original:  tensors.FromFlatDataAndDimensions([]float32{1.0, 2.0, 3.0, 4.0}, 2, 2),
			onnxDType: protos.TensorProto_FLOAT,
			makeProto: func(dims []int64, size int) *protos.TensorProto {
				return &protos.TensorProto{
					Dims:      dims,
					DataType:  int32(protos.TensorProto_FLOAT),
					FloatData: make([]float32, size),
				}
			},
		},
		{
			name:      "Int32_1D",
			original:  tensors.FromFlatDataAndDimensions([]int32{10, 20, 30}, 3),
			onnxDType: protos.TensorProto_INT32,
			makeProto: func(dims []int64, size int) *protos.TensorProto {
				return &protos.TensorProto{
					Dims:      dims,
					DataType:  int32(protos.TensorProto_INT32),
					Int32Data: make([]int32, size),
				}
			},
		},
		{
			name:      "Int64_Scalar",
			original:  tensors.FromFlatDataAndDimensions([]int64{42}, 1),
			onnxDType: protos.TensorProto_INT64,
			makeProto: func(dims []int64, size int) *protos.TensorProto {
				return &protos.TensorProto{
					Dims:      dims,
					DataType:  int32(protos.TensorProto_INT64),
					Int64Data: make([]int64, size),
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer tt.original.FinalizeAll()

			// Convert GoMLX → ONNX
			shape := tt.original.Shape()
			dims := make([]int64, len(shape.Dimensions))
			for i, d := range shape.Dimensions {
				dims[i] = int64(d)
			}
			proto := tt.makeProto(dims, shape.Size())

			err := TensorValueToONNX(tt.original, proto)
			require.NoError(t, err)

			// Convert ONNX → GoMLX
			recovered, err := tensorToGoMLX(backend, proto)
			require.NoError(t, err)
			require.NotNil(t, recovered)
			defer recovered.FinalizeAll()

			// Verify shapes match
			require.Equal(t, tt.original.Shape(), recovered.Shape())

			// Verify data matches based on dtype
			switch tt.onnxDType {
			case protos.TensorProto_FLOAT:
				originalData := tensors.MustCopyFlatData[float32](tt.original)
				recoveredData := tensors.MustCopyFlatData[float32](recovered)
				require.Equal(t, originalData, recoveredData)
			case protos.TensorProto_INT32:
				originalData := tensors.MustCopyFlatData[int32](tt.original)
				recoveredData := tensors.MustCopyFlatData[int32](recovered)
				require.Equal(t, originalData, recoveredData)
			case protos.TensorProto_INT64:
				originalData := tensors.MustCopyFlatData[int64](tt.original)
				recoveredData := tensors.MustCopyFlatData[int64](recovered)
				require.Equal(t, originalData, recoveredData)
			}
		})
	}
}
