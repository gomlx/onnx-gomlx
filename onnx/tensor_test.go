package onnx

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
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

	t.Run("ExternalDataRequiresBaseDir", func(t *testing.T) {
		proto := &protos.TensorProto{
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "external.bin"},
			},
		}
		// tensorToGoMLX should return an error for external data
		// since it doesn't have access to the base directory
		_, err := tensorToGoMLX(backend, proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "external data")
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

// TestParseExternalData tests the parseExternalData function
func TestParseExternalData(t *testing.T) {
	t.Run("NoExternalData", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
		}
		info, err := parseExternalData(proto)
		require.NoError(t, err)
		require.Nil(t, info)
	})

	t.Run("LocationOnly", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
			},
		}
		info, err := parseExternalData(proto)
		require.NoError(t, err)
		require.NotNil(t, info)
		require.Equal(t, "weights.bin", info.location)
		require.Equal(t, int64(0), info.offset)
		require.Equal(t, int64(-1), info.length)
	})

	t.Run("AllFields", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
				{Key: "offset", Value: "1024"},
				{Key: "length", Value: "4096"},
				{Key: "checksum", Value: "abc123"}, // Should be ignored
			},
		}
		info, err := parseExternalData(proto)
		require.NoError(t, err)
		require.NotNil(t, info)
		require.Equal(t, "weights.bin", info.location)
		require.Equal(t, int64(1024), info.offset)
		require.Equal(t, int64(4096), info.length)
	})

	t.Run("MissingLocation", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "offset", Value: "1024"},
			},
		}
		_, err := parseExternalData(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "missing required 'location'")
	})

	t.Run("InvalidOffset", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
				{Key: "offset", Value: "not-a-number"},
			},
		}
		_, err := parseExternalData(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "invalid offset")
	})

	t.Run("InvalidLength", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name: "test",
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
				{Key: "length", Value: "not-a-number"},
			},
		}
		_, err := parseExternalData(proto)
		require.Error(t, err)
		require.Contains(t, err.Error(), "invalid length")
	})
}

// TestTensorToGoMLXWithBaseDir_ExternalData tests external data loading
func TestTensorToGoMLXWithBaseDir_ExternalData(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("BasicExternalData", func(t *testing.T) {
		// Create a temporary directory for the test
		tmpDir := t.TempDir()

		// Create external data file with float32 values
		data := []float32{1.5, 2.5, 3.5, 4.5}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "weights.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		// Create proto with external data reference
		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{4},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
			},
		}

		// Load tensor
		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Float32, tensor.Shape().DType)
		require.Equal(t, []int{4}, tensor.Shape().Dimensions)
		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})

	t.Run("ExternalDataWithOffset", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file with padding + float32 values
		padding := make([]byte, 128) // Some padding at the start
		data := []float32{10.0, 20.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		fileContent := append(padding, rawBytes...)
		externalFile := filepath.Join(tmpDir, "weights.bin")
		err := os.WriteFile(externalFile, fileContent, 0644)
		require.NoError(t, err)

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
				{Key: "offset", Value: "128"},
				{Key: "length", Value: "8"},
			},
		}

		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})

	t.Run("ExternalDataInt32", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file with int32 values
		data := []int32{100, 200, 300}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := uint32(val)
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "data.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{3},
			DataType: int32(protos.TensorProto_INT32),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "data.bin"},
			},
		}

		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Int32, tensor.Shape().DType)
		result := tensors.MustCopyFlatData[int32](tensor)
		require.Equal(t, data, result)
	})

	t.Run("SubdirectoryLocation", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create subdirectory and external data file
		subDir := filepath.Join(tmpDir, "weights")
		err := os.MkdirAll(subDir, 0755)
		require.NoError(t, err)

		data := []float32{5.0, 6.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(subDir, "layer1.bin")
		err = os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights/layer1.bin"},
			},
		}

		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})

	t.Run("MissingFile", func(t *testing.T) {
		tmpDir := t.TempDir()

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "nonexistent.bin"},
			},
		}

		_, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to open external data file")
	})

	t.Run("EmptyBaseDir", func(t *testing.T) {
		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
			},
		}

		_, err := tensorToGoMLXWithBaseDir(backend, proto, "", nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "base directory is required")
	})

	t.Run("SizeMismatch", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create file with wrong size (2 floats instead of 4)
		data := []float32{1.0, 2.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "weights.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{4}, // Expects 4 floats = 16 bytes
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
			},
		}

		_, err = tensorToGoMLXWithBaseDir(backend, proto, tmpDir, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "bytes")
	})
}

// TestExternalDataReader tests the ExternalDataReader mmap functionality
func TestExternalDataReader(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("BasicMmapRead", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file with float32 values
		data := []float32{1.5, 2.5, 3.5, 4.5}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "weights.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		// Create reader and use it
		reader := NewExternalDataReader(tmpDir)
		defer reader.Close()

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{4},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "weights.bin"},
			},
		}

		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, reader)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		require.Equal(t, dtypes.Float32, tensor.Shape().DType)
		require.Equal(t, []int{4}, tensor.Shape().Dimensions)
		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})

	t.Run("MultipleTensorsShareSameFile", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file with multiple tensors' data
		// Tensor 1: 2 floats at offset 0
		// Tensor 2: 3 floats at offset 8
		data1 := []float32{1.0, 2.0}
		data2 := []float32{3.0, 4.0, 5.0}
		rawBytes := make([]byte, (len(data1)+len(data2))*4)

		for i, val := range data1 {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		for i, val := range data2 {
			offset := (len(data1) + i) * 4
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[offset] = byte(bits)
			rawBytes[offset+1] = byte(bits >> 8)
			rawBytes[offset+2] = byte(bits >> 16)
			rawBytes[offset+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "shared.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		// Create reader
		reader := NewExternalDataReader(tmpDir)
		defer reader.Close()

		// Load first tensor
		proto1 := &protos.TensorProto{
			Name:     "tensor1",
			Dims:     []int64{2},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "shared.bin"},
				{Key: "offset", Value: "0"},
				{Key: "length", Value: "8"},
			},
		}
		tensor1, err := tensorToGoMLXWithBaseDir(backend, proto1, tmpDir, reader)
		require.NoError(t, err)
		defer tensor1.FinalizeAll()

		// Load second tensor
		proto2 := &protos.TensorProto{
			Name:     "tensor2",
			Dims:     []int64{3},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "shared.bin"},
				{Key: "offset", Value: "8"},
				{Key: "length", Value: "12"},
			},
		}
		tensor2, err := tensorToGoMLXWithBaseDir(backend, proto2, tmpDir, reader)
		require.NoError(t, err)
		defer tensor2.FinalizeAll()

		// Verify both tensors have correct data
		result1 := tensors.MustCopyFlatData[float32](tensor1)
		require.InDeltaSlice(t, data1, result1, 0.0001)

		result2 := tensors.MustCopyFlatData[float32](tensor2)
		require.InDeltaSlice(t, data2, result2, 0.0001)
	})

	t.Run("ReaderCaching", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file
		data := []float32{1.0, 2.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "cached.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		reader := NewExternalDataReader(tmpDir)
		defer reader.Close()

		info := &externalDataInfo{
			location: "cached.bin",
			offset:   0,
			length:   8,
		}

		// Read multiple times - should reuse cached mmap
		for i := 0; i < 3; i++ {
			dst := make([]byte, 8)
			err := reader.ReadInto(info, dst)
			require.NoError(t, err)
		}

		// Verify only one mapping was created
		require.Len(t, reader.mappings, 1)
	})

	t.Run("MissingFileMmap", func(t *testing.T) {
		tmpDir := t.TempDir()

		reader := NewExternalDataReader(tmpDir)
		defer reader.Close()

		info := &externalDataInfo{
			location: "nonexistent.bin",
			offset:   0,
			length:   8,
		}

		dst := make([]byte, 8)
		err := reader.ReadInto(info, dst)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to mmap external data file")
	})

	t.Run("CloseReleasesResources", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file
		data := []float32{1.0, 2.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		externalFile := filepath.Join(tmpDir, "close_test.bin")
		err := os.WriteFile(externalFile, rawBytes, 0644)
		require.NoError(t, err)

		reader := NewExternalDataReader(tmpDir)

		info := &externalDataInfo{
			location: "close_test.bin",
			offset:   0,
			length:   8,
		}

		// Read to create a mapping
		dst := make([]byte, 8)
		err = reader.ReadInto(info, dst)
		require.NoError(t, err)
		require.Len(t, reader.mappings, 1)

		// Close should release resources
		err = reader.Close()
		require.NoError(t, err)
		require.Nil(t, reader.mappings)
	})

	t.Run("OffsetAndLengthMmap", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create external data file with padding + data
		padding := make([]byte, 256)
		data := []float32{10.0, 20.0, 30.0}
		rawBytes := make([]byte, len(data)*4)
		for i, val := range data {
			bits := *(*uint32)(unsafe.Pointer(&val))
			rawBytes[i*4] = byte(bits)
			rawBytes[i*4+1] = byte(bits >> 8)
			rawBytes[i*4+2] = byte(bits >> 16)
			rawBytes[i*4+3] = byte(bits >> 24)
		}
		fileContent := append(padding, rawBytes...)
		externalFile := filepath.Join(tmpDir, "offset_test.bin")
		err := os.WriteFile(externalFile, fileContent, 0644)
		require.NoError(t, err)

		reader := NewExternalDataReader(tmpDir)
		defer reader.Close()

		proto := &protos.TensorProto{
			Name:     "test_tensor",
			Dims:     []int64{3},
			DataType: int32(protos.TensorProto_FLOAT),
			ExternalData: []*protos.StringStringEntryProto{
				{Key: "location", Value: "offset_test.bin"},
				{Key: "offset", Value: "256"},
				{Key: "length", Value: "12"},
			},
		}

		tensor, err := tensorToGoMLXWithBaseDir(backend, proto, tmpDir, reader)
		require.NoError(t, err)
		require.NotNil(t, tensor)
		defer tensor.FinalizeAll()

		result := tensors.MustCopyFlatData[float32](tensor)
		require.InDeltaSlice(t, data, result, 0.0001)
	})
}
