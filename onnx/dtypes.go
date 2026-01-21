// Package togomlx contains several conversion utilities from ONNX and GoMLX.
package onnx

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// dtypeForONNX converts an ONNX data type to a gomlx data type.
func dtypeForONNX(onnxDType protos.TensorProto_DataType) (dtypes.DType, error) {
	switch onnxDType {
	case protos.TensorProto_FLOAT:
		return dtypes.Float32, nil
	case protos.TensorProto_FLOAT16:
		return dtypes.Float16, nil
	case protos.TensorProto_BFLOAT16:
		return dtypes.BFloat16, nil
	case protos.TensorProto_DOUBLE:
		return dtypes.Float64, nil
	case protos.TensorProto_INT32:
		return dtypes.Int32, nil
	case protos.TensorProto_INT64:
		return dtypes.Int64, nil
	case protos.TensorProto_UINT8:
		return dtypes.Uint8, nil
	case protos.TensorProto_INT8:
		return dtypes.Int8, nil
	case protos.TensorProto_INT16:
		return dtypes.Int16, nil
	case protos.TensorProto_UINT16:
		return dtypes.Uint16, nil
	case protos.TensorProto_UINT32:
		return dtypes.Uint32, nil
	case protos.TensorProto_UINT64:
		return dtypes.Uint64, nil
	case protos.TensorProto_BOOL:
		return dtypes.Bool, nil
	case protos.TensorProto_COMPLEX64:
		return dtypes.Complex64, nil
	case protos.TensorProto_COMPLEX128:
		return dtypes.Complex128, nil
	default:
		return dtypes.InvalidDType, errors.Errorf("unsupported/unknown ONNX data type %v", onnxDType)
	}
}

// promoteToCommonDType converts two nodes to a common dtype based on type promotion rules.
//
// IMPORTANT: This function intentionally deviates from standard NumPy/PyTorch type promotion
// rules in one specific case: when mixing Float16 and Float32, we promote to Float16 instead
// of Float32. This is an intentional optimization to leverage NEON-accelerated FP16 kernels
// on ARM64 platforms (e.g., Apple Silicon, modern ARM servers), which can provide significant
// performance improvements for FP16 operations.
//
// Trade-offs of this design decision:
//   - Pro: Up to 2x throughput on ARM64 with native FP16 SIMD (FMLAL/FMLAL2 instructions)
//   - Pro: Reduced memory bandwidth for large tensors
//   - Con: Potential precision loss compared to Float32 computation
//   - Con: Non-standard behavior may surprise users expecting NumPy-like semantics
//
// For all other mixed-type combinations, standard promotion rules apply:
// Float64 > Float32 > Float16/BFloat16 > Int64 > Int32 > Int16 > Int8 > Uint64 > ...
//
// Note: This behavior applies to ONNX models that have mixed Float16/Float32 tensors,
// which commonly occurs in quantization-aware or mixed-precision trained models.
func promoteToCommonDType(lhs, rhs *Node) (*Node, *Node) {
	lhsDType := lhs.DType()
	rhsDType := rhs.DType()

	// Special case: prefer FP16 over Float32 to leverage NEON-accelerated FP16 kernels.
	// This is an intentional performance optimization for ARM64 platforms.
	// See function documentation for trade-offs and rationale.
	if (lhsDType == dtypes.Float16 && rhsDType == dtypes.Float32) ||
		(lhsDType == dtypes.Float32 && rhsDType == dtypes.Float16) {
		targetDType := dtypes.Float16
		if lhsDType != targetDType {
			lhs = ConvertDType(lhs, targetDType)
		}
		if rhsDType != targetDType {
			rhs = ConvertDType(rhs, targetDType)
		}
		return lhs, rhs
	}

	targetDType := lhsDType
	if dtypePriority(rhsDType) > dtypePriority(lhsDType) {
		targetDType = rhsDType
	}

	if lhsDType != targetDType {
		lhs = ConvertDType(lhs, targetDType)
	}
	if rhsDType != targetDType {
		rhs = ConvertDType(rhs, targetDType)
	}
	return lhs, rhs
}

// dtypePriority returns a priority value for dtype promotion.
// Higher values are preferred in mixed-type operations.
func dtypePriority(dt dtypes.DType) int {
	switch dt {
	case dtypes.Complex128:
		return 110
	case dtypes.Complex64:
		return 105
	case dtypes.Float64:
		return 100
	case dtypes.Float32:
		return 90
	case dtypes.Float16, dtypes.BFloat16:
		return 80
	case dtypes.Int64:
		return 70
	case dtypes.Int32:
		return 60
	case dtypes.Int16:
		return 50
	case dtypes.Int8:
		return 40
	case dtypes.Uint64:
		return 35
	case dtypes.Uint32:
		return 30
	case dtypes.Uint16:
		return 25
	case dtypes.Uint8:
		return 20
	case dtypes.Bool:
		return 10
	default:
		return 0
	}
}
