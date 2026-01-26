// Package togomlx contains several conversion utilities from ONNX and GoMLX.
package onnx

import (
	"github.com/gomlx/exceptions"
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

// DTypePromotionConfig controls how dtype mismatches are handled during ONNX conversion.
type DTypePromotionConfig struct {
	// AllowPromotion enables automatic dtype promotion. If false (default),
	// dtype mismatches will panic per ONNX specification.
	AllowPromotion bool
	// PrioritizeFloat16 prefers Float16 over Float32 when promoting.
	// Only applies when AllowPromotion is true.
	PrioritizeFloat16 bool
}

// promoteToCommonDType converts two nodes to a common dtype based on type promotion rules.
// Panics if promotion is not allowed (config.AllowPromotion=false) and dtypes mismatch.
//
// When PrioritizeFloat16 is enabled, Float16+Float32 promotes to Float16 (for ARM64 optimization).
// Otherwise, standard promotion rules apply: Float64 > Float32 > Float16 > Int64 > ...
func promoteToCommonDType(lhs, rhs *Node, config DTypePromotionConfig) (*Node, *Node) {
	lhsDType := lhs.DType()
	rhsDType := rhs.DType()

	if lhsDType == rhsDType {
		return lhs, rhs
	}

	if !config.AllowPromotion {
		exceptions.Panicf("dtype mismatch: %v vs %v (ONNX does not allow implicit casting; use Model.AllowDTypePromotion() to enable)", lhsDType, rhsDType)
	}

	// Special case: prefer FP16 over Float32 if configured (ARM64/NEON optimization)
	if config.PrioritizeFloat16 {
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
	}

	// Standard promotion: use higher precision type
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
