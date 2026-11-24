// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package fusion

import (
	"encoding/binary"
	"math"
	"slices"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
)

// sdpaMaskSentinelThreshold is the maximum (least negative) scalar value recognized as a
// hard attention-masking sentinel inside structured ONNX mask-builder subgraphs such as:
//   - Where(cond, sentinel, 0) / Where(cond, 0, sentinel) / Where(Cast(bool, x), sentinel, x)
//   - Mul(Sub(1, mask), sentinel) / Mul(Sub(mask, 1), -sentinel)
//   - Mul(Cast(float, boolOrInvertedMask), sentinel)
//
// Common real-world ONNX/HuggingFace sentinels are -10000 (-1e4, classic BERT/RoBERTa/FP16),
// -65504 (float16 min), -1e9, -3.3895e38 (bfloat16 min), -3.4028e38 (float32 min), and -Inf.
// Arbitrary float mask tensors that are not built by these structured subgraphs are left as
// float additive masks so exact additive semantics are preserved.
const sdpaMaskSentinelThreshold = -1e4

type sdpaBoolMaskOp int

const (
	// sdpaBoolMaskLeafDirect uses SourceName as the attend mask (Bool directly, or != 0).
	sdpaBoolMaskLeafDirect sdpaBoolMaskOp = iota
	// sdpaBoolMaskLeafNot inverts SourceName (LogicalNot if Bool, or == 0).
	sdpaBoolMaskLeafNot
	// sdpaBoolMaskLeafEqualOne marks SourceName == 1 (or true if Bool) as attend.
	sdpaBoolMaskLeafEqualOne
	// sdpaBoolMaskAnd combines two boolean attend masks with LogicalAnd.
	sdpaBoolMaskAnd
)

// sdpaBoolMaskExpr describes how to build a GoMLX Bool attention mask (true = attend,
// false = ignore) directly from upstream ONNX tensors, bypassing float sentinel arithmetic
// so backends can use native boolean mask trimming (e.g. sdpaTrimBoolKeyMask) and
// MaskedSoftmax.
type sdpaBoolMaskExpr struct {
	Op         sdpaBoolMaskOp
	SourceName string
	Left       *sdpaBoolMaskExpr
	Right      *sdpaBoolMaskExpr
}

func (e *sdpaBoolMaskExpr) collectExternalInputs(dst *[]string) {
	if e == nil {
		return
	}
	if e.Op == sdpaBoolMaskAnd {
		e.Left.collectExternalInputs(dst)
		e.Right.collectExternalInputs(dst)
		return
	}
	if e.SourceName != "" && !slices.Contains(*dst, e.SourceName) {
		*dst = append(*dst, e.SourceName)
	}
}

func (e *sdpaBoolMaskExpr) cacheKey() string {
	if e == nil {
		return ""
	}
	switch e.Op {
	case sdpaBoolMaskLeafDirect:
		return "D:" + e.SourceName
	case sdpaBoolMaskLeafNot:
		return "N:" + e.SourceName
	case sdpaBoolMaskLeafEqualOne:
		return "E1:" + e.SourceName
	case sdpaBoolMaskAnd:
		return "A(" + e.Left.cacheKey() + "," + e.Right.cacheKey() + ")"
	default:
		return ""
	}
}

func (e *sdpaBoolMaskExpr) emit(g *Graph, convertedOutputs map[string]*Node) *Node {
	if e == nil {
		return nil
	}
	key := "\x00sdpa_bool_mask:" + e.cacheKey()
	if cached, ok := convertedOutputs[key]; ok {
		return cached
	}
	var out *Node
	switch e.Op {
	case sdpaBoolMaskLeafDirect:
		src := convertedOutputs[e.SourceName]
		if src.DType() == dtypes.Bool {
			out = src
		} else {
			out = NotEqual(src, ScalarZero(g, src.DType()))
		}
	case sdpaBoolMaskLeafNot:
		src := convertedOutputs[e.SourceName]
		if src.DType() == dtypes.Bool {
			out = LogicalNot(src)
		} else {
			out = Equal(src, ScalarZero(g, src.DType()))
		}
	case sdpaBoolMaskLeafEqualOne:
		src := convertedOutputs[e.SourceName]
		if src.DType() == dtypes.Bool {
			out = src
		} else {
			out = Equal(src, ScalarOne(g, src.DType()))
		}
	case sdpaBoolMaskAnd:
		lhs := e.Left.emit(g, convertedOutputs)
		rhs := e.Right.emit(g, convertedOutputs)
		bcast := onnxgomlx.RuntimeBroadcast([]*Node{lhs, rhs})
		out = LogicalAnd(bcast[0], bcast[1])
	}
	if out != nil {
		convertedOutputs[key] = out
	}
	return out
}

// matchSDPABoolMask recognizes ONNX subgraphs that convert a binary/boolean attention mask
// into an additive float mask using a large negative sentinel (<= -1e4 or -Inf), and returns
// an equivalent sdpaBoolMaskExpr (true = attend, false = ignore). Returns nil if maskName is
// not a recognized sentinel-mask builder (e.g., arbitrary float bias/mask inputs).
func matchSDPABoolMask(m *onnxgomlx.Model, maskName string) *sdpaBoolMaskExpr {
	if m == nil || maskName == "" {
		return nil
	}
	return matchSDPABoolMaskDepth(m, maskName, 0)
}

func matchSDPABoolMaskDepth(m *onnxgomlx.Model, maskName string, depth int) *sdpaBoolMaskExpr {
	if depth > 8 || maskName == "" {
		return nil
	}
	node, ok := m.NodeOutputToNode[maskName]
	if !ok || node == nil {
		return nil
	}

	switch node.OpType {
	case "Identity":
		if len(node.Input) == 1 {
			return matchSDPABoolMaskDepth(m, node.Input[0], depth+1)
		}
	case "Cast":
		// Unwrap float-to-float Casts (e.g. float32 mask cast to float16/bfloat16).
		if len(node.Input) == 1 && isONNXFloatDType(onnxgomlx.GetIntAttrOr(node, "to", 0)) {
			return matchSDPABoolMaskDepth(m, node.Input[0], depth+1)
		}
	case "Where":
		return matchSDPABoolMaskWhere(m, node, maskName)
	case "Mul":
		return matchSDPABoolMaskMul(m, node, maskName)
	case "Add", "Min":
		// Combined masks such as Add(causal_mask, padding_mask) or Min(maskA, maskB):
		// only convert when BOTH operands are pure sentinel masks (so ALiBi / relative
		// position biases stay additive float masks).
		if len(node.Input) == 2 {
			left := matchSDPABoolMaskDepth(m, node.Input[0], depth+1)
			if left == nil {
				return nil
			}
			right := matchSDPABoolMaskDepth(m, node.Input[1], depth+1)
			if right == nil {
				return nil
			}
			return &sdpaBoolMaskExpr{
				Op:    sdpaBoolMaskAnd,
				Left:  left,
				Right: right,
			}
		}
	}
	return nil
}

// matchSDPABoolMaskWhere matches:
//   - Where(cond, negSentinel, 0) or Where(Cast(bool, x), negSentinel, x) -> attend = !cond
//   - Where(cond, 0, negSentinel) or Where(Equal(x, 0), x, negSentinel)  -> attend = cond
func matchSDPABoolMaskWhere(m *onnxgomlx.Model, node *protos.NodeProto, whereOutName string) *sdpaBoolMaskExpr {
	if len(node.Input) != 3 {
		return nil
	}
	condName, onTrueName, onFalseName := node.Input[0], node.Input[1], node.Input[2]

	// Case 1: onTrue is negative sentinel -> attend when cond is false.
	if sentScalar, ok := isMaskSentinelTensor(m, onTrueName, true); ok {
		zeroScalar, isZero := isZeroConstantTensor(m, onFalseName)
		zeroWhenCondFalse := isZero || isCondNonZeroOf(m, condName, onFalseName)
		if zeroWhenCondFalse {
			needShapeCheck := !sentScalar || (isZero && !zeroScalar)
			if !needShapeCheck || sdpaBoolMaskShapeSafe(m, condName, whereOutName) {
				return simplifyBoolAttendFromCond(m, condName, true, 0)
			}
		}
	}

	// Case 2: onFalse is negative sentinel -> attend when cond is true.
	if sentScalar, ok := isMaskSentinelTensor(m, onFalseName, true); ok {
		zeroScalar, isZero := isZeroConstantTensor(m, onTrueName)
		zeroWhenCondTrue := isZero || isCondZeroOf(m, condName, onTrueName)
		if zeroWhenCondTrue {
			needShapeCheck := !sentScalar || (isZero && !zeroScalar)
			if !needShapeCheck || sdpaBoolMaskShapeSafe(m, condName, whereOutName) {
				return simplifyBoolAttendFromCond(m, condName, false, 0)
			}
		}
	}

	return nil
}

// matchSDPABoolMaskMul matches:
//   - Mul(Sub(1, x), negSentinel) / Mul(negSentinel, Sub(1, x)) -> attend = (x == 1)
//   - Mul(Sub(x, 1), posSentinel) / Mul(Add(x, -1), posSentinel) -> attend = (x == 1)
//   - Mul(Cast(float, boolOrSub1), negSentinel) -> attend = !bool / (x == 1)
func matchSDPABoolMaskMul(m *onnxgomlx.Model, node *protos.NodeProto, mulOutName string) *sdpaBoolMaskExpr {
	if len(node.Input) != 2 {
		return nil
	}
	for _, sentIdx := range []int{1, 0} {
		sentName := node.Input[sentIdx]
		otherName := node.Input[1-sentIdx]

		if sentScalar, ok := isMaskSentinelTensor(m, sentName, true); ok {
			if !sentScalar && !sdpaBoolMaskShapeSafe(m, otherName, mulOutName) {
				continue
			}
			// Classic HuggingFace BERT/RoBERTa: Mul(Sub(1.0, x), -10000 / -3.4e38).
			if yName, ok := matchSubOneMinusX(m, otherName); ok {
				return sdpaBoolMaskFromEqualOne(m, yName)
			}
			// Mul(Cast(float, boolMask), -1e9) or Mul(Cast(float, Sub(1, intMask)), -1e9).
			if otherNode, ok := m.NodeOutputToNode[otherName]; ok && otherNode.OpType == "Cast" && len(otherNode.Input) == 1 &&
				isONNXFloatDType(onnxgomlx.GetIntAttrOr(otherNode, "to", 0)) {
				origName := otherNode.Input[0]
				if yName, ok := matchSubOneMinusX(m, origName); ok {
					return sdpaBoolMaskFromEqualOne(m, yName)
				}
				if isKnownBoolTensor(m, origName) {
					return simplifyBoolAttendFromCond(m, origName, true, 0)
				}
			}
		}

		// Positive sentinel variant: Mul(Sub(x, 1.0), +10000) or Mul(Add(x, -1.0), +10000).
		if sentScalar, ok := isMaskSentinelTensor(m, sentName, false); ok {
			if !sentScalar && !sdpaBoolMaskShapeSafe(m, otherName, mulOutName) {
				continue
			}
			if yName, ok := matchXMinusOne(m, otherName); ok {
				return sdpaBoolMaskFromEqualOne(m, yName)
			}
		}
	}
	return nil
}

// simplifyBoolAttendFromCond returns a simplified sdpaBoolMaskExpr for either `cond`
// (inverted=false) or `!cond` (inverted=true).
func simplifyBoolAttendFromCond(m *onnxgomlx.Model, condName string, inverted bool, depth int) *sdpaBoolMaskExpr {
	if depth > 8 {
		return leafFromCond(condName, inverted)
	}
	node, ok := m.NodeOutputToNode[condName]
	if !ok || node == nil {
		return leafFromCond(condName, inverted)
	}
	switch node.OpType {
	case "Identity":
		if len(node.Input) == 1 {
			return simplifyBoolAttendFromCond(m, node.Input[0], inverted, depth+1)
		}
	case "Not":
		if len(node.Input) == 1 {
			return simplifyBoolAttendFromCond(m, node.Input[0], !inverted, depth+1)
		}
	case "Cast":
		if len(node.Input) == 1 && onnxgomlx.GetIntAttrOr(node, "to", 0) == int(protos.TensorProto_BOOL) {
			inner := node.Input[0]
			// Unwrap redundant bool->bool casts (PyTorch ONNX exports often emit Cast(bool)->Cast(bool)).
			if isKnownBoolTensor(m, inner) {
				return simplifyBoolAttendFromCond(m, inner, inverted, depth+1)
			}
			// !Cast(bool, Sub(1.0, y)) <=> Sub(1.0, y) == 0 <=> y == 1.
			if inverted {
				if yName, ok := matchSubOneMinusX(m, inner); ok {
					return sdpaBoolMaskFromEqualOne(m, yName)
				}
				if yName, ok := matchXMinusOne(m, inner); ok {
					return sdpaBoolMaskFromEqualOne(m, yName)
				}
			}
		}
	case "Equal":
		if len(node.Input) == 2 {
			// !Equal(x, 0) <=> x != 0; Equal(x, 0) <=> x == 0.
			for _, zIdx := range []int{1, 0} {
				if v, ok := tryGetConstantScalarOk(m, node.Input[zIdx]); ok && v == 0 {
					xName := node.Input[1-zIdx]
					if isScalarConstantShape(m, node.Input[zIdx]) {
						if inverted {
							return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafDirect, SourceName: xName}
						}
						return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafNot, SourceName: xName}
					}
				}
			}
		}
	}
	return leafFromCond(condName, inverted)
}

func leafFromCond(condName string, inverted bool) *sdpaBoolMaskExpr {
	if inverted {
		return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafNot, SourceName: condName}
	}
	return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafDirect, SourceName: condName}
}

// sdpaBoolMaskFromEqualOne returns an sdpaBoolMaskExpr testing `x == 1`, unwrapping
// numeric Cast/Identity nodes so e.g. Cast(float32, attention_mask_int64) compares the
// original integer tensor directly (or uses a Bool tensor directly with zero ops).
func sdpaBoolMaskFromEqualOne(m *onnxgomlx.Model, name string) *sdpaBoolMaskExpr {
	cur := name
	for range 8 {
		node, ok := m.NodeOutputToNode[cur]
		if !ok || node == nil || len(node.Input) != 1 {
			break
		}
		if node.OpType == "Identity" {
			cur = node.Input[0]
			continue
		}
		if node.OpType == "Cast" && isONNXNumericDType(onnxgomlx.GetIntAttrOr(node, "to", 0)) {
			cur = node.Input[0]
			continue
		}
		break
	}
	if isKnownBoolTensor(m, cur) {
		return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafDirect, SourceName: cur}
	}
	return &sdpaBoolMaskExpr{Op: sdpaBoolMaskLeafEqualOne, SourceName: cur}
}

// matchSubOneMinusX checks if name is Sub(1, x) where 1 is a scalar constant.
func matchSubOneMinusX(m *onnxgomlx.Model, name string) (string, bool) {
	node, ok := m.NodeOutputToNode[name]
	if !ok || node == nil || node.OpType != "Sub" || len(node.Input) != 2 {
		return "", false
	}
	if v, ok := tryGetConstantScalarOk(m, node.Input[0]); ok && v == 1 && isScalarConstantShape(m, node.Input[0]) {
		return node.Input[1], true
	}
	return "", false
}

// matchXMinusOne checks if name is Sub(x, 1) or Add(x, -1) / Add(-1, x).
func matchXMinusOne(m *onnxgomlx.Model, name string) (string, bool) {
	node, ok := m.NodeOutputToNode[name]
	if !ok || node == nil || len(node.Input) != 2 {
		return "", false
	}
	if node.OpType == "Sub" {
		if v, ok := tryGetConstantScalarOk(m, node.Input[1]); ok && v == 1 && isScalarConstantShape(m, node.Input[1]) {
			return node.Input[0], true
		}
	}
	if node.OpType == "Add" {
		for _, cIdx := range []int{1, 0} {
			if v, ok := tryGetConstantScalarOk(m, node.Input[cIdx]); ok && v == -1 && isScalarConstantShape(m, node.Input[cIdx]) {
				return node.Input[1-cIdx], true
			}
		}
	}
	return "", false
}

// isCondNonZeroOf reports whether condName is true iff targetName != 0
// (e.g. Cast(bool, Cast(bool, targetName)) or Not(Equal(targetName, 0))).
func isCondNonZeroOf(m *onnxgomlx.Model, condName, targetName string) bool {
	if targetName == "" {
		return false
	}
	cur := condName
	strippedBoolCast := false
	for range 8 {
		node, ok := m.NodeOutputToNode[cur]
		if !ok || node == nil {
			break
		}
		if node.OpType == "Identity" && len(node.Input) == 1 {
			cur = node.Input[0]
			continue
		}
		if node.OpType == "Cast" && len(node.Input) == 1 &&
			onnxgomlx.GetIntAttrOr(node, "to", 0) == int(protos.TensorProto_BOOL) {
			strippedBoolCast = true
			cur = node.Input[0]
			if cur == targetName {
				return true
			}
			continue
		}
		if !strippedBoolCast && node.OpType == "Not" && len(node.Input) == 1 {
			return isCondZeroOf(m, node.Input[0], targetName)
		}
		break
	}
	return strippedBoolCast && cur == targetName
}

// isCondZeroOf reports whether condName is true iff targetName == 0
// (e.g. Equal(targetName, 0) or Not(Cast(bool, targetName))).
func isCondZeroOf(m *onnxgomlx.Model, condName, targetName string) bool {
	if targetName == "" {
		return false
	}
	cur := condName
	for range 8 {
		node, ok := m.NodeOutputToNode[cur]
		if !ok || node == nil {
			return false
		}
		switch node.OpType {
		case "Identity":
			if len(node.Input) != 1 {
				return false
			}
			cur = node.Input[0]
		case "Cast":
			if len(node.Input) != 1 || onnxgomlx.GetIntAttrOr(node, "to", 0) != int(protos.TensorProto_BOOL) {
				return false
			}
			cur = node.Input[0]
		case "Equal":
			if len(node.Input) != 2 {
				return false
			}
			for _, tIdx := range []int{0, 1} {
				if node.Input[tIdx] == targetName {
					if v, ok := tryGetConstantScalarOk(m, node.Input[1-tIdx]); ok && v == 0 {
						return true
					}
				}
			}
			return false
		case "Not":
			if len(node.Input) != 1 {
				return false
			}
			return isCondNonZeroOf(m, node.Input[0], targetName)
		default:
			return false
		}
	}
	return false
}

// isMaskSentinelTensor reports whether name is a constant scalar or broadcast/filled constant
// tensor whose value is a mask sentinel (<= -1e4 if negative=true, or >= 1e4 if negative=false).
// isScalar is true if the constant is scalar / single-element broadcast-1.
func isMaskSentinelTensor(m *onnxgomlx.Model, name string, negative bool) (isScalar, ok bool) {
	return matchConstantFilledTensor(m, name, func(v float64) bool {
		if math.IsNaN(v) {
			return false
		}
		if negative {
			return v <= sdpaMaskSentinelThreshold
		}
		return v >= -sdpaMaskSentinelThreshold
	}, 0)
}

// isZeroConstantTensor reports whether name is a constant scalar or broadcast/filled constant
// tensor of zeros (including ConstantOfShape with default zero value and Expand(0, shape)).
func isZeroConstantTensor(m *onnxgomlx.Model, name string) (isScalar, ok bool) {
	return matchConstantFilledTensor(m, name, func(v float64) bool { return v == 0 }, 0)
}

func matchConstantFilledTensor(m *onnxgomlx.Model, name string, pred func(float64) bool, depth int) (isScalar, ok bool) {
	if depth > 8 || name == "" {
		return false, false
	}
	if v, ok := tryGetConstantScalarOk(m, name); ok {
		return true, pred(v)
	}
	node, exists := m.NodeOutputToNode[name]
	if !exists || node == nil {
		return false, false
	}
	switch node.OpType {
	case "Identity", "Cast":
		if len(node.Input) == 1 {
			return matchConstantFilledTensor(m, node.Input[0], pred, depth+1)
		}
	case "Expand":
		if len(node.Input) >= 1 {
			if _, ok := matchConstantFilledTensor(m, node.Input[0], pred, depth+1); ok {
				return false, true
			}
		}
	case "ConstantOfShape":
		// ONNX ConstantOfShape defaults to float32 0.0 when the "value" attribute is omitted.
		val := 0.0
		for _, attr := range node.Attribute {
			if attr.Name == "value" && attr.T != nil {
				v, ok := tensorProtoToScalarOk(attr.T)
				if !ok {
					return false, false
				}
				val = v
				break
			}
		}
		return false, pred(val)
	}
	return false, false
}

// sdpaBoolMaskShapeSafe verifies that replacing maskOutName with srcName preserves rank
// and KV-axis size so broadcasting into attention.Core remains valid.
func sdpaBoolMaskShapeSafe(m *onnxgomlx.Model, srcName, maskOutName string) bool {
	srcShape := m.InferShapeForName(srcName)
	outShape := m.InferShapeForName(maskOutName)

	srcRank := len(srcShape.Dimensions)
	if srcShape.Dimensions == nil {
		srcRank = inferRank(m, srcName, 0)
	}
	outRank := len(outShape.Dimensions)
	if outShape.Dimensions == nil {
		outRank = inferRank(m, maskOutName, 0)
	}
	if srcRank < 2 || outRank < 2 || srcRank != outRank {
		return false
	}

	if srcShape.Dimensions != nil {
		last := len(srcShape.Dimensions) - 1
		if outShape.Dimensions != nil {
			// KV sequence dimension must not shrink from non-1 to 1.
			if srcShape.Dimensions[last] == 1 && outShape.Dimensions[last] != 1 {
				return false
			}
			for i := range last {
				sd, od := srcShape.Dimensions[i], outShape.Dimensions[i]
				if sd != od && sd != 1 {
					return false
				}
			}
		} else if srcShape.Dimensions[last] == 1 {
			// Output shape is unknown; be conservative if src has a unit KV axis.
			return false
		}
	}
	return true
}

func isScalarConstantShape(m *onnxgomlx.Model, name string) bool {
	s := m.InferShapeForName(name)
	if s.Dimensions == nil {
		// Initializer/Constant verified single-element by tryGetConstantScalarOk.
		return true
	}
	return len(s.Dimensions) == 0 || (len(s.Dimensions) == 1 && s.Dimensions[0] == 1)
}

func isKnownBoolTensor(m *onnxgomlx.Model, name string) bool {
	if s := m.InferShapeForName(name); s.DType == dtypes.Bool {
		return true
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok || node == nil {
		return false
	}
	switch node.OpType {
	case "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Not", "And", "Or", "Xor":
		return true
	case "Cast":
		return onnxgomlx.GetIntAttrOr(node, "to", 0) == int(protos.TensorProto_BOOL)
	case "Identity", "Unsqueeze", "Squeeze", "Reshape", "Expand", "Transpose":
		if len(node.Input) > 0 {
			return isKnownBoolTensor(m, node.Input[0])
		}
	}
	return false
}

func isONNXFloatDType(dt int) bool {
	switch protos.TensorProto_DataType(dt) {
	case protos.TensorProto_FLOAT, protos.TensorProto_DOUBLE,
		protos.TensorProto_FLOAT16, protos.TensorProto_BFLOAT16:
		return true
	default:
		return false
	}
}

func isONNXNumericDType(dt int) bool {
	if isONNXFloatDType(dt) {
		return true
	}
	switch protos.TensorProto_DataType(dt) {
	case protos.TensorProto_INT8, protos.TensorProto_INT16, protos.TensorProto_INT32, protos.TensorProto_INT64,
		protos.TensorProto_UINT8, protos.TensorProto_UINT16, protos.TensorProto_UINT32, protos.TensorProto_UINT64:
		return true
	default:
		return false
	}
}

// tryGetConstantScalarOk extracts a single-element constant scalar float64, distinguishing
// an actual constant 0.0 from a non-constant tensor. Unwraps Identity and numeric Casts.
func tryGetConstantScalarOk(m *onnxgomlx.Model, name string) (float64, bool) {
	return tryGetConstantScalarOkDepth(m, name, 0)
}

func tryGetConstantScalarOkDepth(m *onnxgomlx.Model, name string, depth int) (float64, bool) {
	if depth > 8 || name == "" {
		return 0, false
	}
	if tp, ok := m.VariableNameToValue[name]; ok {
		return tensorProtoToScalarOk(tp)
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok || node == nil {
		return 0, false
	}
	switch node.OpType {
	case "Constant":
		for _, attr := range node.Attribute {
			switch attr.Name {
			case "value":
				if attr.T != nil {
					return tensorProtoToScalarOk(attr.T)
				}
			case "value_float":
				return float64(attr.F), true
			case "value_int":
				return float64(attr.I), true
			}
		}
		return 0, false
	case "Identity":
		if len(node.Input) == 1 {
			return tryGetConstantScalarOkDepth(m, node.Input[0], depth+1)
		}
	case "Cast":
		if len(node.Input) == 1 && isONNXNumericDType(onnxgomlx.GetIntAttrOr(node, "to", 0)) {
			return tryGetConstantScalarOkDepth(m, node.Input[0], depth+1)
		}
	}
	return 0, false
}

func tensorProtoToScalarOk(tp *protos.TensorProto) (float64, bool) {
	if tp == nil {
		return 0, false
	}
	totalElements := int64(1)
	for _, d := range tp.Dims {
		totalElements *= d
	}
	if totalElements != 1 {
		return 0, false
	}
	switch protos.TensorProto_DataType(tp.DataType) {
	case protos.TensorProto_FLOAT:
		if len(tp.FloatData) > 0 {
			return float64(tp.FloatData[0]), true
		}
		if len(tp.RawData) >= 4 {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(tp.RawData[:4]))), true
		}
	case protos.TensorProto_DOUBLE:
		if len(tp.DoubleData) > 0 {
			return tp.DoubleData[0], true
		}
		if len(tp.RawData) >= 8 {
			return math.Float64frombits(binary.LittleEndian.Uint64(tp.RawData[:8])), true
		}
	case protos.TensorProto_FLOAT16:
		if len(tp.RawData) >= 2 {
			return float64(float16.Float16(binary.LittleEndian.Uint16(tp.RawData[:2])).Float32()), true
		}
		if len(tp.Int32Data) > 0 {
			return float64(float16.Float16(uint16(tp.Int32Data[0])).Float32()), true
		}
	case protos.TensorProto_BFLOAT16:
		if len(tp.RawData) >= 2 {
			return float64(bfloat16.BFloat16(binary.LittleEndian.Uint16(tp.RawData[:2])).Float32()), true
		}
		if len(tp.Int32Data) > 0 {
			return float64(bfloat16.BFloat16(uint16(tp.Int32Data[0])).Float32()), true
		}
	case protos.TensorProto_INT32:
		if len(tp.Int32Data) > 0 {
			return float64(tp.Int32Data[0]), true
		}
		if len(tp.RawData) >= 4 {
			return float64(int32(binary.LittleEndian.Uint32(tp.RawData[:4]))), true
		}
	case protos.TensorProto_INT64:
		if len(tp.Int64Data) > 0 {
			return float64(tp.Int64Data[0]), true
		}
		if len(tp.RawData) >= 8 {
			return float64(int64(binary.LittleEndian.Uint64(tp.RawData[:8]))), true
		}
	case protos.TensorProto_INT16:
		if len(tp.Int32Data) > 0 {
			return float64(int16(tp.Int32Data[0])), true
		}
		if len(tp.RawData) >= 2 {
			return float64(int16(binary.LittleEndian.Uint16(tp.RawData[:2]))), true
		}
	case protos.TensorProto_INT8:
		if len(tp.Int32Data) > 0 {
			return float64(int8(tp.Int32Data[0])), true
		}
		if len(tp.RawData) >= 1 {
			return float64(int8(tp.RawData[0])), true
		}
	case protos.TensorProto_UINT8:
		if len(tp.Int32Data) > 0 {
			return float64(uint8(tp.Int32Data[0])), true
		}
		if len(tp.RawData) >= 1 {
			return float64(tp.RawData[0]), true
		}
	case protos.TensorProto_BOOL:
		if len(tp.Int32Data) > 0 {
			if tp.Int32Data[0] != 0 {
				return 1, true
			}
			return 0, true
		}
		if len(tp.RawData) >= 1 {
			if tp.RawData[0] != 0 {
				return 1, true
			}
			return 0, true
		}
	}
	return 0, false
}
