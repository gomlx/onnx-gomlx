package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertMod converts an ONNX Mod node to GoMLX.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Mod.html
//
// The fmod attribute (default 0) controls the behavior:
//   - fmod=1: C-style fmod — the result has the same sign as the dividend.
//     This maps directly to GoMLX's Mod/Rem.
//   - fmod=0: Python-style modulo — the result has the same sign as the divisor.
func (m *Model) convertMod(node *protos.NodeProto, inputs []*Node) *Node {
	fmod := GetIntAttrOr(node, "fmod", 0)

	operands := onnxImplicitExpansion([]*Node{inputs[0], inputs[1]})
	lhs, rhs := operands[0], operands[1]
	lhs, rhs = m.checkOrPromoteDTypes(lhs, rhs)

	r := Mod(lhs, rhs)
	if fmod == 1 {
		return r
	}

	// fmod=0: adjust C-style remainder to Python-style modulo.
	// Compare signs directly: multiplying r by rhs can overflow for integers.
	// A zero remainder needs no adjustment.
	zero := ScalarZero(r.Graph(), r.DType())
	differentSigns := LogicalXor(LessThan(r, zero), LessThan(rhs, zero))
	needsAdjust := LogicalAnd(NotEqual(r, zero), differentSigns)
	return Where(needsAdjust, Add(r, rhs), r)
}
