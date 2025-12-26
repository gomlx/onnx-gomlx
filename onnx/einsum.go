package onnx

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// convertEinsum converts the ONNX Einsum operation to GoMLX operations.
// Einsum is a generalized tensor contraction operation using Einstein summation notation.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Einsum.html
func convertEinsum(node *protos.NodeProto, inputs []*Node) *Node {
	// Get the equation attribute
	equation := getStringAttrOr(node, "equation", "")
	if equation == "" {
		exceptions.Panicf("Einsum node %q missing required 'equation' attribute", node.Name)
	}

	// Currently we only support two-operand einsum
	if len(inputs) != 2 {
		exceptions.Panicf("Einsum with %d inputs not yet supported (equation: %q) in node %q", len(inputs), equation, node.Name)
	}

	// Implement specific einsum equations as needed
	switch equation {
	case "BLKD,BCD->BLKC":
		// This is: (B,L,K,D) x (B,C,D) -> (B,L,K,C)
		// Contract over D dimension, batch over B dimension
		// Input shapes:
		//   lhs: [B, L, K, D]
		//   rhs: [B, C, D]
		// Output shape: [B, L, K, C]
		//
		// We use DotGeneral:
		//   - Contract lhs[3] (D) with rhs[2] (D)
		//   - Batch over lhs[0] (B) and rhs[0] (B)
		//   - Output has batch dims, then lhs non-contracting dims (L, K), then rhs non-contracting dims (C)
		lhs := inputs[0]
		rhs := inputs[1]

		result := DotGeneral(
			lhs, []int{3}, []int{0},  // lhs: contract axis 3 (D), batch axis 0 (B)
			rhs, []int{2}, []int{0},  // rhs: contract axis 2 (D), batch axis 0 (B)
		)
		// DotGeneral output order: [batch_dims, lhs_non_contracting, rhs_non_contracting]
		// = [B, L, K, C] which matches our desired output

		return result

	default:
		exceptions.Panicf("Einsum equation %q not yet implemented in node %q", equation, node.Name)
		return nil
	}
}
