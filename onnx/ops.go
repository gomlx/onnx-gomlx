package onnx

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
)

// This file implements the ONNX operators that don't have a direct corresponding GoMLX operator.

// MatMul behaves like numpy.matmul
func MatMul(a, b *Node) *Node {
	if a.Rank() == 0 || b.Rank() == 0 {
		exceptions.Panicf("MatMul expects two tensors with rank > 0, got %v and %v", a.Rank(), b.Rank())
	}
	// Not handling yet cases where a or b are of higher rank than 2.
	return Dot(a, b)
}
