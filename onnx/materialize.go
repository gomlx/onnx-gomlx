package onnx

import (
	"fmt"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/tensors"
)

// materializeConstantExpression materializes a node to its constant expression.
//
// This is required for ONNX ops that take dynamic values (like axes and shapes), but for which GoMLX only accept
// static (materialized) values.
//
// If the node depends on non-constant values (like input parameters) this fails with an exception.
func materializeConstantExpression(node *Node) *tensors.Tensor {
	if node.Type() == NodeTypeConstant {
		return node.ConstantValue()
	}

	// TODO: calculate constant expressions if possible
	fmt.Printf("Node to materialize dependencies:\n")
	isConstant := traverseSubGraph(node, types.MakeSet[*Node]())
	fmt.Printf("\tisConstant=%v\n", isConstant)
	exceptions.Panicf("dynamic value for %s is not a constant and cannot be materialized (isConstant=%v)", node, isConstant)
	return nil
}

func traverseSubGraph(node *Node, visited types.Set[*Node]) bool {
	if visited.Has(node) {
		return true
	}
	fmt.Printf("\t#%d %s\n", node.Id(), node)
	isConstant := node.Type() != NodeTypeParameter
	for _, input := range node.Inputs() {
		isConstant = isConstant && traverseSubGraph(input, visited)
	}
	return isConstant
}
