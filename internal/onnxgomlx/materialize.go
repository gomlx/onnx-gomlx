package onnxgomlx

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
)

// IsConstantExpression returns true if the subgraph producing nodeOutputName
// has no dependencies on graph inputs or non-constant variables.
func (m *Model) IsConstantExpression(nodeOutputName string) bool {
	inputs, vars, ctxNodes := m.nonConstantDependencies(nodeOutputName)
	return len(inputs) == 0 && len(vars) == 0 && len(ctxNodes) == 0
}

// MaterializeConstantExpression materializes a node to its constant expression.
func (m *Model) MaterializeConstantExpression(nodeOutputName string, convertedOutputs map[string]*Node) (*tensors.Tensor, error) {
	return m.materializeConstantExpression(nodeOutputName, convertedOutputs)
}

// MaterializeConstantScalar materializes a scalar node to its constant float64 value.
func (m *Model) MaterializeConstantScalar(nodeOutputName string, convertedOutputs map[string]*Node) (float64, error) {
	t, err := m.materializeConstantExpression(nodeOutputName, convertedOutputs)
	if err != nil {
		return 0, err
	}
	return TensorToScalarFloat64(t), nil
}

// TensorToScalarFloat64 converts a 1-element tensor (scalar or 1D length 1) of any numeric type to float64.
func TensorToScalarFloat64(t *tensors.Tensor) float64 {
	if t.Shape().Size() != 1 {
		return 0
	}
	var val float64
	_ = t.ConstFlatData(func(flat any) {
		switch f := flat.(type) {
		case []float32:
			val = float64(f[0])
		case []float64:
			val = f[0]
		case []int64:
			val = float64(f[0])
		case []int32:
			val = float64(f[0])
		case []int16:
			val = float64(f[0])
		case []int8:
			val = float64(f[0])
		case []uint64:
			val = float64(f[0])
		case []uint32:
			val = float64(f[0])
		case []uint16:
			val = float64(f[0])
		case []uint8:
			val = float64(f[0])
		case []float16.Float16:
			val = f[0].Float64()
		case []bfloat16.BFloat16:
			val = f[0].Float64()
		}
	})
	return val
}

// nonConstantDependencies returns the non-constant dependencies: inputs or variables.
func (m *Model) nonConstantDependencies(nodeOutputName string) (inputs, variables []string, contextNodes []*protos.NodeProto) {
	visitedNodes := sets.Make[string]()
	return m.recursiveNonConstantDependencies(nodeOutputName, visitedNodes, inputs, variables, contextNodes)
}

// recursiveNonConstantDependencies is the recursive implementation of nonConstantDependencies.
// Use nonConstantDependencies.
func (m *Model) recursiveNonConstantDependencies(name string, visitedNodes sets.Set[string],
	nonConstInputs, variables []string, contextNodes []*protos.NodeProto) ([]string, []string, []*protos.NodeProto) {
	visitedNodes.Insert(name)
	if _, found := m.VariableNameToValue[name]; found {
		// Record a variable dependency.
		if m.isVariableConstant(name) {
			// Constant variable, ok.
			return nonConstInputs, variables, contextNodes
		}
		variables = append(variables, name)
		return nonConstInputs, variables, contextNodes
	}
	if m.InputsNameSet.Has(name) {
		// Input dependency is recorded as non-constant only if the input is not fed as a constant.
		if m.InputsAsConstants == nil || m.InputsAsConstants[name] == nil {
			nonConstInputs = append(nonConstInputs, name)
		}
		return nonConstInputs, variables, contextNodes
	}

	// Recurse into the inputs of the node that generated the `name` output.
	node := m.NodeOutputToNode[name]
	if node == nil {
		exceptions.Panicf("nonConstantDepedencies given an unknown node output name %q", name)
		return nil, nil, nil
	}
	if opRequiresContext(node.OpType) {
		contextNodes = append(contextNodes, node)
	}
	if node.OpType == "Shape" || node.OpType == "Size" {
		// Shape and Size ops return static values after converting to GoMLX, independent of inputs.
		// So we don't recurse into their inputs.
		return nonConstInputs, variables, contextNodes
	}
	for _, input := range node.Input {
		if visitedNodes.Has(input) {
			continue
		}
		nonConstInputs, variables, contextNodes = m.recursiveNonConstantDependencies(input, visitedNodes, nonConstInputs, variables, contextNodes)
	}
	return nonConstInputs, variables, contextNodes
}

// isVariableConstant tries to guess if the variable can be used as a constant during the graph construction.
// For instance, as the dimension for a "Reshape" or axis for a "Slice" method.
// Some ONNX models use variables instead of constants.
//
// varName must be an existing variable name.
func (m *Model) isVariableConstant(varName string) bool {
	sizeLimit := 100 // Max size to be accepted as constant.
	lowerName := strings.ToLower(varName)
	if strings.Contains(lowerName, "constant") {
		// If there is "constant" in the name, we assume constant at a higher size.
		sizeLimit = 10_000
	} else if strings.Contains(lowerName, "const") {
		// With less confidence...
		sizeLimit = 1_000
	}
	tensorProto := m.VariableNameToValue[varName]
	shape, err := Shape(tensorProto)
	if err != nil {
		panic(errors.WithMessagef(err, "ONNX variable %q has an invalid shape", varName))
	}
	if shape.Size() > sizeLimit {
		return false
	}
	// Variables with "constant" in the name are treated as constants regardless of
	// dtype — they may be float values that get cast to int via ConvertDType (e.g.
	// when Concat dtype promotion casts a Float32 constant to Int64 for a shape).
	if strings.Contains(lowerName, "const") {
		return true
	}
	// For variables without "constant" in the name, only accept integer types
	// since they're more likely to represent shape/axis metadata.
	return shape.DType.IsInt()
}

// materializeConstantExpression materializes a node to its constant expression.
//
// This is required for ONNX ops that take dynamic values (like axes and shapes), but for which GoMLX only accepts
// static (materialized) values.
//
// If the node depends on non-constant values (like input parameters), this fails with an exception.
func (m *Model) materializeConstantExpression(nodeOutputName string, convertedOutputs map[string]*Node) (*tensors.Tensor, error) {
	// Easy reply: if the node is already a constant.
	node := convertedOutputs[nodeOutputName]
	if node == nil {
		return nil, errors.Errorf("node output %q hasn't been converted yet, so we can't materializeConstantExpression!?", nodeOutputName)
	}
	if node.Type() == NodeTypeConstant {
		if cVal := node.ConstantValue(); cVal != nil {
			return cVal, nil
		}
	}

	// See if it is possible: if the subgraph that generated the node is a constant expression.
	nonConstInputs, nonConstVariables, contextNodes := m.nonConstantDependencies(nodeOutputName)
	if len(nonConstInputs) > 0 || len(nonConstVariables) > 0 || len(contextNodes) > 0 {
		// Add shape info for variables.
		varDesc := make([]string, 0, len(nonConstVariables))
		for _, varName := range nonConstVariables {
			// We discard the error because we know this conversion works already, to have reached this point.
			shape, _ := Shape(m.VariableNameToValue[varName])
			varDesc = append(varDesc, fmt.Sprintf("%q (%s)", varName, shape))
		}
		opsDesc := make([]string, 0, len(contextNodes))
		for _, node := range contextNodes {
			// We discard the error because we know this conversion works already, to have reached this point.
			opsDesc = append(opsDesc, node.String())
		}
		return nil, errors.Errorf("cannot materialize constant/static value for %q: it depends on non-constant: inputs=%q, variables: %s, ops with context: %s",
			nodeOutputName, nonConstInputs, strings.Join(varDesc, ", "), strings.Join(opsDesc, ", "))
	}

	// Evaluate constant sub-expression in a newly created sub-
	backend := node.Graph().Backend()
	var result *tensors.Tensor
	err := exceptions.TryCatch[error](func() {
		result = MustExecOnce(backend, func(g *Graph) *Node {
			constConvertedOutputs := make(map[string]*Node)
			m.recursiveMaterializeConstantExpression(nodeOutputName, g, constConvertedOutputs, convertedOutputs)
			return constConvertedOutputs[nodeOutputName]
		})
	})
	if err != nil {
		return nil, errors.WithMessage(err, "while evaluating constant sub-expression")
	}
	return result, nil
}

// recursiveMaterializeConstantExpression creates a GoMLX graph with the constant expressions in constConvertedOutputs.
// It may use the original converted graph in originalConvertedOutput, but it doesn't change it.
func (m *Model) recursiveMaterializeConstantExpression(nodeOutputName string, g *Graph, constConvertedOutputs, originalConvertedOutput map[string]*Node) {
	if _, found := constConvertedOutputs[nodeOutputName]; found {
		// Already converted.
		return
	}

	// Check in the original graph being converted if this node was converted as a constant (for instance, for nodes like "Shape"),
	// in which case we take the constant value and inject it directly in the new constant expression
	if originalNode, found := originalConvertedOutput[nodeOutputName]; found {
		if originalNode.Type() == NodeTypeConstant {
			if cVal := originalNode.ConstantValue(); cVal != nil {
				// Duplicate the constant in the new graph.
				constConvertedOutputs[nodeOutputName] = Const(g, cVal)
				return
			}
		}
	}

	// Check for constant variables.
	if tensorNode, found := m.VariableNameToValue[nodeOutputName]; found {
		if !m.isVariableConstant(nodeOutputName) {
			exceptions.Panicf("attempting to materialize as constant variable %q, which we don't think is constant", nodeOutputName)
		}
		t, err := ONNXTensorToGoMLX(m.Backend, tensorNode, m.getExternalDataReader())
		if err != nil {
			panic(errors.WithMessagef(err, "attempting to materialize variable %q as constant", nodeOutputName))
		}
		constConvertedOutputs[nodeOutputName] = Const(g, t)
		// TODO: mark variable as used for constant-expression and make sure it is also used in the final model, and
		// 		 try to make as such that if it changes, the graph is rebuilt.
		return
	}

	// Find the node generating this output.
	onnxNode, found := m.NodeOutputToNode[nodeOutputName]
	if !found {
		exceptions.Panicf("ONNX node %q not found as the output of an Op, and not a constant either -- is this really a constant expression!?", nodeOutputName)
	}
	if opRequiresContext(onnxNode.OpType) {
		// Operation requires a context, which is not supported when materializing constant sub-expressions.
		exceptions.Panicf("attempting to materialize expression with operation %q, which is not supported for materialization: %s", onnxNode.OpType, onnxNode)
	}

	// Recursively converts the inputs of the onnxNode:
	for _, inputName := range onnxNode.Input {
		m.recursiveMaterializeConstantExpression(inputName, g, constConvertedOutputs, originalConvertedOutput)
	}

	// And now convert the node itself.
	m.convertNode(nil, g, onnxNode, constConvertedOutputs)
}
