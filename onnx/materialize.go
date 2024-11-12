package onnx

import (
	"fmt"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"github.com/pkg/errors"
	"strings"
)

// nonConstantDependencies returns the non-constant dependencies: inputs or variables.
func (m *Model) nonConstantDependencies(nodeOutputName string) (inputs, variables []string) {
	visitedNodes := types.MakeSet[string]()
	return m.recursiveNonConstantDependencies(nodeOutputName, visitedNodes, inputs, variables)
}

// recursiveNonConstantDependencies is the recursive implementation of nonConstantDependencies.
// Use nonConstantDependencies.
func (m *Model) recursiveNonConstantDependencies(name string, visitedNodes types.Set[string], nonConstInputs, variables []string) ([]string, []string) {
	visitedNodes.Insert(name)
	if _, found := m.variableNameToValue[name]; found {
		// Record a variable dependency.
		if m.isVariableConstant(name) {
			// Constant variable, ok.
			return nonConstInputs, variables
		}
		variables = append(variables, name)
		return nonConstInputs, variables
	}
	if m.inputsNameSet.Has(name) {
		// Input dependency is recorded as non-constant only if the input is not fed as a constant.
		if m.inputsAsConstants == nil || m.inputsAsConstants[name] == nil {
			nonConstInputs = append(nonConstInputs, name)
		}
		return nonConstInputs, variables
	}

	// Recurse into the inputs of the node that generated the `name` output.
	node := m.nodeOutputToNode[name]
	if node == nil {
		exceptions.Panicf("nonConstantDepedencies given an unknown node output name %q", name)
		panic(nil) // lint.
	}
	if node.OpType == "Shape" {
		// Shape op returns a static value after converting to GoMLX, independent of inputs.
		// So we don't recurse into its inputs.
		return nonConstInputs, variables
	}
	for _, input := range node.Input {
		if visitedNodes.Has(input) {
			continue
		}
		nonConstInputs, variables = m.recursiveNonConstantDependencies(input, visitedNodes, nonConstInputs, variables)
	}
	return nonConstInputs, variables
}

// isVariableConstant tries to guess if the variable can be used as a constant during the graph construction.
// For instance as the dimension for a "Reshape" or axis for a "Slice" method.
// Some ONNX models use variables instead of constants.
//
// varName must be an existing variable name.
func (m *Model) isVariableConstant(varName string) bool {
	sizeLimit := 100 // Max size to be accepted as constant.
	lowerName := strings.ToLower(varName)
	if strings.Index(lowerName, "constant") >= 0 {
		// If there is "constant" in the name, we assume constant at a higher size.
		sizeLimit = 10_000
	} else if strings.Index(lowerName, "const") >= 0 {
		// With less confidence...
		sizeLimit = 1_000
	}
	tensorProto := m.variableNameToValue[varName]
	shape, err := togomlx.Shape(tensorProto)
	if err != nil {
		panic(errors.WithMessagef(err, "ONNX variable %q has an invalid shape", varName))
	}
	return shape.DType.IsInt() && shape.Size() <= sizeLimit
}

// materializeConstantExpression materializes a node to its constant expression.
//
// This is required for ONNX ops that take dynamic values (like axes and shapes), but for which GoMLX only accept
// static (materialized) values.
//
// If the node depends on non-constant values (like input parameters) this fails with an exception.
func (m *Model) materializeConstantExpression(nodeOutputName string, convertedOutputs map[string]*Node) (*tensors.Tensor, error) {
	// Easy reply: if the node is already a constant.
	node := convertedOutputs[nodeOutputName]
	if node == nil {
		return nil, errors.Errorf("node output %q hasn't been converted yet, so we can't materializeConstantExpression!?", nodeOutputName)
	}
	if node.Type() == NodeTypeConstant {
		return node.ConstantValue(), nil
	}

	// See if it is possible: if subgraph that generated the node is a constant expression.
	nonConstInputs, nonConstVariables := m.nonConstantDependencies(nodeOutputName)
	if len(nonConstInputs) > 0 || len(nonConstVariables) > 0 {
		// Add shape info for variables.
		varDesc := make([]string, 0, len(nonConstVariables))
		for _, varName := range nonConstVariables {
			// We discard the error, because we know this conversion works already, to have reached this point.
			shape, _ := togomlx.Shape(m.variableNameToValue[varName])
			varDesc = append(varDesc, fmt.Sprintf("%q (%s)", varName, shape))
		}
		return nil, errors.Errorf("cannot materialize constant/static value for %q: it depends on non-constant: inputs=%q, variables: %s",
			nodeOutputName, nonConstInputs, strings.Join(varDesc, ", "))
	}

	// Evaluate constant sub-expression in a newly created sub-graph.
	backend := node.Graph().Backend()
	var result *tensors.Tensor
	err := exceptions.TryCatch[error](func() {
		result = ExecOnce(backend, func(g *Graph) *Node {
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

	// Check in the original graph being converted if this node was converted as a constant (for instance for nodes like "Shape"),
	// in which case we take the constant value and inject it directly in the new constant expression graph.
	if originalNode, found := originalConvertedOutput[nodeOutputName]; found {
		if originalNode.Type() == NodeTypeConstant {
			// Duplicate the constant in the new graph.
			constConvertedOutputs[nodeOutputName] = Const(g, originalNode.ConstantValue())
			return
		}
	}

	// Check for constant variables.
	if tensorNode, found := m.variableNameToValue[nodeOutputName]; found {
		if !m.isVariableConstant(nodeOutputName) {
			exceptions.Panicf("attempting to materialize as constant variable %q, which we don't think is constant", nodeOutputName)
		}
		t, err := togomlx.Tensor(tensorNode)
		if err != nil {
			panic(errors.WithMessagef(err, "attempting to materialize variable %q as constant", nodeOutputName))
		}
		constConvertedOutputs[nodeOutputName] = Const(g, t)
		return
	}

	// Find node generating this output.
	onnxNode, found := m.nodeOutputToNode[nodeOutputName]
	if !found {
		exceptions.Panicf("ONNX node %q not found as the output of an Op, and not a constant either -- is this really a constant expression!?", nodeOutputName)
	}

	// Recursively converts the inputs of the onnxNode:
	for _, inputName := range onnxNode.Input {
		m.recursiveMaterializeConstantExpression(inputName, g, constConvertedOutputs, originalConvertedOutput)
	}

	// And now convert the node itself.
	m.convertNode(g, onnxNode, constConvertedOutputs)
}
