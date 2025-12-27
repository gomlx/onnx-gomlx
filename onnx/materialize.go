package onnx

import (
	"strings"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// isShapeManipulationOp returns true if the operation manipulates shape values
// (as opposed to data values). These operations are materializable if their inputs are materializable.
func isShapeManipulationOp(opType string) bool {
	switch opType {
	case "Gather", "Slice", "Concat", "Unsqueeze", "Squeeze", "Cast",
		"Add", "Sub", "Mul", "Div", "Constant", "ConstantOfShape", "Reshape":
		return true
	default:
		return false
	}
}

// nonConstantDependencies returns the non-constant dependencies: inputs or variables.
func (m *Model) nonConstantDependencies(nodeOutputName string, convertedOutputs map[string]*Node) (inputs, variables []string, contextNodes []*protos.NodeProto) {
	visitedNodes := sets.Make[string]()
	return m.recursiveNonConstantDependencies(nodeOutputName, visitedNodes, inputs, variables, contextNodes, convertedOutputs)
}

// recursiveNonConstantDependencies is the recursive implementation of nonConstantDependencies.
// Use nonConstantDependencies.
func (m *Model) recursiveNonConstantDependencies(name string, visitedNodes sets.Set[string],
	nonConstInputs, variables []string, contextNodes []*protos.NodeProto, convertedOutputs map[string]*Node) ([]string, []string, []*protos.NodeProto) {
	visitedNodes.Insert(name)

	// Check if this node was already converted to a constant in the graph
	if convertedNode, found := convertedOutputs[name]; found {
		if convertedNode.Type() == NodeTypeConstant {
			// This node is already a constant, so it doesn't add non-constant dependencies
			return nonConstInputs, variables, contextNodes
		}
		// For certain operations that are known to be materializable (like BroadcastInDim of constants),
		// we don't need to traverse their dependencies. The actual materialization will happen later
		// using tryExtractConstantShape at the StableHLO level.
		node := m.nodeOutputToNode[name]
		if node != nil && (node.OpType == "ConstantOfShape" || node.OpType == "Expand") {
			// These operations use DynamicBroadcastInDim internally, which has tryExtractConstantShape support
			// So we optimistically assume they're materializable without checking all dependencies
			return nonConstInputs, variables, contextNodes
		}
	}

	if _, found := m.variableNameToValue[name]; found {
		// Record a variable dependency.
		if m.isVariableConstant(name) {
			// Constant variable, ok.
			return nonConstInputs, variables, contextNodes
		}
		variables = append(variables, name)
		return nonConstInputs, variables, contextNodes
	}
	if m.inputsNameSet.Has(name) {
		// Input dependency is recorded as non-constant only if the input is not fed as a constant.
		if m.inputsAsConstants == nil || m.inputsAsConstants[name] == nil {
			nonConstInputs = append(nonConstInputs, name)
		}
		return nonConstInputs, variables, contextNodes
	}

	// Recurse into the inputs of the node that generated the `name` output.
	node := m.nodeOutputToNode[name]
	if node == nil {
		exceptions.Panicf("nonConstantDepedencies given an unknown node output name %q", name)
		return nil, nil, nil
	}
	if opRequiresContext(node.OpType) {
		contextNodes = append(contextNodes, node)
	}
	if node.OpType == "Shape" {
		// Shape op returns a static value after converting to GoMLX if the input has concrete dimensions.
		if len(node.Input) == 0 {
			// If we couldn't determine it's materializable, fall through to recurse
		} else if m.inputsNameSet.Has(node.Input[0]) {
			// Check if the input to Shape is a Parameter with all concrete dimensions.
			// First, check the actual converted node shape (which has concrete dims at CallGraph time)
			if inputNode, found := convertedOutputs[node.Input[0]]; found {
				actualShape := inputNode.Shape()
				allConcrete := true
				for _, dim := range actualShape.Dimensions {
					if dim < 0 {
						allConcrete = false
						break
					}
				}
				if allConcrete {
					// Shape of a Parameter with concrete dimensions is materializable
					return nonConstInputs, variables, contextNodes
				}
			}
			// Fall back to model's declared shape (for cases where node isn't in convertedOutputs yet)
			for idx, inputName := range m.InputsNames {
				if inputName == node.Input[0] {
					// Check if all dimensions are concrete (non-negative)
					allConcrete := true
					for _, dim := range m.InputsShapes[idx].Dimensions {
						if dim < 0 {
							allConcrete = false
							break
						}
					}
					if allConcrete {
						// Shape of a Parameter with concrete dimensions is materializable
						return nonConstInputs, variables, contextNodes
					}
					break
				}
			}
		} else if varTensorProto, found := m.variableNameToValue[node.Input[0]]; found {
			// Check if the input to Shape is a variable with all concrete dimensions.
			varShape, err := Shape(varTensorProto)
			if err == nil {
				// Check if all dimensions are concrete (non-negative)
				allConcrete := true
				for _, dim := range varShape.Dimensions {
					if dim < 0 {
						allConcrete = false
						break
					}
				}
				if allConcrete {
					// Shape of a variable with concrete dimensions is materializable
					return nonConstInputs, variables, contextNodes
				}
			}
		} else if _, found := convertedOutputs[node.Input[0]]; found {
			// Input to Shape is an intermediate tensor.
			// DON'T mark as materializable: ONNX models often extract shapes from one tensor
			// and use them to reshape a different tensor. This relies on DynamicReshape.
		}
		// If we couldn't determine it's materializable, fall through to recurse
	}

	// Special handling for Reshape when used in shape manipulation context
	// Reshape has two inputs: data and shape
	// For shape manipulation, we only care about the shape input (input[1]), not the data input (input[0])
	if node.OpType == "Reshape" && len(node.Input) >= 2 {
		// Save current state
		origNonConstInputs := make([]string, len(nonConstInputs))
		copy(origNonConstInputs, nonConstInputs)
		origVariables := make([]string, len(variables))
		copy(origVariables, variables)
		origContextNodes := make([]*protos.NodeProto, len(contextNodes))
		copy(origContextNodes, contextNodes)

		// Only recurse into the shape input (input[1])
		if !visitedNodes.Has(node.Input[1]) {
			nonConstInputs, variables, contextNodes = m.recursiveNonConstantDependencies(node.Input[1], visitedNodes, nonConstInputs, variables, contextNodes, convertedOutputs)
		}

		// Check if anything was added
		if len(nonConstInputs) == len(origNonConstInputs) &&
			len(variables) == len(origVariables) &&
			len(contextNodes) == len(origContextNodes) {
			// No non-constant dependencies were added from the shape input, so this Reshape is materializable
			return nonConstInputs, variables, contextNodes
		}
		// Something was added from the shape input, so this Reshape is not materializable
		return nonConstInputs, variables, contextNodes
	}

	// Operations that work on shape values (Gather, Slice, Concat, etc.) are materializable
	// if all their inputs are materializable. We recurse into inputs and if none add
	// non-constant dependencies, this operation is materializable.
	if isShapeManipulationOp(node.OpType) {
		// Save current state
		origNonConstInputs := make([]string, len(nonConstInputs))
		copy(origNonConstInputs, nonConstInputs)
		origVariables := make([]string, len(variables))
		copy(origVariables, variables)
		origContextNodes := make([]*protos.NodeProto, len(contextNodes))
		copy(origContextNodes, contextNodes)

		// Recurse into inputs
		for _, input := range node.Input {
			if visitedNodes.Has(input) {
				continue
			}
			nonConstInputs, variables, contextNodes = m.recursiveNonConstantDependencies(input, visitedNodes, nonConstInputs, variables, contextNodes, convertedOutputs)
		}

		// Check if anything was added
		if len(nonConstInputs) == len(origNonConstInputs) &&
		   len(variables) == len(origVariables) &&
		   len(contextNodes) == len(origContextNodes) {
			// No non-constant dependencies were added, so this operation is materializable
			return nonConstInputs, variables, contextNodes
		}
		// Something was added, but we already recursed, so just return
		return nonConstInputs, variables, contextNodes
	}
	for _, input := range node.Input {
		if visitedNodes.Has(input) {
			continue
		}
		nonConstInputs, variables, contextNodes = m.recursiveNonConstantDependencies(input, visitedNodes, nonConstInputs, variables, contextNodes, convertedOutputs)
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
	tensorProto := m.variableNameToValue[varName]
	shape, err := Shape(tensorProto)
	if err != nil {
		panic(errors.WithMessagef(err, "ONNX variable %q has an invalid shape", varName))
	}
	return shape.DType.IsInt() && shape.Size() <= sizeLimit
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
		return node.ConstantValue(), nil
	}

	// See if it is possible: if the subgraph that generated the node is a constant expression.
	// Note: We used to check nonConstantDependencies here and fail early, but this was too conservative.
	// With StableHLO-level constant extraction (tryExtractConstantShape), many operations that have
	// variable dependencies in the ONNX graph can still be materialized. So we optimistically try
	// to materialize and only fail if the actual execution fails.
	nonConstInputs, nonConstVariables, contextNodes := m.nonConstantDependencies(nodeOutputName, convertedOutputs)
	_ = nonConstInputs      // Keep for potential future use
	_ = nonConstVariables   // Keep for potential future use
	_ = contextNodes        // Keep for potential future use

	// Uncomment to restore strict dependency checking:
	// if len(nonConstInputs) > 0 || len(nonConstVariables) > 0 || len(contextNodes) > 0 {
	// 	varDesc := make([]string, 0, len(nonConstVariables))
	// 	for _, varName := range nonConstVariables {
	// 		shape, _ := Shape(m.variableNameToValue[varName])
	// 		varDesc = append(varDesc, fmt.Sprintf("%q (%s)", varName, shape))
	// 	}
	// 	opsDesc := make([]string, 0, len(contextNodes))
	// 	for _, node := range contextNodes {
	// 		varDesc = append(opsDesc, node.String())
	// 	}
	// 	return nil, errors.Errorf("cannot materialize constant/static value for %q: it depends on non-constant: inputs=%q, variables: %s, ops with context: %s",
	// 		nodeOutputName, nonConstInputs, strings.Join(varDesc, ", "), strings.Join(opsDesc, ", "))
	// }

	// Evaluate constant sub-expression in a newly created sub-graph.
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
	// HOWEVER: For Shape operations, we need to re-evaluate them because the input tensor might have a different
	// shape in the materialization context than it did in the original graph.
	onnxNode := m.nodeOutputToNode[nodeOutputName]
	if originalNode, found := originalConvertedOutput[nodeOutputName]; found {
		if originalNode.Type() == NodeTypeConstant {
			// Skip reusing constants for Shape operations - they need to be re-evaluated
			// with the correct input tensor shape
			if onnxNode != nil && onnxNode.OpType == "Shape" {
				// Fall through to re-evaluate the Shape operation below
			} else {
				// Duplicate the constant in the new graph.
				constConvertedOutputs[nodeOutputName] = Const(g, originalNode.ConstantValue())
				return
			}
		}
	}

	// Check for constant variables.
	if tensorNode, found := m.variableNameToValue[nodeOutputName]; found {
		if !m.isVariableConstant(nodeOutputName) {
			exceptions.Panicf("attempting to materialize as constant variable %q, which we don't think is constant", nodeOutputName)
		}
		t, err := tensorToGoMLX(m.backend, tensorNode)
		if err != nil {
			panic(errors.WithMessagef(err, "attempting to materialize variable %q as constant", nodeOutputName))
		}
		constConvertedOutputs[nodeOutputName] = Const(g, t)
		// TODO: mark variable as used for constant-expression and make sure it is also used in the final model, and
		// 		 try to make as such that if it changes, the graph is rebuilt.
		return
	}

	// Find the node generating this output (if not already found above).
	if onnxNode == nil {
		var found bool
		onnxNode, found = m.nodeOutputToNode[nodeOutputName]
		if !found {
			exceptions.Panicf("ONNX node %q not found as the output of an Op, and not a constant either -- is this really a constant expression!?", nodeOutputName)
		}
	}
	if opRequiresContext(onnxNode.OpType) {
		// Operation requires a context, which is not supported when materializing constant sub-expressions.
		exceptions.Panicf("attempting to materialize expression with operation %q, which is not supported for materialization: %s", onnxNode.OpType, onnxNode)
	}

	// Special case: Shape op on a Parameter or variable with concrete dimensions
	// IMPORTANT: We should ONLY materialize Shape operations if their input is a Parameter or variable,
	// NOT if it's an intermediate computed tensor, because the shape of intermediate tensors might
	// be data-dependent (e.g., after GatherElements operations) or might have different compile-time
	// vs runtime shapes (e.g., due to bucketing/padding in RNN layers).
	// CRITICAL: This must be checked BEFORE recursively materializing inputs, because we don't need
	// to materialize the variable's value, only its shape!
	if onnxNode.OpType == "Shape" && len(onnxNode.Input) > 0 {
		var dimensions []int
		var hasShape bool

		// Check if input is a Parameter (model input)
		if m.inputsNameSet.Has(onnxNode.Input[0]) {
			// First, check the actual converted node shape (which has concrete dims at CallGraph time)
			if inputNode, found := originalConvertedOutput[onnxNode.Input[0]]; found {
				actualShape := inputNode.Shape()
				allConcrete := true
				for _, dim := range actualShape.Dimensions {
					if dim < 0 {
						allConcrete = false
						break
					}
				}
				if allConcrete {
					dimensions = actualShape.Dimensions
					hasShape = true
				}
			}
			// Fall back to model's declared shape
			if !hasShape {
				for idx, inputName := range m.InputsNames {
					if inputName == onnxNode.Input[0] {
						dimensions = m.InputsShapes[idx].Dimensions
						hasShape = true
						break
					}
				}
			}
		} else if varTensorProto, found := m.variableNameToValue[onnxNode.Input[0]]; found {
			// Check if input is a variable
			varShape, err := Shape(varTensorProto)
			if err == nil {
				dimensions = varShape.Dimensions
				hasShape = true
			}
		} else if _, found := originalConvertedOutput[onnxNode.Input[0]]; found {
			// Input is an intermediate tensor from the original graph.
			// DON'T materialize: ONNX models often extract shapes from one tensor and use them
			// to reshape a different tensor. This only works with DynamicReshape which handles
			// size mismatches. Materializing would force static Reshape which validates sizes.
		}

		if !hasShape {
			// Input is an intermediate tensor that either:
			// 1. Hasn't been converted yet, or
			// 2. Has symbolic dimensions
			exceptions.Panicf("cannot materialize Shape operation %q as constant: input %q is an intermediate tensor that hasn't been converted or has symbolic dims.",
				onnxNode.GetName(), onnxNode.Input[0])
		}


		if hasShape {
			// Check if all dimensions are concrete
			allConcrete := true
			for _, dim := range dimensions {
				if dim < 0 {
					allConcrete = false
					break
				}
			}
			if allConcrete {
				// Create a constant with the shape dimensions
				start := getIntAttrOr(onnxNode, "start", 0)
				if start < 0 {
					start = len(dimensions) + start
				}
				end := getIntAttrOr(onnxNode, "end", 0)
				if end == 0 {
					end = len(dimensions)
				} else if end < 0 {
					end = len(dimensions) + end
				}
				// Extract the shape slice and convert to int64
				dims := make([]int64, end-start)
				for i := start; i < end; i++ {
					dims[i-start] = int64(dimensions[i])
				}
				constConvertedOutputs[nodeOutputName] = Const(g, dims)
				return
			}
		}
	}

	// Recursively converts the inputs of the onnxNode:
	for _, inputName := range onnxNode.Input {
		m.recursiveMaterializeConstantExpression(inputName, g, constConvertedOutputs, originalConvertedOutput)
	}

	// And now convert the node itself.
	m.convertNode(nil, g, onnxNode, constConvertedOutputs)
}
