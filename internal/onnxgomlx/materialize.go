// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
)

// IsConstantExpression returns true if the subgraph producing nodeOutputName
// has no dependencies on graph inputs or non-constant variables based on static
// ONNX model metadata (e.g. when called at Parse time before CallGraph).
func (m *Model) IsConstantExpression(nodeOutputName string) bool {
	inputs, vars, ctxNodes := m.nonConstantDependencies(nodeOutputName, nil)
	return len(inputs) == 0 && len(vars) == 0 && len(ctxNodes) == 0
}

// MaterializeConstantExpression materializes a node to its constant expression.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It is seeded with the graph inputs and populated
// as nodes are converted.
func (m *Model) MaterializeConstantExpression(nodeOutputName string, convertedOutputs map[string]*Node) (*tensors.Tensor, error) {
	return m.materializeConstantExpression(nodeOutputName, convertedOutputs)
}

// MaterializeConstantScalar materializes a scalar node to its constant float64 value.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It is seeded with the graph inputs and populated
// as nodes are converted.
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

// nonConstantDependencies returns the non-constant dependencies (inputs, variables, or context-requiring ops)
// of the subgraph producing nodeOutputName.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It is seeded with the graph inputs and populated
// as nodes are converted. It may be nil when called outside CallGraph (specifically from
// IsConstantExpression during Parse, before any GoMLX graph exists), in which case only
// static ONNX protobuf metadata is consulted.
//
// Why convertedOutputs is needed here: Model is parsed once from the ONNX protobuf and
// reused across compilations, so Model.InferShapeForName reflects static ONNX metadata
// where input and intermediate axes are frequently symbolic (e.g. "batch", "seq" -> -1).
// During CallGraph, however, convertedOutputs holds the GoMLX *[Node]s for the current
// graph compilation, which often have concrete static shapes (e.g. [1, 128] on XLA or
// non-dynamic executions) or have already been folded into GoMLX Constant nodes. Without
// checking convertedOutputs, Shape/Size subgraphs over concrete inputs would be falsely
// classified as non-constant.
func (m *Model) nonConstantDependencies(nodeOutputName string, convertedOutputs map[string]*Node) (inputs, variables []string, contextNodes []*protos.NodeProto) {
	visitedNodes := sets.Make[string]()
	return m.recursiveNonConstantDependencies(nodeOutputName, visitedNodes, convertedOutputs, inputs, variables, contextNodes)
}

// recursiveNonConstantDependencies is the recursive implementation of nonConstantDependencies.
// Use nonConstantDependencies instead.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It may be nil when called before CallGraph, and
// individual entries may hold a nil *[Node] for zero-size constant inputs.
func (m *Model) recursiveNonConstantDependencies(name string, visitedNodes sets.Set[string], convertedOutputs map[string]*Node,
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

	// Short-circuit if this value has already been converted to a GoMLX Constant node whose
	// tensor value is retained in memory (ConstantValue() != nil, which GoMLX keeps for tensors
	// smaller than MinConstValueSizeToKeep, such as Shape/Size outputs).
	// Constants >= MinConstValueSizeToKeep (or donated variable buffers) have ConstantValue() == nil
	// and cannot be introspected back out of the GoMLX *Node, so we must continue walking their ONNX subgraph.
	if convertedOutputs != nil {
		if convertedNode := convertedOutputs[name]; convertedNode != nil &&
			convertedNode.Type() == NodeTypeConstant && convertedNode.ConstantValue() != nil {
			return nonConstInputs, variables, contextNodes
		}
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
		// External/outer-scope or unregistered runtime input.
		nonConstInputs = append(nonConstInputs, name)
		return nonConstInputs, variables, contextNodes
	}
	if opRequiresContext(node.OpType) {
		contextNodes = append(contextNodes, node)
	}
	if node.OpType == "Shape" || node.OpType == "Size" {
		if len(node.Input) > 0 && node.Input[0] != "" {
			inShape := m.InferShapeForName(node.Input[0])
			if convertedOutputs != nil {
				if convertedInput := convertedOutputs[node.Input[0]]; convertedInput != nil {
					inShape = convertedInput.Shape()
				}
			}
			if inShape.Dimensions != nil {
				hasDynamic := false
				if node.OpType == "Size" {
					hasDynamic = inShape.IsDynamic()
				} else {
					for _, d := range shapeSliceDimensions(node, inShape.Dimensions) {
						if d < 0 {
							hasDynamic = true
							break
						}
					}
				}
				if hasDynamic {
					nonConstInputs = append(nonConstInputs, node.Input[0])
				}
			}
		}
		return nonConstInputs, variables, contextNodes
	}
	if node.OpType == "Gather" {
		if xName, constT, hasDynamic, handled := m.inspectGatherFromShape(node, convertedOutputs); handled {
			if constT != nil {
				constT.FinalizeAll()
			}
			if hasDynamic {
				nonConstInputs = append(nonConstInputs, xName)
			}
			return nonConstInputs, variables, contextNodes
		}
	}
	for _, input := range node.Input {
		if input == "" || visitedNodes.Has(input) {
			continue
		}
		nonConstInputs, variables, contextNodes = m.recursiveNonConstantDependencies(input, visitedNodes, convertedOutputs, nonConstInputs, variables, contextNodes)
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
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node]. It is seeded with the graph inputs and populated
// as nodes are converted.
//
// If the node depends on non-constant values (like runtime input parameters), it returns an error.
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
	nonConstInputs, nonConstVariables, contextNodes := m.nonConstantDependencies(nodeOutputName, convertedOutputs)
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

// recursiveMaterializeConstantExpression builds a temporary GoMLX graph g evaluating nodeOutputName,
// storing nodes created in g into constConvertedOutputs.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node] in the caller's main graph (seeded with graph inputs and
// populated as nodes are converted); it is only read here and never mutated.
func (m *Model) recursiveMaterializeConstantExpression(nodeOutputName string, g *Graph, constConvertedOutputs, convertedOutputs map[string]*Node) {
	if _, found := constConvertedOutputs[nodeOutputName]; found {
		// Already converted.
		return
	}

	// Check in the original graph being converted if this node was converted as a constant (for instance, for nodes like "Shape"),
	// in which case we take the constant value and inject it directly in the new constant expression.
	if convertedOutputs != nil {
		if originalNode := convertedOutputs[nodeOutputName]; originalNode != nil && originalNode.Type() == NodeTypeConstant {
			if cVal := originalNode.ConstantValue(); cVal != nil {
				// Duplicate the constant in the new graph.
				constConvertedOutputs[nodeOutputName] = Const(g, cVal)
				return
			}
		}
	}

	// Check for inputs marked as constants (needed when the constant tensor has >= MinConstValueSizeToKeep elements,
	// so originalNode.ConstantValue() above is nil).
	if m.InputsAsConstants != nil {
		if staticVal := m.InputsAsConstants[nodeOutputName]; staticVal != nil {
			constConvertedOutputs[nodeOutputName] = Const(g, staticVal)
			return
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

	if onnxNode.OpType == "Shape" || onnxNode.OpType == "Size" {
		if len(onnxNode.Input) == 0 || onnxNode.Input[0] == "" {
			exceptions.Panicf("%s missing input", onnxNode.OpType)
		}
		inputName := onnxNode.Input[0]
		inShape := m.InferShapeForName(inputName)
		if convertedOutputs != nil {
			if origIn := convertedOutputs[inputName]; origIn != nil {
				inShape = origIn.Shape()
			}
		}
		if inShape.Dimensions == nil {
			exceptions.Panicf("%s(%q) has unknown shape", onnxNode.OpType, inputName)
		}
		if onnxNode.OpType == "Size" {
			if inShape.IsDynamic() {
				exceptions.Panicf("%s depends on runtime dimensions of %q (%s)", onnxNode.Name, inputName, inShape)
			}
			constConvertedOutputs[nodeOutputName] = Const(g, int64(inShape.Size()))
			return
		}
		sliceDims := shapeSliceDimensions(onnxNode, inShape.Dimensions)
		dims := make([]int64, len(sliceDims))
		for i, d := range sliceDims {
			if d < 0 {
				exceptions.Panicf("%s depends on runtime dimensions of %q (%s)", onnxNode.Name, inputName, inShape)
			}
			dims[i] = int64(d)
		}
		constConvertedOutputs[nodeOutputName] = Const(g, tensors.FromFlatDataAndDimensions(dims, len(dims)))
		return
	}
	if onnxNode.OpType == "Gather" {
		if xName, constT, hasDynamic, handled := m.inspectGatherFromShape(onnxNode, convertedOutputs); handled {
			if hasDynamic || constT == nil {
				if constT != nil {
					constT.FinalizeAll()
				}
				exceptions.Panicf("Gather(%q) depends on runtime dimensions of %q", onnxNode.Name, xName)
			}
			defer constT.FinalizeAll()
			constConvertedOutputs[nodeOutputName] = ConstTensor(g, constT)
			return
		}
	}

	// Recursively converts the inputs of the onnxNode:
	for _, inputName := range onnxNode.Input {
		if inputName != "" {
			m.recursiveMaterializeConstantExpression(inputName, g, constConvertedOutputs, convertedOutputs)
		}
	}

	// And now convert the node itself.
	m.convertNode(nil, g, onnxNode, constConvertedOutputs)
}

// shapeSliceDimensions returns dims[start:end] according to ONNX Shape's optional start/end attributes.
func shapeSliceDimensions(shapeNode *protos.NodeProto, dims []int) []int {
	rank := len(dims)
	start := GetIntAttrOr(shapeNode, "start", 0)
	end := GetIntAttrOr(shapeNode, "end", rank)
	if start < 0 {
		start += rank
	}
	if end < 0 {
		end += rank
	}
	start = min(max(start, 0), rank)
	end = min(max(end, start), rank)
	return dims[start:end]
}

// evalConstantTensor evaluates name to a constant Tensor if its subgraph has no non-constant dependencies.
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node], and may be nil when called at Parse time.
// When owned is true, the caller owns the returned Tensor and should FinalizeAll it when done.
func (m *Model) evalConstantTensor(name string, convertedOutputs map[string]*Node) (t *tensors.Tensor, owned, ok bool) {
	if convertedOutputs != nil {
		if n := convertedOutputs[name]; n != nil && n.Type() == NodeTypeConstant {
			if cVal := n.ConstantValue(); cVal != nil {
				return cVal, false, true
			}
		}
	}
	nonConstInputs, nonConstVars, ctxNodes := m.nonConstantDependencies(name, convertedOutputs)
	if len(nonConstInputs) > 0 || len(nonConstVars) > 0 || len(ctxNodes) > 0 {
		return nil, false, false
	}
	backend := m.Backend
	if convertedOutputs != nil {
		if n := convertedOutputs[name]; n != nil {
			backend = n.Graph().Backend()
		}
	}
	if backend == nil {
		return nil, false, false
	}
	var res *tensors.Tensor
	err := exceptions.TryCatch[error](func() {
		res = MustExecOnce(backend, func(g *Graph) *Node {
			constOut := make(map[string]*Node)
			m.recursiveMaterializeConstantExpression(name, g, constOut, convertedOutputs)
			return constOut[name]
		})
	})
	if err != nil || res == nil {
		return nil, false, false
	}
	return res, true, true
}

// inspectGatherFromShape checks whether node is a Gather(Shape(x), const_indices) along axis 0/-1.
//
// This allows extracting statically known axes (e.g. hidden/head dimensions) as constants even when
// other axes of x (e.g. batch or sequence length) are dynamic (-1).
//
// convertedOutputs maps ONNX value names (graph inputs, variables, and node outputs)
// to their converted GoMLX *[Node], and may be nil when called at Parse time.
// If handled is true and hasDynamic is false, constResult (when non-nil) holds the newly allocated
// Int64 result tensor (which the caller owns and must FinalizeAll).
func (m *Model) inspectGatherFromShape(node *protos.NodeProto, convertedOutputs map[string]*Node) (xName string, constResult *tensors.Tensor, hasDynamic, handled bool) {
	if m == nil || node == nil || node.OpType != "Gather" || len(node.Input) < 2 || node.Input[0] == "" || node.Input[1] == "" {
		return "", nil, false, false
	}
	// Shape outputs a 1D tensor, so valid Gather axis is 0 or -1.
	if axis := GetIntAttrOr(node, "axis", 0); axis != 0 && axis != -1 {
		return "", nil, false, false
	}
	shapeNode := m.NodeOutputToNode[node.Input[0]]
	if shapeNode == nil || shapeNode.OpType != "Shape" || len(shapeNode.Input) == 0 || shapeNode.Input[0] == "" {
		return "", nil, false, false
	}
	xName = shapeNode.Input[0]

	idxT, owned, ok := m.evalConstantTensor(node.Input[1], convertedOutputs)
	if !ok {
		return "", nil, false, false
	}
	if owned {
		defer idxT.FinalizeAll()
	}
	if !idxT.DType().IsInt() {
		return "", nil, false, false
	}

	inShape := m.InferShapeForName(xName)
	if convertedOutputs != nil {
		if convertedIn := convertedOutputs[xName]; convertedIn != nil {
			inShape = convertedIn.Shape()
		}
	}
	if inShape.Dimensions == nil {
		// At Parse time (convertedOutputs == nil) when intermediate ValueInfo is missing,
		// optimistically treat as potentially constant (matching Shape handling) so deferred
		// consumers such as SDPA scale can materialize once CallGraph provides shapes.
		if convertedOutputs == nil {
			return xName, nil, false, true
		}
		return "", nil, false, false
	}

	sliceDims := shapeSliceDimensions(shapeNode, inShape.Dimensions)
	indices := tensorToInts(idxT)
	outVals := make([]int64, len(indices))
	for i, idx := range indices {
		if idx < 0 {
			idx += len(sliceDims)
		}
		if idx < 0 || idx >= len(sliceDims) {
			return "", nil, false, false
		}
		if sliceDims[idx] < 0 {
			return xName, nil, true, true
		}
		outVals[i] = int64(sliceDims[idx])
	}
	return xName, tensors.FromFlatDataAndDimensions(outVals, idxT.Shape().Dimensions...), false, true
}
