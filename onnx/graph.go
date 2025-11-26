package onnx

import (
	"fmt"
	"runtime"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// sliceMap executes the given function sequentially for every element on in and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// CallGraph calls the ONNX graph, and hence are building it with GoMLX ops.
// This can be used for inference or training.
//
// If the model has any variables, call Model.VariablesToContext first (only once) to upload all
// variable values from the ONNX model to the context -- or load them from a checkpoint if you saved one.
//
// If the model has no variables, the context in ctx can be set to nil.
//
// The inputs (a map of the input name to its graph.Node) can be given as normal input parameters to the graph or as
// static constants -- see WithInputsAsConstants.
// Set the inputs as constants if they are meant to be interpreted as constants (static) values, that won't change
// in different inference/training steps.
//
// If outputNames is not given, it will output the model's registered outputs. Alternatively, you can select
// any list of node outputs to generate. It will return the values for the selected outputs.
//
// The graph being built is given in g.
//
// You can pass a nil context (ctx) if the model has no variables.
//
// As in GoMLX graph building (symbolic) functions, it panics (throws exceptions) in case of errors.
func (m *Model) CallGraph(ctx *context.Context, g *Graph, inputs map[string]*Node, outputNames ...string) (outputs []*Node) {
	if ctx != nil {
		ctx = ctx.In(ModelScope).Checked(false)
	}

	// Sanity check of things we don't support yet.
	if len(m.Proto.Functions) > 0 {
		exceptions.Panicf("onnx.CallGraph does not support ONNX functions")
	}
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.CallGraph does not support ONNX SparseTensors")
	}

	// If no outputNames were given, take the model outputs.
	if len(outputNames) == 0 {
		outputNames = m.OutputsNames
	}

	// Map the given inputs to the corresponding ONNX inputs and report (throw exception) if there are
	// any discrepancies.
	// Also, initialize convertedOutputs with the given/converted inputs.
	convertedOutputs := make(map[string]*Node)
	missingInputs := sets.Make[string]()
	repeatedInputs := sets.Make[string]()
	unknownInputs := sets.Make[string]()
	for inputIdx, inputName := range m.InputsNames {
		if inputName == "" {
			inputName = fmt.Sprintf("#%d", inputIdx)
		}
		inputN := inputs[inputName]
		if inputN == nil {
			staticValue := m.inputsAsConstants[inputName]
			if staticValue != nil {
				inputN = Const(g, staticValue)
			} else {
				missingInputs.Insert(inputName)
				continue
			}
		} else {
			if _, found := m.inputsAsConstants[inputName]; found {
				repeatedInputs.Insert(inputName)
			}
		}
		convertedOutputs[inputName] = inputN
	}
	for givenName := range inputs {
		if _, found := convertedOutputs[givenName]; !found {
			unknownInputs.Insert(givenName)
		}
	}
	for givenName := range m.inputsAsConstants {
		if _, found := convertedOutputs[givenName]; !found {
			unknownInputs.Insert(givenName)
		}
	}
	if len(missingInputs) > 0 || len(unknownInputs) > 0 {
		exceptions.Panicf("onnx.CallGraph() called with wrong inputs: missing inputs=%q; unknown given inputs=%q; inputs given normally and as constant inputs=%q",
			missingInputs, unknownInputs, repeatedInputs)
	}

	// Validate the input shapes.
	err := m.ValidateInputs(sliceMap(m.InputsNames, func(inputName string) shapes.Shape { return convertedOutputs[inputName].Shape() })...)
	if err != nil {
		panic(err)
	}

	// Convert variables: create the GoMLX nodes corresponding to the ONNX model variables.
	if len(m.Proto.Graph.Initializer) > 0 && ctx == nil {
		exceptions.Panicf("onnx.CallGraph(): model has variables, but a nil context was give")
		panic(nil) // for lint benefit.
	}

	// Convert all nodes recursively, which will implicitly yield a topological order.
	for _, target := range outputNames {
		m.recursiveCallGraph(ctx, g, target, convertedOutputs)
	}

	// Pick the outputs.
	outputs = make([]*Node, len(outputNames))
	var found bool
	for outputIdx, nodeName := range outputNames {
		outputs[outputIdx], found = convertedOutputs[nodeName]
		if !found {
			exceptions.Panicf("output node %q not found", nodeName)
		}
	}

	// Makes sure all the temporarily allocated on-device tensors are freed.
	for _ = range 3 {
		runtime.GC()
	}
	return outputs
}

// recursiveCallGraph recursively creates a GoMLX graph for the target output name.
// The convertedOutputs are used both as input and as output to store the converted nodes.
//
// The ctx may be nil if no variables are used.
func (m *Model) recursiveCallGraph(ctx *context.Context, g *Graph, nodeOutputName string, convertedOutputs map[string]*Node) {
	if _, found := convertedOutputs[nodeOutputName]; found {
		// Already converted.
		return
	}

	// Is it the output of a variable?
	if _, found := m.variableNameToValue[nodeOutputName]; found {
		if ctx == nil {
			exceptions.Panicf("onnx.CallGraph(): model has variables, but a nil context was given")
			panic(nil) // for lint benefit.
		}
		varName := SafeVarName(nodeOutputName)
		v := ctx.GetVariable(varName)
		if v == nil {
			exceptions.Panicf("variable %q (named %q in ONNX) has not been uploaded yet to context -- did you forget to call onnx.Model.VariablesToContext?",
				varName, nodeOutputName)
			panic(nil) // for lint benefit.
		}
		convertedOutputs[nodeOutputName] = v.ValueGraph(g)
		return
	}

	onnxNode, found := m.nodeOutputToNode[nodeOutputName]
	if !found {
		exceptions.Panicf("ONNX node output %q not found as the output of any Op, and not a variable or input either -- could it be a node name, and note a node **output** name ?", nodeOutputName)
	}

	// Recursively converts the inputs of the onnxNode:
	for _, inputName := range onnxNode.Input {
		if inputName == "" {
			// Probably an optional parameter, not used. LSTM nodes have this.
			continue
		}
		m.recursiveCallGraph(ctx, g, inputName, convertedOutputs)
	}

	// Convert the node itself.
	m.convertNode(ctx, g, onnxNode, convertedOutputs)
}

// convertSubGraph converts an ONNX sub-graph (used in control flow ops like If) to GoMLX nodes.
// It takes the parent graph g and the sub-graph proto, along with the current convertedOutputs mapping.
// Returns a slice of output nodes from the sub-graph in the order they appear in the sub-graph's output list.
func (m *Model) convertSubGraph(g *Graph, subGraphProto *protos.GraphProto, parentConvertedOutputs map[string]*Node) []*Node {
	// Create a new local context for the sub-graph
	// Note: Sub-graphs in ONNX can reference outputs from the parent graph
	localConvertedOutputs := make(map[string]*Node)

	// Copy parent outputs into local context so sub-graph can reference them
	for name, node := range parentConvertedOutputs {
		localConvertedOutputs[name] = node
	}

	// Convert sub-graph initializers (constants) to GoMLX nodes
	// Also temporarily add them to model's variableNameToValue for materializeConstantExpression
	subGraphInitializers := make(map[string]*protos.TensorProto)
	for _, initializerProto := range subGraphProto.Initializer {
		initializerName := initializerProto.Name
		if initializerName == "" {
			continue
		}
		// Convert the initializer tensor to a GoMLX constant
		tensor, err := tensorToGoMLX(g.Backend(), initializerProto)
		if err != nil {
			exceptions.Panicf("failed to convert sub-graph initializer %q: %v", initializerName, err)
		}
		localConvertedOutputs[initializerName] = Const(g, tensor)
		subGraphInitializers[initializerName] = initializerProto
		m.variableNameToValue[initializerName] = initializerProto
	}

	// Build a mapping from output name to the node that produces it (for this sub-graph only)
	subGraphNodeOutputToNode := make(map[string]*protos.NodeProto)
	for _, node := range subGraphProto.Node {
		for _, outputName := range node.Output {
			if outputName != "" {
				subGraphNodeOutputToNode[outputName] = node
			}
		}
	}

	// Temporarily add sub-graph nodes to the model's nodeOutputToNode map
	// This is needed for materializeConstantExpression to work with sub-graph nodes
	for outputName, node := range subGraphNodeOutputToNode {
		m.nodeOutputToNode[outputName] = node
	}

	// Consolidated cleanup: remove all temporary entries from model's maps when done
	// Using a single defer with recovery handling to ensure cleanup always happens
	defer func() {
		// Clean up sub-graph initializers from model's variableNameToValue map
		for initName := range subGraphInitializers {
			delete(m.variableNameToValue, initName)
		}
		// Clean up sub-graph nodes from model's nodeOutputToNode map
		for outputName := range subGraphNodeOutputToNode {
			delete(m.nodeOutputToNode, outputName)
		}
	}()

	// Recursive helper to convert a node output within the sub-graph
	var convertSubGraphOutput func(outputName string)
	convertSubGraphOutput = func(outputName string) {
		// Empty output name means optional output
		if outputName == "" {
			return
		}

		// Already converted?
		if _, found := localConvertedOutputs[outputName]; found {
			return
		}

		// Check if it's a model-level initializer (variable)
		if initializerProto, found := m.variableNameToValue[outputName]; found {
			// Convert the model-level initializer to a constant in the sub-graph
			tensor, err := tensorToGoMLX(g.Backend(), initializerProto)
			if err != nil {
				exceptions.Panicf("failed to convert model initializer %q in sub-graph: %v", outputName, err)
			}
			localConvertedOutputs[outputName] = Const(g, tensor)
			return
		}

		// Is it a sub-graph node output?
		node, found := subGraphNodeOutputToNode[outputName]
		if !found {
			// Not found in sub-graph nodes - might be in parent scope
			if _, foundInParent := parentConvertedOutputs[outputName]; foundInParent {
				// Already available from parent - nothing to do
				return
			}

			// Not in parent outputs yet - try to find and convert it from the main model
			if mainNode, foundInMain := m.nodeOutputToNode[outputName]; foundInMain {
				// This is a main model node that hasn't been converted yet
				// Recursively convert its inputs first
				for _, inputName := range mainNode.Input {
					if inputName == "" {
						continue
					}
					convertSubGraphOutput(inputName)
				}
				// Now convert this main model node and add to local outputs
				m.convertNode(nil, g, mainNode, localConvertedOutputs)

				// Also add to parent outputs so other branches/sub-graphs can reuse it
				parentConvertedOutputs[outputName] = localConvertedOutputs[outputName]
				return
			}

			// Not found anywhere - this is an error
			exceptions.Panicf("sub-graph output %q not found in sub-graph nodes, parent outputs, model initializers, or main model nodes", outputName)
		}

		// Recursively convert all inputs first
		for _, inputName := range node.Input {
			if inputName == "" {
				// Optional input not provided
				continue
			}

			// Try to convert the input
			convertSubGraphOutput(inputName)
		}

		// Verify all required inputs are available before converting the node
		for i, inputName := range node.Input {
			if inputName == "" {
				// Optional input - skip verification
				continue
			}
			if _, found := localConvertedOutputs[inputName]; !found {
				exceptions.Panicf("input[%d] %q for sub-graph node %q (%s) not found after conversion attempt",
					i, inputName, node.Name, node.OpType)
			}
		}

		// Now convert this node
		m.convertNode(nil, g, node, localConvertedOutputs)
	}

	// Convert all output nodes recursively (which will convert their dependencies)
	for _, output := range subGraphProto.Output {
		convertSubGraphOutput(output.Name)
	}

	// Collect the sub-graph outputs
	outputs := make([]*Node, len(subGraphProto.Output))
	for i, output := range subGraphProto.Output {
		outputNode, found := localConvertedOutputs[output.Name]
		if !found {
			exceptions.Panicf("sub-graph output %q not found after conversion", output.Name)
		}
		outputs[i] = outputNode
	}

	return outputs
}

// opRequiresContext checks if the given operation type requires a context.
// Currently only LSTM.
func opRequiresContext(opType string) bool {
	return opType == "LSTM"
}

// convertNode converts a single ONNX node to a GoMLX node.
//
// Previously converted nodes are given in convertedNodes.
// The converted output(s) are updated into `convertedNodes`.
//
// It panics (throw exceptions) in case of errors.
//
// TODO: One of ONNX broadcasting rule is not applied by default in GoMLX/XLA for binary operators, namely:
//
//	"The tensors that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2."
//
// See the definitions in:
// . https://openxla.org/xla/broadcasting
// . https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
func (m *Model) convertNode(_ *context.Context, g *Graph, node *protos.NodeProto, convertedOutputs map[string]*Node) {
	if node.Overload != "" {
		exceptions.Panicf("overload %q to in-model function in ONNX model not implemented in node %q", node.Overload, node.Name)
	}

	// Convert the node: the usual case is that there is only one output.
	// If the result is not nil, it is set to convertedOutputs[output[0]].
	// Anything different must be implemented by the specific op switch.
	var result *Node
	inputs := sliceMap(node.Input, func(n string) *Node { return convertedOutputs[n] })
	switch node.OpType {
	// Binary operators: see the note on differences on default broadcasting.
	case "Add":
		result = convertBinaryOp(Add, inputs[0], inputs[1])
	case "Sub":
		result = convertBinaryOp(Sub, inputs[0], inputs[1])
	case "Mul":
		result = convertBinaryOp(Mul, inputs[0], inputs[1])
	case "Div":
		result = convertBinaryOp(Div, inputs[0], inputs[1])
	case "Pow":
		//result = convertBinaryOp(Pow, inputs[0], inputs[1])
		result = convertPow(m, convertedOutputs, node, inputs)
	case "And":
		result = convertBinaryOp(LogicalAnd, inputs[0], inputs[1])
	case "Or":
		result = convertBinaryOp(LogicalOr, inputs[0], inputs[1])
	case "Xor":
		result = convertBinaryOp(LogicalXor, inputs[0], inputs[1])
	case "BitwiseAnd":
		result = convertBinaryOp(BitwiseAnd, inputs[0], inputs[1])
	case "BitwiseOr":
		result = convertBinaryOp(BitwiseOr, inputs[0], inputs[1])
	case "BitwiseXor":
		result = convertBinaryOp(BitwiseXor, inputs[0], inputs[1])
	case "Equal":
		result = convertBinaryOp(Equal, inputs[0], inputs[1])
	case "Less":
		result = convertBinaryOp(LessThan, inputs[0], inputs[1])
	case "LessOrEqual":
		result = convertBinaryOp(LessOrEqual, inputs[0], inputs[1])
	case "Greater":
		result = convertBinaryOp(GreaterThan, inputs[0], inputs[1])
	case "GreaterOrEqual":
		result = convertBinaryOp(GreaterOrEqual, inputs[0], inputs[1])

	// Unary operators
	case "Sqrt":
		result = Sqrt(inputs[0])
	case "Exp":
		result = Exp(inputs[0])
	case "Log":
		result = Log(inputs[0])
	case "Erf":
		result = Erf(inputs[0])
	case "Relu":
		result = activations.Relu(inputs[0])
	case "Abs":
		result = Abs(inputs[0])
	case "Neg":
		result = Neg(inputs[0])
	case "Sign":
		result = Sign(inputs[0])
	case "Ceil":
		result = Ceil(inputs[0])
	case "Floor":
		result = Floor(inputs[0])
	case "Identity":
		result = Identity(inputs[0])
	case "Not":
		result = LogicalNot(inputs[0])
	case "BitwiseNot":
		result = BitwiseNot(inputs[0])
	case "Tanh":
		result = Tanh(inputs[0])
	case "Sin":
		result = Sin(inputs[0])
	case "Cos":
		result = Cos(inputs[0])

		// Ops with equivalents:
	case "MatMul":
		result = MatMul(inputs[0], inputs[1])

	// Ops with special behavior:
	case "Clip":
		result = convertClip(node, inputs)
	case "Where":
		result = convertWhere(node, inputs)
	case "Min":
		result = convertMin(inputs)
	case "Max":
		result = convertMax(inputs)

		// Ops with attributes:
	case "Constant":
		result = convertConstant(m, node, g)
	case "Gather":
		result = convertGather(node, inputs)
	case "GatherElements":
		result = convertGatherElements(node, inputs)
	case "Shape":
		result = convertShape(node, inputs)
	case "Concat":
		result = convertConcat(node, inputs)
	case "Softmax":
		result = convertSoftmax(node, inputs)
	case "Cast":
		result = convertCast(node, inputs)
	case "Transpose":
		result = convertTranspose(node, inputs)
	case "Gemm":
		result = convertGemm(node, inputs)
	case "Flatten":
		result = convertFlatten(node, inputs)
	case "DequantizeLinear":
		result = convertDequantizeLinear(node, inputs)
	case "MatMulInteger":
		result = convertMatMulInteger(node, inputs)

		// Ops that require constant sub-expression materialization:
		// they take dynamic (graph) values in ONNX but only take static values in XLA
	case "Squeeze":
		result = convertSqueeze(m, convertedOutputs, node, inputs)
	case "Unsqueeze":
		result = convertUnsqueeze(m, convertedOutputs, node, inputs)
	case "Slice":
		result = convertSlice(m, convertedOutputs, node, inputs)
	case "Reshape":
		result = convertReshape(m, convertedOutputs, node, inputs)
	case "ReduceMean":
		result = convertReduceMean(m, convertedOutputs, node, inputs)
	case "ConstantOfShape":
		result = convertConstantOfShape(m, convertedOutputs, node, inputs)
	case "Expand":
		result = convertExpand(m, convertedOutputs, node, inputs)
	case "Tile":
		result = convertTile(m, convertedOutputs, node, inputs)
	case "Range":
		result = convertRange(m, convertedOutputs, node, inputs)
	case "CumSum":
		result = convertCumSum(m, convertedOutputs, node, inputs)

	// Full ML layers ops:
	case "LSTM":
		result = convertLSTM(m, convertedOutputs, node, inputs)
	case "Conv":
		result = convertConv(m, convertedOutputs, node, inputs)
	case "AveragePool":
		result = convertAveragePool(m, convertedOutputs, node, inputs)
	case "MaxPool":
		result = convertMaxPool(m, convertedOutputs, node, inputs)
	case "GlobalAveragePool":
		result = convertGlobalAveragePool(m, convertedOutputs, node, inputs)
	case "BatchNormalization":
		result = convertBatchNormalization(m, convertedOutputs, node, inputs)
	case "LayerNormalization":
		result = convertLayerNormalization(m, convertedOutputs, node, inputs)

	// Multiple outputs ops:
	case "Pad":
		result = convertPad(m, convertedOutputs, node, inputs)
	case "DynamicQuantizeLinear":
		result = convertDynamicQuantizeLinear(convertedOutputs, node, inputs)
	case "Split":
		result = convertSplit(m, convertedOutputs, node, inputs)
	case "Trilu":
		result = convertTrilu(m, convertedOutputs, node, inputs)
	case "ScatterND":
		result = convertScatterND(m, convertedOutputs, node, inputs)

	// Control flow ops:
	case "If":
		result = convertIf(m, convertedOutputs, node, inputs)

		// Ops not implemented:
	default:
		exceptions.Panicf("unimplemented ONNX op %q in %s", node.OpType, nodeToString(node))
	}
	if result != nil {
		convertedOutputs[node.Output[0]] = result
	} else {
		exceptions.Panicf("nil output for ONNX node %q", node.Name)
	}
}
