package onnx

import (
	"fmt"
	"runtime"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
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
				// Check if the static value is a zero-size tensor (e.g., empty KV cache).
				// Zero-size tensors cannot be represented as constants in some backends (e.g., XLA),
				// so we leave them as nil in convertedOutputs. Op converters must handle nil inputs.
				if t, ok := staticValue.(*tensors.Tensor); ok && t.Shape().Size() == 0 {
					convertedOutputs[inputName] = nil
				} else {
					inputN = Const(g, staticValue)
					convertedOutputs[inputName] = inputN
				}
			} else {
				missingInputs.Insert(inputName)
				continue
			}
		} else {
			if _, found := m.inputsAsConstants[inputName]; found {
				repeatedInputs.Insert(inputName)
			}
			convertedOutputs[inputName] = inputN
		}
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

	// Validate the input shapes, skipping nil entries from zero-size constants.
	var validShapes []shapes.Shape
	var validIndices []int
	for idx, inputName := range m.InputsNames {
		if n := convertedOutputs[inputName]; n != nil {
			validShapes = append(validShapes, n.Shape())
			validIndices = append(validIndices, idx)
		}
	}
	// Build a temporary model view with only the non-nil inputs for validation.
	if len(validShapes) < len(m.InputsNames) {
		tmpModel := &Model{
			InputsNames:  make([]string, len(validIndices)),
			InputsShapes: make([]DynamicShape, len(validIndices)),
		}
		for i, idx := range validIndices {
			tmpModel.InputsNames[i] = m.InputsNames[idx]
			tmpModel.InputsShapes[i] = m.InputsShapes[idx]
		}
		err := tmpModel.ValidateInputs(validShapes...)
		if err != nil {
			panic(err)
		}
	} else {
		err := m.ValidateInputs(validShapes...)
		if err != nil {
			panic(err)
		}
	}

	// Convert variables: create the GoMLX nodes corresponding to the ONNX model variables.
	if len(m.Proto.Graph.Initializer) > 0 && ctx == nil {
		exceptions.Panicf("onnx.CallGraph(): model has variables, but a nil context was give")
		return
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
	for range 3 {
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

	// Check if this output belongs to a fusion group.
	if fg := m.isFusionGroupOutput(nodeOutputName); fg != nil {
		m.ensureFusionGroupConverted(ctx, g, fg, convertedOutputs)
		return
	}

	// Is it the output of a variable?
	if _, found := m.variableNameToValue[nodeOutputName]; found {
		if ctx == nil {
			exceptions.Panicf("onnx.CallGraph(): model has variables, but a nil context was given")
			return
		}
		varName := SafeVarName(nodeOutputName)
		v := ctx.GetVariable(varName)
		if v == nil {
			exceptions.Panicf("variable %q (named %q in ONNX) has not been uploaded yet to context -- did you forget to call onnx.Model.VariablesToContext?",
				varName, nodeOutputName)
			return
		}

		// Check if the backend prefers constants for variables (e.g., CoreML).
		// This enables optimizations like blob storage for weights and avoids
		// passing hundreds of weight tensors as inputs per inference.
		backend := g.Backend()
		if backend != nil && backend.Capabilities().PreferConstantsForVariables {
			// Get the variable value and create a constant node
			value, err := v.Value()
			if err != nil {
				exceptions.Panicf("failed to get value for variable %q: %v", varName, err)
				return
			}
			convertedOutputs[nodeOutputName] = Const(g, value)
		} else {
			// Default behavior: create a parameter that will be filled at execution time
			convertedOutputs[nodeOutputName] = v.ValueGraph(g)
		}
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
	reader := m.getExternalDataReader()
	for _, initializerProto := range subGraphProto.Initializer {
		initializerName := initializerProto.Name
		if initializerName == "" {
			continue
		}
		// Convert the initializer tensor to a GoMLX constant
		tensor, err := tensorToGoMLXWithBaseDir(g.Backend(), initializerProto, m.baseDir(), reader)
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
			tensor, err := tensorToGoMLXWithBaseDir(g.Backend(), initializerProto, m.baseDir(), reader)
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
		result = m.convertBinaryOp(Add, inputs[0], inputs[1])
	case "Sub":
		result = m.convertBinaryOp(Sub, inputs[0], inputs[1])
	case "Mul":
		result = m.convertBinaryOp(Mul, inputs[0], inputs[1])
	case "Div":
		result = m.convertBinaryOp(Div, inputs[0], inputs[1])
	case "Pow":
		result = m.convertPow(convertedOutputs, node, inputs)
	case "And":
		result = m.convertBinaryOp(LogicalAnd, inputs[0], inputs[1])
	case "Or":
		result = m.convertBinaryOp(LogicalOr, inputs[0], inputs[1])
	case "Xor":
		result = m.convertBinaryOp(LogicalXor, inputs[0], inputs[1])
	case "BitwiseAnd":
		result = m.convertBinaryOp(BitwiseAnd, inputs[0], inputs[1])
	case "BitwiseOr":
		result = m.convertBinaryOp(BitwiseOr, inputs[0], inputs[1])
	case "BitwiseXor":
		result = m.convertBinaryOp(BitwiseXor, inputs[0], inputs[1])
	case "Equal":
		result = m.convertBinaryOp(Equal, inputs[0], inputs[1])
	case "Less":
		result = m.convertBinaryOp(LessThan, inputs[0], inputs[1])
	case "LessOrEqual":
		result = m.convertBinaryOp(LessOrEqual, inputs[0], inputs[1])
	case "Greater":
		result = m.convertBinaryOp(GreaterThan, inputs[0], inputs[1])
	case "GreaterOrEqual":
		result = m.convertBinaryOp(GreaterOrEqual, inputs[0], inputs[1])

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
	case "Gelu":
		result = activations.Gelu(inputs[0])
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
	case "Sigmoid":
		result = Sigmoid(inputs[0])

	// Ops with equivalents:
	case "MatMul":
		result = m.convertMatMul(inputs[0], inputs[1])

	// Ops with special behavior:
	case "Clip":
		result = m.convertClip(node, inputs)
	case "Where":
		result = m.convertWhere(node, inputs)
	case "Min":
		result = m.convertMin(inputs)
	case "Max":
		result = m.convertMax(inputs)

	// Ops with attributes:
	case "Constant":
		result = convertConstant(m, node, g)
	case "Gather":
		result = convertGather(node, inputs)
	case "GatherElements":
		result = convertGatherElements(node, inputs)
	case "GatherND":
		result = convertGatherND(node, inputs)
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
		result = m.convertGemm(node, inputs)
	case "Flatten":
		result = convertFlatten(node, inputs)
	case "DequantizeLinear":
		result = m.convertDequantizeLinear(convertedOutputs, node, inputs)
	case "QuantizeLinear":
		result = convertQuantizeLinear(node, inputs)
	case "MatMulInteger":
		result = convertMatMulInteger(node, inputs)
	case "QLinearMatMul":
		result = convertQLinearMatMul(node, inputs)

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
	case "ReduceMax":
		result = convertReduceMax(m, convertedOutputs, node, inputs)
	case "ReduceMin":
		result = convertReduceMin(m, convertedOutputs, node, inputs)
	case "ReduceSum":
		result = convertReduceSum(m, convertedOutputs, node, inputs)
	case "ReduceProd":
		result = convertReduceProd(m, convertedOutputs, node, inputs)
	case "ReduceL2":
		result = convertReduceL2(m, convertedOutputs, node, inputs)
	case "NonZero":
		result = convertNonZero(m, convertedOutputs, node, inputs)
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
	case "Resize":
		result = convertResize(m, convertedOutputs, node, inputs)
	case "BatchNormalization":
		result = convertBatchNormalization(m, convertedOutputs, node, inputs)
	case "LayerNormalization":
		result = convertLayerNormalization(m, convertedOutputs, node, inputs)
	case "SimplifiedLayerNormalization":
		result = convertSimplifiedLayerNormalization(m, convertedOutputs, node, inputs)
	case "RotaryEmbedding":
		result = convertRotaryEmbedding(m, convertedOutputs, node, inputs)
	case "MultiHeadAttention":
		result = convertMultiHeadAttention(m, convertedOutputs, node, inputs)
	case "GroupQueryAttention":
		result = convertGroupQueryAttention(m, convertedOutputs, node, inputs)

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

	// Sorting/ranking ops:
	case "TopK":
		result = convertTopK(m, convertedOutputs, node, inputs)
	case "ArgMax":
		result = convertArgMax(node, inputs)
	case "ArgMin":
		result = convertArgMin(node, inputs)

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
