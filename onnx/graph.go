package onnx

import (
	"fmt"
	"runtime"

	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// CallGraph calls the ONNX graph, and hence building it with GoMLX ops.
// This can be used for inference or training.
//
// If the model has any variables, call Model.VariablesToContext first (only once) to upload all
// variable values from the ONNX model to the context -- or load them from a checkpoint if you saved one.
//
// If the model has no variables, the context in ctx can be set to nil.
//
// The inputs (a map of input name to its graph.Node) can be given as normal input parameters to the graph or as
// static constants -- see WithInputsAsConstants.
// Set the inputs as constants if they are meant to be interpreted as constants (static) values, that won't change
// in different inference/training steps.
//
// If outputNames is not given, it will output the model's registered outputs. Alternatively, you can select
// any list of node outputs to generate. It will return the values for the selected outputs.
//
// The graph being built is given in g.
//
// As in GoMLX graph functions, it panics (throws exceptions) in case of errors.
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

	// Map the given inputs to the corresponding ONNX inputs, and report (throw exception) if there are
	// any discrepancies.
	// Also initialize convertedOutputs with the given/converted inputs.
	convertedOutputs := make(map[string]*Node)
	missingInputs := types.MakeSet[string]()
	repeatedInputs := types.MakeSet[string]()
	unknownInputs := types.MakeSet[string]()
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

	// Makes sure all temporarily allocated tensor on device are freed.
	for _ = range 3 {
		runtime.GC()
	}
	return outputs
}

// recursiveCallGraph recursively creates a GoMLX graph for the target output name.
// The convertedOutputs is used both as input, and as output to store the converted nodes.
func (m *Model) recursiveCallGraph(ctx *context.Context, g *Graph, nodeOutputName string, convertedOutputs map[string]*Node) {
	if _, found := convertedOutputs[nodeOutputName]; found {
		// Already converted.
		return
	}

	// Is it the output of a variable ?
	if _, found := m.variableNameToValue[nodeOutputName]; found {
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
func (m *Model) convertNode(ctx *context.Context, g *Graph, node *protos.NodeProto, convertedOutputs map[string]*Node) {
	if node.Overload != "" {
		exceptions.Panicf("overload %q to in-model function in ONNX model not implemented in node %q", node.Overload, node.Name)
	}

	// Convert the node: the usual case is that there is only one output.
	// If res is not nil, it is set to convertedOutputs[output[0]].
	// Anything different must be implemented by the specific op switch.
	var res *Node
	inputs := sliceMap(node.Input, func(n string) *Node { return convertedOutputs[n] })
	switch node.OpType {
	// Binary operators: see note on differences on default broadcasting.
	case "Add":
		res = convertBinaryOp(Add, inputs[0], inputs[1])
	case "Sub":
		res = convertBinaryOp(Sub, inputs[0], inputs[1])
	case "Mul":
		res = convertBinaryOp(Mul, inputs[0], inputs[1])
	case "Div":
		res = convertBinaryOp(Div, inputs[0], inputs[1])
	case "Pow":
		//res = convertBinaryOp(Pow, inputs[0], inputs[1])
		res = convertPow(m, convertedOutputs, node, inputs)
	case "And":
		res = convertBinaryOp(LogicalAnd, inputs[0], inputs[1])
	case "Or":
		res = convertBinaryOp(LogicalOr, inputs[0], inputs[1])
	case "Xor":
		res = convertBinaryOp(LogicalXor, inputs[0], inputs[1])
	case "BitwiseAnd":
		res = convertBinaryOp(BitwiseAnd, inputs[0], inputs[1])
	case "BitwiseOr":
		res = convertBinaryOp(BitwiseOr, inputs[0], inputs[1])
	case "BitwiseXor":
		res = convertBinaryOp(BitwiseXor, inputs[0], inputs[1])
	case "Equal":
		res = convertBinaryOp(Equal, inputs[0], inputs[1])
	case "Less":
		res = convertBinaryOp(LessThan, inputs[0], inputs[1])
	case "LessOrEqual":
		res = convertBinaryOp(LessOrEqual, inputs[0], inputs[1])
	case "Greater":
		res = convertBinaryOp(GreaterThan, inputs[0], inputs[1])
	case "GreaterOrEqual":
		res = convertBinaryOp(GreaterOrEqual, inputs[0], inputs[1])

	// Unary operators
	case "Sqrt":
		res = Sqrt(inputs[0])
	case "Exp":
		res = Exp(inputs[0])
	case "Log":
		res = Log(inputs[0])
	case "Erf":
		res = Erf(inputs[0])
	case "Relu":
		res = activations.Relu(inputs[0])
	case "Abs":
		res = Abs(inputs[0])
	case "Neg":
		res = Neg(inputs[0])
	case "Sign":
		res = Sign(inputs[0])
	case "Ceil":
		res = Ceil(inputs[0])
	case "Floor":
		res = Floor(inputs[0])
	case "Identity":
		res = Identity(inputs[0])
	case "Not":
		res = LogicalNot(inputs[0])
	case "BitwiseNot":
		res = BitwiseNot(inputs[0])
	case "Tanh":
		res = Tanh(inputs[0])
	case "Sin":
		res = Sin(inputs[0])
	case "Cos":
		res = Cos(inputs[0])

		// Ops with equivalents:
	case "MatMul":
		res = MatMul(inputs[0], inputs[1])

	// Ops with special behavior:
	case "Clip":
		res = convertClip(node, inputs)
	case "Where":
		res = convertWhere(node, inputs)
	case "Min":
		res = convertMin(inputs)
	case "Max":
		res = convertMax(inputs)

		// Ops with attributes:
	case "Constant":
		res = convertConstant(m, node, g)
	case "Gather":
		res = convertGather(node, inputs)
	case "GatherElements":
		res = convertGatherElements(node, inputs)
	case "Shape":
		res = convertShape(node, inputs)
	case "Concat":
		res = convertConcat(node, inputs)
	case "Softmax":
		res = convertSoftmax(node, inputs)
	case "Cast":
		res = convertCast(node, inputs)
	case "Transpose":
		res = convertTranspose(node, inputs)
	case "Gemm":
		res = convertGemm(node, inputs)
	case "Flatten":
		res = convertFlatten(node, inputs)
	case "DequantizeLinear":
		res = convertDequantizeLinear(node, inputs)

		// Ops that require constant sub-expression materialization:
		// they take dynamic (graph) values in ONNX, but only take static values in XLA
	case "Squeeze":
		res = convertSqueeze(m, convertedOutputs, node, inputs)
	case "Unsqueeze":
		res = convertUnsqueeze(m, convertedOutputs, node, inputs)
	case "Slice":
		res = convertSlice(m, convertedOutputs, node, inputs)
	case "Reshape":
		res = convertReshape(m, convertedOutputs, node, inputs)
	case "ReduceMean":
		res = convertReduceMean(m, convertedOutputs, node, inputs)
	case "ConstantOfShape":
		res = convertConstantOfShape(m, convertedOutputs, node, inputs)
	case "Expand":
		res = convertExpand(m, convertedOutputs, node, inputs)
	case "Tile":
		res = convertTile(m, convertedOutputs, node, inputs)
	case "Range":
		res = convertRange(m, convertedOutputs, node, inputs)
	case "CumSum":
		res = convertCumSum(m, convertedOutputs, node, inputs)

	// Full ML layers ops:
	case "LSTM":
		res = convertLSTM(m, convertedOutputs, node, inputs)

	// Multiple outputs ops:
	case "DynamicQuantizeLinear":
		res = convertDynamicQuantizeLinear(convertedOutputs, node, inputs)
	case "Trilu":
		fmt.Println("entered trilu")
		res = convertTrilu(m, convertedOutputs, node, inputs)
	case "ScatterND":
		fmt.Println("entered ScatterND")
		res = convertScatterND(m, convertedOutputs, node, inputs)

		// Ops not implemented:
	default:
		exceptions.Panicf("unimplemented ONNX op %q in %s", node.OpType, nodeToString(node))
	}
	if res != nil {
		convertedOutputs[node.Output[0]] = res
	} else {
		exceptions.Panicf("nil output for ONNX node %q", node.Name)
	}
}
