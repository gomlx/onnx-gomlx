package onnx

import (
	"fmt"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"github.com/pkg/errors"
	"maps"
	"slices"
	"strings"
)

// ModelScope is the default model scope to use when for the ONNX model variables when converting to GoMLX.
var ModelScope = "ONNX"

// This file defines the methods that build the computation graph using GoMLX.

// VariablesToContext will create variables in the context (within scope ModelScope) from
// all variables present in the model initializer list.
//
// Call this once in your context, before using the model with Model.CallGraph.
// Alternatively, if you have already checkpoint-ed your model, load the variables from a checkpoint and don't call this.
func (m *Model) VariablesToContext(ctx *context.Context) error {
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.VariablesToContext does not support ONNX SparseTensors")
	}
	ctx = ctx.In(ModelScope).Checked(false)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensor, err := togomlx.Tensor(tensorProto)
		if err != nil {
			return errors.WithMessagef(err, "Model.VariablesToContext()")
		}
		tensorName := SafeVarName(tensorProto.Name)
		ctx.VariableWithValue(tensorName, tensor)
	}
	return nil
}

// SafeVarName converts an ONNX variable name to a GoMLX safe variable name by replacing the scope separator with a "|".
func SafeVarName(onnxName string) (gomlxName string) {
	return strings.ReplaceAll(onnxName, context.ScopeSeparator, "|")
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
// The graph being built is given in g.
//
// As in GoMLX graph functions, it panics (throws exceptions) in case of errors.
func (m *Model) CallGraph(ctx *context.Context, g *Graph, inputs map[string]*Node) (outputs []*Node) {
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
	if len(m.Proto.Graph.Initializer) > 0 {
		if ctx == nil {
			exceptions.Panicf("onnx.CallGraph(): model has variables, but a nil context was give")
			panic(nil) // for lint benefit.
		}
		for _, tensorProto := range m.Proto.Graph.Initializer {
			varName := SafeVarName(tensorProto.Name)
			v := ctx.InspectVariableInScope(varName)
			if v == nil {
				exceptions.Panicf("variable %q (from the ONNX model %q) has not been uploaded yet to context -- did you forget to call onnx.Model.VariablesToContext?",
					varName, tensorProto.Name)
				panic(nil) // for lint benefit.
			}
			convertedOutputs[tensorProto.Name] = v.ValueGraph(g)
		}
	}

	// Convert all nodes in topological order.
	sortedNodes := m.sortedGraph()
	for ii, node := range sortedNodes {
		err := exceptions.TryCatch[error](func() { m.convertNode(g, node, convertedOutputs) })
		if err != nil {
			err = errors.WithMessagef(err, "while converting node %d out of %d", ii, len(sortedNodes))
			panic(err)
		}
	}

	// Pick the outputs.
	outputs = make([]*Node, len(m.OutputsNames))
	var found bool
	for outputIdx, nodeName := range m.OutputsNames {
		outputs[outputIdx], found = convertedOutputs[nodeName]
		if !found {
			exceptions.Panicf("output node %q not found", nodeName)
		}
	}
	return outputs
}

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// sortedGraph returns a DAG sorting of the graph, so the returned nodes can be converted in order.
//
// It assumes the inputs and variables are given.
//
// Careful not to mix up node.Name and node.Output (there can be more than one output).
func (m *Model) sortedGraph() []*protos.NodeProto {
	sortedNodes := make([]*protos.NodeProto, 0, len(m.Proto.Graph.Node))

	// Build reverse dependency map.
	outputToDependants := make(map[string]types.Set[*protos.NodeProto])
	for _, node := range m.Proto.Graph.Node {
		for _, input := range node.Input {
			deps, found := outputToDependants[input]
			if !found {
				deps = types.SetWith(node)
				outputToDependants[input] = deps
			} else {
				deps.Insert(node)
			}
		}
	}

	// Check whether node is done.
	doneOutputs := types.MakeSet[string]() // It includes both: Node.Name and Node.Output.
	isReady := func(node *protos.NodeProto) bool {
		for _, input := range node.Input {
			_, found := doneOutputs[input]
			if !found {
				return false
			}
		}
		return true
	}

	// Tabs on finished nodes, and process of marking one node as done.
	nextDoneScan := types.MakeSet[string]()
	markDone := func(outputName string) {
		deps, found := outputToDependants[outputName]
		if !found {
			return
		}
		delete(outputToDependants, outputName)
		for dep := range maps.Keys(deps) {
			if doneOutputs.Has(dep.Name) {
				// This dependant is already marked as done.
				continue
			}
			if !isReady(dep) {
				// This dependant has other dependencies and is not done yet.
				continue
			}
			// One of the dependents is ready, so mark this node as done.
			sortedNodes = append(sortedNodes, dep)
			doneOutputs.Insert(dep.Name)
			for _, output := range dep.Output {
				doneOutputs.Insert(output)
				nextDoneScan.Insert(output)
			}
		}
	}

	// Mark inputs (inputs names and outputs are the same), variables and nodes without any inputs as finished.
	for _, input := range m.InputsNames {
		doneOutputs.Insert(input)
		nextDoneScan.Insert(input)
	}
	for _, tensorProto := range m.Proto.Graph.Initializer {
		// Mark variable node as done.
		doneOutputs.Insert(tensorProto.Name)
		nextDoneScan.Insert(tensorProto.Name)
	}
	for _, node := range m.Proto.Graph.Node {
		if len(node.Input) > 0 {
			continue
		}
		// No inputs: mark as done and append to sortedNodes.
		sortedNodes = append(sortedNodes, node)
		doneOutputs.Insert(node.Name)
		for _, output := range node.Output {
			doneOutputs.Insert(output)
			nextDoneScan.Insert(output)
		}
	}

	// Loop marking nodes as done, and collecting nextDoneScan for the next iteration.
	for len(nextDoneScan) > 0 {
		nextDoneScanSlice := slices.Collect(maps.Keys(nextDoneScan))
		clear(nextDoneScan) // Clear for next batch.
		for _, nodeName := range nextDoneScanSlice {
			markDone(nodeName)
		}
	}
	//fmt.Printf("nodes: %v\n", sliceMap(m.Proto.Graph.Node, func(n *protos.NodeProto) string { return n.Name }))
	//fmt.Printf("sortedNodes: %v\n", sliceMap(sortedNodes, func(n *protos.NodeProto) string { return n.Name }))
	if len(sortedNodes) != len(m.Proto.Graph.Node) {
		exceptions.Panicf("sorting operations graph failed: found %d nodes connected to inputs, but there were %d nodes!?",
			len(sortedNodes), len(m.Proto.Graph.Node))
	}
	return sortedNodes
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
func (m *Model) convertNode(g *Graph, node *protos.NodeProto, convertedOutputs map[string]*Node) {
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
		res = convertBinaryOp(Pow, inputs[0], inputs[1])
	case "Equal":
		res = convertBinaryOp(Equal, inputs[0], inputs[1])

	// Unary operators
	case "Sqrt":
		res = Sqrt(inputs[0])
	case "Exp":
		res = Exp(inputs[0])
	case "Log":
		res = Log(inputs[0])
	case "Erf":
		res = Erf(inputs[0])

		// Ops with equivalents:
	case "MatMul":
		res = MatMul(inputs[0], inputs[1])
	case "Where":
		res = Where(inputs[0], inputs[1], inputs[2])

		// Ops with attributes:
	case "Constant":
		res = convertConstant(node, g)
	case "Gather":
		res = convertGather(node, inputs)
	case "Shape":
		res = convertShape(node, inputs)
	case "Concat":
		res = convertConcat(node, inputs)
	case "Softmax":
		res = convertSoftmax(node, inputs)
	case "Cast":
		res = convertCast(node, inputs)

		// Ops that require contant-expression materialization:
		// they take dynamic (graph) values in ONNX, but only take static values in XLA
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

	default:
		exceptions.Panicf("unimplemented ONNX %s", nodeToString(node))
	}
	if res != nil {
		convertedOutputs[node.Output[0]] = res
	}
}
