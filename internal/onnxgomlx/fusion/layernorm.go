package fusion

import (
	"math"

	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
	"github.com/gomlx/compute-onnx/support/protos"
)

// LayerNormParams holds parameters for fused LayerNorm.
type LayerNormParams struct {
	XInputName string
	WeightName string // empty if no scale
	BiasName   string // empty if no bias
	OutputName string // final output of LayerNorm
	Axes       []int
	Epsilon    float64
}

// layerNormCandidate implements onnxgomlx.FusionCandidate for LayerNorm.
type layerNormCandidate struct {
	params          *LayerNormParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *layerNormCandidate) Name() string                     { return "LayerNorm" }
func (c *layerNormCandidate) Score() float32                   { return 70.0 }
func (c *layerNormCandidate) OutputNames() []string            { return []string{c.params.OutputName} }
func (c *layerNormCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *layerNormCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *layerNormCandidate) Emit(_ *model.Scope, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params
	x := convertedOutputs[p.XInputName]

	var weight, bias *Node
	if p.WeightName != "" {
		weight = convertedOutputs[p.WeightName]
	}
	if p.BiasName != "" {
		bias = convertedOutputs[p.BiasName]
	}

	axes := make([]int, len(p.Axes))
	for i, a := range p.Axes {
		if a < 0 {
			axes[i] = x.Rank() + a
		} else {
			axes[i] = a
		}
	}

	result := nn.LayerNorm(x, axes, p.Epsilon, weight, bias, nil)
	convertedOutputs[p.OutputName] = result
}

func init() {
	onnxgomlx.RegisterFusionDetector(detectLayerNormCandidates)
}

func detectLayerNormCandidates(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	consumers := m.Consumers
	var candidates []onnxgomlx.FusionCandidate

	// 1. Explicit LayerNormalization nodes (opset 17+)
	for _, node := range m.Proto.Graph.Node {
		if node.OpType == "LayerNormalization" && len(node.Input) >= 1 && len(node.Output) > 0 {
			if cand := matchExplicitLayerNorm(node); cand != nil {
				candidates = append(candidates, cand)
			}
		}
	}

	// 2. Decomposed LayerNorm patterns starting from Sub nodes:
	// x -> ReduceMean(x) -> Sub(x, mean) -> [Pow(2) or Mul(diff, diff)] -> ReduceMean -> Add(eps) -> Sqrt -> Div(diff, sqrt) -> [Mul(weight)] -> [Add(bias)]
	for _, node := range m.Proto.Graph.Node {
		if node.OpType == "Sub" && len(node.Input) == 2 && len(node.Output) > 0 {
			if cand := tryMatchDecomposedLayerNorm(m, consumers, node); cand != nil {
				candidates = append(candidates, cand)
			}
		}
	}

	return candidates
}

func matchExplicitLayerNorm(node *protos.NodeProto) *layerNormCandidate {
	xName := node.Input[0]
	var weightName, biasName string
	externalInputs := []string{xName}
	if len(node.Input) > 1 && node.Input[1] != "" {
		weightName = node.Input[1]
		externalInputs = append(externalInputs, weightName)
	}
	if len(node.Input) > 2 && node.Input[2] != "" {
		biasName = node.Input[2]
		externalInputs = append(externalInputs, biasName)
	}
	axis := onnxgomlx.GetIntAttrOr(node, "axis", -1)
	epsilon := float64(onnxgomlx.GetFloatAttrOr(node, "epsilon", 1e-5))

	return &layerNormCandidate{
		params: &LayerNormParams{
			XInputName: xName,
			WeightName: weightName,
			BiasName:   biasName,
			OutputName: node.Output[0],
			Axes:       []int{int(axis)},
			Epsilon:    epsilon,
		},
		internalOutputs: map[string]bool{},
		externalInputs:  externalInputs,
	}
}

func tryMatchDecomposedLayerNorm(m *onnxgomlx.Model, consumers map[string][]*protos.NodeProto, subNode *protos.NodeProto) *layerNormCandidate {
	xName := subNode.Input[0]
	meanName := subNode.Input[1]

	meanNode := m.NodeOutputToNode[meanName]
	if meanNode == nil || meanNode.OpType != "ReduceMean" || len(meanNode.Input) == 0 {
		return nil
	}
	if meanNode.Input[0] != xName {
		return nil
	}

	axes := extractAxes(meanNode)
	if len(axes) == 0 {
		axes = []int{-1}
	}

	subOut := subNode.Output[0]
	subConsumers := consumers[subOut]
	// subOut must be consumed by:
	// 1) variance calculation (either Pow(diff, 2) or Mul(diff, diff))
	// 2) divNode (Div(diff, sqrt))
	if len(subConsumers) != 2 {
		return nil
	}

	var varInputNode *protos.NodeProto
	var divNode *protos.NodeProto

	for _, c := range subConsumers {
		if c.OpType == "Pow" && len(c.Input) == 2 {
			other := onnxgraph.OtherBinaryOpInput(c, subOut)
			if other != "" && m.IsConstant(other) && math.Abs(m.TryGetConstantScalar(other)-2.0) < 1e-4 {
				varInputNode = c
				continue
			}
		} else if c.OpType == "Mul" && len(c.Input) == 2 && c.Input[0] == subOut && c.Input[1] == subOut {
			varInputNode = c
			continue
		}
		if c.OpType == "Div" {
			divNode = c
		}
	}

	if varInputNode == nil || divNode == nil || len(varInputNode.Output) == 0 || len(divNode.Output) == 0 {
		return nil
	}

	// Downstream of varInputNode: must be ReduceMean with same axes
	varMeanNode := onnxgraph.SoleConsumer(consumers, varInputNode.Output[0])
	if varMeanNode == nil || varMeanNode.OpType != "ReduceMean" || len(varMeanNode.Output) == 0 {
		return nil
	}
	varAxes := extractAxes(varMeanNode)
	if len(varAxes) == 0 {
		varAxes = []int{-1}
	}
	if !equalIntSlices(axes, varAxes) {
		return nil
	}

	// Downstream of varMeanNode: Add(epsilon)
	addEpsNode := onnxgraph.SoleConsumer(consumers, varMeanNode.Output[0])
	if addEpsNode == nil || addEpsNode.OpType != "Add" || len(addEpsNode.Output) == 0 {
		return nil
	}
	epsName := onnxgraph.OtherBinaryOpInput(addEpsNode, varMeanNode.Output[0])
	if epsName == "" || !m.IsConstant(epsName) {
		return nil
	}
	epsilon := m.TryGetConstantScalar(epsName)
	if epsilon <= 0 {
		epsilon = 1e-5
	}

	// Downstream of addEpsNode: Sqrt
	sqrtNode := onnxgraph.SoleConsumer(consumers, addEpsNode.Output[0])
	if sqrtNode == nil || sqrtNode.OpType != "Sqrt" || len(sqrtNode.Output) == 0 {
		return nil
	}

	// divNode must divide subOut by sqrtNode.Output[0]
	if divNode.Input[0] != subOut || divNode.Input[1] != sqrtNode.Output[0] {
		return nil
	}

	internalNodes := map[*protos.NodeProto]bool{
		meanNode:     true,
		subNode:      true,
		varInputNode: true,
		varMeanNode:  true,
		addEpsNode:   true,
		sqrtNode:     true,
		divNode:      true,
	}
	internalOutputs := map[string]bool{
		meanName:               true,
		subOut:                 true,
		varInputNode.Output[0]: true,
		varMeanNode.Output[0]:  true,
		addEpsNode.Output[0]:   true,
		sqrtNode.Output[0]:     true,
	}

	currOut := divNode.Output[0]
	var weightName, biasName string
	externalInputs := []string{xName}

	// Check optional Mul(weight)
	mulWeightNode := onnxgraph.SoleConsumer(consumers, currOut)
	if mulWeightNode != nil && mulWeightNode.OpType == "Mul" && len(mulWeightNode.Output) > 0 {
		wName := onnxgraph.OtherBinaryOpInput(mulWeightNode, currOut)
		if wName != "" && m.IsConstant(wName) {
			weightName = wName
			internalNodes[mulWeightNode] = true
			internalOutputs[currOut] = true
			currOut = mulWeightNode.Output[0]
			externalInputs = append(externalInputs, weightName)

			// Check optional Add(bias)
			addBiasNode := onnxgraph.SoleConsumer(consumers, currOut)
			if addBiasNode != nil && addBiasNode.OpType == "Add" && len(addBiasNode.Output) > 0 {
				bName := onnxgraph.OtherBinaryOpInput(addBiasNode, currOut)
				if bName != "" && m.IsConstant(bName) {
					biasName = bName
					internalNodes[addBiasNode] = true
					internalOutputs[currOut] = true
					currOut = addBiasNode.Output[0]
					externalInputs = append(externalInputs, biasName)
				}
			}
		}
	}

	if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	return &layerNormCandidate{
		params: &LayerNormParams{
			XInputName: xName,
			WeightName: weightName,
			BiasName:   biasName,
			OutputName: currOut,
			Axes:       axes,
			Epsilon:    epsilon,
		},
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
	}
}

func extractAxes(node *protos.NodeProto) []int {
	return onnxgomlx.GetIntsAttrOr(node, "axes", nil)
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
