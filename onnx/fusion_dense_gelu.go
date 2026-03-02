package onnx

import (
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// DenseActivationParams holds parameters for fused MatMul + optional bias + activation.
type DenseActivationParams struct {
	XInputName     string
	WeightName     string
	BiasName       string // empty if no bias
	OutputName     string // final output after activation
}

// denseActivationCandidate implements FusionCandidate for fused Dense+Activation.
type denseActivationCandidate struct {
	params          *DenseActivationParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *denseActivationCandidate) Name() string                    { return "DenseGelu" }
func (c *denseActivationCandidate) Score() float32                   { return 50.0 }
func (c *denseActivationCandidate) OutputNames() []string            { return []string{c.params.OutputName} }
func (c *denseActivationCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *denseActivationCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *denseActivationCandidate) Emit(_ *context.Context, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	x := convertedOutputs[p.XInputName]
	weight := convertedOutputs[p.WeightName]

	var bias *Node
	if p.BiasName != "" {
		bias = convertedOutputs[p.BiasName]
	}

	result := nn.Dense(x, weight, bias, activations.TypeGelu)
	convertedOutputs[p.OutputName] = result
}

func init() {
	RegisterFusionDetector(detectDenseActivationCandidates)
}

// detectDenseActivationCandidates scans the ONNX graph for:
//
//	MatMul(x, W) → [Add(·, bias)] → Gelu(·)
//
// and returns FusionCandidates for each match.
func detectDenseActivationCandidates(m *Model) []FusionCandidate {
	consumers := m.consumers
	var candidates []FusionCandidate
	for _, node := range m.Proto.Graph.Node {
		if node.OpType != "MatMul" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		if cand := m.tryMatchDenseActivation(consumers, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMatchDenseActivation attempts to match MatMul → [Add bias] → Gelu starting from a MatMul node.
func (m *Model) tryMatchDenseActivation(consumers map[string][]*protos.NodeProto, matmulNode *protos.NodeProto) *denseActivationCandidate {
	xName := matmulNode.Input[0]
	weightName := matmulNode.Input[1]

	// Weight must be a constant.
	if !m.isConstant(weightName) {
		return nil
	}

	matmulOut := matmulNode.Output[0]
	next := onnxgraph.SoleConsumer(consumers, matmulOut)
	if next == nil {
		return nil
	}

	// Track internal nodes and outputs for external consumer check.
	internalNodes := map[*protos.NodeProto]bool{matmulNode: true}
	internalOutputs := map[string]bool{}

	switch next.OpType {
	case "Add":
		// MatMul → Add(bias) → Gelu?
		biasName := onnxgraph.OtherBinaryOpInput(next, matmulOut)
		if biasName == "" || !m.isConstant(biasName) {
			return nil
		}
		if len(next.Output) == 0 {
			return nil
		}
		internalNodes[next] = true
		internalOutputs[matmulOut] = true
		afterBiasOut := next.Output[0]

		// Now look for Gelu after Add.
		geluNode := onnxgraph.SoleConsumer(consumers, afterBiasOut)
		if geluNode == nil || geluNode.OpType != "Gelu" {
			return nil
		}
		if len(geluNode.Output) == 0 {
			return nil
		}
		internalNodes[geluNode] = true
		internalOutputs[afterBiasOut] = true

		if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName, biasName}
		return &denseActivationCandidate{
			params: &DenseActivationParams{
				XInputName: xName,
				WeightName: weightName,
				BiasName:   biasName,
				OutputName: geluNode.Output[0],
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}

	case "Gelu":
		// MatMul → Gelu (no bias).
		if len(next.Output) == 0 {
			return nil
		}
		internalNodes[next] = true
		internalOutputs[matmulOut] = true

		if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName}
		return &denseActivationCandidate{
			params: &DenseActivationParams{
				XInputName: xName,
				WeightName: weightName,
				OutputName: next.Output[0],
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}
	}

	return nil
}
