package onnx

import (
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// DenseGeluParams holds parameters for fused MatMul + optional bias + Gelu.
type DenseGeluParams struct {
	XInputName     string
	WeightName     string
	BiasName       string // empty if no bias
	GeluOutputName string // final output after Gelu
}

// denseGeluCandidate implements FusionCandidate for fused Dense+Gelu.
type denseGeluCandidate struct {
	params          *DenseGeluParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *denseGeluCandidate) Name() string                    { return "DenseGelu" }
func (c *denseGeluCandidate) Score() float32                   { return 50.0 }
func (c *denseGeluCandidate) OutputNames() []string            { return []string{c.params.GeluOutputName} }
func (c *denseGeluCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *denseGeluCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *denseGeluCandidate) Emit(_ *context.Context, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	x := convertedOutputs[p.XInputName]
	weight := convertedOutputs[p.WeightName]

	var bias *Node
	if p.BiasName != "" {
		bias = convertedOutputs[p.BiasName]
	}

	result := nn.Dense(x, weight, bias, activations.TypeGelu)
	convertedOutputs[p.GeluOutputName] = result
}

func init() {
	RegisterFusionDetector(detectDenseGeluCandidates)
}

// detectDenseGeluCandidates scans the ONNX graph for:
//
//	MatMul(x, W) → [Add(·, bias)] → Gelu(·)
//
// and returns FusionCandidates for each match.
func detectDenseGeluCandidates(m *Model, graph *protos.GraphProto, consumers map[string][]*protos.NodeProto) []FusionCandidate {
	var candidates []FusionCandidate
	for _, node := range graph.Node {
		if node.OpType != "MatMul" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		if cand := m.tryMatchDenseGelu(graph, consumers, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMatchDenseGelu attempts to match MatMul → [Add bias] → Gelu starting from a MatMul node.
func (m *Model) tryMatchDenseGelu(graph *protos.GraphProto, consumers map[string][]*protos.NodeProto, matmulNode *protos.NodeProto) *denseGeluCandidate {
	xName := matmulNode.Input[0]
	weightName := matmulNode.Input[1]

	// Weight must be a constant.
	if !m.isConstant(weightName) {
		return nil
	}

	matmulOut := matmulNode.Output[0]
	next := soleConsumer(consumers, matmulOut)
	if next == nil {
		return nil
	}

	// Track internal nodes and outputs for external consumer check.
	internalNodes := map[*protos.NodeProto]bool{matmulNode: true}
	internalOutputs := map[string]bool{}

	switch next.OpType {
	case "Add":
		// MatMul → Add(bias) → Gelu?
		biasName := otherAddInput(next, matmulOut)
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
		geluNode := soleConsumer(consumers, afterBiasOut)
		if geluNode == nil || geluNode.OpType != "Gelu" {
			return nil
		}
		if len(geluNode.Output) == 0 {
			return nil
		}
		internalNodes[geluNode] = true
		internalOutputs[afterBiasOut] = true

		if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName, biasName}
		return &denseGeluCandidate{
			params: &DenseGeluParams{
				XInputName:     xName,
				WeightName:     weightName,
				BiasName:       biasName,
				GeluOutputName: geluNode.Output[0],
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

		if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName}
		return &denseGeluCandidate{
			params: &DenseGeluParams{
				XInputName:     xName,
				WeightName:     weightName,
				GeluOutputName: next.Output[0],
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}
	}

	return nil
}
