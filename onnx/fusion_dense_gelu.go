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
	XInputName      string
	WeightName      string
	BiasName        string // empty if no bias
	GeluOutputName  string // final output after Gelu
}

// detectDenseGeluPatterns scans the ONNX graph for:
//
//	MatMul(x, W) → [Add(·, bias)] → Gelu(·)
//
// and registers FusionGroups for each match.
func (m *Model) detectDenseGeluPatterns(graph *protos.GraphProto, consumers map[string][]*protos.NodeProto) {
	for _, node := range graph.Node {
		if node.OpType != "MatMul" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		m.tryMatchDenseGelu(graph, consumers, node)
	}
}

// tryMatchDenseGelu attempts to match MatMul → [Add bias] → Gelu starting from a MatMul node.
func (m *Model) tryMatchDenseGelu(graph *protos.GraphProto, consumers map[string][]*protos.NodeProto, matmulNode *protos.NodeProto) {
	xName := matmulNode.Input[0]
	weightName := matmulNode.Input[1]

	// Weight must be a constant.
	if !m.isConstant(weightName) {
		return
	}

	matmulOut := matmulNode.Output[0]
	next := soleConsumer(consumers, matmulOut)
	if next == nil {
		return
	}

	// Track internal nodes and outputs for external consumer check.
	internalNodes := map[*protos.NodeProto]bool{matmulNode: true}
	internalOutputs := map[string]bool{}

	switch next.OpType {
	case "Add":
		// MatMul → Add(bias) → Gelu?
		biasName := otherAddInput(next, matmulOut)
		if biasName == "" || !m.isConstant(biasName) {
			return
		}
		if len(next.Output) == 0 {
			return
		}
		internalNodes[next] = true
		internalOutputs[matmulOut] = true
		afterBiasOut := next.Output[0]

		// Now look for Gelu after Add.
		geluNode := soleConsumer(consumers, afterBiasOut)
		if geluNode == nil || geluNode.OpType != "Gelu" {
			return
		}
		if len(geluNode.Output) == 0 {
			return
		}
		internalNodes[geluNode] = true
		internalOutputs[afterBiasOut] = true

		if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return
		}

		m.registerDenseGeluFusion(xName, weightName, biasName, geluNode.Output[0], internalOutputs)
		return

	case "Gelu":
		// MatMul → Gelu (no bias).
		if len(next.Output) == 0 {
			return
		}
		internalNodes[next] = true
		internalOutputs[matmulOut] = true

		if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return
		}

		m.registerDenseGeluFusion(xName, weightName, "", next.Output[0], internalOutputs)
		return
	}
}

func (m *Model) registerDenseGeluFusion(xName, weightName, biasName, geluOutputName string, internalOutputs map[string]bool) {
	params := &DenseGeluParams{
		XInputName:     xName,
		WeightName:     weightName,
		BiasName:       biasName,
		GeluOutputName: geluOutputName,
	}

	externalInputs := []string{xName, weightName}
	if biasName != "" {
		externalInputs = append(externalInputs, biasName)
	}

	fg := &FusionGroup{
		Type:                FusionDenseGelu,
		RootOutputName:      geluOutputName,
		InternalOutputNames: internalOutputs,
		ExternalInputNames:  externalInputs,
		Params:              params,
	}

	m.detectedFusionGroups[geluOutputName] = fg
}

// emitDenseGelu emits a FusedDense op with GELU activation for the given fusion group.
func (m *Model) emitDenseGelu(_ *context.Context, g *Graph, fg *FusionGroup, convertedOutputs map[string]*Node) {
	p := fg.Params.(*DenseGeluParams)

	x := convertedOutputs[p.XInputName]
	weight := convertedOutputs[p.WeightName]

	var bias *Node
	if p.BiasName != "" {
		bias = convertedOutputs[p.BiasName]
	}

	result := nn.Dense(x, weight, bias, activations.TypeGelu)
	convertedOutputs[p.GeluOutputName] = result
}
