package onnx

import (
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// QKVDenseParams holds parameters for fused QKV projection.
type QKVDenseParams struct {
	SharedInputName                       string
	WQName, WKName, WVName                string
	BiasQName, BiasKName, BiasVName       string
	QOutputName, KOutputName, VOutputName string
	QDim, KVDim                           int
}

// qkvDenseCandidate implements FusionCandidate for fused QKV projection.
type qkvDenseCandidate struct {
	params          *QKVDenseParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *qkvDenseCandidate) Name() string       { return "QKVDense" }
func (c *qkvDenseCandidate) Score() float32      { return 80.0 }
func (c *qkvDenseCandidate) OutputNames() []string {
	return []string{c.params.QOutputName, c.params.KOutputName, c.params.VOutputName}
}
func (c *qkvDenseCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *qkvDenseCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *qkvDenseCandidate) Emit(_ *context.Context, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	x := convertedOutputs[p.SharedInputName]
	wQ := convertedOutputs[p.WQName]
	wK := convertedOutputs[p.WKName]
	wV := convertedOutputs[p.WVName]

	// ONNX MatMul computes x @ W where W is [inFeatures, outDim].
	// QKVProjection expects wQKV shape [inFeatures, qDim+2*kvDim] — same ONNX convention.
	// Concatenate along the last axis (output dimension).
	wQKV := Concatenate([]*Node{wQ, wK, wV}, -1)

	var biasQ, biasK, biasV *Node
	if p.BiasQName != "" {
		biasQ = convertedOutputs[p.BiasQName]
	}
	if p.BiasKName != "" {
		biasK = convertedOutputs[p.BiasKName]
	}
	if p.BiasVName != "" {
		biasV = convertedOutputs[p.BiasVName]
	}

	q, k, v := attention.QKVProjection(x, wQKV, biasQ, biasK, biasV, p.QDim, p.KVDim)
	convertedOutputs[p.QOutputName] = q
	convertedOutputs[p.KOutputName] = k
	convertedOutputs[p.VOutputName] = v
}

func init() {
	RegisterFusionDetector(detectQKVDenseCandidates)
}

// detectQKVDenseCandidates scans for three MatMul nodes sharing the same first input (x)
// with constant weight second inputs, optionally followed by bias Add nodes.
func detectQKVDenseCandidates(m *Model, graph *protos.GraphProto, consumers map[string][]*protos.NodeProto) []FusionCandidate {
	// Group MatMul nodes by their first input.
	matmulsByInput := make(map[string][]*protos.NodeProto)
	for _, node := range graph.Node {
		if node.OpType != "MatMul" || len(node.Input) < 2 {
			continue
		}
		firstInput := node.Input[0]
		if firstInput == "" {
			continue
		}
		matmulsByInput[firstInput] = append(matmulsByInput[firstInput], node)
	}

	var candidates []FusionCandidate
	for sharedInput, matmuls := range matmulsByInput {
		if len(matmuls) != 3 {
			continue
		}
		if cand := m.tryMatchQKVDense(graph, consumers, sharedInput, matmuls); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMatchQKVDense attempts to match a QKV Dense fusion pattern from 3 MatMul nodes sharing input x.
func (m *Model) tryMatchQKVDense(graph *protos.GraphProto, consumers map[string][]*protos.NodeProto, sharedInput string, matmuls []*protos.NodeProto) *qkvDenseCandidate {
	// All three MatMuls must have constant weight (second input).
	type projection struct {
		matmul     *protos.NodeProto
		weightName string
		biasName   string
		outputName string
		dim        int
	}

	projs := make([]projection, 3)
	for i, mm := range matmuls {
		weightName := mm.Input[1]
		if !m.isConstant(weightName) {
			return nil
		}
		dim := m.getWeightOutputDim(graph, weightName)
		if dim <= 0 {
			return nil
		}
		if len(mm.Output) == 0 {
			return nil
		}

		projs[i] = projection{
			matmul:     mm,
			weightName: weightName,
			outputName: mm.Output[0],
			dim:        dim,
		}

		// Check for optional bias Add after each MatMul.
		biasConsumer := soleConsumer(consumers, mm.Output[0])
		if biasConsumer != nil && biasConsumer.OpType == "Add" {
			biasName := otherAddInput(biasConsumer, mm.Output[0])
			if biasName != "" && m.isConstant(biasName) {
				projs[i].biasName = biasName
				if len(biasConsumer.Output) > 0 {
					projs[i].outputName = biasConsumer.Output[0]
				}
			}
		}
	}

	// Verify no internal output (MatMul outputs when bias is used) is consumed externally.
	internalNodes := make(map[*protos.NodeProto]bool)
	internalOutputs := make(map[string]bool)
	for _, p := range projs {
		internalNodes[p.matmul] = true
		if p.biasName != "" {
			// The MatMul output is internal (consumed only by the bias Add).
			internalOutputs[p.matmul.Output[0]] = true
			// The bias Add node is also internal.
			biasAddNode := soleConsumer(consumers, p.matmul.Output[0])
			if biasAddNode != nil {
				internalNodes[biasAddNode] = true
			}
		}
	}

	if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	// Determine Q, K, V ordering. We use dim sizes: Q typically has the largest dim,
	// or equal dims. Without explicit ordering info, we assign by order of appearance.
	// The caller can also rely on the output names matching their model's convention.
	qIdx, kIdx, vIdx := 0, 1, 2

	// If two projections have the same dim and one differs, the differing one is Q (or they're all equal).
	if projs[0].dim == projs[1].dim && projs[0].dim != projs[2].dim {
		// projs[0] and [1] are KV, [2] is Q
		qIdx, kIdx, vIdx = 2, 0, 1
	} else if projs[0].dim == projs[2].dim && projs[0].dim != projs[1].dim {
		// projs[0] and [2] are KV, [1] is Q
		qIdx, kIdx, vIdx = 1, 0, 2
	} else if projs[1].dim == projs[2].dim && projs[1].dim != projs[0].dim {
		// projs[1] and [2] are KV, [0] is Q
		qIdx, kIdx, vIdx = 0, 1, 2
	}
	// If all equal, keep default ordering.

	qProj := projs[qIdx]
	kProj := projs[kIdx]
	vProj := projs[vIdx]

	// kvDim must be equal for K and V.
	if kProj.dim != vProj.dim {
		return nil
	}

	params := &QKVDenseParams{
		SharedInputName: sharedInput,
		WQName:          qProj.weightName,
		WKName:          kProj.weightName,
		WVName:          vProj.weightName,
		BiasQName:       qProj.biasName,
		BiasKName:       kProj.biasName,
		BiasVName:       vProj.biasName,
		QOutputName:     qProj.outputName,
		KOutputName:     kProj.outputName,
		VOutputName:     vProj.outputName,
		QDim:            qProj.dim,
		KVDim:           kProj.dim,
	}

	externalInputs := []string{sharedInput, qProj.weightName, kProj.weightName, vProj.weightName}
	if qProj.biasName != "" {
		externalInputs = append(externalInputs, qProj.biasName)
	}
	if kProj.biasName != "" {
		externalInputs = append(externalInputs, kProj.biasName)
	}
	if vProj.biasName != "" {
		externalInputs = append(externalInputs, vProj.biasName)
	}

	return &qkvDenseCandidate{
		params:          params,
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
	}
}

// isConstant checks if a name refers to a constant (initializer or Constant node output).
func (m *Model) isConstant(name string) bool {
	if _, ok := m.variableNameToValue[name]; ok {
		return true
	}
	if node, ok := m.nodeOutputToNode[name]; ok && node.OpType == "Constant" {
		return true
	}
	return false
}

// getWeightOutputDim returns the output dimension of a weight matrix.
// For a MatMul x @ W where x is [batch, inFeatures] and W is [inFeatures, outFeatures],
// returns outFeatures. Returns -1 if unknown.
func (m *Model) getWeightOutputDim(graph *protos.GraphProto, weightName string) int {
	dims := m.getShapeDims(graph, weightName)
	if len(dims) < 2 {
		return -1
	}
	// Weight shape is [inFeatures, outFeatures] for standard MatMul.
	return dims[len(dims)-1]
}
