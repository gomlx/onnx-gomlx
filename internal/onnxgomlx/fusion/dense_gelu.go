package fusion

import (
	"math"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
	"github.com/gomlx/compute-onnx/support/protos"
)

// DenseActivationParams holds parameters for fused MatMul + optional bias + activation.
type DenseActivationParams struct {
	XInputName     string
	WeightName     string
	BiasName       string // empty if no bias
	OutputName     string // final output after activation
	ActivationType activation.Type
}

// denseActivationCandidate implements onnxgomlx.FusionCandidate for fused Dense+Activation.
type denseActivationCandidate struct {
	params          *DenseActivationParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *denseActivationCandidate) Name() string                     { return "Dense" + c.params.ActivationType.String() }
func (c *denseActivationCandidate) Score() float32                   { return 50.0 }
func (c *denseActivationCandidate) OutputNames() []string            { return []string{c.params.OutputName} }
func (c *denseActivationCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *denseActivationCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *denseActivationCandidate) Emit(_ *model.Scope, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	x := convertedOutputs[p.XInputName]
	weight := convertedOutputs[p.WeightName]

	var bias *Node
	if p.BiasName != "" {
		bias = convertedOutputs[p.BiasName]
	}

	result := nn.Dense(x, weight, bias, compute.DenseLayoutInputOutputs, p.ActivationType)
	convertedOutputs[p.OutputName] = result
}

func init() {
	onnxgomlx.RegisterFusionDetector(detectDenseActivationCandidates)
}

// detectDenseActivationCandidates scans the ONNX graph for:
//
//	MatMul(x, W) → [Add(·, bias)] → Activation(·)
//
// and returns FusionCandidates for each match.
func detectDenseActivationCandidates(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	consumers := m.Consumers
	var candidates []onnxgomlx.FusionCandidate
	for _, node := range m.Proto.Graph.Node {
		if node.OpType != "MatMul" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		if cand := tryMatchDenseActivation(m, consumers, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMatchDenseActivation attempts to match MatMul → [Add bias] → Activation starting from a MatMul node.
func tryMatchDenseActivation(m *onnxgomlx.Model, consumers map[string][]*protos.NodeProto, matmulNode *protos.NodeProto) *denseActivationCandidate {
	xName := matmulNode.Input[0]
	weightName := matmulNode.Input[1]

	// Weight must be a constant.
	if !m.IsConstant(weightName) {
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

	if next.OpType == "Add" {
		// MatMul → Add(bias) → Activation?
		biasName := onnxgraph.OtherBinaryOpInput(next, matmulOut)
		if biasName == "" || !m.IsConstant(biasName) {
			return nil
		}
		if len(next.Output) == 0 {
			return nil
		}
		internalNodes[next] = true
		internalOutputs[matmulOut] = true
		afterBiasOut := next.Output[0]

		// Now look for supported Activation after Add.
		actNode := onnxgraph.SoleConsumer(consumers, afterBiasOut)
		actType := denseActivationType(actNode)
		if actType == activation.TypeNone {
			return nil
		}
		if len(actNode.Output) == 0 {
			return nil
		}
		internalNodes[actNode] = true
		internalOutputs[afterBiasOut] = true

		if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName, biasName}
		return &denseActivationCandidate{
			params: &DenseActivationParams{
				XInputName:     xName,
				WeightName:     weightName,
				BiasName:       biasName,
				OutputName:     actNode.Output[0],
				ActivationType: actType,
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}
	}

	// MatMul → Activation (no bias).
	actType := denseActivationType(next)
	if actType != activation.TypeNone {
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
				XInputName:     xName,
				WeightName:     weightName,
				OutputName:     next.Output[0],
				ActivationType: actType,
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}
	}

	return nil
}

// denseActivationType returns the activation type for an ONNX node that can be fused into Dense,
// or TypeNone if the node is nil or not a recognized activation.
func denseActivationType(node *protos.NodeProto) activation.Type {
	if node == nil {
		return activation.TypeNone
	}
	switch node.OpType {
	case "Relu":
		return activation.TypeRelu
	case "Gelu":
		return activation.TypeGelu
	case "FastGelu":
		return activation.TypeGeluApprox
	case "Sigmoid":
		return activation.TypeSigmoid
	case "Tanh":
		return activation.TypeTanh
	case "Swish", "Silu":
		return activation.TypeSilu
	case "HardSwish":
		return activation.TypeHardSwish
	case "HardSigmoid":
		alpha := onnxgomlx.GetFloatAttrOr(node, "alpha", 0.2)
		beta := onnxgomlx.GetFloatAttrOr(node, "beta", 0.5)
		if math.Abs(float64(alpha)-0.2) < 1e-4 && math.Abs(float64(beta)-0.5) < 1e-4 {
			return activation.TypeHardSigmoid
		}
		return activation.TypeNone
	case "LeakyRelu":
		alpha := onnxgomlx.GetFloatAttrOr(node, "alpha", 0.01)
		if math.Abs(float64(alpha)-0.3) < 1e-4 {
			return activation.TypeLeakyRelu
		}
		return activation.TypeNone
	case "Selu":
		alpha := float64(onnxgomlx.GetFloatAttrOr(node, "alpha", float32(activation.SeluAlpha)))
		gamma := float64(onnxgomlx.GetFloatAttrOr(node, "gamma", float32(activation.SeluScale)))
		if math.Abs(alpha-activation.SeluAlpha) < 1e-4 && math.Abs(gamma-activation.SeluScale) < 1e-4 {
			return activation.TypeSelu
		}
		return activation.TypeNone
	default:
		return activation.TypeNone
	}
}
