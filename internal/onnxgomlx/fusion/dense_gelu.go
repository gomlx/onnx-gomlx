package fusion

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
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
	m               *onnxgomlx.Model
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

	actType := p.ActivationType
	if c.m != nil && c.m.ForceApproximateGeluEnabled() && actType == activation.TypeGelu {
		actType = activation.TypeGeluApprox
	}

	result := nn.Dense(x, weight, bias, compute.DenseLayoutInputOutputs, actType)
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

		// Now look for supported Activation after Add (single node or decomposed GELU).
		actNode := onnxgraph.SoleConsumer(consumers, afterBiasOut)
		actType := denseActivationType(actNode)
		var outputName string
		if actType != activation.TypeNone && len(actNode.Output) > 0 {
			internalNodes[actNode] = true
			internalOutputs[afterBiasOut] = true
			outputName = actNode.Output[0]
		} else if geluOut, geluNodes, geluOutputs, ok := tryMatchDecomposedGelu(m, consumers, afterBiasOut); ok {
			actType = activation.TypeGelu
			outputName = geluOut
			internalOutputs[afterBiasOut] = true
			for _, gn := range geluNodes {
				internalNodes[gn] = true
			}
			for _, goName := range geluOutputs {
				internalOutputs[goName] = true
			}
		} else {
			return nil
		}

		if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}

		externalInputs := []string{xName, weightName, biasName}
		return &denseActivationCandidate{
			m: m,
			params: &DenseActivationParams{
				XInputName:     xName,
				WeightName:     weightName,
				BiasName:       biasName,
				OutputName:     outputName,
				ActivationType: actType,
			},
			internalOutputs: internalOutputs,
			externalInputs:  externalInputs,
		}
	}

	// MatMul → Activation (no bias).
	actType := denseActivationType(next)
	var outputName string
	if actType != activation.TypeNone && len(next.Output) > 0 {
		internalNodes[next] = true
		internalOutputs[matmulOut] = true
		outputName = next.Output[0]
	} else if geluOut, geluNodes, geluOutputs, ok := tryMatchDecomposedGelu(m, consumers, matmulOut); ok {
		actType = activation.TypeGelu
		outputName = geluOut
		internalOutputs[matmulOut] = true
		for _, gn := range geluNodes {
			internalNodes[gn] = true
		}
		for _, goName := range geluOutputs {
			internalOutputs[goName] = true
		}
	} else {
		return nil
	}

	if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	externalInputs := []string{xName, weightName}
	return &denseActivationCandidate{
		m: m,
		params: &DenseActivationParams{
			XInputName:     xName,
			WeightName:     weightName,
			OutputName:     outputName,
			ActivationType: actType,
		},
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
	}
}

// tryMatchDecomposedGelu checks if inName feeds a decomposed GELU activation:
//
//	x -> Div(x, sqrt(2)) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5)
//
// or:
//
//	x -> [Div(x, sqrt(2)) -> Erf -> Add(1)] and [Mul(x, 0.5)] -> Mul(...)
func tryMatchDecomposedGelu(m *onnxgomlx.Model, consumers map[string][]*protos.NodeProto, inName string) (
	outputName string,
	geluNodes []*protos.NodeProto,
	geluOutputs []string,
	ok bool,
) {
	consumersOfIn := consumers[inName]
	if len(consumersOfIn) != 2 {
		return "", nil, nil, false
	}

	var divOrMulSqrt *protos.NodeProto // Div(x, sqrt(2)) or Mul(x, 1/sqrt(2))
	var mulWithX *protos.NodeProto     // Mul(x, ...)

	for _, n := range consumersOfIn {
		if (n.OpType == "Div" || n.OpType == "Mul") && divOrMulSqrt == nil {
			other := onnxgraph.OtherBinaryOpInput(n, inName)
			if other != "" && m.IsConstant(other) {
				val := m.TryGetConstantScalar(other)
				if (n.OpType == "Div" && math.Abs(val-math.Sqrt(2)) < 1e-2) ||
					(n.OpType == "Mul" && math.Abs(val-1.0/math.Sqrt(2)) < 1e-2) {
					divOrMulSqrt = n
					continue
				}
			}
		}
		if n.OpType == "Mul" {
			mulWithX = n
		}
	}

	if divOrMulSqrt == nil || mulWithX == nil {
		return "", nil, nil, false
	}

	// 1. divOrMulSqrt -> Erf
	if len(divOrMulSqrt.Output) == 0 {
		return "", nil, nil, false
	}
	erfNode := onnxgraph.SoleConsumer(consumers, divOrMulSqrt.Output[0])
	if erfNode == nil || erfNode.OpType != "Erf" || len(erfNode.Output) == 0 {
		return "", nil, nil, false
	}

	// 2. Erf -> Add(1.0)
	addOneNode := onnxgraph.SoleConsumer(consumers, erfNode.Output[0])
	if addOneNode == nil || addOneNode.OpType != "Add" || len(addOneNode.Output) == 0 {
		return "", nil, nil, false
	}
	otherAdd := onnxgraph.OtherBinaryOpInput(addOneNode, erfNode.Output[0])
	if otherAdd == "" || !m.IsConstant(otherAdd) || math.Abs(m.TryGetConstantScalar(otherAdd)-1.0) > 1e-4 {
		return "", nil, nil, false
	}

	// 3. Now check how (1 + Erf) connects to mulWithX and the 0.5 factor.
	var finalMul *protos.NodeProto

	if mulWithX.Input[0] == addOneNode.Output[0] || mulWithX.Input[1] == addOneNode.Output[0] {
		// Variant 1: mulWithX is Mul(x, 1+Erf). Next must be Mul(..., 0.5)
		if len(mulWithX.Output) == 0 {
			return "", nil, nil, false
		}
		mulHalfNode := onnxgraph.SoleConsumer(consumers, mulWithX.Output[0])
		if mulHalfNode == nil || mulHalfNode.OpType != "Mul" || len(mulHalfNode.Output) == 0 {
			return "", nil, nil, false
		}
		otherHalf := onnxgraph.OtherBinaryOpInput(mulHalfNode, mulWithX.Output[0])
		if otherHalf == "" || !m.IsConstant(otherHalf) || math.Abs(m.TryGetConstantScalar(otherHalf)-0.5) > 1e-4 {
			return "", nil, nil, false
		}
		finalMul = mulHalfNode
	} else {
		// Variant 2: mulWithX is Mul(x, 0.5).
		otherHalf := onnxgraph.OtherBinaryOpInput(mulWithX, inName)
		if otherHalf == "" || !m.IsConstant(otherHalf) || math.Abs(m.TryGetConstantScalar(otherHalf)-0.5) > 1e-4 {
			return "", nil, nil, false
		}
		// Then mulWithX and addOneNode must feed a final Mul
		if len(mulWithX.Output) == 0 {
			return "", nil, nil, false
		}
		consumerOfHalf := onnxgraph.SoleConsumer(consumers, mulWithX.Output[0])
		if consumerOfHalf == nil || consumerOfHalf.OpType != "Mul" || len(consumerOfHalf.Output) == 0 {
			return "", nil, nil, false
		}
		if consumerOfHalf.Input[0] != addOneNode.Output[0] && consumerOfHalf.Input[1] != addOneNode.Output[0] {
			return "", nil, nil, false
		}
		finalMul = consumerOfHalf
	}

	nodes := []*protos.NodeProto{divOrMulSqrt, erfNode, addOneNode, mulWithX, finalMul}
	outputs := []string{
		divOrMulSqrt.Output[0],
		erfNode.Output[0],
		addOneNode.Output[0],
		mulWithX.Output[0],
	}

	return finalMul.Output[0], nodes, outputs, true
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
		if onnxgomlx.GetStringAttrOr(node, "approximate", "none") == "tanh" {
			return activation.TypeGeluApprox
		}
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
