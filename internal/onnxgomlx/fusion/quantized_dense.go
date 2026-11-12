package fusion

import (
	"math"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
)

// prepareActivationQuantParams builds the Scale and ZeroPoint nodes for XQuantization in
// nn.QuantizedDense from ONNX MatMulInteger / DynamicQuantizeLinear inputs.
// When aScaleNode is nil (DequantizeLinear variant where the combined scale is folded into bScale),
// a unit float32 scalar scale (1.0) is used.
func prepareActivationQuantParams(g *Graph, a, aScaleNode, aZPNode *Node) (xScale, xZP *Node) {
	if aScaleNode != nil {
		xScale = ConvertDType(aScaleNode, dtypes.Float32)
	} else {
		xScale = ScalarOne(g, dtypes.Float32)
	}
	if aZPNode == nil {
		return xScale, nil
	}
	xZP = aZPNode
	// ONNX MatMulInteger allows a 1D per-row a_zero_point of shape [M] aligning with axis a.Rank()-2.
	// FusedQuantizedDense represents per-row activation parameters as [batch..., 1].
	if !xZP.IsScalar() && xZP.Shape().Size() != 1 && xZP.Rank() == 1 && a.Rank() >= 2 {
		ones := slices.Repeat([]int{1}, a.Rank())
		ones[a.Rank()-2] = xZP.Shape().Dimensions[0]
		zpReshaped := Reshape(xZP, ones...)

		targetSpecs := append(slices.Clone(DimensionSpecsFor(a)[:a.Rank()-1]), StaticDim(1))
		xZP = DynamicBroadcastToDims(zpReshaped, targetSpecs...)
		xScale = DynamicBroadcastToDims(ExpandLeftToRank(xScale, a.Rank()), targetSpecs...)
	}
	return xScale, xZP
}

// QuantizedDenseParams holds parameters for fused DynamicQuantizeLinear + MatMulInteger
// (or MatMulInteger + DequantizeLinear) chains, emitted as nn.QuantizedDense with
// pre-quantized XQuantization and WeightQuantization so DynamicQuantizeLinear stays in the
// graph (computed once and shared across consumers) while integer matmul + scale + bias +
// optional GELU execute in a single fused kernel.
type QuantizedDenseParams struct {
	// AInputName is the quantized uint8/int8 input A to MatMulInteger
	// (e.g. output[0] of DynamicQuantizeLinear).
	AInputName string

	// AScaleName is the activation scale (e.g. output[1] of DynamicQuantizeLinear).
	// Empty in the DequantizeLinear variant where the combined scale is in BScaleName.
	AScaleName string

	// AZeroPointName is the optional activation zero point from MatMulInteger (input[2]).
	AZeroPointName string

	// BWeightName is the constant int8/uint8 weight [K, N].
	BWeightName string

	// BDType is the ONNX TensorProto DataType of BWeightName (INT8 or UINT8).
	BDType int32

	// BScaleName is the constant weight scale: scalar ([] / [1]) or 1D [N] (per-channel).
	BScaleName string

	// BZeroPointName is the optional constant weight zero point from MatMulInteger (input[3]),
	// omitted when absent or all zeros.
	BZeroPointName string

	// BiasName is the optional bias constant, "" if absent.
	BiasName string

	// Weight dimensions extracted at parse time.
	K, N int

	// HasGelu is true when the decomposed GELU pattern follows the output.
	HasGelu bool
}

// quantizedDenseCandidate implements onnxgomlx.FusionCandidate for quantized dense.
type quantizedDenseCandidate struct {
	m               *onnxgomlx.Model
	params          *QuantizedDenseParams
	outputName      string
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *quantizedDenseCandidate) Name() string                     { return "QuantizedDense" }
func (c *quantizedDenseCandidate) Score() float32                   { return 40.0 }
func (c *quantizedDenseCandidate) OutputNames() []string            { return []string{c.outputName} }
func (c *quantizedDenseCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *quantizedDenseCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *quantizedDenseCandidate) Emit(_ *model.Scope, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	a := convertedOutputs[p.AInputName]
	var aScaleNode, aZPNode *Node
	if p.AScaleName != "" {
		aScaleNode = convertedOutputs[p.AScaleName]
	}
	if p.AZeroPointName != "" {
		aZPNode = convertedOutputs[p.AZeroPointName]
	}
	xScale, xZP := prepareActivationQuantParams(g, a, aScaleNode, aZPNode)

	b := convertedOutputs[p.BWeightName]
	bScale := ConvertDType(convertedOutputs[p.BScaleName], dtypes.Float32)

	var bZP *Node
	if p.BZeroPointName != "" {
		bZP = convertedOutputs[p.BZeroPointName]
		// Ensure per-channel [N] bZP is paired with a [N] bScale when bScale was scalar.
		if bZP.Shape().Size() > 1 && bScale.Shape().Size() == 1 {
			bScale = BroadcastToDims(Reshape(bScale, 1), p.N)
		}
	}

	var bias *Node
	if p.BiasName != "" {
		bias = ConvertDType(convertedOutputs[p.BiasName], dtypes.Float32)
	}

	cfg := QuantizedDenseConfig{
		WeightQuantization: &Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     bScale,
			ZeroPoint: bZP,
		},
		XQuantization: &Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     xScale,
			ZeroPoint: xZP,
		},
		WeightLayout: compute.DenseLayoutInputOutputs,
	}

	var result *Node
	if p.HasGelu {
		actType := activation.TypeGelu
		if c.m != nil && c.m.ForceApproximateGeluEnabled() {
			actType = activation.TypeGeluApprox
		}
		result = nn.QuantizedDense(a, b, bias, cfg, actType)
	} else {
		result = nn.QuantizedDense(a, b, bias, cfg)
	}

	convertedOutputs[c.outputName] = result
}

func init() {
	onnxgomlx.RegisterFusionDetector(detectQuantizedDenseCandidates)
}

// detectQuantizedDenseCandidates scans the ONNX graph for:
//
//	DynamicQuantizeLinear(float_x) → (uint8_x, a_scale, a_zp)
//	MatMulInteger(uint8_x, int8_B, a_zp, b_zp) → int32
//	Cast(int32→float32)
//	Mul(a_scale, B_scale) → combined_scale         [B_scale is constant]
//	Mul(float_result, combined_scale) → output
//	  optionally followed by Add(bias) and/or decomposed GELU
//
// (as well as the MatMulInteger → DequantizeLinear variant) and returns FusionCandidates for each match.
func detectQuantizedDenseCandidates(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	var candidates []onnxgomlx.FusionCandidate
	for _, node := range m.Proto.Graph.Node {
		if node.OpType != "MatMulInteger" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		if cand := tryMatchQuantizedDense(m, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// validateWeightZeroPoint checks matMulInputs[3] (b_zero_point):
//   - If absent or a zero initializer, returns ("", true) so zero-point subtraction is skipped.
//   - If a constant scalar ([] or [1]) or 1D [N] int8/uint8 tensor, returns (name, true).
//   - Otherwise returns ("", false).
func validateWeightZeroPoint(m *onnxgomlx.Model, matMulInputs []string, N int) (string, bool) {
	if len(matMulInputs) <= 3 || matMulInputs[3] == "" {
		return "", true
	}
	bZPName := matMulInputs[3]
	if m.IsZeroInitializer(bZPName) {
		return "", true
	}
	bZPTP, ok := m.VariableNameToValue[bZPName]
	if !ok || bZPTP == nil {
		return "", false
	}
	if bZPTP.DataType != int32(protos.TensorProto_INT8) && bZPTP.DataType != int32(protos.TensorProto_UINT8) {
		return "", false
	}
	switch {
	case len(bZPTP.Dims) == 0:
		return bZPName, true
	case len(bZPTP.Dims) == 1 && (bZPTP.Dims[0] == 1 || bZPTP.Dims[0] == int64(N)):
		return bZPName, true
	default:
		return "", false
	}
}

// tryMatchQuantizedDense attempts to match the DynamicQuantizeLinear → MatMulInteger →
// Cast → Mul(combined_scale) chain while keeping DynamicQuantizeLinear in the graph so its
// quantized uint8 activations, scale, and zero point are preserved and shared across consumers.
func tryMatchQuantizedDense(m *onnxgomlx.Model, matMulNode *protos.NodeProto) *quantizedDenseCandidate {
	consumers := m.Consumers
	matMulInputs := matMulNode.Input
	if len(matMulInputs) < 2 {
		return nil
	}

	aName := matMulInputs[0] // quantized output of DynamicQuantizeLinear
	bName := matMulInputs[1] // constant int8/uint8 weight

	// B (weight) must be a constant 2D int8 or uint8 tensor.
	if !m.IsConstant(bName) {
		return nil
	}
	bTP, bFound := m.VariableNameToValue[bName]
	if !bFound || bTP == nil || len(bTP.Dims) != 2 {
		return nil
	}
	if bTP.DataType != int32(protos.TensorProto_INT8) && bTP.DataType != int32(protos.TensorProto_UINT8) {
		return nil
	}
	K := int(bTP.Dims[0])
	N := int(bTP.Dims[1])

	bZeroPointName, bZPOK := validateWeightZeroPoint(m, matMulInputs, N)
	if !bZPOK {
		return nil
	}

	// Check if A comes from DynamicQuantizeLinear.
	dynamicQuantizeNode, ok := m.NodeOutputToNode[aName]
	if !ok || dynamicQuantizeNode.OpType != "DynamicQuantizeLinear" || len(dynamicQuantizeNode.Input) == 0 || len(dynamicQuantizeNode.Output) < 2 {
		// DynamicQuantizeLinear not found — try DequantizeLinear variant instead.
		return tryMatchQuantizedDenseDequantLinear(m, matMulNode, bName, bTP.DataType, bZeroPointName, K, N)
	}
	aScaleName := dynamicQuantizeNode.Output[1] // dynamic per-tensor scale

	var aZeroPointName string
	if len(matMulInputs) > 2 && matMulInputs[2] != "" {
		aZeroPointName = matMulInputs[2]
	}

	// Follow forward: MatMulInteger → sole consumer Cast → sole consumer Mul(combined_scale).
	matMulOut := matMulNode.Output[0]
	castNode := onnxgraph.SoleConsumer(consumers, matMulOut)
	if castNode == nil || castNode.OpType != "Cast" || len(castNode.Output) == 0 {
		return nil
	}
	castOut := castNode.Output[0]

	resultMulNode := onnxgraph.SoleConsumer(consumers, castOut)
	if resultMulNode == nil || resultMulNode.OpType != "Mul" || len(resultMulNode.Input) < 2 || len(resultMulNode.Output) == 0 {
		return nil
	}

	// Identify the combined scale input to the result Mul.
	combinedScaleName := onnxgraph.OtherBinaryOpInput(resultMulNode, castOut)
	if combinedScaleName == "" {
		return nil
	}

	// The combined scale must be produced by Mul(a_scale, B_scale) where B_scale is constant.
	scaleMulNode, ok := m.NodeOutputToNode[combinedScaleName]
	if !ok || scaleMulNode.OpType != "Mul" || len(scaleMulNode.Input) < 2 {
		return nil
	}
	bScaleName := identifyConstantPeer(m, scaleMulNode, aScaleName)
	if bScaleName == "" {
		return nil
	}
	// B scale must be scalar (per-tensor) or 1D [N] (per-channel).
	bScaleTP, bScaleFound := m.VariableNameToValue[bScaleName]
	if !bScaleFound || bScaleTP == nil {
		return nil
	}
	switch {
	case len(bScaleTP.Dims) == 0:
		// scalar (no dimensions)
	case len(bScaleTP.Dims) == 1 && bScaleTP.Dims[0] == 1:
		// scalar (single-element 1-D)
	case len(bScaleTP.Dims) == 1 && bScaleTP.Dims[0] == int64(N):
		// per-channel [N]
	default:
		return nil // unsupported scale shape
	}

	resultMulOut := resultMulNode.Output[0]

	// Collect internal nodes/outputs for the base pattern.
	// DynamicQuantizeLinear is NOT internal: it stays in the graph so quantized x, x_scale, and x_zp
	// are computed once and can be shared across multiple projections (e.g. Q, K, V).
	internalNodes := map[*protos.NodeProto]bool{
		matMulNode:    true,
		castNode:      true,
		scaleMulNode:  true,
		resultMulNode: true,
	}
	internalOutputs := map[string]bool{
		matMulOut:         true,
		castOut:           true,
		combinedScaleName: true,
	}

	// Check for optional bias Add after the result Mul.
	var biasName string
	currentOut := resultMulOut
	biasConsumer := onnxgraph.SoleConsumer(consumers, currentOut)
	if biasConsumer != nil && biasConsumer.OpType == "Add" && len(biasConsumer.Output) > 0 {
		otherInput := onnxgraph.OtherBinaryOpInput(biasConsumer, currentOut)
		if otherInput != "" && m.IsConstant(otherInput) {
			biasName = otherInput
			internalOutputs[currentOut] = true
			internalNodes[biasConsumer] = true
			currentOut = biasConsumer.Output[0]
		}
	}

	// Try to detect decomposed GELU following the current output.
	hasGelu, geluFinalOut, geluInternalNodes, geluInternalOutputs := tryMatchDecomposedGELU(m, consumers, currentOut)

	if hasGelu {
		for n := range geluInternalNodes {
			internalNodes[n] = true
		}
		for o := range geluInternalOutputs {
			internalOutputs[o] = true
		}
		internalOutputs[currentOut] = true
		currentOut = geluFinalOut
	}

	// Check no internal outputs leak to external consumers.
	if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	// Put aName (DynamicQuantizeLinear output[0]) first so converting it populates aScaleName and aZeroPointName too.
	externalInputs := []string{aName, aScaleName, bName, bScaleName}
	if aZeroPointName != "" {
		externalInputs = append(externalInputs, aZeroPointName)
	}
	if bZeroPointName != "" {
		externalInputs = append(externalInputs, bZeroPointName)
	}
	if biasName != "" {
		externalInputs = append(externalInputs, biasName)
	}

	params := &QuantizedDenseParams{
		AInputName:     aName,
		AScaleName:     aScaleName,
		AZeroPointName: aZeroPointName,
		BWeightName:    bName,
		BDType:         bTP.DataType,
		BScaleName:     bScaleName,
		BZeroPointName: bZeroPointName,
		BiasName:       biasName,
		K:              K,
		N:              N,
		HasGelu:        hasGelu,
	}

	return &quantizedDenseCandidate{
		m:               m,
		params:          params,
		outputName:      currentOut,
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
	}
}

// tryMatchQuantizedDenseDequantLinear attempts to match the alternative pattern:
//
//	MatMulInteger(A, int8_B, a_zp?, b_zp?) → int32
//	  sole consumer → DequantizeLinear(int32_out, scalar_scale, no_zp) → float output
//
// This variant is emitted by some ONNX exporters instead of the DynamicQuantizeLinear-based
// Cast+Mul(a_scale*B_scale) chain.
func tryMatchQuantizedDenseDequantLinear(
	m *onnxgomlx.Model,
	matMulNode *protos.NodeProto,
	bName string,
	bDType int32,
	bZeroPointName string,
	K, N int,
) *quantizedDenseCandidate {
	consumers := m.Consumers
	matMulOut := matMulNode.Output[0]
	matMulInputs := matMulNode.Input

	// MatMulInteger output must have a sole consumer that is DequantizeLinear.
	dequantNode := onnxgraph.SoleConsumer(consumers, matMulOut)
	if dequantNode == nil || dequantNode.OpType != "DequantizeLinear" || len(dequantNode.Input) < 2 || len(dequantNode.Output) == 0 {
		return nil
	}

	// DequantizeLinear must have no zero point.
	if len(dequantNode.Input) > 2 && dequantNode.Input[2] != "" {
		return nil
	}

	// DequantizeLinear scale must be a constant scalar.
	scaleName := dequantNode.Input[1]
	if !m.IsConstant(scaleName) {
		return nil
	}
	scaleTP, scaleFound := m.VariableNameToValue[scaleName]
	if !scaleFound || scaleTP == nil {
		return nil
	}
	isScalar := true
	for _, d := range scaleTP.Dims {
		if d != 1 {
			isScalar = false
			break
		}
	}
	if !isScalar {
		return nil
	}

	dequantOut := dequantNode.Output[0]

	// Collect A input name and optional A zero point.
	aInputName := matMulInputs[0]
	var aZeroPointName string
	if len(matMulInputs) > 2 && matMulInputs[2] != "" {
		aZeroPointName = matMulInputs[2]
	}

	internalNodes := map[*protos.NodeProto]bool{
		matMulNode:  true,
		dequantNode: true,
	}
	internalOutputs := map[string]bool{
		matMulOut: true,
	}

	// Check for optional bias Add after DequantizeLinear.
	var biasName string
	currentOut := dequantOut
	biasConsumer := onnxgraph.SoleConsumer(consumers, currentOut)
	if biasConsumer != nil && biasConsumer.OpType == "Add" && len(biasConsumer.Output) > 0 {
		otherInput := onnxgraph.OtherBinaryOpInput(biasConsumer, currentOut)
		if otherInput != "" && m.IsConstant(otherInput) {
			biasName = otherInput
			internalOutputs[currentOut] = true
			internalNodes[biasConsumer] = true
			currentOut = biasConsumer.Output[0]
		}
	}

	// Try to detect decomposed GELU following the current output.
	hasGelu, geluFinalOut, geluInternalNodes, geluInternalOutputs := tryMatchDecomposedGELU(m, consumers, currentOut)

	if hasGelu {
		for n := range geluInternalNodes {
			internalNodes[n] = true
		}
		for o := range geluInternalOutputs {
			internalOutputs[o] = true
		}
		internalOutputs[currentOut] = true
		currentOut = geluFinalOut
	}

	if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	externalInputs := []string{aInputName, bName, scaleName}
	if aZeroPointName != "" {
		externalInputs = append(externalInputs, aZeroPointName)
	}
	if bZeroPointName != "" {
		externalInputs = append(externalInputs, bZeroPointName)
	}
	if biasName != "" {
		externalInputs = append(externalInputs, biasName)
	}

	params := &QuantizedDenseParams{
		AInputName:     aInputName,
		AZeroPointName: aZeroPointName,
		BWeightName:    bName,
		BDType:         bDType,
		BScaleName:     scaleName,
		BZeroPointName: bZeroPointName,
		BiasName:       biasName,
		K:              K,
		N:              N,
		HasGelu:        hasGelu,
	}

	return &quantizedDenseCandidate{
		m:               m,
		params:          params,
		outputName:      currentOut,
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
	}
}

// identifyConstantPeer checks a Mul node with two inputs: one must be knownDynamic
// (runtime-computed, e.g. a_scale from DynamicQuantizeLinear), the other must be a constant. Returns
// the constant input name, or "" if the pattern doesn't match.
func identifyConstantPeer(m *onnxgomlx.Model, mulNode *protos.NodeProto, knownDynamic string) string {
	if len(mulNode.Input) < 2 {
		return ""
	}
	in0, in1 := mulNode.Input[0], mulNode.Input[1]
	if in0 == knownDynamic && m.IsConstant(in1) {
		return in1
	}
	if in1 == knownDynamic && m.IsConstant(in0) {
		return in0
	}
	return ""
}

// tryMatchDecomposedGELU detects the decomposed GELU pattern starting from xOutputName:
//
//	x → Div(x, √2) → Erf → Add(·, 1) → Mul(x, ·) → Mul(·, 0.5)
//
// where x is consumed by both the Div and the second Mul (the skip connection).
//
// Returns hasGelu, finalOutputName, internalNodes, internalOutputs.
func tryMatchDecomposedGELU(m *onnxgomlx.Model, consumers map[string][]*protos.NodeProto, xOutputName string) (
	hasGelu bool, finalOutputName string, internalNodes map[*protos.NodeProto]bool, internalOutputs map[string]bool,
) {
	xConsumers := consumers[xOutputName]
	if len(xConsumers) != 2 {
		return
	}

	// Identify the Div and the skip-connection Mul among the two consumers.
	var divNode, skipMulNode *protos.NodeProto
	for _, c := range xConsumers {
		switch c.OpType {
		case "Div":
			if divNode == nil {
				divNode = c
			}
		case "Mul":
			if skipMulNode == nil {
				skipMulNode = c
			}
		}
	}
	if divNode == nil || skipMulNode == nil {
		return
	}

	// Div must divide by √2 (≈1.4142).
	if !isDivBySqrt2(m, divNode, xOutputName) {
		return
	}
	if len(divNode.Output) == 0 {
		return
	}
	divOut := divNode.Output[0]

	// Div output → sole consumer must be Erf.
	erfNode := onnxgraph.SoleConsumer(consumers, divOut)
	if erfNode == nil || erfNode.OpType != "Erf" || len(erfNode.Output) == 0 {
		return
	}

	erfOut := erfNode.Output[0]

	// Erf output → sole consumer must be Add(erfOut, 1.0).
	addNode := onnxgraph.SoleConsumer(consumers, erfOut)
	if addNode == nil || addNode.OpType != "Add" || len(addNode.Output) == 0 {
		return
	}
	addOther := onnxgraph.OtherBinaryOpInput(addNode, erfOut)
	if addOther == "" {
		return
	}
	addConstVal := m.TryGetConstantScalar(addOther)
	if math.Abs(addConstVal-1.0) > 1e-3 {
		return
	}
	addOut := addNode.Output[0]

	// Add output → must be consumed by the skipMulNode: Mul(xOutputName, addOut).
	if !nodeHasInputs(skipMulNode, xOutputName, addOut) {
		return
	}
	if len(skipMulNode.Output) == 0 {
		return
	}
	skipMulOut := skipMulNode.Output[0]

	// skipMul output → sole consumer must be Mul(., 0.5).
	halfMulNode := onnxgraph.SoleConsumer(consumers, skipMulOut)
	if halfMulNode == nil || halfMulNode.OpType != "Mul" || len(halfMulNode.Input) < 2 || len(halfMulNode.Output) == 0 {
		return
	}
	halfVal := getOtherMulConstant(m, halfMulNode, skipMulOut)
	if math.Abs(halfVal-0.5) > 1e-3 {
		return
	}

	hasGelu = true
	finalOutputName = halfMulNode.Output[0]
	internalNodes = map[*protos.NodeProto]bool{
		divNode:     true,
		erfNode:     true,
		addNode:     true,
		skipMulNode: true,
		halfMulNode: true,
	}
	internalOutputs = map[string]bool{
		divOut:     true,
		erfOut:     true,
		addOut:     true,
		skipMulOut: true,
	}
	return
}

// isDivBySqrt2 checks if a Div node divides xOutputName by approximately √2.
func isDivBySqrt2(m *onnxgomlx.Model, divNode *protos.NodeProto, xOutputName string) bool {
	if len(divNode.Input) < 2 {
		return false
	}
	// x must be the numerator.
	if divNode.Input[0] != xOutputName {
		return false
	}
	divisor := m.TryGetConstantScalar(divNode.Input[1])
	return math.Abs(divisor-math.Sqrt2) < 1e-3
}

// nodeHasInputs checks if a node has exactly the two given inputs (in either order).
func nodeHasInputs(node *protos.NodeProto, a, b string) bool {
	if len(node.Input) < 2 {
		return false
	}
	return (node.Input[0] == a && node.Input[1] == b) ||
		(node.Input[0] == b && node.Input[1] == a)
}

// getOtherMulConstant returns the scalar constant value of the Mul input that is NOT knownInput.
// Returns 0 if the other input is not a scalar constant.
func getOtherMulConstant(m *onnxgomlx.Model, mulNode *protos.NodeProto, knownInput string) float64 {
	if len(mulNode.Input) < 2 {
		return 0
	}
	var otherName string
	if mulNode.Input[0] == knownInput {
		otherName = mulNode.Input[1]
	} else if mulNode.Input[1] == knownInput {
		otherName = mulNode.Input[0]
	} else {
		return 0
	}
	return m.TryGetConstantScalar(otherName)
}
