package onnx

import (
	"math"

	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// SDPAParams holds parameters for fused scaled dot-product attention.
type SDPAParams struct {
	QInputName, KInputName, VInputName string
	MaskInputName                      string  // empty if no mask
	Scale                              float64 // 1/sqrt(headDim)
	NumHeads                           int
	NumKVHeads                         int
	// KNeedsHeadsFirst is true when K is in [batch, kvLen, numKVHeads, headDim] layout
	// and needs a [0,2,1,3] transpose to become [batch, numKVHeads, kvLen, headDim].
	KNeedsHeadsFirst bool
}

// sdpaCandidate implements FusionCandidate for scaled dot-product attention.
type sdpaCandidate struct {
	params          *SDPAParams
	outputName      string
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *sdpaCandidate) Name() string                    { return "SDPA" }
func (c *sdpaCandidate) Score() float32                   { return 100.0 }
func (c *sdpaCandidate) OutputNames() []string            { return []string{c.outputName} }
func (c *sdpaCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *sdpaCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *sdpaCandidate) Emit(ctx *context.Context, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params

	q := convertedOutputs[p.QInputName]
	k := convertedOutputs[p.KInputName]
	v := convertedOutputs[p.VInputName]

	// When K is in [batch, kvLen, numKVHeads, headDim] (e.g. from Snowflake-style models),
	// transpose to [batch, numKVHeads, kvLen, headDim] as expected by attention.Core.
	if p.KNeedsHeadsFirst {
		k = TransposeAllDims(k, 0, 2, 1, 3)
	}

	var mask *Node
	if p.MaskInputName != "" {
		mask = convertedOutputs[p.MaskInputName]
	}

	output, _ := attention.Core(ctx, q, k, v, p.Scale, mask, 0, attention.LayoutBHSD, false, false)
	convertedOutputs[c.outputName] = output
}

func init() {
	RegisterFusionDetector(detectSDPACandidates)
}

// detectSDPACandidates scans the ONNX graph for decomposed scaled dot-product attention:
//
//	MatMul(Q, K^T) → Div/Mul(·, scale) → [Add(·, mask)] → Softmax(·, axis=-1) → MatMul(·, V)
//
// and returns FusionCandidates for each match.
func detectSDPACandidates(m *Model, graph *protos.GraphProto, consumers map[string][]*protos.NodeProto) []FusionCandidate {
	var candidates []FusionCandidate
	for _, node := range graph.Node {
		if node.OpType != "MatMul" {
			continue
		}
		if cand := m.tryMatchSDPA(graph, consumers, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMatchSDPA attempts to match an SDPA chain starting from matmul1 (the Q@K^T multiplication).
// It supports two patterns:
//
// Post-scaled (standard): MatMul(Q, K^T) → Div/Mul(·, scale) → [Add(mask)] → Softmax(-1) → MatMul(·, V)
// Pre-scaled:             MatMul(Mul(Q, s), Mul(K^T, s)) → [Add(mask)] → Softmax(-1) → MatMul(·, V)
//
// In the pre-scaled pattern, both Q and K inputs to MatMul1 come from Mul(·, scalar)
// with the same scalar constant, and the effective scale is scalar².
func (m *Model) tryMatchSDPA(graph *protos.GraphProto, consumers map[string][]*protos.NodeProto, matmul1 *protos.NodeProto) *sdpaCandidate {
	if len(matmul1.Output) == 0 {
		return nil
	}
	m1Out := matmul1.Output[0]

	if len(matmul1.Input) < 2 {
		return nil
	}

	// Try to match K^T: either direct Transpose, or Mul(Transpose(...), scalar).
	kTransposeNode, kInputName := m.matchKTranspose(matmul1.Input[1])
	var kPreScaleMulNode *protos.NodeProto
	var kNeedsHeadsFirst bool
	if kTransposeNode == nil {
		// Try pre-scaled K: Mul(Transpose(...), scalar)
		kPreScaleMulNode, kTransposeNode, kInputName, kNeedsHeadsFirst = m.matchPreScaledKTranspose(matmul1.Input[1])
		if kTransposeNode == nil {
			return nil
		}
	}

	// Follow chain from MatMul1 output.
	scaleConsumer := soleConsumer(consumers, m1Out)
	if scaleConsumer == nil {
		return nil
	}

	var scale float64
	var afterScaleOut string
	var scaleNode *protos.NodeProto // non-nil only for post-scale pattern

	switch scaleConsumer.OpType {
	case "Div":
		// Post-scaled: MatMul → Div
		scale = m.extractScaleFromDiv(scaleConsumer)
		scaleNode = scaleConsumer
	case "Mul":
		// Could be post-scaled: MatMul → Mul(·, scalar)
		// Check if this Mul has a constant scalar input (post-scale).
		postScale := m.extractScaleFromMul(scaleConsumer)
		if postScale != 0 {
			scale = postScale
			scaleNode = scaleConsumer
		} else {
			return nil
		}
	case "Add", "Softmax":
		// No post-scale node. Check for pre-scaled Q/K pattern.
		if kPreScaleMulNode == nil {
			return nil // K wasn't pre-scaled, and there's no post-scale → not SDPA
		}
		scale = m.extractPreScale(matmul1.Input[0], kPreScaleMulNode)
		if scale == 0 {
			return nil
		}
		// afterScaleOut is the MatMul output itself (no separate scale node).
		afterScaleOut = m1Out
	default:
		return nil
	}
	if scale == 0 {
		return nil
	}

	// For post-scale pattern, advance past the scale node.
	if scaleNode != nil {
		if len(scaleNode.Output) == 0 {
			return nil
		}
		afterScaleOut = scaleNode.Output[0]
	}

	// Next consumer: either Add (mask) then Softmax, or directly Softmax.
	var nextNode *protos.NodeProto
	if scaleNode != nil {
		nextNode = soleConsumer(consumers, afterScaleOut)
	} else {
		// Pre-scale pattern: scaleConsumer IS the next node (Add or Softmax).
		nextNode = scaleConsumer
	}
	if nextNode == nil {
		return nil
	}

	var maskInputName string
	var softmaxNode *protos.NodeProto
	var addNode *protos.NodeProto

	switch nextNode.OpType {
	case "Add":
		addNode = nextNode
		// Mask add: one of the Add inputs is afterScaleOut, the other is the mask.
		maskInputName = otherAddInput(addNode, afterScaleOut)
		if maskInputName == "" {
			return nil
		}
		if !m.isMaskRankAcceptable(graph, maskInputName) {
			return nil
		}
		if len(addNode.Output) == 0 {
			return nil
		}
		softmaxNode = soleConsumer(consumers, addNode.Output[0])
		if softmaxNode == nil || softmaxNode.OpType != "Softmax" {
			return nil
		}
	case "Softmax":
		softmaxNode = nextNode
	default:
		return nil
	}

	// Verify softmax axis is -1 (last axis).
	softmaxAxis := getIntAttrOr(softmaxNode, "axis", -1)
	if softmaxAxis != -1 {
		return nil
	}

	if len(softmaxNode.Output) == 0 {
		return nil
	}
	softmaxOut := softmaxNode.Output[0]

	// Final MatMul: Softmax output @ V
	matmul2 := soleConsumer(consumers, softmaxOut)
	if matmul2 == nil || matmul2.OpType != "MatMul" {
		return nil
	}
	if len(matmul2.Input) < 2 || len(matmul2.Output) == 0 {
		return nil
	}
	if matmul2.Input[0] != softmaxOut {
		return nil
	}
	vInputName := matmul2.Input[1]
	rootOutput := matmul2.Output[0]

	// Collect all internal nodes and their outputs.
	internalNodes := map[*protos.NodeProto]bool{
		matmul1:     true,
		softmaxNode: true,
		matmul2:     true,
	}
	internalOutputs := map[string]bool{
		m1Out:      true,
		softmaxOut: true,
	}
	if scaleNode != nil {
		internalNodes[scaleNode] = true
		internalOutputs[afterScaleOut] = true
	}
	if addNode != nil {
		internalNodes[addNode] = true
		for _, out := range addNode.Output {
			internalOutputs[out] = true
		}
	}

	// Verify no internal output is consumed outside the group.
	if hasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	// Extract numHeads from Q and K shapes.
	// For the pre-scale pattern, the MatMul inputs are Mul outputs — look through to the
	// Transpose input for shape info.
	qShapeName := matmul1.Input[0]
	kShapeName := kInputName
	if kPreScaleMulNode != nil {
		// Pre-scaled: look through Mul → Transpose for shape.
		qShapeName = m.lookThroughMulForShapeName(matmul1.Input[0])
		// kInputName is already the pre-transpose K input from matchPreScaledKTranspose.
	}
	var numHeads, numKVHeads int
	if kNeedsHeadsFirst {
		// K is in [batch, kvLen, numKVHeads, headDim] — heads are at axis 2.
		numHeads = m.extractDimFromShape(graph, qShapeName, 1)
		numKVHeads = m.extractDimFromShape(graph, kShapeName, 2)
		if numHeads <= 0 {
			numHeads = 1
		}
		if numKVHeads <= 0 {
			numKVHeads = numHeads
		}
	} else {
		numHeads, numKVHeads = m.extractHeadCounts(graph, qShapeName, kShapeName)
	}

	// Build external inputs list.
	// For the pre-scale pattern, use pre-Mul inputs (the Mul nodes carry the scale
	// which is already captured in the scale parameter).
	qInputName := matmul1.Input[0]
	if kPreScaleMulNode != nil {
		// Q input is the non-scalar input to Q's Mul.
		qInputName = m.lookThroughMulForShapeName(matmul1.Input[0])
	}
	externalInputs := []string{qInputName, kInputName, vInputName}
	if maskInputName != "" {
		externalInputs = append(externalInputs, maskInputName)
	}

	return &sdpaCandidate{
		outputName:      rootOutput,
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
		params: &SDPAParams{
			QInputName:       qInputName,
			KInputName:       kInputName,
			VInputName:       vInputName,
			MaskInputName:    maskInputName,
			Scale:            scale,
			NumHeads:         numHeads,
			NumKVHeads:       numKVHeads,
			KNeedsHeadsFirst: kNeedsHeadsFirst,
		},
	}
}

// matchKTranspose checks if inputName comes from a Transpose node that swaps the last two axes.
// Returns the Transpose node and the original (pre-transpose) input name, or nil if not matched.
func (m *Model) matchKTranspose(inputName string) (*protos.NodeProto, string) {
	node, ok := m.nodeOutputToNode[inputName]
	if !ok || node.OpType != "Transpose" {
		return nil, ""
	}
	if len(node.Input) == 0 {
		return nil, ""
	}

	perm := getIntsAttrOr(node, "perm", nil)
	if perm == nil {
		// Default transpose reverses all axes. For rank ≥ 2, this swaps last two.
		// We accept this as K^T.
		return node, node.Input[0]
	}

	// Check that perm swaps the last two axes and leaves others unchanged.
	rank := len(perm)
	if rank < 2 {
		return nil, ""
	}
	for i := 0; i < rank-2; i++ {
		if perm[i] != i {
			return nil, ""
		}
	}
	if perm[rank-2] != rank-1 || perm[rank-1] != rank-2 {
		return nil, ""
	}

	return node, node.Input[0]
}

// matchPreScaledKTranspose checks if inputName comes from Mul(Transpose(...), scalar)
// where the Transpose produces K^T (headDim and kvLen in the last two positions).
// Returns the Mul node, Transpose node, and the pre-transpose input name; or nil if not matched.
// kNeedsHeadsFirst is true when K_raw needs a [0,2,1,3] transpose to get to [batch, heads, kvLen, headDim].
func (m *Model) matchPreScaledKTranspose(inputName string) (mulNode, transposeNode *protos.NodeProto, preTransposeInput string, kNeedsHeadsFirst bool) {
	node, ok := m.nodeOutputToNode[inputName]
	if !ok || node.OpType != "Mul" {
		return nil, nil, "", false
	}
	if len(node.Input) < 2 {
		return nil, nil, "", false
	}

	// One input should be a Transpose, the other a scalar constant.
	for _, transposeIdx := range []int{0, 1} {
		scalarIdx := 1 - transposeIdx
		scalar := m.tryGetConstantScalar(node.Input[scalarIdx])
		if scalar == 0 {
			continue
		}

		// First try: standard K^T (last two axes swapped, e.g. [0,1,3,2]).
		tNode, preInput := m.matchKTranspose(node.Input[transposeIdx])
		if tNode != nil {
			return node, tNode, preInput, false
		}

		// Second try: combined heads-first + K^T (e.g. [0,2,3,1] on [batch, seqLen, heads, headDim]).
		// This rearranges to [batch, heads, headDim, seqLen] in one step.
		tNode, preInput, ok := m.matchCombinedKTranspose(node.Input[transposeIdx])
		if ok {
			return node, tNode, preInput, true
		}
	}
	return nil, nil, "", false
}

// matchCombinedKTranspose matches Transpose with perm [0,2,3,1] which combines
// heads-first reordering and K^T in one operation.
// Input: [batch, kvLen, numKVHeads, headDim] → Output: [batch, numKVHeads, headDim, kvLen]
// Returns the Transpose node and pre-transpose input name, or (nil, "", false).
func (m *Model) matchCombinedKTranspose(inputName string) (*protos.NodeProto, string, bool) {
	node, ok := m.nodeOutputToNode[inputName]
	if !ok || node.OpType != "Transpose" {
		return nil, "", false
	}
	if len(node.Input) == 0 {
		return nil, "", false
	}

	perm := getIntsAttrOr(node, "perm", nil)
	if perm == nil || len(perm) != 4 {
		return nil, "", false
	}

	// Accept [0, 2, 3, 1]: batch stays, middle two axes move left, last axis wraps to position 1.
	if perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1 {
		return node, node.Input[0], true
	}

	return nil, "", false
}

// extractPreScale extracts the effective scale when both Q and K inputs to MatMul
// come from Mul(·, scalar) with the same constant scalar. Returns scalar² or 0 if not matched.
func (m *Model) extractPreScale(qMulOutputName string, kMulNode *protos.NodeProto) float64 {
	// Q input should also come from a Mul(·, scalar).
	qNode, ok := m.nodeOutputToNode[qMulOutputName]
	if !ok || qNode.OpType != "Mul" {
		return 0
	}
	if len(qNode.Input) < 2 {
		return 0
	}

	// Extract scalar from Q's Mul.
	qScalar := m.tryGetConstantScalar(qNode.Input[1])
	if qScalar == 0 {
		// Try the other input.
		qScalar = m.tryGetConstantScalar(qNode.Input[0])
	}
	if qScalar == 0 {
		return 0
	}

	// Extract scalar from K's Mul.
	if len(kMulNode.Input) < 2 {
		return 0
	}
	kScalar := m.tryGetConstantScalar(kMulNode.Input[1])
	if kScalar == 0 {
		kScalar = m.tryGetConstantScalar(kMulNode.Input[0])
	}
	if kScalar == 0 {
		return 0
	}

	// Effective scale is qScalar * kScalar (typically both are the same, so scalar²).
	return qScalar * kScalar
}

// lookThroughMulForShapeName returns the non-scalar input to a Mul node, which typically
// has shape info (e.g. from a Transpose). Falls back to the original name.
func (m *Model) lookThroughMulForShapeName(name string) string {
	node, ok := m.nodeOutputToNode[name]
	if !ok || node.OpType != "Mul" {
		return name
	}
	if len(node.Input) < 2 {
		return name
	}
	// Return the input that is NOT a scalar constant.
	if m.tryGetConstantScalar(node.Input[1]) != 0 {
		return node.Input[0]
	}
	if m.tryGetConstantScalar(node.Input[0]) != 0 {
		return node.Input[1]
	}
	return name
}

// extractScaleFromDiv extracts the scale factor from a Div node: result = x / divisor → scale = 1/divisor.
func (m *Model) extractScaleFromDiv(node *protos.NodeProto) float64 {
	if len(node.Input) < 2 {
		return 0
	}
	divisor := m.tryGetConstantScalar(node.Input[1])
	if divisor == 0 {
		return 0
	}
	return 1.0 / divisor
}

// extractScaleFromMul extracts the scale factor from a Mul node: result = x * scale.
func (m *Model) extractScaleFromMul(node *protos.NodeProto) float64 {
	if len(node.Input) < 2 {
		return 0
	}
	return m.tryGetConstantScalar(node.Input[1])
}

// tryGetConstantScalar attempts to read a scalar float64 from a constant/initializer.
func (m *Model) tryGetConstantScalar(name string) float64 {
	// Check initializers (variables).
	if tp, ok := m.variableNameToValue[name]; ok {
		return tensorProtoToScalar(tp)
	}
	// Check if it's a Constant node output.
	if node, ok := m.nodeOutputToNode[name]; ok && node.OpType == "Constant" {
		return constantNodeToScalar(node)
	}
	return 0
}

// tensorProtoToScalar extracts a scalar float64 from a TensorProto.
func tensorProtoToScalar(tp *protos.TensorProto) float64 {
	// Check dims: must be scalar (empty dims or [1]).
	totalElements := int64(1)
	for _, d := range tp.Dims {
		totalElements *= d
	}
	if totalElements != 1 {
		return 0
	}

	switch tp.DataType {
	case int32(protos.TensorProto_FLOAT):
		if len(tp.FloatData) > 0 {
			return float64(tp.FloatData[0])
		}
		if len(tp.RawData) >= 4 {
			bits := uint32(tp.RawData[0]) | uint32(tp.RawData[1])<<8 | uint32(tp.RawData[2])<<16 | uint32(tp.RawData[3])<<24
			return float64(math.Float32frombits(bits))
		}
	case int32(protos.TensorProto_DOUBLE):
		if len(tp.DoubleData) > 0 {
			return tp.DoubleData[0]
		}
		if len(tp.RawData) >= 8 {
			bits := uint64(tp.RawData[0]) | uint64(tp.RawData[1])<<8 | uint64(tp.RawData[2])<<16 |
				uint64(tp.RawData[3])<<24 | uint64(tp.RawData[4])<<32 | uint64(tp.RawData[5])<<40 |
				uint64(tp.RawData[6])<<48 | uint64(tp.RawData[7])<<56
			return math.Float64frombits(bits)
		}
	case int32(protos.TensorProto_FLOAT16):
		if len(tp.RawData) >= 2 {
			bits := uint16(tp.RawData[0]) | uint16(tp.RawData[1])<<8
			return float64(math.Float32frombits(halfToFloat32Bits(bits)))
		}
	}
	return 0
}

// halfToFloat32Bits converts a float16 bit pattern to float32 bits.
func halfToFloat32Bits(h uint16) uint32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	if exp == 0 {
		if mant == 0 {
			return sign << 31
		}
		// Denormalized
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		return (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	} else if exp == 0x1f {
		return (sign << 31) | (0xff << 23) | (mant << 13)
	}
	return (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
}

// constantNodeToScalar extracts a scalar from a Constant op node.
func constantNodeToScalar(node *protos.NodeProto) float64 {
	for _, attr := range node.Attribute {
		if attr.Name == "value" && attr.T != nil {
			return tensorProtoToScalar(attr.T)
		}
		if attr.Name == "value_float" {
			return float64(attr.F)
		}
	}
	return 0
}

// isMaskRankAcceptable checks that the mask is rank ≤ 4.
// Rank-2 masks are shared across batches and heads. Rank-3/4 masks are broadcast
// per batch/head by the backend using strides computed from the mask shape.
// Returns false if rank is unknown (be conservative and skip fusion).
func (m *Model) isMaskRankAcceptable(graph *protos.GraphProto, maskName string) bool {
	rank := m.getOutputRank(graph, maskName)
	if rank < 0 {
		// Unknown rank, be conservative and skip fusion.
		return false
	}
	return rank <= 4
}

// getOutputRank tries to determine the rank of a named output from ValueInfo, inputs, or initializers.
// Returns -1 if unknown.
func (m *Model) getOutputRank(graph *protos.GraphProto, name string) int {
	dims := m.getShapeDims(graph, name)
	if dims == nil {
		return -1
	}
	return len(dims)
}

// extractHeadCounts tries to determine numHeads and numKVHeads from Q and K shapes.
// The expected shape is [batch, numHeads, seqLen, headDim].
// Falls back to numHeads=1, numKVHeads=numHeads if shape info is unavailable.
func (m *Model) extractHeadCounts(graph *protos.GraphProto, qName, kName string) (numHeads, numKVHeads int) {
	numHeads = m.extractDimFromShape(graph, qName, 1)
	numKVHeads = m.extractDimFromShape(graph, kName, 1)
	if numHeads <= 0 {
		numHeads = 1
	}
	if numKVHeads <= 0 {
		numKVHeads = numHeads
	}
	return
}

// extractDimFromShape extracts a specific dimension value from the shape of a named output.
// Returns -1 if unknown or dynamic.
func (m *Model) extractDimFromShape(graph *protos.GraphProto, name string, dimIdx int) int {
	shape := m.getShapeDims(graph, name)
	if shape == nil || dimIdx >= len(shape) {
		return -1
	}
	return shape[dimIdx]
}

// getShapeDims returns the static dimension sizes for a named output, or nil if unknown.
// Dynamic dimensions are returned as -1.
func (m *Model) getShapeDims(graph *protos.GraphProto, name string) []int {
	sources := [][]*protos.ValueInfoProto{graph.ValueInfo, graph.Input, graph.Output}
	for _, vis := range sources {
		for _, vi := range vis {
			if vi.Name == name {
				return extractDimsFromValueInfo(vi)
			}
		}
	}
	if tp, ok := m.variableNameToValue[name]; ok {
		dims := make([]int, len(tp.Dims))
		for i, d := range tp.Dims {
			dims[i] = int(d)
		}
		return dims
	}
	return nil
}

// extractDimsFromValueInfo extracts dimension sizes from a ValueInfoProto.
func extractDimsFromValueInfo(vi *protos.ValueInfoProto) []int {
	tt, ok := vi.Type.Value.(*protos.TypeProto_TensorType)
	if !ok || tt.TensorType.Shape == nil {
		return nil
	}
	dims := make([]int, len(tt.TensorType.Shape.Dim))
	for i, d := range tt.TensorType.Shape.Dim {
		if dv, ok := d.Value.(*protos.TensorShapeProto_Dimension_DimValue); ok {
			dims[i] = int(dv.DimValue)
		} else {
			dims[i] = -1 // dynamic
		}
	}
	return dims
}
