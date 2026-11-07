package fusion

// sdpa.go implements detection and emission of Scaled Dot-Product Attention (SDPA)
// fusion patterns. It covers the following model families and variants:
//
// Standard (post-scaled) SDPA:
//
//	MatMul(Q, K^T) → Div/Mul(·, scale) → [Add(·, mask)] → Softmax(-1) → MatMul(·, V)
//	Used by: BERT, GPT-2, RoBERTa, DistilBERT, standard Hugging Face transformers
//
// Pre-scaled SDPA:
//
//	Mul(Q, s), Mul(K^T, s) → MatMul(·, ·) → [Add(mask)] → Softmax(-1) → MatMul(·, V)
//	Used by: Snowflake arctic-embed models, some ONNX exports from PyTorch with pre-scaling
//
// Combined heads-first + K^T transpose (perm [0,2,3,1]):
//
//	Used by: Models exported with heads in non-standard axis layout (Snowflake variants)
//
// Grouped Query Attention (GQA) is supported when NumHeads != NumKVHeads.

import (
	"encoding/binary"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"

	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
	"github.com/gomlx/compute-onnx/support/protos"
)

// SDPAParams holds parameters for fused scaled dot-product attention.
type SDPAParams struct {
	QInputName, KInputName, VInputName string
	MaskInputName                      string  // empty if no mask or if converted to SeqMaskInputName
	SeqMaskInputName                   string  // rank-2 [batch, seq] attention_mask input for QuerySeqLen/KVSeqLen
	Scale                              float64 // 1/sqrt(headDim)
	QScaleNodeName                     string  // non-empty if Q scale is deferred constant expression
	KScaleNodeName                     string  // non-empty if K scale is deferred constant expression
	ScaleNodeName                      string  // non-empty if post-scale is deferred constant expression
	IsDivScale                         bool    // true if ScaleNodeName is a divisor (scale = 1/s)
	NumHeads                           int
	NumKVHeads                         int
	// KNeedsHeadsFirst is true when K is in [batch, kvLen, numKVHeads, headDim] layout
	// and needs a [0,2,1,3] transpose to become [batch, numKVHeads, kvLen, headDim].
	KNeedsHeadsFirst bool
	// UseBSHD is true when Q, K, V, and output [0,2,1,3] transposes are absorbed so
	// attention.Core runs directly in [batch, seq, heads, headDim] (LayoutBSHD).
	UseBSHD bool
}

// sdpaCandidate implements onnxgomlx.FusionCandidate for scaled dot-product attention.
type sdpaCandidate struct {
	model           *onnxgomlx.Model
	params          *SDPAParams
	outputName      string
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *sdpaCandidate) Name() string                     { return "SDPA" }
func (c *sdpaCandidate) Score() float32                   { return 100.0 }
func (c *sdpaCandidate) OutputNames() []string            { return []string{c.outputName} }
func (c *sdpaCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *sdpaCandidate) ExternalInputs() []string         { return c.externalInputs }

func (c *sdpaCandidate) Emit(scope *model.Scope, g *Graph, convertedOutputs map[string]*Node) {
	p := c.params
	m := c.model

	scale := p.Scale
	if scale == 0 && m != nil {
		if p.QScaleNodeName != "" && p.KScaleNodeName != "" {
			sq, err1 := m.MaterializeConstantScalar(p.QScaleNodeName, convertedOutputs)
			sk, err2 := m.MaterializeConstantScalar(p.KScaleNodeName, convertedOutputs)
			if err1 == nil && err2 == nil && sq != 0 && sk != 0 {
				scale = sq * sk
			}
		} else if p.ScaleNodeName != "" {
			s, err := m.MaterializeConstantScalar(p.ScaleNodeName, convertedOutputs)
			if err == nil && s != 0 {
				if p.IsDivScale {
					scale = 1.0 / s
				} else {
					scale = s
				}
			}
		}
	}

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

	var querySeqLen, kvSeqLen *Node
	layout := attention.LayoutBHSD
	if p.UseBSHD {
		layout = attention.LayoutBSHD
		if p.SeqMaskInputName != "" {
			if seqMask := convertedOutputs[p.SeqMaskInputName]; seqMask != nil && seqMask.Rank() == 2 {
				cacheKey := p.SeqMaskInputName + "#sdpa_seqlen"
				seqLen := convertedOutputs[cacheKey]
				if seqLen == nil {
					seqLen = ReduceSum(ConvertDType(seqMask, dtypes.Int32), 1)
					convertedOutputs[cacheKey] = seqLen
				}
				querySeqLen = seqLen
				kvSeqLen = seqLen
			}
		} else if mask != nil && mask.Rank() == 4 && (mask.Shape().Dim(1) != 1 || mask.Shape().Dim(2) != 1) {
			mask = TransposeAllDims(mask, 0, 2, 1, 3)
		}
	}

	output, _ := attention.Core(q, k, v, layout, attention.CoreOptions{
		Scale:         scale,
		AttentionMask: mask,
		QuerySeqLen:   querySeqLen,
		KVSeqLen:      kvSeqLen,
	})
	convertedOutputs[c.outputName] = output
}

func init() {
	onnxgomlx.RegisterFusionDetector(detectSDPACandidates)
}

func detectSDPACandidates(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	return DetectSDPA(m)
}

// detectSDPACandidates scans the ONNX graph for decomposed scaled dot-product attention
// and returns onnxgomlx.FusionCandidates for each match.
func DetectSDPA(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	var candidates []onnxgomlx.FusionCandidate
	for _, node := range m.Proto.Graph.Node {
		if node.OpType != "MatMul" {
			continue
		}
		if cand := sdpaTryMatch(m, node); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// sdpaTryMatch attempts to match an SDPA chain starting from matmul1 (the Q@K^T multiplication).
// It supports two patterns:
//
// Post-scaled (standard): MatMul(Q, K^T) → Div/Mul(·, scale) → [Add(mask)] → Softmax(-1) → MatMul(·, V)
// Pre-scaled:             MatMul(Mul(Q, s), Mul(K^T, s)) → [Add(mask)] → Softmax(-1) → MatMul(·, V)
//
// In the pre-scaled pattern, both Q and K inputs to MatMul1 come from Mul(·, scalar)
// with the same scalar constant, and the effective scale is scalar².
func sdpaTryMatch(m *onnxgomlx.Model, matmul1 *protos.NodeProto) *sdpaCandidate {
	consumers := m.Consumers
	if len(matmul1.Output) == 0 {
		return nil
	}
	m1Out := matmul1.Output[0]

	if len(matmul1.Input) < 2 {
		return nil
	}

	// Try to match K^T: direct Transpose ([0,1,3,2] or combined [0,2,3,1]), or Mul(Transpose(...), scalar).
	kTransposeNode, kInputName := matchKTranspose(m, matmul1.Input[1])
	var kPreScaleMulNode *protos.NodeProto
	var kNeedsHeadsFirst bool
	var kScaleNodeName string
	if kTransposeNode == nil {
		if tNode, preInput, ok := matchCombinedKTranspose(m, matmul1.Input[1]); ok {
			kTransposeNode = tNode
			kInputName = preInput
			kNeedsHeadsFirst = true
		} else {
			// Try pre-scaled K: Mul(Transpose(...), scalar)
			kPreScaleMulNode, kTransposeNode, kInputName, kNeedsHeadsFirst, kScaleNodeName = matchPreScaledKTranspose(m, matmul1.Input[1])
			if kTransposeNode == nil {
				return nil
			}
		}
	}
	if kTransposeNode != nil && len(kTransposeNode.Output) > 0 && isGraphOutput(m, kTransposeNode.Output[0]) {
		return nil
	}
	if kPreScaleMulNode != nil && len(kPreScaleMulNode.Output) > 0 && isGraphOutput(m, kPreScaleMulNode.Output[0]) {
		return nil
	}

	// Follow chain from MatMul1 output.
	scaleConsumer := onnxgraph.SoleConsumer(consumers, m1Out)
	if scaleConsumer == nil {
		return nil
	}

	var scale float64
	var afterScaleOut string
	var scaleNode *protos.NodeProto // non-nil only for post-scale pattern
	var qScaleNodeName, scaleNodeName string
	var isDivScale bool

	switch scaleConsumer.OpType {
	case "Div":
		// Post-scaled: MatMul → Div
		scale, scaleNodeName = sdpaExtractScaleFromDiv(m, scaleConsumer)
		isDivScale = true
		scaleNode = scaleConsumer
	case "Mul":
		// Could be post-scaled: MatMul → Mul(·, scalar)
		// Check if this Mul has a constant scalar input (post-scale).
		scale, scaleNodeName = sdpaExtractScaleFromMul(m, scaleConsumer)
		if scale != 0 || scaleNodeName != "" {
			scaleNode = scaleConsumer
		} else {
			return nil
		}
	case "Add", "Softmax":
		// No post-scale node. Check for pre-scaled Q/K pattern.
		if kPreScaleMulNode == nil {
			return nil // K wasn't pre-scaled, and there's no post-scale → not SDPA
		}
		scale, qScaleNodeName, kScaleNodeName = sdpaExtractPreScale(m, matmul1.Input[0], kPreScaleMulNode, kScaleNodeName)
		if scale == 0 && (qScaleNodeName == "" || kScaleNodeName == "") {
			return nil
		}
		// afterScaleOut is the MatMul output itself (no separate scale node).
		afterScaleOut = m1Out
	default:
		return nil
	}
	if scale == 0 && scaleNodeName == "" && qScaleNodeName == "" {
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
		nextNode = onnxgraph.SoleConsumer(consumers, afterScaleOut)
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
		maskInputName = onnxgraph.OtherBinaryOpInput(addNode, afterScaleOut)
		if maskInputName == "" {
			return nil
		}
		if !isMaskRankAcceptable(m, maskInputName) {
			return nil
		}
		if len(addNode.Output) == 0 {
			return nil
		}
		softmaxNode = onnxgraph.SoleConsumer(consumers, addNode.Output[0])
		if softmaxNode == nil || softmaxNode.OpType != "Softmax" {
			return nil
		}
	case "Softmax":
		softmaxNode = nextNode
	default:
		return nil
	}

	// Verify softmax axis is -1 (last axis).
	softmaxAxis := onnxgomlx.GetIntAttrOr(softmaxNode, "axis", -1)
	if softmaxAxis != -1 {
		return nil
	}

	if len(softmaxNode.Output) == 0 {
		return nil
	}
	softmaxOut := softmaxNode.Output[0]

	// Final MatMul: Softmax output @ V
	matmul2 := onnxgraph.SoleConsumer(consumers, softmaxOut)
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
	if kTransposeNode != nil && len(kTransposeNode.Output) > 0 {
		internalNodes[kTransposeNode] = true
		internalOutputs[kTransposeNode.Output[0]] = true
	}
	if kPreScaleMulNode != nil && len(kPreScaleMulNode.Output) > 0 {
		internalNodes[kPreScaleMulNode] = true
		internalOutputs[kPreScaleMulNode.Output[0]] = true
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
	if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
		return nil
	}

	// Extract numHeads from Q and K shapes.
	// For the pre-scale pattern, the MatMul inputs are Mul outputs — look through to the
	// Transpose input for shape info.
	qShapeName := matmul1.Input[0]
	kShapeName := kInputName
	if kPreScaleMulNode != nil {
		// Pre-scaled: look through Mul → Transpose for shape.
		qShapeName = sdpaLookThroughMulForShapeName(m, matmul1.Input[0])
		// kInputName is already the pre-transpose K input from matchPreScaledKTranspose.
	}
	var numHeads, numKVHeads int
	if kNeedsHeadsFirst {
		// K is in [batch, kvLen, numKVHeads, headDim] — heads are at axis 2.
		qShape := m.ShapeForName(qShapeName)
		kShape := m.ShapeForName(kShapeName)
		if len(qShape.Dimensions) > 1 {
			numHeads = qShape.Dimensions[1]
		}
		if len(kShape.Dimensions) > 2 {
			numKVHeads = kShape.Dimensions[2]
		}
		if numHeads <= 0 {
			numHeads = 1
		}
		if numKVHeads <= 0 {
			numKVHeads = numHeads
		}
	} else {
		numHeads, numKVHeads = extractHeadCounts(m, qShapeName, kShapeName)
	}

	// Build external inputs list.
	// For the pre-scale pattern, use pre-Mul inputs (the Mul nodes carry the scale
	// which is already captured in the scale parameter).
	qInputName := matmul1.Input[0]
	var qPreScaleMulNode *protos.NodeProto
	if kPreScaleMulNode != nil {
		if qNode, ok := m.NodeOutputToNode[matmul1.Input[0]]; ok && qNode.OpType == "Mul" {
			qPreScaleMulNode = qNode
		}
		// Q input is the non-scalar input to Q's Mul.
		qInputName = sdpaLookThroughMulForShapeName(m, matmul1.Input[0])
	}
	if qPreScaleMulNode != nil && len(qPreScaleMulNode.Output) > 0 {
		if isGraphOutput(m, qPreScaleMulNode.Output[0]) {
			return nil
		}
		internalNodes[qPreScaleMulNode] = true
		internalOutputs[qPreScaleMulNode.Output[0]] = true
		if onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes) {
			return nil
		}
	}

	// Check if Q, K, V inputs and the SDPA output are all wrapped in [0,2,1,3] transposes
	// between BSHD [batch, seq, heads, headDim] and BHSD [batch, heads, seq, headDim].
	// Note: PyTorch ONNX exports often use `key_layer.size(-1)` (i.e. Shape -> Slice on qInputName
	// or kInputName) to compute `1/sqrt(headDim)`. Those `Shape` consumers are part of the scale
	// constant expression and only inspect metadata at build time (axis 3 is unchanged by [0,2,1,3]).
	var useBSHD bool
	qBSHDNode, qBSHDInput := matchHeadsTranspose0213(m, qInputName)
	vBSHDNode, vBSHDInput := matchHeadsTranspose0213(m, vInputName)
	outBSHDNode, outBSHDName := matchConsumerHeadsTranspose0213(m, consumers, rootOutput)
	var kBSHDNode *protos.NodeProto
	kBSHDInput := kInputName
	if !kNeedsHeadsFirst {
		kBSHDNode, kBSHDInput = matchHeadsTranspose0213(m, kInputName)
	}
	if qBSHDNode != nil && (kNeedsHeadsFirst || kBSHDNode != nil) && vBSHDNode != nil && outBSHDNode != nil &&
		!isGraphOutput(m, qInputName) && (kNeedsHeadsFirst || !isGraphOutput(m, kInputName)) &&
		!isGraphOutput(m, vInputName) && !isGraphOutput(m, rootOutput) {
		extraNodes := []*protos.NodeProto{qBSHDNode, vBSHDNode, outBSHDNode}
		extraOutputs := []string{qInputName, vInputName, rootOutput}
		if kBSHDNode != nil {
			extraNodes = append(extraNodes, kBSHDNode)
			extraOutputs = append(extraOutputs, kInputName)
		}
		var scaleShapeNodes []*protos.NodeProto
		for _, scaleExpr := range []string{scaleNodeName, qScaleNodeName, kScaleNodeName} {
			if scaleExpr == "" {
				continue
			}
			for _, out := range extraOutputs {
				for _, c := range consumers[out] {
					if c.OpType == "Shape" && !internalNodes[c] {
						scaleShapeNodes = append(scaleShapeNodes, c)
					}
				}
			}
		}
		for _, n := range extraNodes {
			internalNodes[n] = true
		}
		for _, n := range scaleShapeNodes {
			internalNodes[n] = true
		}
		for _, out := range extraOutputs {
			internalOutputs[out] = true
		}
		hasExt := onnxgraph.HasExternalConsumers(internalOutputs, consumers, internalNodes)
		for _, n := range scaleShapeNodes {
			delete(internalNodes, n)
		}
		if !hasExt {
			// Redirect any Shape node that read the BHSD transpose output to read its BSHD input instead
			// so MaterializeConstantScalar can evaluate it when the BHSD transpose node is skipped.
			if len(scaleShapeNodes) > 0 {
				redirect := map[string]string{
					qInputName: qBSHDInput,
					vInputName: vBSHDInput,
				}
				if kBSHDNode != nil {
					redirect[kInputName] = kBSHDInput
				}
				for _, sn := range scaleShapeNodes {
					if len(sn.Input) > 0 {
						if newIn, ok := redirect[sn.Input[0]]; ok && newIn != "" {
							sn.Input[0] = newIn
						}
					}
				}
			}
			useBSHD = true
			qInputName = qBSHDInput
			kInputName = kBSHDInput
			vInputName = vBSHDInput
			rootOutput = outBSHDName
			kNeedsHeadsFirst = false
		} else {
			for _, n := range extraNodes {
				delete(internalNodes, n)
			}
			for _, out := range extraOutputs {
				delete(internalOutputs, out)
			}
		}
	}

	var seqMaskInputName string
	if useBSHD && maskInputName != "" {
		qShape := m.ShapeForName(qInputName)
		kShape := m.ShapeForName(kInputName)
		sameSeqDim := len(qShape.Dimensions) != 4 || len(kShape.Dimensions) != 4 || qShape.Dimensions[1] == kShape.Dimensions[1]
		if sameSeqDim {
			if seqMask := matchKeyPaddingMaskFrom2DAttentionMask(m, maskInputName); seqMask != "" {
				seqMaskInputName = seqMask
				maskInputName = ""
			}
		}
	}

	externalInputs := []string{qInputName, kInputName, vInputName}
	if maskInputName != "" {
		externalInputs = append(externalInputs, maskInputName)
	}
	if seqMaskInputName != "" {
		externalInputs = append(externalInputs, seqMaskInputName)
	}
	if qScaleNodeName != "" {
		externalInputs = append(externalInputs, qScaleNodeName)
	}
	if kScaleNodeName != "" && kScaleNodeName != qScaleNodeName {
		externalInputs = append(externalInputs, kScaleNodeName)
	}
	if scaleNodeName != "" {
		externalInputs = append(externalInputs, scaleNodeName)
	}

	return &sdpaCandidate{
		model:           m,
		outputName:      rootOutput,
		internalOutputs: internalOutputs,
		externalInputs:  externalInputs,
		params: &SDPAParams{
			QInputName:       qInputName,
			KInputName:       kInputName,
			VInputName:       vInputName,
			MaskInputName:    maskInputName,
			SeqMaskInputName: seqMaskInputName,
			Scale:            scale,
			QScaleNodeName:   qScaleNodeName,
			KScaleNodeName:   kScaleNodeName,
			ScaleNodeName:    scaleNodeName,
			IsDivScale:       isDivScale,
			NumHeads:         numHeads,
			NumKVHeads:       numKVHeads,
			KNeedsHeadsFirst: kNeedsHeadsFirst,
			UseBSHD:          useBSHD,
		},
	}
}

// matchKeyPaddingMaskFrom2DAttentionMask detects when an additive 4D attention mask is derived
// from a 2D [batch, seq] attention_mask input via standard HuggingFace / PyTorch ONNX patterns:
//   - Mul(Sub(1.0, Cast?(Expand?(Unsqueeze+(attention_mask)))), neg_const)
//   - Where(Cast+(Sub(1.0, Cast?(Expand?(Unsqueeze+(attention_mask))))), neg_const, Sub(...))
//
// Returns the 2D input name (e.g. "attention_mask") if matched, or "" otherwise.
func matchKeyPaddingMaskFrom2DAttentionMask(m *onnxgomlx.Model, maskInputName string) string {
	topNode, ok := m.NodeOutputToNode[maskInputName]
	if !ok {
		return ""
	}

	var subOutName string
	switch topNode.OpType {
	case "Mul":
		if len(topNode.Input) != 2 {
			return ""
		}
		if s := tryGetConstantScalar(m, topNode.Input[1]); s <= -1e3 {
			subOutName = topNode.Input[0]
		} else if s := tryGetConstantScalar(m, topNode.Input[0]); s <= -1e3 {
			subOutName = topNode.Input[1]
		} else {
			return ""
		}
	case "Where":
		if len(topNode.Input) != 3 {
			return ""
		}
		if s := tryGetConstantScalar(m, topNode.Input[1]); s > -1e3 {
			return ""
		}
		condBase := unwrapCasts(m, topNode.Input[0])
		falseBase := unwrapCasts(m, topNode.Input[2])
		if condBase == "" || condBase != falseBase {
			return ""
		}
		subOutName = falseBase
	default:
		return ""
	}

	// Unwrap optional Cast / Expand before Sub(1.0, x).
	curr := subOutName
	for range 8 {
		n, ok := m.NodeOutputToNode[curr]
		if !ok {
			return ""
		}
		if n.OpType == "Cast" || n.OpType == "Expand" {
			if len(n.Input) == 0 {
				return ""
			}
			curr = n.Input[0]
			continue
		}
		break
	}

	subNode, ok := m.NodeOutputToNode[curr]
	if !ok || subNode.OpType != "Sub" || len(subNode.Input) != 2 {
		return ""
	}
	if tryGetConstantScalar(m, subNode.Input[0]) != 1.0 {
		return ""
	}

	// Walk from subNode.Input[1] through Cast, Expand, and Unsqueeze(axes in {1, 2}) up to a rank-2 input.
	curr = subNode.Input[1]
	unsqueezedAxes := 0
	for range 12 {
		n, ok := m.NodeOutputToNode[curr]
		if !ok {
			break
		}
		switch n.OpType {
		case "Cast", "Expand":
			if len(n.Input) == 0 {
				return ""
			}
			curr = n.Input[0]
		case "Unsqueeze":
			if len(n.Input) == 0 {
				return ""
			}
			axes, ok := extractUnsqueezeAxes(m, n)
			if !ok || len(axes) == 0 {
				return ""
			}
			for _, ax := range axes {
				if ax != 1 && ax != 2 {
					return ""
				}
			}
			unsqueezedAxes += len(axes)
			curr = n.Input[0]
		default:
			return ""
		}
	}

	if unsqueezedAxes != 2 || !m.InputsNameSet.Has(curr) {
		return ""
	}
	inShape := m.ShapeForName(curr)
	if len(inShape.Dimensions) != 2 {
		return ""
	}
	return curr
}

func unwrapCasts(m *onnxgomlx.Model, name string) string {
	curr := name
	for range 6 {
		n, ok := m.NodeOutputToNode[curr]
		if !ok || n.OpType != "Cast" || len(n.Input) == 0 {
			return curr
		}
		curr = n.Input[0]
	}
	return curr
}

func extractUnsqueezeAxes(m *onnxgomlx.Model, node *protos.NodeProto) ([]int, bool) {
	if axes := onnxgomlx.GetIntsAttrOr(node, "axes", nil); len(axes) > 0 {
		return axes, true
	}
	if len(node.Input) < 2 {
		return nil, false
	}
	axesName := node.Input[1]
	var tp *protos.TensorProto
	if initTP, ok := m.VariableNameToValue[axesName]; ok {
		tp = initTP
	} else if cNode, ok := m.NodeOutputToNode[axesName]; ok && cNode.OpType == "Constant" {
		for _, attr := range cNode.Attribute {
			if attr.Name == "value" && attr.T != nil {
				tp = attr.T
				break
			}
		}
	}
	if tp == nil {
		return nil, false
	}
	if len(tp.Int64Data) > 0 {
		out := make([]int, len(tp.Int64Data))
		for i, v := range tp.Int64Data {
			out[i] = int(v)
		}
		return out, true
	}
	if len(tp.Int32Data) > 0 {
		out := make([]int, len(tp.Int32Data))
		for i, v := range tp.Int32Data {
			out[i] = int(v)
		}
		return out, true
	}
	switch tp.DataType {
	case int32(protos.TensorProto_INT64):
		if len(tp.RawData) >= 8 && len(tp.RawData)%8 == 0 {
			n := len(tp.RawData) / 8
			out := make([]int, n)
			for i := range n {
				out[i] = int(int64(binary.LittleEndian.Uint64(tp.RawData[i*8:])))
			}
			return out, true
		}
	case int32(protos.TensorProto_INT32):
		if len(tp.RawData) >= 4 && len(tp.RawData)%4 == 0 {
			n := len(tp.RawData) / 4
			out := make([]int, n)
			for i := range n {
				out[i] = int(int32(binary.LittleEndian.Uint32(tp.RawData[i*4:])))
			}
			return out, true
		}
	}
	return nil, false
}

// isGraphOutput reports whether name is one of the ONNX graph's declared outputs.
func isGraphOutput(m *onnxgomlx.Model, name string) bool {
	if m == nil || m.Proto.Graph == nil {
		return false
	}
	for _, out := range m.Proto.Graph.Output {
		if out.Name == name {
			return true
		}
	}
	return false
}

// matchHeadsTranspose0213 checks if inputName is produced by a rank-4 Transpose with perm=[0,2,1,3].
func matchHeadsTranspose0213(m *onnxgomlx.Model, inputName string) (*protos.NodeProto, string) {
	node, ok := m.NodeOutputToNode[inputName]
	if !ok || node.OpType != "Transpose" || len(node.Input) == 0 {
		return nil, ""
	}
	perm := onnxgomlx.GetIntsAttrOr(node, "perm", nil)
	if len(perm) == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3 {
		return node, node.Input[0]
	}
	return nil, ""
}

// matchConsumerHeadsTranspose0213 checks if outputName's sole consumer is a rank-4 Transpose with perm=[0,2,1,3].
func matchConsumerHeadsTranspose0213(m *onnxgomlx.Model, consumers map[string][]*protos.NodeProto, outputName string) (*protos.NodeProto, string) {
	node := onnxgraph.SoleConsumer(consumers, outputName)
	if node == nil || node.OpType != "Transpose" || len(node.Output) == 0 {
		return nil, ""
	}
	perm := onnxgomlx.GetIntsAttrOr(node, "perm", nil)
	if len(perm) == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3 {
		return node, node.Output[0]
	}
	return nil, ""
}

// matchKTranspose checks if inputName comes from a Transpose node that swaps the last two axes.
// Returns the Transpose node and the original (pre-transpose) input name, or nil if not matched.
func matchKTranspose(m *onnxgomlx.Model, inputName string) (*protos.NodeProto, string) {
	node, ok := m.NodeOutputToNode[inputName]
	if !ok || node.OpType != "Transpose" {
		return nil, ""
	}
	if len(node.Input) == 0 {
		return nil, ""
	}

	perm := onnxgomlx.GetIntsAttrOr(node, "perm", nil)
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
// Returns the Mul node, Transpose node, the pre-transpose input name, and the scalar node name; or nil if not matched.
// kNeedsHeadsFirst is true when K_raw needs a [0,2,1,3] transpose to get to [batch, heads, kvLen, headDim].
func matchPreScaledKTranspose(m *onnxgomlx.Model, inputName string) (mulNode, transposeNode *protos.NodeProto, preTransposeInput string, kNeedsHeadsFirst bool, scalarNodeName string) {
	node, ok := m.NodeOutputToNode[inputName]
	if !ok || node.OpType != "Mul" {
		return nil, nil, "", false, ""
	}
	if len(node.Input) < 2 {
		return nil, nil, "", false, ""
	}

	// One input should be a Transpose, the other a scalar constant (or constant expression).
	for _, transposeIdx := range []int{0, 1} {
		scalarIdx := 1 - transposeIdx
		sName := node.Input[scalarIdx]
		scalar := tryGetConstantScalar(m, sName)
		if scalar == 0 && !m.IsConstantExpression(sName) {
			continue
		}

		// First try: standard K^T (last two axes swapped, e.g. [0,1,3,2]).
		tNode, preInput := matchKTranspose(m, node.Input[transposeIdx])
		if tNode != nil {
			return node, tNode, preInput, false, sName
		}

		// Second try: combined heads-first + K^T (e.g. [0,2,3,1] on [batch, seqLen, heads, headDim]).
		// This rearranges to [batch, heads, headDim, seqLen] in one step.
		tNode, preInput, ok := matchCombinedKTranspose(m, node.Input[transposeIdx])
		if ok {
			return node, tNode, preInput, true, sName
		}
	}
	return nil, nil, "", false, ""
}

// matchCombinedKTranspose matches Transpose with perm [0,2,3,1] which combines
// heads-first reordering and K^T in one operation.
// Input: [batch, kvLen, numKVHeads, headDim] → Output: [batch, numKVHeads, headDim, kvLen]
// Returns the Transpose node and pre-transpose input name, or (nil, "", false).
func matchCombinedKTranspose(m *onnxgomlx.Model, inputName string) (*protos.NodeProto, string, bool) {
	node, ok := m.NodeOutputToNode[inputName]
	if !ok || node.OpType != "Transpose" {
		return nil, "", false
	}
	if len(node.Input) == 0 {
		return nil, "", false
	}

	perm := onnxgomlx.GetIntsAttrOr(node, "perm", nil)
	if perm == nil || len(perm) != 4 {
		return nil, "", false
	}

	// Accept [0, 2, 3, 1]: batch stays, middle two axes move left, last axis wraps to position 1.
	if perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1 {
		return node, node.Input[0], true
	}

	return nil, "", false
}

// sdpaExtractPreScale extracts the effective scale when both Q and K inputs to MatMul
// come from Mul(·, scalar) with the same constant scalar or constant expression.
// Returns (scale, qScaleNodeName, kScaleNodeName).
func sdpaExtractPreScale(m *onnxgomlx.Model, qMulOutputName string, kMulNode *protos.NodeProto, kScaleNodeName string) (scale float64, qScaleName, kScaleName string) {
	// Q input should also come from a Mul(·, scalar).
	qNode, ok := m.NodeOutputToNode[qMulOutputName]
	if !ok || qNode.OpType != "Mul" {
		return 0, "", ""
	}
	if len(qNode.Input) < 2 {
		return 0, "", ""
	}

	// Extract scalar from Q's Mul.
	qScaleName, qScalar := sdpaGetScalarOrExpr(m, qNode)
	if qScaleName == "" {
		return 0, "", ""
	}

	kScalar := tryGetConstantScalar(m, kScaleNodeName)
	if qScalar != 0 && kScalar != 0 {
		return qScalar * kScalar, "", ""
	}
	return 0, qScaleName, kScaleNodeName
}

// sdpaGetScalarOrExpr returns the name of a scalar input (either constant or constant expression),
// and the scalar float64 value if it's already an immediate constant.
func sdpaGetScalarOrExpr(m *onnxgomlx.Model, node *protos.NodeProto) (string, float64) {
	if len(node.Input) < 2 {
		return "", 0
	}
	for _, idx := range []int{1, 0} {
		name := node.Input[idx]
		if s := tryGetConstantScalar(m, name); s != 0 {
			return name, s
		}
		if m.IsConstantExpression(name) {
			return name, 0
		}
	}
	return "", 0
}

// sdpaLookThroughMulForShapeName returns the non-scalar input to a Mul node, which typically
// has shape info (e.g. from a Transpose). Falls back to the original name.
func sdpaLookThroughMulForShapeName(m *onnxgomlx.Model, name string) string {
	node, ok := m.NodeOutputToNode[name]
	if !ok || node.OpType != "Mul" {
		return name
	}
	if len(node.Input) < 2 {
		return name
	}
	// Return the input that is NOT a scalar constant or constant expression.
	if tryGetConstantScalar(m, node.Input[1]) != 0 || m.IsConstantExpression(node.Input[1]) {
		return node.Input[0]
	}
	if tryGetConstantScalar(m, node.Input[0]) != 0 || m.IsConstantExpression(node.Input[0]) {
		return node.Input[1]
	}
	return name
}

// sdpaExtractScaleFromDiv extracts the scale factor from a Div node: result = x / divisor → scale = 1/divisor.
func sdpaExtractScaleFromDiv(m *onnxgomlx.Model, node *protos.NodeProto) (scale float64, scaleNodeName string) {
	if len(node.Input) < 2 {
		return 0, ""
	}
	divisor := tryGetConstantScalar(m, node.Input[1])
	if divisor != 0 {
		return 1.0 / divisor, ""
	}
	if m.IsConstantExpression(node.Input[1]) {
		return 0, node.Input[1]
	}
	return 0, ""
}

// sdpaExtractScaleFromMul extracts the scale factor from a Mul node: result = x * scale.
// The scalar constant may appear as either input.
func sdpaExtractScaleFromMul(m *onnxgomlx.Model, node *protos.NodeProto) (scale float64, scaleNodeName string) {
	if len(node.Input) < 2 {
		return 0, ""
	}
	name, s := sdpaGetScalarOrExpr(m, node)
	if s != 0 {
		return s, ""
	}
	if name != "" {
		return 0, name
	}
	return 0, ""
}

// tryGetConstantScalar attempts to read a scalar float64 from a constant/initializer.
func tryGetConstantScalar(m *onnxgomlx.Model, name string) float64 {
	// Check initializers (variables).
	if tp, ok := m.VariableNameToValue[name]; ok {
		return onnxgomlx.TensorProtoToScalar(tp)
	}
	// Check if it's a Constant node output.
	if node, ok := m.NodeOutputToNode[name]; ok && node.OpType == "Constant" {
		return onnxgomlx.ConstantNodeToScalar(node)
	}
	return 0
}

// isMaskRankAcceptable checks that the mask is rank ≤ 4.
// Rank-2 masks are shared across batches and heads. Rank-3/4 masks are broadcast
// per batch/head by the backend using strides computed from the mask shape.
// Returns false if rank is unknown (be conservative and skip fusion).
func isMaskRankAcceptable(m *onnxgomlx.Model, maskName string) bool {
	s := m.ShapeForName(maskName)
	if s.Dimensions != nil {
		return len(s.Dimensions) <= 4
	}
	rank := inferRank(m, maskName, 0)
	if rank < 0 {
		return false
	}
	return rank <= 4
}

func inferRank(m *onnxgomlx.Model, name string, depth int) int {
	if depth > 20 {
		return -1
	}
	s := m.ShapeForName(name)
	if s.Dimensions != nil {
		return len(s.Dimensions)
	}
	node, ok := m.NodeOutputToNode[name]
	if !ok {
		return -1
	}
	switch node.OpType {
	case "Where", "Cast", "Sub", "Add", "Mul", "Div", "Neg", "Abs":
		for _, inp := range node.Input {
			if r := inferRank(m, inp, depth+1); r >= 0 {
				return r
			}
		}
	case "Expand":
		if len(node.Input) > 1 {
			if shapeNode, ok := m.NodeOutputToNode[node.Input[1]]; ok && shapeNode.OpType == "Concat" {
				return len(shapeNode.Input)
			}
		}
		if len(node.Input) > 0 {
			return inferRank(m, node.Input[0], depth+1)
		}
	case "Unsqueeze":
		if len(node.Input) > 0 {
			r := inferRank(m, node.Input[0], depth+1)
			if r >= 0 {
				axes := onnxgomlx.GetIntsAttrOr(node, "axes", nil)
				if len(node.Input) > 1 {
					if axesTensor, ok := m.VariableNameToValue[node.Input[1]]; ok {
						return r + len(axesTensor.Int64Data)
					}
					if _, ok := m.NodeOutputToNode[node.Input[1]]; ok {
						return r + 1
					}
				}
				if axes != nil {
					return r + len(axes)
				}
				return r + 1
			}
		}
	case "Reshape":
		if len(node.Input) > 1 {
			if shapeNode, ok := m.NodeOutputToNode[node.Input[1]]; ok && shapeNode.OpType == "Concat" {
				return len(shapeNode.Input)
			}
		}
	}
	return -1
}

// extractHeadCounts tries to determine numHeads and numKVHeads from Q and K shapes.
// The expected shape is [batch, numHeads, seqLen, headDim].
// Falls back to numHeads=1, numKVHeads=numHeads if shape info is unavailable.
func extractHeadCounts(m *onnxgomlx.Model, qName, kName string) (numHeads, numKVHeads int) {
	qShape := m.ShapeForName(qName)
	kShape := m.ShapeForName(kName)
	if len(qShape.Dimensions) > 1 {
		numHeads = qShape.Dimensions[1]
	}
	if len(kShape.Dimensions) > 1 {
		numKVHeads = kShape.Dimensions[1]
	}
	if numHeads <= 0 {
		numHeads = 1
	}
	if numKVHeads <= 0 {
		numKVHeads = numHeads
	}
	return
}
