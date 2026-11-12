package fusion

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph" //nolint
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
)

// QuantizedQKVDenseParams holds parameters for fused Q/K/V int8/uint8 projections sharing
// the same pre-quantized activation input (e.g. from a single DynamicQuantizeLinear).
// Instead of 3 separate QuantizedDense calls, weights/scales/biases are concatenated along
// the output feature axis and executed in a single fused QuantizedDense call.
type QuantizedQKVDenseParams struct {
	AInputName     string
	AScaleName     string
	AZeroPointName string

	WQName, WKName, WVName             string
	ScaleQName, ScaleKName, ScaleVName string
	ZPQName, ZPKName, ZPVName          string
	BiasQName, BiasKName, BiasVName    string

	QOutputName, KOutputName, VOutputName string

	K     int // input features (shared)
	QDim  int // Q output features
	KVDim int // K and V output features (must be equal)
}

// quantizedQKVDenseCandidate implements onnxgomlx.FusionCandidate for fused quantized QKV projection.
type quantizedQKVDenseCandidate struct {
	params          *QuantizedQKVDenseParams
	internalOutputs map[string]bool
	externalInputs  []string
}

func (c *quantizedQKVDenseCandidate) Name() string   { return "QuantizedQKVDense" }
func (c *quantizedQKVDenseCandidate) Score() float32 { return 70.0 }
func (c *quantizedQKVDenseCandidate) OutputNames() []string {
	return []string{c.params.QOutputName, c.params.KOutputName, c.params.VOutputName}
}
func (c *quantizedQKVDenseCandidate) InternalOutputs() map[string]bool { return c.internalOutputs }
func (c *quantizedQKVDenseCandidate) ExternalInputs() []string         { return c.externalInputs }

func expandWeightParam1D(node *Node, targetDType dtypes.DType, dim int) *Node {
	p := ConvertDType(node, targetDType)
	if p.Rank() == 1 && p.Shape().Dimensions[0] == dim {
		return p
	}
	return BroadcastToDims(Reshape(p, 1), dim)
}

func (c *quantizedQKVDenseCandidate) Emit(_ *model.Scope, g *Graph, convertedOutputs map[string]*Node) {
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

	wQ := convertedOutputs[p.WQName]
	wK := convertedOutputs[p.WKName]
	wV := convertedOutputs[p.WVName]

	// Concatenate quantized weights: [K, QDim] + [K, KVDim] + [K, KVDim] → [K, totalN]
	wQKV := Concatenate([]*Node{wQ, wK, wV}, 1)

	// Expand scalar or 1D scales to 1D [dim] and concatenate to 1D [totalN].
	scaleQ := expandWeightParam1D(convertedOutputs[p.ScaleQName], dtypes.Float32, p.QDim)
	scaleK := expandWeightParam1D(convertedOutputs[p.ScaleKName], dtypes.Float32, p.KVDim)
	scaleV := expandWeightParam1D(convertedOutputs[p.ScaleVName], dtypes.Float32, p.KVDim)
	fusedScales := Concatenate([]*Node{scaleQ, scaleK, scaleV}, 0)

	// Concatenate weight zero points to 1D [totalN] if any projection has a non-zero zero point.
	var fusedBZP *Node
	if p.ZPQName != "" || p.ZPKName != "" || p.ZPVName != "" {
		zpDType := wQ.DType()
		zpOrZero := func(name string, dim int) *Node {
			if name == "" {
				return Zeros(g, shapes.Make(zpDType, dim))
			}
			return expandWeightParam1D(convertedOutputs[name], zpDType, dim)
		}
		fusedBZP = Concatenate([]*Node{
			zpOrZero(p.ZPQName, p.QDim),
			zpOrZero(p.ZPKName, p.KVDim),
			zpOrZero(p.ZPVName, p.KVDim),
		}, 0)
	}

	// Concatenate biases if present: [QDim] + [KVDim] + [KVDim] → [totalN]
	var bias *Node
	if p.BiasQName != "" {
		biasQ := ConvertDType(convertedOutputs[p.BiasQName], dtypes.Float32)
		biasK := ConvertDType(convertedOutputs[p.BiasKName], dtypes.Float32)
		biasV := ConvertDType(convertedOutputs[p.BiasVName], dtypes.Float32)
		bias = Concatenate([]*Node{biasQ, biasK, biasV}, 0)
	}

	cfg := QuantizedDenseConfig{
		WeightQuantization: &Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     fusedScales,
			ZeroPoint: fusedBZP,
		},
		XQuantization: &Quantization{
			Scheme:    compute.QuantLinear,
			Scale:     xScale,
			ZeroPoint: xZP,
		},
		WeightLayout: compute.DenseLayoutInputOutputs,
	}
	result := nn.QuantizedDense(a, wQKV, bias, cfg)

	// Slice output along last axis into Q, K, V (supports both MHA QDim==KVDim and GQA QDim!=KVDim,
	// as well as dynamic leading batch/sequence dimensions).
	totalOut := p.QDim + 2*p.KVDim
	lastAxis := result.Rank() - 1
	sliceLastAxis := func(start, end int) *Node {
		specs := make([]SliceAxisSpec, result.Rank())
		for i := range lastAxis {
			specs[i] = AxisRange()
		}
		specs[lastAxis] = AxisRange(start, end)
		return Slice(result, specs...)
	}
	convertedOutputs[p.QOutputName] = sliceLastAxis(0, p.QDim)
	convertedOutputs[p.KOutputName] = sliceLastAxis(p.QDim, p.QDim+p.KVDim)
	convertedOutputs[p.VOutputName] = sliceLastAxis(p.QDim+p.KVDim, totalOut)
}

func init() {
	onnxgomlx.RegisterFusionDetector(detectQuantizedQKVDenseCandidates)
}

type sharedQuantActKey struct {
	aInput string
	aScale string
	aZP    string
}

// detectQuantizedQKVDenseCandidates runs the individual quantized dense detector internally,
// then merges triplets of QuantizedDense candidates sharing the same quantized activation
// input (e.g. from a shared DynamicQuantizeLinear) into a single QuantizedQKVDense candidate.
func detectQuantizedQKVDenseCandidates(m *onnxgomlx.Model) []onnxgomlx.FusionCandidate {
	var qdCandidates []*quantizedDenseCandidate
	for _, node := range m.Proto.Graph.Node {
		if node.OpType != "MatMulInteger" || len(node.Input) < 2 || len(node.Output) == 0 {
			continue
		}
		if cand := tryMatchQuantizedDense(m, node); cand != nil {
			qdCandidates = append(qdCandidates, cand)
		}
	}

	// Group by shared quantized activation (A, AScale, AZP). Skip GELU-fused FFN projections.
	byInput := make(map[sharedQuantActKey][]*quantizedDenseCandidate)
	for _, cand := range qdCandidates {
		p := cand.params
		if p.HasGelu || p.AInputName == "" {
			continue
		}
		key := sharedQuantActKey{
			aInput: p.AInputName,
			aScale: p.AScaleName,
			aZP:    p.AZeroPointName,
		}
		byInput[key] = append(byInput[key], cand)
	}

	var candidates []onnxgomlx.FusionCandidate
	for _, entries := range byInput {
		if len(entries) != 3 {
			continue
		}
		if cand := tryMergeQuantizedQKV(entries); cand != nil {
			candidates = append(candidates, cand)
		}
	}
	return candidates
}

// tryMergeQuantizedQKV attempts to merge 3 QuantizedDense candidates into one QKV candidate.
func tryMergeQuantizedQKV(entries []*quantizedDenseCandidate) *quantizedQKVDenseCandidate {
	params := make([]*QuantizedDenseParams, 3)
	for i, e := range entries {
		params[i] = e.params
	}

	// All must share the same K (input features) and quantized weight DType.
	K := params[0].K
	bDType := params[0].BDType
	for _, p := range params[1:] {
		if p.K != K || p.BDType != bDType {
			return nil
		}
	}

	// Determine Q, K, V ordering by dimension. If two share the same N and
	// one differs, the differing one is Q (GQA). If all equal, keep appearance order.
	qIdx, kIdx, vIdx := 0, 1, 2
	d0, d1, d2 := params[0].N, params[1].N, params[2].N
	if d0 == d1 && d0 != d2 {
		qIdx, kIdx, vIdx = 2, 0, 1
	} else if d0 == d2 && d0 != d1 {
		qIdx, kIdx, vIdx = 1, 0, 2
	} else if d1 == d2 && d1 != d0 {
		qIdx, kIdx, vIdx = 0, 1, 2
	}

	qP := params[qIdx]
	kP := params[kIdx]
	vP := params[vIdx]

	// K and V must have equal N.
	if kP.N != vP.N {
		return nil
	}

	// Bias must be all-or-nothing.
	allHaveBias := qP.BiasName != "" && kP.BiasName != "" && vP.BiasName != ""
	noneHasBias := qP.BiasName == "" && kP.BiasName == "" && vP.BiasName == ""
	if !allHaveBias && !noneHasBias {
		return nil
	}

	mergedInternalOutputs := make(map[string]bool)
	for _, e := range entries {
		for name := range e.internalOutputs {
			mergedInternalOutputs[name] = true
		}
	}

	// Put AInputName first so DynamicQuantizeLinear populates AScaleName and AZeroPointName on conversion.
	externalInputs := []string{qP.AInputName}
	if qP.AScaleName != "" {
		externalInputs = append(externalInputs, qP.AScaleName)
	}
	if qP.AZeroPointName != "" {
		externalInputs = append(externalInputs, qP.AZeroPointName)
	}
	externalInputs = append(externalInputs,
		qP.BWeightName, kP.BWeightName, vP.BWeightName,
		qP.BScaleName, kP.BScaleName, vP.BScaleName,
	)
	for _, zpName := range []string{qP.BZeroPointName, kP.BZeroPointName, vP.BZeroPointName} {
		if zpName != "" {
			externalInputs = append(externalInputs, zpName)
		}
	}
	if allHaveBias {
		externalInputs = append(externalInputs, qP.BiasName, kP.BiasName, vP.BiasName)
	}

	qkvParams := &QuantizedQKVDenseParams{
		AInputName:     qP.AInputName,
		AScaleName:     qP.AScaleName,
		AZeroPointName: qP.AZeroPointName,
		WQName:         qP.BWeightName,
		WKName:         kP.BWeightName,
		WVName:         vP.BWeightName,
		ScaleQName:     qP.BScaleName,
		ScaleKName:     kP.BScaleName,
		ScaleVName:     vP.BScaleName,
		ZPQName:        qP.BZeroPointName,
		ZPKName:        kP.BZeroPointName,
		ZPVName:        vP.BZeroPointName,
		QOutputName:    entries[qIdx].outputName,
		KOutputName:    entries[kIdx].outputName,
		VOutputName:    entries[vIdx].outputName,
		K:              K,
		QDim:           qP.N,
		KVDim:          kP.N,
	}
	if allHaveBias {
		qkvParams.BiasQName = qP.BiasName
		qkvParams.BiasKName = kP.BiasName
		qkvParams.BiasVName = vP.BiasName
	}

	return &quantizedQKVDenseCandidate{
		params:          qkvParams,
		internalOutputs: mergedInternalOutputs,
		externalInputs:  externalInputs,
	}
}
