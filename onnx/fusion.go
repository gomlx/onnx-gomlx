package onnx

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// FusionType identifies the kind of fused operation.
type FusionType int

const (
	FusionSDPA FusionType = iota
	FusionQKVDense
	FusionDenseGelu
)

// FusionGroup describes a set of ONNX nodes that can be replaced by a single fused GoMLX op.
type FusionGroup struct {
	Type                FusionType
	RootOutputName      string          // final output of the fused pattern
	InternalOutputNames map[string]bool // outputs produced inside the group (not to be converted individually)
	ExternalInputNames  []string        // inputs from outside the group
	Params              any             // SDPAParams or QKVDenseParams
}

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

// QKVDenseParams holds parameters for fused QKV projection.
type QKVDenseParams struct {
	SharedInputName                       string
	WQName, WKName, WVName                string
	BiasQName, BiasKName, BiasVName       string
	QOutputName, KOutputName, VOutputName string
	QDim, KVDim                           int
}

// buildConsumerMap builds a map from output name to all NodeProto nodes that consume it as input.
func buildConsumerMap(graph *protos.GraphProto) map[string][]*protos.NodeProto {
	consumers := make(map[string][]*protos.NodeProto)
	for _, node := range graph.Node {
		for _, inputName := range node.Input {
			if inputName == "" {
				continue
			}
			consumers[inputName] = append(consumers[inputName], node)
		}
	}
	return consumers
}

// soleConsumer returns the single consumer of outputName, or nil if there are 0 or 2+ consumers.
func soleConsumer(consumers map[string][]*protos.NodeProto, outputName string) *protos.NodeProto {
	list := consumers[outputName]
	if len(list) == 1 {
		return list[0]
	}
	return nil
}

// detectFusionPatterns scans the ONNX graph and populates m.detectedFusionGroups with detected patterns.
func (m *Model) detectFusionPatterns() {
	graph := m.Proto.Graph
	consumers := buildConsumerMap(graph)
	m.detectedFusionGroups = make(map[string]*FusionGroup)

	// Run all pattern matchers. Each appends to m.detectedFusionGroups.
	m.detectSDPAPatterns(graph, consumers)
	m.detectQKVDensePatterns(graph, consumers)
	m.detectDenseGeluPatterns(graph, consumers)
}

// buildActiveFusionGroups returns the subset of detectedFusionGroups whose required ops
// are supported by the backend. It does not modify detectedFusionGroups.
func (m *Model) buildActiveFusionGroups(caps backends.Capabilities) map[string]*FusionGroup {
	active := make(map[string]*FusionGroup, len(m.detectedFusionGroups))
	for name, fg := range m.detectedFusionGroups {
		var supported bool
		switch fg.Type {
		case FusionSDPA:
			supported = caps.Operations[backends.OpTypeFusedMultiHeadSDPA]
		case FusionQKVDense:
			supported = caps.Operations[backends.OpTypeFusedQKVDense]
		case FusionDenseGelu:
			supported = caps.Operations[backends.OpTypeFusedDense]
		}
		if supported {
			active[name] = fg
		}
	}
	return active
}

// emitFusionGroup converts a FusionGroup into fused GoMLX ops, storing results in convertedOutputs.
func (m *Model) emitFusionGroup(ctx *context.Context, g *Graph, fg *FusionGroup, convertedOutputs map[string]*Node) {
	switch fg.Type {
	case FusionSDPA:
		m.emitSDPA(ctx, g, fg, convertedOutputs)
	case FusionQKVDense:
		m.emitQKVDense(ctx, g, fg, convertedOutputs)
	case FusionDenseGelu:
		m.emitDenseGelu(ctx, g, fg, convertedOutputs)
	}
}

// otherAddInput returns the input to an Add node that is not knownInput.
// Returns "" if the node doesn't have exactly 2 inputs or knownInput isn't one of them.
func otherAddInput(addNode *protos.NodeProto, knownInput string) string {
	if len(addNode.Input) < 2 {
		return ""
	}
	if addNode.Input[0] == knownInput {
		return addNode.Input[1]
	}
	if addNode.Input[1] == knownInput {
		return addNode.Input[0]
	}
	return ""
}

// hasExternalConsumers checks whether any of the internal outputs of a candidate fusion
// group are consumed by a node outside the group.
func hasExternalConsumers(internalOutputs map[string]bool, consumers map[string][]*protos.NodeProto, internalNodes map[*protos.NodeProto]bool) bool {
	for outputName := range internalOutputs {
		for _, consumer := range consumers[outputName] {
			if !internalNodes[consumer] {
				return true
			}
		}
	}
	return false
}
