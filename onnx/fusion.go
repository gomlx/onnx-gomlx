package onnx

import (
	"sort"

	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
)

// FusionCandidate represents a detected fusion pattern that can replace multiple ONNX nodes
// with a single fused GoMLX operation.
type FusionCandidate interface {
	// Name returns the fusion type name (e.g. "SDPA", "QKVDense", "DenseGelu").
	Name() string
	// Score returns the priority of this fusion. Higher scores are preferred when fusions overlap.
	Score() float32
	// OutputNames returns the output names that this fusion produces.
	// These are registered in the fusion map and trigger the fusion when requested.
	OutputNames() []string
	// InternalOutputs returns intermediate output names produced inside the group
	// that should not be converted individually. These are distinct from OutputNames.
	InternalOutputs() map[string]bool
	// ExternalInputs returns the input names from outside the group that must be converted
	// before the fusion can be emitted.
	ExternalInputs() []string
	// Emit converts the fusion into GoMLX ops, storing results in convertedOutputs.
	Emit(ctx *context.Context, g *Graph, convertedOutputs map[string]*Node)
}

// FusionDetector scans an ONNX graph and returns detected fusion candidates.
// The detector uses m.Proto.Graph for the graph and m.consumers for the consumer map.
type FusionDetector func(m *Model) []FusionCandidate

var registeredDetectors []FusionDetector

// RegisterFusionDetector adds a fusion detector to the global registry.
func RegisterFusionDetector(d FusionDetector) {
	registeredDetectors = append(registeredDetectors, d)
}

// detectFusionPatterns runs all registered detectors, sorts candidates by score descending,
// then greedily selects non-overlapping fusions, populating m.detectedFusions.
func (m *Model) detectFusionPatterns() {
	m.consumers = onnxgraph.BuildConsumerMap(m.Proto.Graph)
	m.detectedFusions = make(map[string]FusionCandidate)

	// Collect all candidates from all detectors.
	var allCandidates []FusionCandidate
	for _, detector := range registeredDetectors {
		allCandidates = append(allCandidates, detector(m)...)
	}

	// Sort by score descending for greedy selection.
	sort.Slice(allCandidates, func(i, j int) bool {
		return allCandidates[i].Score() > allCandidates[j].Score()
	})

	// Greedily select non-overlapping fusions.
	claimed := make(map[string]bool)
	for _, cand := range allCandidates {
		// Check if any output or internal node is already claimed.
		overlap := false
		for _, name := range cand.OutputNames() {
			if claimed[name] {
				overlap = true
				break
			}
		}
		if !overlap {
			for name := range cand.InternalOutputs() {
				if claimed[name] {
					overlap = true
					break
				}
			}
		}
		if overlap {
			continue
		}

		// Claim all outputs and internals.
		for _, name := range cand.OutputNames() {
			claimed[name] = true
			m.detectedFusions[name] = cand
		}
		for name := range cand.InternalOutputs() {
			claimed[name] = true
		}
	}
}

// ensureFusionGroupConverted ensures all external inputs of a fusion candidate are converted,
// then emits the fused op. This is called when any output of the group is requested.
func (m *Model) ensureFusionGroupConverted(ctx *context.Context, g *Graph, cand FusionCandidate, convertedOutputs map[string]*Node) {
	// Check if already emitted (any output already in convertedOutputs).
	for _, name := range cand.OutputNames() {
		if _, done := convertedOutputs[name]; done {
			return
		}
	}

	// Convert all external inputs first.
	for _, inputName := range cand.ExternalInputs() {
		m.recursiveCallGraph(ctx, g, inputName, convertedOutputs)
	}

	// Emit the fused op.
	cand.Emit(ctx, g, convertedOutputs)
}

// isFusionGroupOutput checks if nodeOutputName is an output of any detected fusion candidate.
// Returns the candidate if found, nil otherwise.
func (m *Model) isFusionGroupOutput(nodeOutputName string) FusionCandidate {
	if m.detectedFusions == nil {
		return nil
	}
	return m.detectedFusions[nodeOutputName]
}

// DisableFusion clears all detected fusions, forcing normal (unfused) conversion.
func (m *Model) DisableFusion() *Model {
	m.detectedFusions = nil
	return m
}
