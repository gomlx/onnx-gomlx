package onnxgomlx

import (
	"sort"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/onnx-gomlx/internal/onnxgraph"
	"github.com/pkg/errors"
)

// Pass represents a graph-level optimization pass over the parsed ONNX model.
// Optimization passes run in priority order (highest priority first) during Parse,
// prior to fusion pattern detection.
type Pass interface {
	// Name returns the unique identifier for the optimization pass.
	Name() string

	// Apply executes the optimization pass on the model, mutating m.Proto.Graph in place.
	// It returns modified=true if the graph was changed.
	Apply(m *Model) (modified bool, err error)
}

type priorityPass struct {
	priority int
	pass     Pass
}

var registeredPasses []priorityPass

// RegisterPass registers an optimization pass to be executed during model parsing.
// Passes with higher priority values run before passes with lower priority values.
func RegisterPass(priority int, pass Pass) {
	registeredPasses = append(registeredPasses, priorityPass{
		priority: priority,
		pass:     pass,
	})
	sort.SliceStable(registeredPasses, func(i, j int) bool {
		return registeredPasses[i].priority > registeredPasses[j].priority
	})
}

// ParseOption configures how an ONNX model is parsed and optimized.
type ParseOption func(m *Model)

// WithoutOptimizationPasses disables all graph optimization passes during ParseWithOptions.
func WithoutOptimizationPasses() ParseOption {
	return func(m *Model) {
		m.disableAllPasses = true
	}
}

// WithPassEnabled enables or disables a specific optimization pass by name during ParseWithOptions.
func WithPassEnabled(name string, enabled bool) ParseOption {
	return func(m *Model) {
		m.SetPassEnabled(name, enabled)
	}
}

// DisableOptimizationPasses disables all optimization passes on this Model.
func (m *Model) DisableOptimizationPasses() {
	m.disableAllPasses = true
}

// SetPassEnabled enables or disables a specific optimization pass by name on this Model.
func (m *Model) SetPassEnabled(name string, enabled bool) {
	if m.disabledPasses == nil {
		m.disabledPasses = make(map[string]bool)
	}
	if enabled {
		delete(m.disabledPasses, name)
	} else {
		m.disabledPasses[name] = true
	}
}

// IsPassEnabled reports whether the named optimization pass is enabled on this Model.
func (m *Model) IsPassEnabled(name string) bool {
	if m.disableAllPasses {
		return false
	}
	if m.disabledPasses != nil && m.disabledPasses[name] {
		return false
	}
	return true
}

// RebuildGraphIndex rebuilds m.NodeOutputToNode and m.Consumers from m.Proto.Graph.Node,
// verifying that no two nodes produce the same non-empty output name.
func (m *Model) RebuildGraphIndex() error {
	m.NodeOutputToNode = make(map[string]*protos.NodeProto, len(m.Proto.Graph.Node))
	for _, node := range m.Proto.Graph.Node {
		for _, outputName := range node.GetOutput() {
			if outputName == "" {
				continue
			}
			if otherNode, found := m.NodeOutputToNode[outputName]; found {
				return errors.Errorf("invalid graph: node output name %q used by 2 different nodes: (1) %s, (2) %s",
					outputName, NodeToString(otherNode), NodeToString(node))
			}
			m.NodeOutputToNode[outputName] = node
		}
	}
	m.Consumers = onnxgraph.BuildConsumerMap(m.Proto.Graph)
	return nil
}

// applyOptimizationPasses executes all enabled optimization passes in priority order.
func (m *Model) applyOptimizationPasses() error {
	if m.disableAllPasses || len(registeredPasses) == 0 {
		return nil
	}
	m.Consumers = onnxgraph.BuildConsumerMap(m.Proto.Graph)
	for _, entry := range registeredPasses {
		if !m.IsPassEnabled(entry.pass.Name()) {
			continue
		}
		modified, err := entry.pass.Apply(m)
		if err != nil {
			return errors.WithMessagef(err, "optimization pass %q failed", entry.pass.Name())
		}
		if modified {
			if err := m.RebuildGraphIndex(); err != nil {
				return errors.WithMessagef(err, "after optimization pass %q", entry.pass.Name())
			}
		}
	}
	return nil
}
