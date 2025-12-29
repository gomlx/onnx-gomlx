package onnx

import (
	. "github.com/gomlx/gomlx/pkg/core/graph" //nolint
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/gomlx/gomlx/pkg/support/sets"
)

// ShapeProvenance tracks where shape information comes from.
// This is used to determine whether dynamic shapes need bounds.
type ShapeProvenance int

const (
	// ProvenanceUnknown - provenance not determined yet.
	ProvenanceUnknown ShapeProvenance = iota

	// ProvenanceConstant - shape comes from a constant or model weights.
	ProvenanceConstant

	// ProvenanceInputShape - shape depends on input tensor shapes (known at graph build time).
	ProvenanceInputShape

	// ProvenanceDataDependent - shape depends on tensor values at runtime (e.g., NonZero, Where).
	ProvenanceDataDependent
)

// String returns a human-readable name for the provenance.
func (p ShapeProvenance) String() string {
	switch p {
	case ProvenanceUnknown:
		return "unknown"
	case ProvenanceConstant:
		return "constant"
	case ProvenanceInputShape:
		return "input_shape"
	case ProvenanceDataDependent:
		return "data_dependent"
	default:
		return "invalid"
	}
}

// ShapeInfo holds information about a shape tensor's provenance and bounds.
type ShapeInfo struct {
	Provenance ShapeProvenance

	// Dims holds known dimensions (-1 for unknown).
	Dims []int

	// Bounds holds upper bounds for each dimension.
	// Used when Provenance is ProvenanceDataDependent.
	Bounds []int

	// SourceOp is the operation that produced this shape (for debugging).
	SourceOp string
}

// IsDataDependent returns true if this shape depends on tensor values at runtime.
func (si *ShapeInfo) IsDataDependent() bool {
	return si.Provenance == ProvenanceDataDependent
}

// IsFullyConcrete returns true if all dimensions are known at compile time.
func (si *ShapeInfo) IsFullyConcrete() bool {
	for _, d := range si.Dims {
		if d < 0 {
			return false
		}
	}
	return true
}

// dataDependentOps is the set of ONNX operations whose output shape depends on tensor values.
var dataDependentOps sets.Set[string]

func init() {
	dataDependentOps = sets.Make[string]()
	dataDependentOps.Insert("NonZero")   // Output shape depends on how many non-zero elements
	dataDependentOps.Insert("Where")     // Output shape can depend on condition values (for ONNX Where with 1 input)
	dataDependentOps.Insert("Compress")  // Output shape depends on condition values
	dataDependentOps.Insert("Unique")    // Output shape depends on number of unique elements
	dataDependentOps.Insert("TopK")      // Output shape is fixed but selection depends on values
	dataDependentOps.Insert("NMS")       // NonMaxSuppression - output depends on scores
}

// IsDataDependentOp returns true if the operation produces data-dependent shapes.
func IsDataDependentOp(opType string) bool {
	return dataDependentOps.Has(opType)
}

// analyzeShapeProvenance analyzes where a shape tensor comes from.
// Returns the provenance and any inferred bounds.
func (m *Model) analyzeShapeProvenance(nodeOutputName string, convertedOutputs map[string]*Node) *ShapeInfo {
	info := &ShapeInfo{
		Provenance: ProvenanceUnknown,
	}

	// Check if already a constant
	if node, found := convertedOutputs[nodeOutputName]; found {
		if node.Type() == NodeTypeConstant {
			info.Provenance = ProvenanceConstant
			return info
		}
	}

	// Find the ONNX node that produced this output
	onnxNode := m.nodeOutputToNode[nodeOutputName]
	if onnxNode == nil {
		// Check if it's a variable or input
		if m.inputsNameSet.Has(nodeOutputName) {
			info.Provenance = ProvenanceInputShape
			return info
		}
		if _, found := m.variableNameToValue[nodeOutputName]; found {
			info.Provenance = ProvenanceConstant
			return info
		}
		return info
	}

	// Check if this operation produces data-dependent shapes
	if IsDataDependentOp(onnxNode.OpType) {
		info.Provenance = ProvenanceDataDependent
		info.SourceOp = onnxNode.OpType
		info.Bounds = m.inferBoundsForDataDependentOp(onnxNode, convertedOutputs)
		return info
	}

	// For shape manipulation ops, trace back to their sources
	info = m.traceShapeProvenance(onnxNode, convertedOutputs, sets.Make[string]())
	return info
}

// traceShapeProvenance recursively traces shape provenance through shape manipulation operations.
func (m *Model) traceShapeProvenance(node *protos.NodeProto, convertedOutputs map[string]*Node, visited sets.Set[string]) *ShapeInfo {
	info := &ShapeInfo{
		Provenance: ProvenanceUnknown,
		SourceOp:   node.OpType,
	}

	// Avoid infinite loops
	if visited.Has(node.GetOutput()[0]) {
		return info
	}
	visited.Insert(node.GetOutput()[0])

	switch node.OpType {
	case "Shape":
		// Shape op - check if input has data-dependent shape
		if len(node.Input) > 0 {
			inputProvenance := m.analyzeShapeProvenance(node.Input[0], convertedOutputs)
			if inputProvenance.IsDataDependent() {
				info.Provenance = ProvenanceDataDependent
				info.Bounds = inputProvenance.Bounds
			} else if m.inputsNameSet.Has(node.Input[0]) {
				// Shape of a model input - known at graph build time
				info.Provenance = ProvenanceInputShape
			} else if _, found := m.variableNameToValue[node.Input[0]]; found {
				// Shape of a model weight/constant
				info.Provenance = ProvenanceConstant
			} else {
				// Shape of an intermediate tensor - trace further
				if parentNode := m.nodeOutputToNode[node.Input[0]]; parentNode != nil {
					if IsDataDependentOp(parentNode.OpType) {
						info.Provenance = ProvenanceDataDependent
						info.Bounds = m.inferBoundsForDataDependentOp(parentNode, convertedOutputs)
					} else {
						// Could be further traced, but for now treat as unknown
						info.Provenance = ProvenanceUnknown
					}
				}
			}
		}

	case "Gather", "Slice", "Squeeze", "Unsqueeze":
		// These ops extract or transform parts of shape tensors
		if len(node.Input) > 0 {
			parentInfo := m.analyzeShapeProvenance(node.Input[0], convertedOutputs)
			info.Provenance = parentInfo.Provenance
			info.Bounds = parentInfo.Bounds
		}

	case "Concat":
		// Concat - data-dependent if ANY input is data-dependent
		maxProvenance := ProvenanceConstant
		var allBounds [][]int
		for _, inputName := range node.Input {
			parentInfo := m.analyzeShapeProvenance(inputName, convertedOutputs)
			if parentInfo.Provenance > maxProvenance {
				maxProvenance = parentInfo.Provenance
			}
			if parentInfo.Bounds != nil {
				allBounds = append(allBounds, parentInfo.Bounds)
			}
		}
		info.Provenance = maxProvenance
		// Merge bounds if any
		if len(allBounds) > 0 {
			// Simple merge: concatenate bounds
			for _, b := range allBounds {
				info.Bounds = append(info.Bounds, b...)
			}
		}

	case "Constant", "ConstantOfShape":
		info.Provenance = ProvenanceConstant

	case "Cast", "Reshape":
		// Pass-through - inherit from first input
		if len(node.Input) > 0 {
			parentInfo := m.analyzeShapeProvenance(node.Input[0], convertedOutputs)
			info.Provenance = parentInfo.Provenance
			info.Bounds = parentInfo.Bounds
		}

	default:
		// Unknown operation - conservatively assume unknown provenance
		info.Provenance = ProvenanceUnknown
	}

	return info
}

// inferBoundsForDataDependentOp infers upper bounds for data-dependent operations.
func (m *Model) inferBoundsForDataDependentOp(node *protos.NodeProto, convertedOutputs map[string]*Node) []int {
	switch node.OpType {
	case "NonZero":
		// NonZero output shape is [rank, num_nonzeros]
		// num_nonzeros ≤ total input elements
		return m.inferNonZeroBounds(node, convertedOutputs)

	case "Where":
		// Where output shape depends on condition true elements
		// Similar to NonZero
		return m.inferWhereBounds(node, convertedOutputs)

	case "Compress":
		// Compress output depends on condition
		return m.inferCompressBounds(node, convertedOutputs)

	default:
		// Unknown data-dependent op - use conservative default
		return []int{4096}
	}
}

// inferNonZeroBounds infers bounds for NonZero operation.
// NonZero output is [rank, num_nonzeros] where num_nonzeros ≤ total_elements.
func (m *Model) inferNonZeroBounds(node *protos.NodeProto, convertedOutputs map[string]*Node) []int {
	if len(node.Input) == 0 {
		return []int{4096, 4096} // Conservative default
	}

	inputName := node.Input[0]

	// Try to get input shape from converted node
	if inputNode, found := convertedOutputs[inputName]; found {
		shape := inputNode.Shape()
		rank := len(shape.Dimensions)
		totalElements := 1
		for _, d := range shape.Dimensions {
			if d < 0 {
				// Symbolic dimension - use conservative estimate
				totalElements *= 1024
			} else {
				totalElements *= d
			}
		}
		// NonZero output: [rank, num_nonzeros]
		return []int{rank, totalElements}
	}

	// Fallback to model input shapes
	for idx, inputNameCheck := range m.InputsNames {
		if inputNameCheck == inputName {
			shape := m.InputsShapes[idx]
			rank := len(shape.Dimensions)
			totalElements := 1
			for _, d := range shape.Dimensions {
				if d < 0 {
					totalElements *= 1024
				} else {
					totalElements *= d
				}
			}
			return []int{rank, totalElements}
		}
	}

	return []int{4, 4096} // Conservative default
}

// inferWhereBounds infers bounds for Where operation.
func (m *Model) inferWhereBounds(node *protos.NodeProto, convertedOutputs map[string]*Node) []int {
	// Where with 3 inputs is element-wise select (shape is same as inputs)
	// Where with 1 input is like NonZero
	if len(node.Input) == 1 {
		return m.inferNonZeroBounds(node, convertedOutputs)
	}
	// For 3-input Where, output shape matches broadcast shape of inputs
	// This is typically known at compile time
	return nil
}

// inferCompressBounds infers bounds for Compress operation.
func (m *Model) inferCompressBounds(node *protos.NodeProto, convertedOutputs map[string]*Node) []int {
	// Compress output size depends on condition, bounded by input size
	if len(node.Input) < 2 {
		return []int{4096}
	}

	// Get input shape for bound
	inputName := node.Input[0]
	if inputNode, found := convertedOutputs[inputName]; found {
		shape := inputNode.Shape()
		axis := getIntAttrOr(node, "axis", -1)
		if axis < 0 || axis >= len(shape.Dimensions) {
			return []int{shape.Size()}
		}
		return []int{shape.Dimensions[axis]}
	}

	return []int{4096}
}
