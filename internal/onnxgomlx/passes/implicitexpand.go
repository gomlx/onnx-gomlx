// Package passes implements graph-level optimization passes for ONNX models prior to GoMLX conversion.
package passes

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/onnx-gomlx/internal/onnxgomlx"
)

// ImplicitExpandPushdownPassName is the unique name of the ImplicitExpandPushdown optimization pass.
const ImplicitExpandPushdownPassName = "ImplicitExpandPushdown"

func init() {
	onnxgomlx.RegisterPass(100, &implicitExpandPushdownPass{})
}

type implicitExpandPushdownPass struct{}

func (p *implicitExpandPushdownPass) Name() string {
	return ImplicitExpandPushdownPassName
}

// expandInfo tracks an Expand node's output and its pre-expansion provenance.
type expandInfo struct {
	unexpandedInput string
	targetShapeName string
	rootSource      string
	unexpandedShape shapes.Shape
}

// Apply pushes Expand operations down through elementwise operations:
//
//	f(Expand(x, targetShape), c) -> Expand(f(x, c), targetShape)
//
// where f is a unary, binary, or ternary (Where) elementwise operation, and any other
// operands are either:
//   - produced by an Expand to the same targetShape from a source with the same pre-expand shape, or
//   - constants that are scalars or broadcast-compatible (all dimensions equal to 1 and rank <= rank(x)), or
//   - tensors with the exact same static shape as x.
//
// Intermediate Expand nodes that become unused after pushdown are pruned from the graph.
func (p *implicitExpandPushdownPass) Apply(m *onnxgomlx.Model) (bool, error) {
	if m == nil || m.Proto.Graph == nil || len(m.Proto.Graph.Node) == 0 {
		return false, nil
	}

	expandMap := make(map[string]expandInfo)
	newNodes := make([]*protos.NodeProto, 0, len(m.Proto.Graph.Node))
	modified := false

	for _, node := range m.Proto.Graph.Node {
		if node.OpType == "Expand" && len(node.Input) == 2 && len(node.Output) == 1 && node.Output[0] != "" {
			inName := node.Input[0]
			targetShape := node.Input[1]
			outName := node.Output[0]
			if prev, ok := expandMap[inName]; ok && prev.targetShapeName == targetShape {
				expandMap[outName] = prev
			} else {
				expandMap[outName] = expandInfo{
					unexpandedInput: inName,
					targetShapeName: targetShape,
					rootSource:      inName,
					unexpandedShape: m.InferShapeForName(inName),
				}
			}
			newNodes = append(newNodes, node)
			continue
		}

		if !isElementwisePushdownOp(node) || len(node.Output) != 1 || node.Output[0] == "" {
			newNodes = append(newNodes, node)
			continue
		}

		firstExp, ok := canPushExpandThrough(m, node, expandMap)
		if !ok {
			newNodes = append(newNodes, node)
			continue
		}

		// Rewrite inputs that came from Expand to use their unexpanded counterparts.
		for i, inp := range node.Input {
			if info, isExp := expandMap[inp]; isExp {
				node.Input[i] = info.unexpandedInput
			}
		}

		origOut := node.Output[0]
		unexpOut := origOut + "#unexpanded"
		node.Output[0] = unexpOut
		m.NodeOutputToNode[unexpOut] = node

		unexpShape := m.InferShapeForName(unexpOut)
		if unexpShape.Dimensions == nil {
			unexpShape = firstExp.unexpandedShape
		}

		newExpand := &protos.NodeProto{
			Name:   node.Name + "#expand",
			OpType: "Expand",
			Input:  []string{unexpOut, firstExp.targetShapeName},
			Output: []string{origOut},
		}
		m.NodeOutputToNode[origOut] = newExpand

		expandMap[origOut] = expandInfo{
			unexpandedInput: unexpOut,
			targetShapeName: firstExp.targetShapeName,
			rootSource:      firstExp.rootSource,
			unexpandedShape: unexpShape,
		}

		newNodes = append(newNodes, node, newExpand)
		modified = true
	}

	if !modified {
		return false, nil
	}

	// Prune dead Expand nodes whose outputs are not graph outputs and have no consumers.
	graphOutputs := make(map[string]bool, len(m.Proto.Graph.Output))
	for _, out := range m.Proto.Graph.Output {
		graphOutputs[out.Name] = true
	}

	useCount := make(map[string]int, len(newNodes)*2)
	for _, node := range newNodes {
		for _, inp := range node.Input {
			if inp != "" {
				useCount[inp]++
			}
		}
	}

	dead := make([]bool, len(newNodes))
	for i := len(newNodes) - 1; i >= 0; i-- {
		node := newNodes[i]
		if node.OpType != "Expand" || len(node.Output) != 1 {
			continue
		}
		outName := node.Output[0]
		if !graphOutputs[outName] && useCount[outName] == 0 {
			dead[i] = true
			for _, inp := range node.Input {
				if inp != "" {
					useCount[inp]--
				}
			}
		}
	}

	prunedNodes := make([]*protos.NodeProto, 0, len(newNodes))
	for i, node := range newNodes {
		if !dead[i] {
			prunedNodes = append(prunedNodes, node)
		}
	}
	m.Proto.Graph.Node = prunedNodes
	return true, nil
}

// canPushExpandThrough checks whether node has at least one input in expandMap and all
// inputs of node are compatible with pushing that Expand after node.
func canPushExpandThrough(m *onnxgomlx.Model, node *protos.NodeProto, expandMap map[string]expandInfo) (expandInfo, bool) {
	var firstExp expandInfo
	foundExp := false
	for _, inp := range node.Input {
		if inp == "" {
			continue
		}
		if info, ok := expandMap[inp]; ok {
			firstExp = info
			foundExp = true
			break
		}
	}
	if !foundExp {
		return expandInfo{}, false
	}

	for _, inp := range node.Input {
		if inp == "" {
			continue
		}
		if info, ok := expandMap[inp]; ok {
			if info.targetShapeName != firstExp.targetShapeName {
				return expandInfo{}, false
			}
			if !samePreExpandShape(info, firstExp) {
				return expandInfo{}, false
			}
			continue
		}

		// Non-Expand input: allow scalar/broadcast-1 constants (whose rank does not exceed
		// the unexpanded tensor's rank) or inputs with the exact same static shape.
		if isCompatibleBroadcastConstant(m, inp, firstExp.unexpandedShape) {
			continue
		}
		if sameStaticDimensions(m.InferShapeForName(inp), firstExp.unexpandedShape) {
			continue
		}
		return expandInfo{}, false
	}

	return firstExp, true
}

func samePreExpandShape(a, b expandInfo) bool {
	if a.rootSource != "" && a.rootSource == b.rootSource {
		return true
	}
	return sameStaticDimensions(a.unexpandedShape, b.unexpandedShape)
}

func sameStaticDimensions(a, b shapes.Shape) bool {
	if a.Dimensions == nil || b.Dimensions == nil || len(a.Dimensions) != len(b.Dimensions) {
		return false
	}
	for i, d := range a.Dimensions {
		if d <= 0 || d != b.Dimensions[i] {
			return false
		}
	}
	return true
}

func isCompatibleBroadcastConstant(m *onnxgomlx.Model, name string, targetUnexpandedShape shapes.Shape) bool {
	if !m.IsScalarOrBroadcast1Constant(name) {
		return false
	}
	cShape := m.InferShapeForName(name)
	if cShape.Dimensions == nil {
		return false
	}
	// Rank-0 scalar constants broadcast with any rank without altering dimensions.
	if len(cShape.Dimensions) == 0 {
		return true
	}
	// Non-scalar all-1s constants are safe if their rank does not exceed the unexpanded shape's rank.
	if targetUnexpandedShape.Dimensions != nil && len(cShape.Dimensions) <= len(targetUnexpandedShape.Dimensions) {
		return true
	}
	return false
}

func isElementwisePushdownOp(node *protos.NodeProto) bool {
	switch node.OpType {
	case "Cast", "Neg", "Abs", "Exp", "Log", "Sqrt", "Erf",
		"Relu", "LeakyRelu", "Elu", "Celu", "Selu", "Sigmoid", "HardSigmoid",
		"Tanh", "Softplus", "Softsign", "Not", "Floor", "Ceil", "Round",
		"Sign", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
		"Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Reciprocal", "Identity", "Gelu":
		return len(node.Input) == 1
	case "Clip":
		return len(node.Input) >= 1 && len(node.Input) <= 3
	case "Add", "Sub", "Mul", "Div", "Mod", "Pow",
		"BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor",
		"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual",
		"And", "Or", "Xor", "Min", "Max", "PRelu":
		return len(node.Input) == 2
	case "Where":
		return len(node.Input) == 3
	default:
		return false
	}
}
