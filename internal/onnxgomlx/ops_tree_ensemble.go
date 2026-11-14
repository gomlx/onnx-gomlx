// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"sort"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// convertTreeEnsembleClassifier lowers numeric-label ONNX-ML TreeEnsembleClassifier
// nodes to vectorized GoMLX comparisons and masked leaf score accumulations.
// Node IDs are local to each tree and need not be contiguous or topologically ordered.
func convertTreeEnsembleClassifier(node *protos.NodeProto, inputs []*Node, converted map[string]*Node) *Node {
	if len(inputs) < 1 {
		exceptions.Panicf("TreeEnsembleClassifier requires 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	if x.Rank() != 2 {
		exceptions.Panicf("TreeEnsembleClassifier: expected rank 2 input, got %s", x.Shape())
	}
	labelAttr := GetNodeAttr(node, "classlabels_int64s", true)
	labels := labelAttr.Ints
	if len(labels) == 0 {
		exceptions.Panicf("TreeEnsembleClassifier: integer class labels are required")
	}

	trees := GetIntsAttrOr(node, "nodes_treeids", nil)
	ids := GetIntsAttrOr(node, "nodes_nodeids", nil)
	features := GetIntsAttrOr(node, "nodes_featureids", nil)
	trueIDs := GetIntsAttrOr(node, "nodes_truenodeids", nil)
	falseIDs := GetIntsAttrOr(node, "nodes_falsenodeids", nil)
	missingTrue := GetIntsAttrOr(node, "nodes_missing_value_tracks_true", nil)
	modes := GetStringsAttrOr(node, "nodes_modes", nil)
	values := GetFloatsAttrOr(node, "nodes_values", nil)
	count := len(ids)
	if count == 0 || len(trees) != count || len(features) != count || len(trueIDs) != count ||
		len(falseIDs) != count || len(modes) != count || len(values) != count ||
		(missingTrue != nil && len(missingTrue) != count) {
		exceptions.Panicf("TreeEnsembleClassifier: inconsistent node attributes (tensor-valued attributes are not supported)")
	}

	type key struct{ tree, id int }
	index := make(map[key]int, count)
	children := make(map[key]bool)
	for i, id := range ids {
		k := key{trees[i], id}
		if _, found := index[k]; found {
			exceptions.Panicf("TreeEnsembleClassifier: duplicate node %v", k)
		}
		index[k] = i
		if modes[i] != "LEAF" {
			children[key{trees[i], trueIDs[i]}], children[key{trees[i], falseIDs[i]}] = true, true
		}
	}
	roots := make(map[int]key)
	for k := range index {
		if !children[k] {
			if _, found := roots[k.tree]; found {
				exceptions.Panicf("TreeEnsembleClassifier: multiple roots in tree %d", k.tree)
			}
			roots[k.tree] = k
		}
	}
	if len(roots) == 0 {
		exceptions.Panicf("TreeEnsembleClassifier: no tree roots")
	}

	classIDs := GetIntsAttrOr(node, "class_ids", nil)
	classNodes := GetIntsAttrOr(node, "class_nodeids", nil)
	classTrees := GetIntsAttrOr(node, "class_treeids", nil)
	classWeights := GetFloatsAttrOr(node, "class_weights", nil)
	if len(classIDs) != len(classNodes) || len(classIDs) != len(classTrees) || len(classIDs) != len(classWeights) {
		exceptions.Panicf("TreeEnsembleClassifier: inconsistent class attributes")
	}

	usedClasses := make(map[int]bool)
	for _, class := range classIDs {
		if class < 0 || class >= len(labels) {
			exceptions.Panicf("TreeEnsembleClassifier: invalid class ID %d", class)
		}
		usedClasses[class] = true
	}
	compactBinary := len(labels) == 2 && len(usedClasses) == 1 && usedClasses[0]
	if len(labels) == 2 && len(usedClasses) == 1 && !usedClasses[0] {
		exceptions.Panicf("TreeEnsembleClassifier: compact binary class weights must target class_id 0")
	}
	scoreCols := len(labels)
	if compactBinary {
		scoreCols = 1
	}

	weights := make(map[key][]float32)
	for i, class := range classIDs {
		k := key{classTrees[i], classNodes[i]}
		idx, found := index[k]
		if !found || modes[idx] != "LEAF" {
			exceptions.Panicf("TreeEnsembleClassifier: class weight refers to a non-leaf node %v", k)
		}
		if weights[k] == nil {
			weights[k] = make([]float32, scoreCols)
		}
		weights[k][class] += classWeights[i]
	}

	g := x.Graph()
	x = ConvertDType(x, dtypes.Float32)
	zero := ScalarZero(g, dtypes.Float32)
	scores := DynamicBroadcastInDim(zero, nil, DimensionSpecFor(x, 0), StaticDim(scoreCols))

	var walk func(key, *Node, map[key]bool)
	walk = func(k key, mask *Node, path map[key]bool) {
		i, found := index[k]
		if !found || path[k] {
			exceptions.Panicf("TreeEnsembleClassifier: missing or cyclic node %v", k)
		}
		path[k] = true
		defer delete(path, k)
		if modes[i] == "LEAF" {
			if contribution := weights[k]; contribution != nil {
				scores = Add(scores, Where(mask, InsertAxes(Const(g, contribution), 0), zero))
			}
			return
		}
		feature := features[i]
		if feature < 0 || (x.Shape().Dim(1) >= 0 && feature >= x.Shape().Dim(1)) {
			exceptions.Panicf("TreeEnsembleClassifier: invalid feature %d for input %s", feature, x.Shape())
		}
		v := Slice(x, AxisRange(), AxisRange(feature, feature+1))
		threshold := Scalar(g, dtypes.Float32, values[i])
		var predicate *Node
		switch modes[i] {
		case "BRANCH_LEQ":
			predicate = LessOrEqual(v, threshold)
		case "BRANCH_LT":
			predicate = LessThan(v, threshold)
		case "BRANCH_GTE":
			predicate = GreaterOrEqual(v, threshold)
		case "BRANCH_GT":
			predicate = GreaterThan(v, threshold)
		case "BRANCH_EQ":
			predicate = Equal(v, threshold)
		case "BRANCH_NEQ":
			predicate = NotEqual(v, threshold)
		default:
			exceptions.Panicf("TreeEnsembleClassifier: unsupported node mode %q", modes[i])
		}
		missing := missingTrue != nil && missingTrue[i] != 0
		predicate = Where(IsNaN(v), Const(g, missing), predicate)
		walk(key{k.tree, trueIDs[i]}, LogicalAnd(mask, predicate), path)
		walk(key{k.tree, falseIDs[i]}, LogicalAnd(mask, LogicalNot(predicate)), path)
	}

	treeOrder := make([]int, 0, len(roots))
	for tree := range roots {
		treeOrder = append(treeOrder, tree)
	}
	sort.Ints(treeOrder)
	for _, tree := range treeOrder {
		walk(roots[tree], Const(g, true), make(map[key]bool))
	}

	base := GetFloatsAttrOr(node, "base_values", nil)
	if len(base) != 0 {
		if len(base) != scoreCols {
			exceptions.Panicf("TreeEnsembleClassifier: invalid base_values length %d (expected %d)", len(base), scoreCols)
		}
		scores = Add(scores, InsertAxes(Const(g, base), 0))
	}

	transform := GetStringAttrOr(node, "post_transform", "NONE")
	if compactBinary {
		switch transform {
		case "NONE":
			scores = Concatenate([]*Node{OneMinus(scores), scores}, 1)
		case "LOGISTIC":
			p1 := Sigmoid(scores)
			scores = Concatenate([]*Node{OneMinus(p1), p1}, 1)
		default:
			exceptions.Panicf("TreeEnsembleClassifier: unsupported compact binary post_transform %q", transform)
		}
	} else {
		switch transform {
		case "NONE":
		case "SOFTMAX":
			scores = Softmax(scores, 1)
		case "LOGISTIC":
			scores = Sigmoid(scores)
		default:
			exceptions.Panicf("TreeEnsembleClassifier: unsupported post_transform %q", transform)
		}
	}

	predicted := onnxGather(Const(g, labels), ArgMax(scores, 1, dtypes.Int64), 0)
	if len(node.Output) > 1 && node.Output[1] != "" {
		converted[node.Output[1]] = scores
	}
	return predicted
}
