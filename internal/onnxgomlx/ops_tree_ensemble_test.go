// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"math"
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func floatsAttr(name string, values ...float32) *protos.AttributeProto {
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_FLOATS, Floats: values}
}

func stringsAttr(name string, values ...string) *protos.AttributeProto {
	raw := make([][]byte, len(values))
	for i, v := range values {
		raw[i] = []byte(v)
	}
	return &protos.AttributeProto{Name: name, Type: protos.AttributeProto_STRINGS, Strings: raw}
}

func TestTreeEnsembleClassifier(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	m := createTestModelWithDTypePromoConfig(false, false)

	t.Run("multiclass_softmax_missing_nan_and_dynamic_batch", func(t *testing.T) {
		// Single tree with non-contiguous node IDs (0 -> true:10, false:20), missing_value_tracks_true=1.
		// Leaf 10 adds 2.0 to class 0; leaf 20 adds 3.0 to class 2.
		node := &protos.NodeProto{
			OpType: "TreeEnsembleClassifier",
			Input:  []string{"x"},
			Output: []string{"label", "probs"},
			Attribute: []*protos.AttributeProto{
				intsAttr("classlabels_int64s", 100, 200, 300),
				intsAttr("nodes_treeids", 0, 0, 0),
				intsAttr("nodes_nodeids", 0, 10, 20),
				intsAttr("nodes_featureids", 0, 0, 0),
				floatsAttr("nodes_values", 0.5, 0, 0),
				stringsAttr("nodes_modes", "BRANCH_LEQ", "LEAF", "LEAF"),
				intsAttr("nodes_truenodeids", 10, 0, 0),
				intsAttr("nodes_falsenodeids", 20, 0, 0),
				intsAttr("nodes_missing_value_tracks_true", 1, 0, 0),
				intsAttr("class_treeids", 0, 0),
				intsAttr("class_nodeids", 10, 20),
				intsAttr("class_ids", 0, 2),
				floatsAttr("class_weights", 2.0, 3.0),
				stringAttr("post_transform", "SOFTMAX"),
			},
		}

		exec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
			outs := convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x})
			return outs[0], outs[1]
		}).WithDynamicAxes([]string{"batch", ""})
		defer exec.Finalize()

		outs := exec.MustExec([][]float32{
			{0.2, 9.0},               // <= 0.5 -> leaf 10 -> class 100
			{0.9, 1.0},               // > 0.5  -> leaf 20 -> class 300
			{float32(math.NaN()), 0}, // NaN tracks true -> leaf 10 -> class 100
		})
		for _, o := range outs {
			defer o.FinalizeAll()
		}
		require.Equal(t, []int64{100, 300, 100}, outs[0].Value())
		probs := outs[1].Value().([][]float32)
		for row := range probs {
			sum := probs[row][0] + probs[row][1] + probs[row][2]
			require.InDelta(t, 1.0, sum, 1e-5)
		}
		require.Greater(t, probs[0][0], probs[0][2])
		require.Greater(t, probs[1][2], probs[1][0])
	})

	t.Run("compact_binary_logistic", func(t *testing.T) {
		// Binary classifier emitting compact single-score weights (class_id=0 in ONNX-ML) with LOGISTIC:
		// probs = [1 - sigmoid(score), sigmoid(score)].
		node := &protos.NodeProto{
			OpType: "TreeEnsembleClassifier",
			Input:  []string{"x"},
			Output: []string{"label", "probs"},
			Attribute: []*protos.AttributeProto{
				intsAttr("classlabels_int64s", 0, 1),
				intsAttr("nodes_treeids", 0, 0, 0),
				intsAttr("nodes_nodeids", 0, 1, 2),
				intsAttr("nodes_featureids", 0, 0, 0),
				floatsAttr("nodes_values", 0.0, 0, 0),
				stringsAttr("nodes_modes", "BRANCH_GT", "LEAF", "LEAF"),
				intsAttr("nodes_truenodeids", 1, 0, 0),
				intsAttr("nodes_falsenodeids", 2, 0, 0),
				intsAttr("class_treeids", 0, 0),
				intsAttr("class_nodeids", 1, 2),
				intsAttr("class_ids", 0, 0),
				floatsAttr("class_weights", 2.0, -2.0),
				stringAttr("post_transform", "LOGISTIC"),
			},
		}

		exec := MustNewExec(backend, func(x *Node) (*Node, *Node) {
			outs := convertTestNode(m, x.Graph(), node, map[string]*Node{"x": x})
			return outs[0], outs[1]
		}).WithDynamicAxes([]string{"batch", ""})
		defer exec.Finalize()

		outs := exec.MustExec([][]float32{{-1.0}, {1.0}})
		for _, o := range outs {
			defer o.FinalizeAll()
		}
		require.Equal(t, []int64{0, 1}, outs[0].Value())
		probs := outs[1].Value().([][]float32)
		pPos := float32(1.0 / (1.0 + math.Exp(-2.0)))
		require.InDeltaSlice(t, []float32{pPos, 1 - pPos}, probs[0], 1e-5)
		require.InDeltaSlice(t, []float32{1 - pPos, pPos}, probs[1], 1e-5)
	})
}
