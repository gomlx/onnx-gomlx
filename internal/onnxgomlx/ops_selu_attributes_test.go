package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestSeluPreservesExplicitNearDefaultGamma(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	for _, gamma := range []float32{1.05070098, 1.05075098, 1.05065098, 1.05079098} {
		exec := MustNewExec(backend, func(x *Node) *Node {
			m := &Model{}
			node := &protos.NodeProto{OpType: "Selu", Attribute: []*protos.AttributeProto{{Name: "gamma", Type: protos.AttributeProto_FLOAT, F: gamma}}}
			return m.convertSelu(node, []*Node{x})
		})
		got := exec.MustExec1([]float32{1})
		require.Equal(t, []float32{gamma}, got.Value(), "For x=1, SELU output must equal explicitly supplied float32 gamma")
		got.FinalizeAll()
		exec.Finalize()
	}
}
