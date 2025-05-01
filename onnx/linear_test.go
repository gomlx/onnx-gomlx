package onnx

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"testing"
)

import _ "github.com/gomlx/gomlx/backends/default"

// TestEndToEnd based on the `linear_test.onnx` minimalistic model.
// Only a couple of ops tested, but from end-to-end, including if changes can be saved
// again (ContextToONNX).
func TestEndToEnd(t *testing.T) {
	model, err := ReadFile("linear_test.onnx")
	fmt.Printf("%s\n", model)
	require.NoError(t, err)
	require.Len(t, model.InputsNames, 1)
	require.Equal(t, "X", model.InputsNames[0])
	require.Len(t, model.OutputsNames, 1)
	require.Equal(t, "Y", model.OutputsNames[0])

	require.Equal(t, model.OutputsShapes[0].Rank(), 1)
	require.Equal(t, "batch_size", model.OutputsShapes[0].Names[0])

	// Verify correct setting of variables.
	ctx := context.New()
	require.NoError(t, model.VariablesToContext(ctx))
	for v := range ctx.IterVariables() {
		fmt.Printf("\tVariable %q: %s\n", v.ScopeAndName(), v.Value())
	}
	vA := ctx.In(ModelScope).GetVariable("A")
	require.NotNil(t, vA)
	require.Equal(t, 1, vA.Shape().Rank())
	require.Equal(t, 5, vA.Shape().Dim(0))
	require.Equal(t, []float32{100, 10, 1, 0.1, 0.01}, tensors.CopyFlatData[float32](vA.Value()))
	vB := ctx.In(ModelScope).GetVariable("B")
	require.NotNil(t, vB)
	require.Equal(t, 0, vB.Shape().Rank())
	require.Equal(t, float32(7000), tensors.ToScalar[float32](vB.Value()))

	// Check conversion.
	backend := graphtest.BuildTestBackend()
	y := context.ExecOnce(backend, ctx, func(ctx *context.Context, x *graph.Node) *graph.Node {
		g := x.Graph()
		outputs := model.CallGraph(ctx, g, map[string]*graph.Node{"X": x})
		vB = ctx.In(ModelScope).GetVariable("B")
		vB.SetValueGraph(graph.OnePlus(vB.ValueGraph(g)))
		return outputs[0]
	}, [][]float32{{1, 2, 3, 4, 5}}) // BatchSize = 1
	require.NoError(t, y.Shape().Check(dtypes.Float32, 1))
	require.InDeltaSlice(t, []float32{7123.45}, tensors.CopyFlatData[float32](y), 0.1)

	// Save change of variable "B" to ONNX model.
	require.NoError(t, model.ContextToONNX(ctx))
	tensorProto, found := model.variableNameToValue["B"]
	require.True(t, found, "Didn't find B variable")
	require.Equal(t, []float32{7001}, tensorProto.FloatData, "ONNX variable B initial value was not updated.")
}
