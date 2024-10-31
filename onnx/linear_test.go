package onnx

import (
	"fmt"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/stretchr/testify/require"
	"testing"
)

// Tests based on the `linear_test.onnx` minimalistic model.
func TestParse(t *testing.T) {
	m, err := ReadFile("linear_test.onnx")
	require.NoError(t, err)
	require.Len(t, m.InputsNames, 1)
	require.Len(t, m.OutputsNames, 1)

	require.Equal(t, m.OutputsShapes[0].Rank(), 1)
	require.Equal(t, "batch_size", m.OutputsShapes[0].Names[0])

	sortedNodes := m.sortedGraph()
	require.Len(t, sortedNodes, 2)
	require.Equal(t, "XA", sortedNodes[0].Name)
	require.Equal(t, "Y", sortedNodes[1].Name)

	// Verify correct setting of variables.
	ctx := context.New()
	require.NoError(t, m.VariablesToContext(ctx))
	for v := range ctx.IterVariables() {
		fmt.Printf("\tVariable %q: %s\n", v.ScopeAndName(), v.Shape())
	}
	vA := ctx.In(ModelScope).GetVariable("A")
	require.NotNil(t, vA)
	require.Equal(t, 1, vA.Shape().Rank())
	require.Equal(t, 5, vA.Shape().Dim(0))
	vB := ctx.In(ModelScope).GetVariable("B")
	require.NotNil(t, vB)
	require.Equal(t, 0, vB.Shape().Rank())
}
