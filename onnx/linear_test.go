package onnx

import (
	"github.com/stretchr/testify/require"
	"testing"
)

// Tests based on the `linear_test.onnx` minimalistic model.
func TestParse(t *testing.T) {
	m, err := ReadFile("linear_test.onnx")
	require.NoError(t, err)
	require.Len(t, m.InputsNames, 3)
	require.Len(t, m.OutputsNames, 1)

	require.Equal(t, m.OutputsShapes[0].Rank(), 1)
	require.Equal(t, "batch_size", m.OutputsShapes[0].Names[0])
}
