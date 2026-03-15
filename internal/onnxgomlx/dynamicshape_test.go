package onnxgomlx

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateInputs(t *testing.T) {
	m := &Model{
		InputsNames: []string{"i0", "i1"},
		InputsShapes: []DynamicShape{
			{
				DType:      dtypes.Float32,
				Dimensions: []int{-1, -1},
				Names:      []string{"batch_size", "feature_dim"},
			},
			{
				DType:      dtypes.Int32,
				Dimensions: []int{-1, 3},
				Names:      []string{"batch_size", "other"},
			},
		},
	}

	// Example valid input, batch_size=5
	require.NoError(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32, 5, 3)))

	// Wrong dtype:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7, 1),
		shapes.Make( /**/ dtypes.Int64, 5, 3)))

	// Wrong rank:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7 /**/, 1),
		shapes.Make(dtypes.Int32, 5, 3)))

	// Fixed dimension not matching:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32, 5 /**/, 4)))

	// Dynamic dimension not matching:
	require.Error(t, m.ValidateInputs(
		shapes.Make(dtypes.Float32, 5, 7),
		shapes.Make(dtypes.Int32 /**/, 6, 3)))
}

// mockBackend implements just enough of backends.Backend for DynamicAxesConfig tests.
type mockBackend struct {
	backends.Backend
	dynamicAxes bool
}

func (b *mockBackend) Capabilities() backends.Capabilities {
	return backends.Capabilities{DynamicAxes: b.dynamicAxes}
}

func TestDynamicAxesConfig(t *testing.T) {
	m := &Model{
		InputsNames: []string{"input", "mask"},
		InputsShapes: []DynamicShape{
			{DType: dtypes.Float32, Dimensions: []int{-1, 128}, Names: []string{"batch", "128"}},
			{DType: dtypes.Int32, Dimensions: []int{-1, -1}, Names: []string{"batch", "seq_len"}},
		},
	}

	// Backend that supports dynamic axes → returns config.
	dynBackend := &mockBackend{dynamicAxes: true}
	config := m.DynamicAxesConfig(dynBackend)
	require.NotNil(t, config)
	assert.Equal(t, [][]string{
		{"batch", ""},
		{"batch", "seq_len"},
	}, config)

	// Backend that does NOT support dynamic axes → nil.
	staticBackend := &mockBackend{dynamicAxes: false}
	assert.Nil(t, m.DynamicAxesConfig(staticBackend))

	// Nil backend → nil.
	assert.Nil(t, m.DynamicAxesConfig(nil))

	// ForceStaticShapes → nil even with dynamic backend.
	m.forceStaticShapes = true
	assert.Nil(t, m.DynamicAxesConfig(dynBackend))
	m.forceStaticShapes = false

	// All-static model → nil (no dynamic axes).
	staticModel := &Model{
		InputsNames: []string{"x"},
		InputsShapes: []DynamicShape{
			{DType: dtypes.Float32, Dimensions: []int{3, 4}, Names: []string{"3", "4"}},
		},
	}
	assert.Nil(t, staticModel.DynamicAxesConfig(dynBackend))

	// Unnamed dynamic dimension → gets positional name.
	unnamedModel := &Model{
		InputsNames: []string{"x"},
		InputsShapes: []DynamicShape{
			{DType: dtypes.Float32, Dimensions: []int{-1, 10}, Names: []string{"?", "10"}},
		},
	}
	config = unnamedModel.DynamicAxesConfig(dynBackend)
	require.NotNil(t, config)
	assert.Equal(t, [][]string{{"dyn_0", ""}}, config)
}
