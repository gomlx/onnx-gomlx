package fusion

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestExplicitLayerNormTrailingAxes(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	all := []float32{-1.4638501, -.8783101, -.2927700, .2927700, .8783101, 1.4638501}
	rows := []float32{-1.2247449, 0, 1.2247449, -1.2247449, 0, 1.2247449}
	tests := []struct {
		name  string
		shape []int
		data  []float32
		axis  int64
		want  []float32
	}{
		{name: "all axes", shape: []int{2, 3}, data: []float32{1, 2, 3, 4, 5, 6}, axis: 0, want: all},
		{name: "negative all axes", shape: []int{2, 3}, data: []float32{1, 2, 3, 4, 5, 6}, axis: -2, want: all},
		{name: "last axis control", shape: []int{2, 3}, data: []float32{1, 2, 3, 4, 5, 6}, axis: -1, want: rows},
		{name: "two trailing axes", shape: []int{1, 2, 3}, data: []float32{1, 2, 3, 4, 5, 6}, axis: 1, want: all},
		{name: "negative two trailing axes", shape: []int{1, 2, 3}, data: []float32{1, 2, 3, 4, 5, 6}, axis: -2, want: all},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				node := &protos.NodeProto{OpType: "LayerNormalization", Input: []string{"x", "scale"}, Output: []string{"result"}, Attribute: []*protos.AttributeProto{
					{Name: "axis", Type: protos.AttributeProto_INT, I: test.axis},
					{Name: "epsilon", Type: protos.AttributeProto_FLOAT, F: 0},
				}}
				axis := MustAdjustAxis(int(test.axis), x)
				values := make([]float32, len(test.data))
				for i := range values {
					values[i] = 1
				}
				// Scale has the dimensions being normalized, as required by ONNX.
				scaleSize := 1
				for _, dim := range test.shape[axis:] {
					scaleSize *= dim
				}
				scale := tensors.FromFlatDataAndDimensions(values[:scaleSize], test.shape[axis:]...)
				defer scale.FinalizeAll()
				converted := map[string]*Node{"x": x, "scale": ConstTensor(x.Graph(), scale)}
				candidate := matchExplicitLayerNorm(node)
				require.NotNil(t, candidate)
				candidate.Emit(nil, x.Graph(), converted)
				return converted["result"]
			})
			defer exec.Finalize()
			data := tensors.FromFlatDataAndDimensions(test.data, test.shape...)
			defer data.FinalizeAll()
			got := exec.MustExec1(data)
			defer got.FinalizeAll()
			require.InDeltaSlice(t, test.want, tensors.MustCopyFlatData[float32](got), 1e-5)
		})
	}
}
