package onnxgomlx

import (
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestResizeCoordinateSemantics(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name, mode, coordinate, nearest string
		data, want                      []float32
		size                            int64
		scales                          []float32
	}{
		{name: "nearest floor", mode: "nearest", coordinate: "half_pixel", nearest: "floor", data: []float32{0, 10, 20}, size: 5, want: []float32{0, 0, 10, 10, 20}},
		{name: "nearest ceil", mode: "nearest", coordinate: "half_pixel", nearest: "ceil", data: []float32{0, 10, 20}, size: 5, want: []float32{0, 10, 10, 20, 20}},
		{name: "tie prefer floor", mode: "nearest", coordinate: "half_pixel", nearest: "round_prefer_floor", data: []float32{0, 10}, size: 3, want: []float32{0, 0, 10}},
		{name: "tie prefer ceil", mode: "nearest", coordinate: "half_pixel", nearest: "round_prefer_ceil", data: []float32{0, 10}, size: 3, want: []float32{0, 10, 10}},
		{name: "align corners ceil", mode: "nearest", coordinate: "align_corners", nearest: "ceil", data: []float32{0, 10, 20}, size: 5, want: []float32{0, 10, 10, 20, 20}},
		{name: "asymmetric floor", mode: "nearest", coordinate: "asymmetric", nearest: "floor", data: []float32{0, 10, 20}, size: 5, want: []float32{0, 0, 10, 10, 20}},
		{name: "pytorch singleton linear", mode: "linear", coordinate: "pytorch_half_pixel", data: []float32{0, 10, 20, 30}, size: 1, want: []float32{0}},
		{name: "pytorch singleton nearest", mode: "nearest", coordinate: "pytorch_half_pixel", nearest: "floor", data: []float32{0, 10, 20, 30}, size: 1, want: []float32{0}},
		{name: "half pixel singleton", mode: "linear", coordinate: "half_pixel", data: []float32{0, 10, 20, 30}, size: 1, want: []float32{15}},
		{name: "fractional scale", mode: "linear", coordinate: "half_pixel", data: []float32{0, 10, 20}, scales: []float32{1.7}, want: []float32{0, 3.823529, 9.705882, 15.588235, 20}},
		{name: "scale preserves output size", mode: "linear", coordinate: "half_pixel", data: []float32{0, 10, 20}, scales: []float32{1.1}, want: []float32{0, 10, 20}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) *Node {
				g := x.Graph()
				node := &protos.NodeProto{OpType: "Resize", Input: []string{"x", "", "", "sizes"}, Output: []string{"result"}, Attribute: []*protos.AttributeProto{
					{Name: "mode", Type: protos.AttributeProto_STRING, S: []byte(test.mode)},
					{Name: "coordinate_transformation_mode", Type: protos.AttributeProto_STRING, S: []byte(test.coordinate)},
					{Name: "nearest_mode", Type: protos.AttributeProto_STRING, S: []byte(test.nearest)},
				}}
				converted := map[string]*Node{"sizes": Const(g, []int64{test.size})}
				if test.scales != nil {
					node.Input = []string{"x", "", "scales"}
					converted = map[string]*Node{"scales": Const(g, test.scales)}
				}
				m := &Model{VariableNameToValue: make(map[string]*protos.TensorProto), NodeOutputToNode: make(map[string]*protos.NodeProto)}
				return convertResize(m, converted, node, []*Node{x})
			})
			defer exec.Finalize()
			got := exec.MustExec1(test.data)
			defer got.FinalizeAll()
			require.InDeltaSlice(t, test.want, got.Value(), 1e-4)
		})
	}
}
