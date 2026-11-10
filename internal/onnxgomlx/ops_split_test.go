package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestSplitLegacySizes(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name  string
		axis  int64
		sizes []int64
		want  []any
	}{
		{
			name: "unequal columns", axis: 1, sizes: []int64{1, 3},
			want: []any{[][]float32{{1}, {5}}, [][]float32{{2, 3, 4}, {6, 7, 8}}},
		},
		{
			name: "negative column axis", axis: -1, sizes: []int64{3, 1},
			want: []any{[][]float32{{1, 2, 3}, {5, 6, 7}}, [][]float32{{4}, {8}}},
		},
		{
			name: "three columns", axis: 1, sizes: []int64{1, 1, 2},
			want: []any{[][]float32{{1}, {5}}, [][]float32{{2}, {6}}, [][]float32{{3, 4}, {7, 8}}},
		},
		{
			name: "equal rows control", axis: 0,
			want: []any{[][]float32{{1, 2, 3, 4}}, [][]float32{{5, 6, 7, 8}}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exec := MustNewExec(backend, func(x *Node) []*Node {
				node := &protos.NodeProto{Output: []string{"first", "second"}, Attribute: []*protos.AttributeProto{
					{Name: "axis", Type: protos.AttributeProto_INT, I: test.axis},
				}}
				if len(test.want) == 3 {
					node.Output = append(node.Output, "third")
				}
				if test.sizes != nil {
					node.Attribute = append(node.Attribute, &protos.AttributeProto{Name: "split", Type: protos.AttributeProto_INTS, Ints: test.sizes})
				}
				converted := make(map[string]*Node)
				convertSplit(nil, converted, node, []*Node{x})
				outputs := make([]*Node, len(node.Output))
				for i, name := range node.Output {
					outputs[i] = converted[name]
				}
				return outputs
			})
			defer exec.Finalize()
			got := exec.MustExec([][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}})
			for i, tensor := range got {
				defer tensor.FinalizeAll()
				require.Equal(t, test.want[i], tensor.Value())
			}
		})
	}
}
