package onnxgomlx

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestModSignAdjustment(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	tests := []struct {
		name                    string
		x, y, modulo, remainder any
	}{
		{
			name:      "int32 ordinary",
			x:         []int32{5, -5, 5, -5, 6, -6, 0},
			y:         []int32{3, 3, -3, -3, -3, 3, -3},
			modulo:    []int32{2, 1, -1, -2, 0, 0, 0},
			remainder: []int32{2, -2, 2, -2, 0, 0, 0},
		},
		{
			name:      "int32 product overflow",
			x:         []int32{2, -2, 2, -2, 0},
			y:         []int32{1 << 30, 1 << 30, -1 << 30, -1 << 30, -1 << 30},
			modulo:    []int32{2, 1<<30 - 2, 2 - 1<<30, -2, 0},
			remainder: []int32{2, -2, 2, -2, 0},
		},
		{
			name:      "int64 product overflow",
			x:         []int64{1 << 32, -1 << 32, 1 << 32, -1 << 32, 0},
			y:         []int64{1 << 40, 1 << 40, -1 << 40, -1 << 40, -1 << 40},
			modulo:    []int64{1 << 32, 1<<40 - 1<<32, 1<<32 - 1<<40, -1 << 32, 0},
			remainder: []int64{1 << 32, -1 << 32, 1 << 32, -1 << 32, 0},
		},
		{
			name:      "int32 broadcast",
			x:         []int32{2, -2, 0},
			y:         int32(1 << 30),
			modulo:    []int32{2, 1<<30 - 2, 0},
			remainder: []int32{2, -2, 0},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, fmod := range []int64{0, 1} {
				exec := MustNewExec(backend, func(x, y *Node) *Node {
					m := &Model{}
					node := &protos.NodeProto{Attribute: []*protos.AttributeProto{{Name: "fmod", Type: protos.AttributeProto_INT, I: fmod}}}
					return m.convertMod(node, []*Node{x, y})
				})
				defer exec.Finalize()
				got := exec.MustExec1(test.x, test.y)
				defer got.FinalizeAll()
				want := test.modulo
				if fmod == 1 {
					want = test.remainder
				}
				require.Equal(t, want, got.Value(), "fmod=%d", fmod)
			}
		})
	}
}
