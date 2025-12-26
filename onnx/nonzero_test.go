package onnx

import (
	"testing"
	
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
)

func TestNonZeroBasic(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	
	// Test with a simple 2D input
	exec, err := NewExec(backend, func(g *Graph) *Node {
		// Create a test input: [[1, 0, 3], [0, 5, 0]]
		input := Const(g, [][]int32{
			{1, 0, 3},
			{0, 5, 0},
		})
		
		// This should return:
		// [[0, 0, 1],   <- row indices
		//  [0, 2, 1]]   <- col indices
		// For elements at positions (0,0)=1, (0,2)=3, (1,1)=5
		
		m := &Model{}
		result := convertNonZero(m, nil, nil, []*Node{input})
		
		return result
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}
	
	result := exec.MustExec1()
	t.Logf("NonZero result shape: %v", result.Shape())
	t.Logf("NonZero result:\n%v", result.Value())
}
