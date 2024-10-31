package onnx

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"github.com/pkg/errors"
	"strings"
)

// This file defines importing variables from ONNX and (TODO) saving them back to the ONNX model file.

// ModelScope is the default model scope to use when for the ONNX model variables when converting to GoMLX.
var ModelScope = "ONNX"

// This file defines the methods that build the computation graph using GoMLX.

// VariablesToContext will create variables in the context (within scope ModelScope) from
// all variables present in the model initializer list.
//
// Call this once in your context, before using the model with Model.CallGraph.
// Alternatively, if you have already checkpoint-ed your model, load the variables from a checkpoint and don't call this.
func (m *Model) VariablesToContext(ctx *context.Context) error {
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.VariablesToContext does not support ONNX SparseTensors")
	}
	ctx = ctx.In(ModelScope).Checked(false)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensor, err := togomlx.Tensor(tensorProto)
		if err != nil {
			return errors.WithMessagef(err, "Model.VariablesToContext()")
		}
		tensorName := SafeVarName(tensorProto.Name)
		ctx.VariableWithValue(tensorName, tensor)
	}
	return nil
}

// SafeVarName converts an ONNX variable name to a GoMLX safe variable name by replacing the scope separator with a "|".
func SafeVarName(onnxName string) (gomlxName string) {
	return strings.ReplaceAll(onnxName, context.ScopeSeparator, "|")
}
