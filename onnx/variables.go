package onnx

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/context"
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
//
// See also ContextToONNX, if after converting and fine-tuning an ONNX model, you want to update its weights.
func (m *Model) VariablesToContext(ctx *context.Context) error {
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.VariablesToContext does not support ONNX SparseTensors")
	}
	ctx = ctx.In(ModelScope).Checked(false)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensor, err := tensorToGoMLX(tensorProto)
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

// ContextToONNX converts the variables in the context back to the ONNX model.
// Do this before saving the ONNX model back to disk.
//
// It's the inverse of VariablesToContext, and the context given must be set in the same scope as when
// VariablesToContext was first called.
//
// Only those variables present in the original ONNX model are converted -- so new variables (e.g.: optimizers (ADAM)
// moving averages) are converted.
func (m *Model) ContextToONNX(ctx *context.Context) error {
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.VariablesToContext does not support ONNX SparseTensors")
	}
	ctx = ctx.In(ModelScope).Checked(false)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensorName := SafeVarName(tensorProto.Name)
		gomlxVar := ctx.GetVariable(tensorName)
		if gomlxVar == nil {
			return errors.Errorf("ONNX variable '%s' not found in context scope %q --"+
				" maybe you used a different scope when Model.VariablesToContext() was used ?",
				tensorName, ctx.Scope())
		}
		err := TensorValueToONNX(gomlxVar.Value(), tensorProto)
		if err != nil {
			return errors.WithMessagef(err, "Model.ContextToONNX() converting tensor %q", tensorName)
		}
	}
	return nil

}
