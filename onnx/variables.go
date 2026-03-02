package onnx

import (
	"strings"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
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
	reader := m.getExternalDataReader()
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensor, err := tensorToGoMLXWithBaseDir(m.backend, tensorProto, m.baseDir(), reader)
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

// FreeUnusedVariables removes initializers that are not referenced by any node input
// in the graph. This is useful after fusion detection creates new combined initializers,
// leaving the original individual weights unused.
func (m *Model) FreeUnusedVariables() {
	// Collect all names referenced as node inputs.
	referenced := make(map[string]bool)
	for _, node := range m.Proto.Graph.Node {
		for _, input := range node.Input {
			if input != "" {
				referenced[input] = true
			}
		}
	}

	// Also keep initializers referenced by fusion external inputs.
	for _, cand := range m.detectedFusions {
		for _, name := range cand.ExternalInputs() {
			referenced[name] = true
		}
	}

	// Filter initializers.
	kept := m.Proto.Graph.Initializer[:0]
	for _, init := range m.Proto.Graph.Initializer {
		if referenced[init.Name] {
			kept = append(kept, init)
		} else {
			delete(m.variableNameToValue, init.Name)
		}
	}
	m.Proto.Graph.Initializer = kept
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
	ctx = ctx.In(ModelScope)
	for _, tensorProto := range m.Proto.Graph.Initializer {
		tensorName := SafeVarName(tensorProto.Name)
		gomlxVar := ctx.GetVariable(tensorName)
		if gomlxVar == nil {
			return errors.Errorf("ONNX variable '%s' not found in context scope %q --"+
				" maybe you used a different scope when Model.VariablesToContext() was used ?",
				tensorName, ctx.Scope())
		}
		gomlxValue, err := gomlxVar.Value()
		if err != nil {
			return errors.WithMessagef(err, "Model.ContextToONNX() getting value of variable %q", tensorName)
		}
		err = TensorValueToONNX(gomlxValue, tensorProto)
		if err != nil {
			return errors.WithMessagef(err, "Model.ContextToONNX() converting tensor %q", tensorName)
		}
	}
	return nil
}
