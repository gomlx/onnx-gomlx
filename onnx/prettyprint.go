package onnx

import (
	"bytes"
	"fmt"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/gomlx/onnx-gomlx/internal/togomlx"
	"maps"
	"slices"
	"strconv"
	"strings"
)

// String implements fmt.Stringer, and pretty prints model information.
func (m *Model) String() string {
	// Convenient writing to buffer that will hold result.
	var buf bytes.Buffer
	w := func(format string, args ...any) { buf.WriteString(fmt.Sprintf(format, args...)) }

	// Model header:
	w("ONNX Model:\n")
	if m.Proto.DocString != "" {
		w("# %s\n", m.Proto.DocString)
	}
	if m.Proto.ModelVersion != 0 {
		w("\tVersion:\t%d\n", m.Proto.ModelVersion)
	}
	if m.Proto.ProducerName != "" {
		w("\tProducer:\t%s / %s\n", m.Proto.ProducerName, m.Proto.ProducerVersion)
	}

	// Graph information:
	w("\t# inputs:\t%d\n", len(m.Proto.Graph.Input))
	for ii, input := range m.Proto.Graph.Input {
		w("\t\t[#%d] %s\n", ii, ppValueInfo(input))
	}
	w("\t# outputs:\t%d\n", len(m.Proto.Graph.Output))
	for ii, output := range m.Proto.Graph.Output {
		w("\t\t[#%d] %s\n", ii, ppValueInfo(output))
	}
	w("\t# nodes:\t%d\n", len(m.Proto.Graph.Node))

	// Tensors (variables):
	if len(m.Proto.Graph.Initializer) > 0 {
		w("\t# tensors (variables):\t%d\n", len(m.Proto.Graph.Initializer))
		for _, t := range m.Proto.Graph.Initializer {
			shape, _ := togomlx.Shape(t)
			w("\t\t%q: %s", t.Name, shape)
			if t.DocString != "" {
				w(" # %s", t.DocString)
			}
			w("\n")
		}
	}
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		w("\t# sparse tensors (variables):\t%d\n", len(m.Proto.Graph.SparseInitializer))
		for _, st := range m.Proto.Graph.SparseInitializer {
			shape, _ := togomlx.SparseShape(st)
			w("\t\t%q: dense shape=%d\n", st.Values.Name, shape)
		}
	}

	// List op-types used.
	opTypesSet := types.MakeSet[string]()
	for _, n := range m.Proto.Graph.Node {
		opTypesSet.Insert(n.GetOpType())
	}
	w("\tOp types:\t%#v\n", slices.Sorted(maps.Keys(opTypesSet)))

	// Training Info.
	if len(m.Proto.TrainingInfo) > 0 {
		w("\t# training info:\t%d\n", len(m.Proto.TrainingInfo))
	}

	// Extra functions:
	if len(m.Proto.Functions) > 0 {
		fnSet := types.MakeSet[string]()
		for _, f := range m.Proto.Functions {
			fnSet.Insert(f.Name)
		}
		w("\tFunctions:\t%#v\n", slices.Sorted(maps.Keys(fnSet)))
	}

	// Versions.
	w("\tIR Version:\t%d\n", m.Proto.IrVersion)
	w("\tOperator Sets:\t[")
	for ii, opSetId := range m.Proto.OpsetImport {
		if ii > 0 {
			w(", ")
		}
		if opSetId.Domain != "" {
			w("v%d (%s)", opSetId.Version, opSetId.Domain)
		} else {
			w("v%d", opSetId.Version)
		}
	}
	w("]\n")

	// Extra meta-data.
	if len(m.Proto.MetadataProps) > 0 {
		w("\tMetadata: [")
		for ii, prop := range m.Proto.MetadataProps {
			if ii > 0 {
				w(", ")
			}
			w("%s=%s", prop.Key, prop.Value)
		}
		w("]\n")
	}
	return buf.String()
}

func ppValueInfo(vi *protos.ValueInfoProto) string {
	if vi.DocString != "" {
		return fmt.Sprintf("%s: %s  # %s", vi.Name, ppType(vi.Type), vi.DocString)
	}
	return fmt.Sprintf("%s: %s", vi.Name, ppType(vi.Type))
}

func ppType(t *protos.TypeProto) string {
	if seq := t.GetSequenceType(); seq != nil {
		return ppSeqType(seq)
	} else if tensor := t.GetTensorType(); tensor != nil {
		return ppTensorType(tensor)
	}
	return "??type??"
}

func ppSeqType(seq *protos.TypeProto_Sequence) string {
	return fmt.Sprintf("(%s...)", ppType(seq.ElemType))
}

func ppTensorType(t *protos.TypeProto_Tensor) string {
	dtype, _ := togomlx.DType(protos.TensorProto_DataType(t.GetElemType()))
	dims := make([]string, len(t.GetShape().Dim))
	for ii, dProto := range t.GetShape().Dim {
		if dim, ok := dProto.GetValue().(*protos.TensorShapeProto_Dimension_DimValue); ok {
			dims[ii] = strconv.Itoa(int(dim.DimValue))
		} else if dimParam, ok := dProto.GetValue().(*protos.TensorShapeProto_Dimension_DimParam); ok {
			dims[ii] = dimParam.DimParam
		} else {
			dims[ii] = "???"
		}
	}
	if len(dims) == 0 {
		return fmt.Sprintf("(%s)", dtype)
	}
	return fmt.Sprintf("(%s) [%s]", dtype, strings.Join(dims, ", "))
}
