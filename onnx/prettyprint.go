package onnx

import (
	"bytes"
	"fmt"
	"github.com/gomlx/gomlx/types"
	"maps"
	"slices"
)

// String implements fmt.Stringer, and pretty prints model information.
func (m *Model) String() string {
	var buf bytes.Buffer
	// w writes lines to the
	w := func(format string, args ...any) {
		if len(args) == 0 {
			buf.WriteString(format)
		} else {
			buf.WriteString(fmt.Sprintf(format, args...))
		}
	}
	w("ONNX Model:\n")
	if m.Proto.DocString != "" {
		w("%s\n", m.Proto.DocString)
	}
	if m.Proto.ModelVersion != 0 {
		w("\tVersion:\t%d\n", m.Proto.ModelVersion)
	}
	if m.Proto.ProducerName != "" {
		w("\tProducer:\t%s / %s\n", m.Proto.ProducerName, m.Proto.ProducerVersion)
	}
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

	w("\t# nodes:\t%d\n", len(m.Proto.Graph.Node))
	opTypesSet := types.MakeSet[string]()
	for _, n := range m.Proto.Graph.Node {
		opTypesSet.Insert(n.GetOpType())
	}
	w("\tOp types:\t%#v\n", slices.Sorted(maps.Keys(opTypesSet)))

	if len(m.Proto.TrainingInfo) > 0 {
		w("\t# training info:\t%d\n", len(m.Proto.TrainingInfo))
	}

	if len(m.Proto.Functions) > 0 {
		fnSet := types.MakeSet[string]()
		for _, f := range m.Proto.Functions {
			fnSet.Insert(f.Name)
		}
		w("\tFunctions:\t%#v\n", slices.Sorted(maps.Keys(fnSet)))
	}

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
