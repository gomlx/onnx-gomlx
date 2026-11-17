package fusion

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/stretchr/testify/require"
)

func TestDenseActivationPreservesCustomAttributes(t *testing.T) {
	for _, test := range []struct {
		op, attr string
		value    float32
	}{{"HardSigmoid", "alpha", .20005}, {"HardSigmoid", "beta", .50005}, {"LeakyRelu", "alpha", .30005}, {"Selu", "alpha", 1.67331324}, {"Selu", "gamma", 1.05075098}} {
		node := &protos.NodeProto{OpType: test.op, Attribute: []*protos.AttributeProto{{Name: test.attr, Type: protos.AttributeProto_FLOAT, F: test.value}}}
		require.Equal(t, activation.TypeNone, denseActivationType(node), "parameterless activation kind cannot preserve %s %s=%v", test.op, test.attr, test.value)
	}
	for _, test := range []struct {
		op    string
		attrs []*protos.AttributeProto
		kind  activation.Type
	}{
		{op: "HardSigmoid", kind: activation.TypeHardSigmoid},
		{op: "LeakyRelu", attrs: []*protos.AttributeProto{{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.3}}, kind: activation.TypeLeakyRelu},
		{op: "LeakyRelu", kind: activation.TypeNone},
		{op: "Selu", kind: activation.TypeSelu},
	} {
		require.Equal(t, test.kind, denseActivationType(&protos.NodeProto{OpType: test.op, Attribute: test.attrs}))
	}
}
