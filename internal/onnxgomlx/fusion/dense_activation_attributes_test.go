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
		op   string
		kind activation.Type
	}{{"HardSigmoid", activation.TypeHardSigmoid}, {"Selu", activation.TypeSelu}} {
		require.Equal(t, test.kind, denseActivationType(&protos.NodeProto{OpType: test.op}))
	}
}
