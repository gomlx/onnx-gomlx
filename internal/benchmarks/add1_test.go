package benchmarks

import (
	"flag"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/janpfeifer/must"
	ort "github.com/yalue/onnxruntime_go"
	"os"
	"testing"
)

import _ "github.com/gomlx/gomlx/backends/xla"

var flagVerify = flag.Bool("verify", false, "verify generated values")

var Add1Shapes = []shapes.Shape{
	shapes.Make(dtypes.Float32, 1, 1),
	shapes.Make(dtypes.Float32, 10, 10),
	shapes.Make(dtypes.Float32, 100, 100),
	shapes.Make(dtypes.Float32, 1000, 1000),
}

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// BenchmarkAdd1XLAExec based on the `add1.onnx` minimalistic model: it adds 1 to a rank-2 tensor.
// We expect this time to be dominated by the tensor conversion inputs/outputs.
//
// GoMLX: v0.15.3 / GoPJRT: 0.4.9
// Results with CPU: go test . -test.bench=.
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1XLAExec/(Float32)[1_1]-24             68131             15034 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[10_10]-24           73905             15140 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[100_100]-24         48411             24641 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[1000_1000]-24        1280            897389 ns/op
//
// Results with GPU (NVidia 2080ti): XLA_BACKEND="xla:cuda" go test . -test.bench=.
//
//	BenchmarkAdd1XLAExec/(Float32)[1_1]-24             80978             14561 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[10_10]-24           70262             15892 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[100_100]-24         48589             24244 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[1000_1000]-24        1215            903928 ns/op
func BenchmarkAdd1XLAExec(b *testing.B) {
	model := must.M1(onnx.ReadFile("add1.onnx"))

	// Check conversion.
	var isDuringBenchmark bool
	backend := graphtest.BuildTestBackend()
	exec := graph.NewExec(backend, func(x *graph.Node) *graph.Node {
		if isDuringBenchmark {
			exceptions.Panicf("Graph building function called during benchmark: this shouldn't happen, as all graphs should have been built in startup")
		}
		g := x.Graph()
		outputs := model.CallGraph(nil, g, map[string]*graph.Node{"X": x})
		return outputs[0]
	})

	// Pre-allocate tensors.
	numShapes := len(Add1Shapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range Add1Shapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		tensors.MutableFlatData[float32](x, func(flat []float32) {
			for ii := range flat {
				flat[ii] = v
			}
		})

		tmpOutput := exec.Call(graph.DonateTensorBuffer(x, backend))[0]
		outputTensors[shapeIdx].CopyFrom(tmpOutput) // Re-use local storage for contents.
		tmpOutput.FinalizeAll()

		// Verify results
		if *flagVerify {
			vWant := v + 1
			tensors.ConstFlatData[float32](outputTensors[shapeIdx], func(flat []float32) {
				for _, vOut := range flat {
					if vOut != vWant {
						exceptions.Panicf("Wanted %f, got %f instead!?", vWant, vOut)
					}
				}
			})
		}
	}

	// Warmup for each shape.
	for shapeIdx := range Add1Shapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	isDuringBenchmark = true
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

// BenchmarkAdd1XLADirect based on the `add1.onnx` minimalistic model: it adds 1 to a rank-2 tensor.
// We expect this time to be dominated by the tensor conversion inputs/outputs.
//
// GoMLX: v0.15.3 / GoPJRT: 0.4.9
// Results with CPU: go test . -test.bench=.
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1XLADirect/(Float32)[1_1]-24                   81979             14303 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[10_10]-24                 79243             15004 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[100_100]-24               48580             24136 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[1000_1000]-24              1294            911609 ns/op
//
// Results with GPU (NVidia 2080ti): XLA_BACKEND="xla:cuda" go test . -test.bench=.
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1XLADirect/(Float32)[1_1]-24                   85447             13832 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[10_10]-24                 72364             14863 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[100_100]-24               49515             22744 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[1000_1000]-24              1207            916167 ns/op
func BenchmarkAdd1XLADirect(b *testing.B) {
	model := must.M1(onnx.ReadFile("add1.onnx"))

	// Create executables.
	backend := graphtest.BuildTestBackend()
	numShapes := len(Add1Shapes)
	graphPerShape := make([]*graph.Graph, numShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range Add1Shapes {
		g := graph.NewGraph(backend, fmt.Sprintf("Graph #%d", shapeIdx))
		x := graph.Parameter(g, "X", s)
		y := model.CallGraph(nil, g, map[string]*graph.Node{"X": x})[0]
		g.Compile(y)
		graphPerShape[shapeIdx] = g
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		tensors.MutableFlatData[float32](x, func(flat []float32) {
			for ii := range flat {
				flat[ii] = v
			}
		})
		xBuf := x.Buffer(backend)

		// Run with given input.
		g := graphPerShape[shapeIdx]
		tmpOutput := g.RunWithBuffers(
			[]backends.Buffer{xBuf},
			[]bool{false})[0]
		//[]backends.Buffer{x.DonateBuffer(backend)},
		//[]bool{true})[0]
		//outputTensors[shapeIdx].CopyFrom(tmpOutput)
		tmpOutput.FinalizeAll()

		// Verify results
		if *flagVerify {
			vWant := v + 1
			tensors.ConstFlatData[float32](outputTensors[shapeIdx], func(flat []float32) {
				for _, vOut := range flat {
					if vOut != vWant {
						exceptions.Panicf("Wanted %f, got %f instead!?", vWant, vOut)
					}
				}
			})
		}
	}

	// Warmup for each shape.
	for shapeIdx := range Add1Shapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}

// BenchmarkAdd1ONNXRuntime
//
// # ONNX Runtime v1.20.1
//
// Results with CPU:
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1ONNXRuntime/(Float32)[1_1]-24               1581852               748.7 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[10_10]-24             1397054               856.4 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[100_100]-24            163327              7051 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[1000_1000]-24            1921            584694 ns/op
//
// Results with GPU (NVidia 2080ti):
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1ONNXRuntime/(Float32)[1_1]-24               1476309               810.2 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[10_10]-24             1321719               905.3 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[100_100]-24            168486              7001 ns/op
//	BenchmarkAdd1ONNXRuntime/(Float32)[1000_1000]-24            1940            621650 ns/op
func BenchmarkAdd1ONNXRT(b *testing.B) {
	ortPath := os.Getenv("ORT_SO_PATH")
	if ortPath == "" {
		exceptions.Panicf("Please set environment ORT_SO_PATH with the path to your ONNX Runtime dynamic linked library")
	}
	ort.SetSharedLibraryPath(ortPath)
	must.M(ort.InitializeEnvironment())
	defer func() { _ = ort.DestroyEnvironment() }()

	// Create a session for each tensor shape:
	numShapes := len(Add1Shapes)
	sessions := make([]*ort.AdvancedSession, 0, numShapes)
	inputsPerShape := make([]*ort.Tensor[float32], 0, numShapes)
	outputsPerShape := make([]*ort.Tensor[float32], 0, numShapes)
	for _, s := range Add1Shapes {
		inputData := make([]float32, s.Size())
		dims64 := sliceMap(s.Dimensions, func(dim int) int64 { return int64(dim) })
		inputShape := ort.NewShape(dims64...)
		inputTensor := must.M1(ort.NewTensor(inputShape, inputData))
		inputsPerShape = append(inputsPerShape, inputTensor)
		outputTensor := must.M1(ort.NewEmptyTensor[float32](inputShape))
		outputsPerShape = append(outputsPerShape, outputTensor)
		session := must.M1(ort.NewAdvancedSession(
			"add1.onnx",
			[]string{"X"},
			[]string{"Y"},
			[]ort.Value{inputTensor},
			[]ort.Value{outputTensor},
			nil))
		sessions = append(sessions, session)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		// Set value.
		input := inputsPerShape[shapeIdx]
		data := input.GetData()
		for ii := range data {
			data[ii] = v
		}

		// Run session
		session := sessions[shapeIdx]
		must.M(session.Run())

		// Check output.
		if *flagVerify {
			output := outputsPerShape[shapeIdx]
			vWant := v + 1
			data = output.GetData()
			for idx, vOut := range data {
				if vOut != vWant {
					exceptions.Panicf("Value #%d: Wanted %f, got %f instead!?", idx, vWant, vOut)
				}
			}
		}
	}

	// Warmup for each shape.
	for idxShape := range Add1Shapes {
		for i := range 10 {
			benchShape(float32(i), idxShape)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for idxShape, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), idxShape)
			}
		})
	}
}

func BenchmarkAdd1PureGo(b *testing.B) {
	// Pre-allocate tensors.
	numShapes := len(Add1Shapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range Add1Shapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int) {
		x := inputTensors[shapeIdx]
		y := outputTensors[shapeIdx]
		tensors.MutableFlatData[float32](x, func(flat []float32) {
			for ii := range flat {
				flat[ii] = v
			}
		})
		tensors.ConstFlatData[float32](x, func(xFlat []float32) {
			tensors.MutableFlatData[float32](y, func(yFlat []float32) {
				for ii, v := range xFlat {
					yFlat[ii] = v
				}
			})
		})
	}

	// Warmup for each shape.
	for shapeIdx := range Add1Shapes {
		for i := range 10 {
			benchShape(float32(i), shapeIdx)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx)
			}
		})
	}
}
