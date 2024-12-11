package benchmarks

import (
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

	_ "github.com/gomlx/gomlx/backends/xla"
)

// Benchmark results for this file using:
//
// - GoMLX: v0.15.4rc / GoPJRT: 0.4.10rc
// - ONNX Runtime v1.20.1
//
// Results with CPU: go test . -test.bench=.
//
//	cpu: 12th Gen Intel(R) Core(TM) i9-12900K
//	BenchmarkAdd1XLAExec/(Float32)[1_1]-24            352744              2848 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[10_10]-24          380127              2691 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[100_100]-24        281173              4254 ns/op
//	BenchmarkAdd1XLAExec/(Float32)[1000_1000]-24       28095             42152 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[1_1]-24          403384              2771 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[10_10]-24        450536              2616 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[100_100]-24      292851              4083 ns/op
//	BenchmarkAdd1XLADirect/(Float32)[1000_1000]-24     28430             42605 ns/op
//	BenchmarkAdd1ONNXRT/(Float32)[1_1]-24            1537388               780.8 ns/op
//	BenchmarkAdd1ONNXRT/(Float32)[10_10]-24          1485952               805.4 ns/op
//	BenchmarkAdd1ONNXRT/(Float32)[100_100]-24         849921              1410 ns/op
//	BenchmarkAdd1ONNXRT/(Float32)[1000_1000]-24        78459             14764 ns/op
//	BenchmarkAdd1PureGo/(Float32)[1_1]-24           39167090                30.53 ns/op
//	BenchmarkAdd1PureGo/(Float32)[10_10]-24         20961847                54.28 ns/op
//	BenchmarkAdd1PureGo/(Float32)[100_100]-24         551452              2029 ns/op
//	BenchmarkAdd1PureGo/(Float32)[1000_1000]-24         5839            197961 ns/op

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
// We try not to count the time for tensor transfers in and out.
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
	benchShape := func(v float32, shapeIdx int, isWarmUp bool) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		if isWarmUp {
			tensors.MutableFlatData[float32](x, func(flat []float32) {
				for ii := range flat {
					flat[ii] = v
				}
			})
		}

		tmpOutput := exec.Call(x)[0]
		if isWarmUp {
			outputTensors[shapeIdx].CopyFrom(tmpOutput) // Re-use local storage for contents.
		}
		tmpOutput.FinalizeAll()

		// Verify results
		if isWarmUp {
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
			benchShape(float32(i), shapeIdx, true)
		}
	}

	// Reset timer and start actual benchmark
	isDuringBenchmark = true
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx, false)
			}
		})
	}
}

// BenchmarkAdd1XLADirect based on the `add1.onnx` minimalistic model: it adds 1 to a rank-2 tensor.
// We try not to count the time for tensor transfers in and out.
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
		if shapeIdx == 3 {
			fmt.Printf("Add1 graph:\n%s\n", g.String())
		}
		g.Compile(y)
		graphPerShape[shapeIdx] = g
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run test for a shape
	benchShape := func(v float32, shapeIdx int, isWarmUp bool) {
		// Set input to value of v.
		x := inputTensors[shapeIdx]
		if isWarmUp {
			tensors.MutableFlatData[float32](x, func(flat []float32) {
				for ii := range flat {
					flat[ii] = v
				}
			})
		}
		xBuf := x.Buffer(backend)

		// Run with given input.
		g := graphPerShape[shapeIdx]
		tmpOutput := g.RunWithBuffers(
			[]backends.Buffer{xBuf},
			[]bool{false})[0]
		//[]backends.Buffer{x.DonateBuffer(backend)},
		//[]bool{true})[0]
		if isWarmUp {
			outputTensors[shapeIdx].CopyFrom(tmpOutput)
		}
		tmpOutput.FinalizeAll()

		// Verify results
		if isWarmUp {
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
			benchShape(float32(i), shapeIdx, true)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx, false)
			}
		})
	}
}

// BenchmarkAdd1ONNXRuntime
// We try not to count the time for tensor transfers in and out.
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
	benchShape := func(v float32, shapeIdx int, isWarmUp bool) {
		// Set value.
		input := inputsPerShape[shapeIdx]
		if isWarmUp {
			data := input.GetData()
			for ii := range data {
				data[ii] = v
			}
		}

		// Run session
		session := sessions[shapeIdx]
		must.M(session.Run())

		// Check output.
		if isWarmUp {
			output := outputsPerShape[shapeIdx]
			vWant := v + 1
			data := output.GetData()
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
			benchShape(float32(i), idxShape, true)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for idxShape, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), idxShape, false)
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
	benchShape := func(v float32, shapeIdx int, isWarmUp bool) {
		x := inputTensors[shapeIdx]
		y := outputTensors[shapeIdx]
		if isWarmUp {
			tensors.MutableFlatData[float32](x, func(flat []float32) {
				for ii := range flat {
					flat[ii] = v
				}
			})
		}
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
			benchShape(float32(i), shapeIdx, true)
		}
	}

	// Reset timer and start actual benchmark
	b.ResetTimer()

	// Test each shape.
	for shapeIdx, s := range Add1Shapes {
		b.Run(s.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				benchShape(float32(i), shapeIdx, false)
			}
		})
	}
}
