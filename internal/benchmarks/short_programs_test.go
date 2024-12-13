package benchmarks

// Benchmark results for this file using:
//
// - GoMLX: v0.15.4rc / GoPJRT: 0.4.10rc
// - ONNX Runtime v1.20.1
// - Command used:
//	go test . -test.bench=.
//
// - cpu: 12th Gen Intel(R) Core(TM) i9-12900K: GOMAXPROC=24, 12 cores (4P, 8E), 24 hyperthread cores.
// - results: https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=0#gid=0

import (
	"fmt"
	"github.com/chewxy/math32"
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
	"k8s.io/klog/v2"
	"os"
	"runtime"
	"sync"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func init() {
	klog.InitFlags(nil)
}

var (
	TestShapes = []shapes.Shape{
		shapes.Make(dtypes.Float32, 1, 1),
		shapes.Make(dtypes.Float32, 10, 10),
		shapes.Make(dtypes.Float32, 100, 100),
		shapes.Make(dtypes.Float32, 1000, 1000),
	}
	numShapes = len(TestShapes)

	TestPrograms = [][2]string{
		{"add1.onnx", "f(x)=x+1"},
		{"add1div2.onnx", "f(x)=(x+1)/2"},
		{"sqrt_add1div2.onnx", "f(x)=Sqrt((x+1)/2)"},
	}
	numPrograms = len(TestPrograms)

	// testGoPrograms is a Go version of the TestPrograms, as a per-element function.
	testGoPrograms = []goVectorFunc{
		parallelizeGoVectorFunc(func(inputs, outputs []float32) {
			for ii, v := range inputs {
				outputs[ii] = v + 1
			}
		}),
		parallelizeGoVectorFunc(func(inputs, outputs []float32) {
			for ii, v := range inputs {
				outputs[ii] = (v + 1) * 0.5
			}
		}),
		parallelizeGoVectorFunc(func(inputs, outputs []float32) {
			for ii, v := range inputs {
				outputs[ii] = math32.Sqrt((v + 1) * 0.5)
			}
		}),
	}

	benchmarkNameSuffix = "|GOMAXPROCS"
)

// goVectorFunc defines the signature for functions that process slices.
type goVectorFunc func(inputs, outputs []float32)

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// parallelizeGoVectorFunc takes a goVectorFunc and parallelizes its execution if the input size is large enough.
func parallelizeGoVectorFunc(fn goVectorFunc) goVectorFunc {
	return func(inputs, outputs []float32) {
		numInputs := len(inputs)
		if numInputs < 100_000 { // Threshold for parallelization. Tune this value.
			fn(inputs, outputs)
			return
		}

		numCPU := runtime.NumCPU()
		chunkSize := numInputs / numCPU
		var wg sync.WaitGroup
		wg.Add(numCPU)
		for i := 0; i < numCPU; i++ {
			start := i * chunkSize
			end := (i + 1) * chunkSize
			if i == numCPU-1 {
				end = numInputs // Handle any remainder
			}
			go func(start, end int) {
				defer wg.Done()
				fn(inputs[start:end], outputs[start:end])
			}(start, end)
		}
		wg.Wait()
	}
}

// BenchmarkXLAExec executes TestPrograms on XLA using the normal Exec method.
// We try not to count the time for tensor transfers in and out.
func BenchmarkXLAExec(b *testing.B) {
	var isDuringBenchmark bool

	// Check conversion.
	backend := graphtest.BuildTestBackend()
	execs := make([]*graph.Exec, numPrograms)
	for progIdx, program := range TestPrograms {
		model := must.M1(onnx.ReadFile(program[0]))
		execs[progIdx] = graph.NewExec(backend, func(x *graph.Node) *graph.Node {
			if isDuringBenchmark {
				exceptions.Panicf("Graph building function called during benchmark: this shouldn't happen, as all graphs should have been built in startup")
			}
			g := x.Graph()
			outputs := model.CallGraph(nil, g, map[string]*graph.Node{"X": x})
			return outputs[0]
		})
	}

	// Pre-allocate tensors.
	numShapes := len(TestShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range TestShapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run tests for each shape/program combination.
	for shapeIdx, s := range TestShapes {
		for progIdx, program := range TestPrograms {
			exec := execs[progIdx]
			b.Run(fmt.Sprintf("shape=%s/%s%s", s, program[1], benchmarkNameSuffix),
				func(b *testing.B) {
					// Set input to value of v.
					x := inputTensors[shapeIdx]
					tensors.MutableFlatData[float32](x, func(flat []float32) {
						for ii := range flat {
							flat[ii] = float32(shapeIdx*numPrograms + progIdx + 1)
						}
					})

					// WarmUp:
					for _ = range 10 {
						tmpOutput := exec.Call(x)[0]
						tmpOutput.FinalizeAll()
					}
					b.ResetTimer()

					for _ = range b.N {
						tmpOutput := exec.Call(x)[0]
						tmpOutput.FinalizeAll()
					}
				})
		}
	}
}

// BenchmarkXLADirect benchmarks TestPrograms using direct GoMLX execution.
// We try not to count the time for tensor transfers in and out.
func BenchmarkXLADirect(b *testing.B) {
	// Create executables.
	backend := graphtest.BuildTestBackend()
	numShapes := len(TestShapes)
	graphPerShapePerProgram := make([][]*graph.Graph, numShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range TestShapes {
		graphPerShapePerProgram[shapeIdx] = make([]*graph.Graph, numPrograms)
		for progIdx, program := range TestPrograms {
			model := must.M1(onnx.ReadFile(program[0]))
			g := graph.NewGraph(backend, fmt.Sprintf("Graph #%d", shapeIdx))
			x := graph.Parameter(g, "X", s)
			y := model.CallGraph(nil, g, map[string]*graph.Node{"X": x})[0]
			g.Compile(y)
			graphPerShapePerProgram[shapeIdx][progIdx] = g
			inputTensors[shapeIdx] = tensors.FromShape(s)
			outputTensors[shapeIdx] = tensors.FromShape(s)
		}
	}

	// Run tests for each shape/program combination.
	for shapeIdx, s := range TestShapes {
		for progIdx, program := range TestPrograms {
			b.Run(fmt.Sprintf("shape=%s/%s%s", s, program[1], benchmarkNameSuffix),
				func(b *testing.B) {
					// Set input to value of v.
					x := inputTensors[shapeIdx]
					tensors.MutableFlatData[float32](x, func(flat []float32) {
						for ii := range flat {
							flat[ii] = float32(shapeIdx*numPrograms + progIdx + 1)
						}
					})
					xBuf := x.Buffer(backend)
					g := graphPerShapePerProgram[shapeIdx][progIdx]

					// WarmUp:
					for _ = range 10 {
						tmpOutput := g.RunWithBuffers(
							[]backends.Buffer{xBuf},
							[]bool{false})[0]
						tmpOutput.FinalizeAll()
					}
					b.ResetTimer()

					for _ = range b.N {
						tmpOutput := g.RunWithBuffers(
							[]backends.Buffer{xBuf},
							[]bool{false})[0]
						tmpOutput.FinalizeAll()
					}
				})
		}
	}
}

// BenchmarkAdd1ONNXRuntime
// We try not to count the time for tensor transfers in and out.
func BenchmarkONNXRT(b *testing.B) {
	ortPath := os.Getenv("ORT_SO_PATH")
	if ortPath == "" {
		exceptions.Panicf("Please set environment ORT_SO_PATH with the path to your ONNX Runtime dynamic linked library")
	}
	ort.SetSharedLibraryPath(ortPath)
	must.M(ort.InitializeEnvironment())
	defer func() { _ = ort.DestroyEnvironment() }()

	// Create a session for each tensor shape:
	numShapes := len(TestShapes)
	sessions := make([][]*ort.AdvancedSession, numShapes)
	inputsPerShape := make([]*ort.Tensor[float32], 0, numShapes)
	outputsPerShape := make([]*ort.Tensor[float32], 0, numShapes)
	for shapeIdx, s := range TestShapes {
		inputData := make([]float32, s.Size())
		dims64 := sliceMap(s.Dimensions, func(dim int) int64 { return int64(dim) })
		inputShape := ort.NewShape(dims64...)
		inputTensor := must.M1(ort.NewTensor(inputShape, inputData))
		inputsPerShape = append(inputsPerShape, inputTensor)
		outputTensor := must.M1(ort.NewEmptyTensor[float32](inputShape))
		outputsPerShape = append(outputsPerShape, outputTensor)

		sessions[shapeIdx] = make([]*ort.AdvancedSession, numPrograms)
		for progIdx, program := range TestPrograms {
			sessions[shapeIdx][progIdx] = must.M1(ort.NewAdvancedSession(
				program[0],
				[]string{"X"},
				[]string{"Y"},
				[]ort.Value{inputTensor},
				[]ort.Value{outputTensor},
				nil))
		}
	}

	// Run tests for each shape/program combination.
	for shapeIdx, s := range TestShapes {
		for progIdx, program := range TestPrograms {
			b.Run(fmt.Sprintf("shape=%s/%s%s", s, program[1], benchmarkNameSuffix),
				func(b *testing.B) {
					// Set input to value of v.
					input := inputsPerShape[shapeIdx]
					data := input.GetData()
					for ii := range data {
						data[ii] = float32(shapeIdx*numPrograms + progIdx + 1)
					}
					session := sessions[shapeIdx][progIdx]

					// WarmUp:
					for _ = range 10 {
						must.M(session.Run())
					}
					b.ResetTimer()

					for _ = range b.N {
						must.M(session.Run())
					}
				})
		}
	}
}

func BenchmarkPureGo(b *testing.B) {
	// Pre-allocate tensors.
	numShapes := len(TestShapes)
	inputTensors := make([]*tensors.Tensor, numShapes)
	outputTensors := make([]*tensors.Tensor, numShapes)
	for shapeIdx, s := range TestShapes {
		inputTensors[shapeIdx] = tensors.FromShape(s)
		outputTensors[shapeIdx] = tensors.FromShape(s)
	}

	// Run tests for each shape/program combination.
	for shapeIdx, s := range TestShapes {
		x := inputTensors[shapeIdx]
		y := outputTensors[shapeIdx]
		for progIdx, program := range TestPrograms {
			b.Run(fmt.Sprintf("shape=%s/%s%s", s, program[1], benchmarkNameSuffix),
				func(b *testing.B) {
					// Set value:
					tensors.MutableFlatData[float32](x, func(flat []float32) {
						for ii := range flat {
							flat[ii] = float32(shapeIdx*numPrograms + progIdx + 1)
						}
					})
					testProgram := testGoPrograms[progIdx]

					// Warm-up:
					for _ = range 10 {
						tensors.ConstFlatData[float32](x, func(inputs []float32) {
							tensors.MutableFlatData[float32](y, func(outputs []float32) {
								testProgram(inputs, outputs)
							})
						})
					}

					// Benchmark
					b.ResetTimer()
					for _ = range b.N {
						tensors.ConstFlatData[float32](x, func(inputs []float32) {
							tensors.MutableFlatData[float32](y, func(outputs []float32) {
								testProgram(inputs, outputs)
							})
						})
					}
				})
		}
	}
}
