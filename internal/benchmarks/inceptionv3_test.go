package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/janpfeifer/go-benchmarks"
	"github.com/janpfeifer/must"
)

func TestBenchInceptionV3(t *testing.T) {
	if testing.Short() {
		fmt.Printf("Skipping InceptionV3 benchmark test: --short is set\n")
		t.SkipNow()
	}
	if *flagBenchDuration == 0 {
		fmt.Printf("Skipping InceptionV3 benchmark test: --bench_duration is not set\n")
		t.SkipNow()
	}
	t.Run("ONNX-GoMLX", benchONNXGoMLXInceptionV3)
}

var (
	inceptionV3RepoID        = "recursionerr/nsfw_01"
	inceptionV3ModelFileName = "inception_v3.onnx"
)

func benchONNXGoMLXInceptionV3(t *testing.T) {
	fmt.Printf("\n%s\n", inceptionV3RepoID)
	repo := hub.New(inceptionV3RepoID).WithAuth(hfAuthToken)
	onnxModelPath := must.M1(repo.DownloadFile(inceptionV3ModelFileName))
	fmt.Printf("\tmodel path: %s\n", onnxModelPath)
	model := must.M1(onnx.ReadFile(onnxModelPath))
	fmt.Printf("%s\n", model)
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	must.M(model.VariablesToContext(ctx))
	ctx = ctx.Reuse()
	inputName := model.InputsNames[0]
	outputName := model.OutputsNames[0]
	for batchIdx, batchSize := range BatchSizes {
		//t.Run(fmt.Sprintf("batchSize=%02d", batchSize), func(t *testing.T) {
		exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, images *Node) *Node {
			g := images.Graph()
			outputs := model.CallGraph(ctx, g,
				map[string]*Node{
					inputName: images,
				}, outputName)
			if *flagPrintXLAGraph {
				fmt.Printf("Graph:\n%s\n", g)
			}
			return outputs[0]
		})

		// Create random images
		r := rand.New(rand.NewPCG(42, 0))
		inputImages := tensors.FromShape(shapes.Make(dtypes.Float32, batchSize, 299, 299, 3))
		tensors.MutableFlatData[float32](inputImages, func(flat []float32) {
			for i := range flat {
				flat[i] = r.Float32()
			}
		})

		runIdx := 0
		benchFn := benchmarks.NamedFunction{
			Name: fmt.Sprintf("%s/batchSize=%02d", t.Name(), batchSize),
			Func: func() {
				output := exec.MustExec1(inputImages)
				tensors.ConstFlatData(output, func(flat []float32) {
					if runIdx == 0 {
						fmt.Printf("\t> Last value of result: %v\n", flat[len(flat)-1])
					}
				})
				output.FinalizeAll()
				runIdx++
			},
		}

		runtime.LockOSThread()
		benchmarks.New(benchFn).
			WithWarmUps(128).
			WithDuration(*flagBenchDuration).
			WithHeader(batchIdx == 0).
			Done()
		runtime.UnlockOSThread()
		exec.Finalize()
	}
}
