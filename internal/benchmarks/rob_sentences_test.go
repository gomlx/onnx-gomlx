package benchmarks

// This file is an extension of knights_sbert_test but defining the test sentences on robSentences.

import (
	"fmt"
	"runtime"
	"sync"
	"testing"

	dtok "github.com/daulet/tokenizers"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/janpfeifer/go-benchmarks"
	"github.com/janpfeifer/must"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	robSentences = []string{
		"robert smith junior",
		"francis ford coppola",
		"robert smith",
		"Tech Innovators Inc. Launches Revolutionary AI Platform",
		"Green Energy Solutions Unveils Next-Gen Solar Panels",
		"Global Ventures Co. Secures $2 Billion in Funding",
		"Creative Minds Studio Launches Virtual Creativity Hub",
		"Healthcare Partners Ltd. Introduces AI-Driven Diagnostics",
		"Future Finance Group Predicts Key Market Trends for 2024",
		"Premier Logistics LLC Expands Into New International Markets",
		"Dynamic Marketing Agency Announces Strategic Partnership",
		"Eco-Friendly Products Corp. Debuts Sustainable Tech Line",
		"Blue Ocean Enterprises Leads the Way in Marine Technology",
		"NextGen Software Solutions Rolls Out New Cloud Suite",
		"Innovative Construction Co. Breaks Ground on Green Projects",
		"Precision Engineering Ltd. Redefines Robotics Efficiency",
		"Elite Consulting Group Forecasts Industry Growth in 2024",
		"Urban Development LLC Transforms City Skylines Nationwide",
		"Digital Media Concepts Sets New Standards for AI Content Delivery",
		"Community Builders Inc. Wins National Housing Award",
		"Trusted Insurance Brokers Introduces Smart Policy Options",
		"Advanced Manufacturing Corp. Showcases Cutting-Edge Automation",
		"Visionary Design Studio Redefines Modern Architecture",
		"Strategic Investment Partners Reveals Key Acquisitions",
		"Modern Retail Solutions Integrates AI Shopping Experiences",
		"Efficient Energy Systems Revolutionizes Grid Technology",
		"High-Tech Components Inc. Develops Next-Gen Processors",
		"Education Outreach Network Empowers Communities with New Programs",
		"Healthcare Innovations Ltd. Drives Breakthrough in Medical Research",
		"Creative Film Productions Wins Prestigious Global Awards",
		"Global Trade Services Expands Globalized Shipping Network",
		"NextLevel Sports Management Signs High-Profile Athletes",
		//"Sustainable Agriculture Group Promotes Organic Farming",
		//"Tech Innovators Inc. to Host Annual Tech Summit This Fall",
		//"Cloud Based Solutions Unveils New Secure Data Services",
	}
)

// initializeRobSentences tokenizes the fixed robSentences (as opposed to using FineWeb, the default)
// and trims any padding.
func initializeRobSentences(minNumExamples int) []tokenizedSentence {
	numSentences := len(robSentences)
	results := make([]tokenizedSentence, max(numSentences, minNumExamples))

	// Create tokenizer: it is configured by the "tokenizer.json" to a max_length of 128, with padding.
	repoTokenizer := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
	localFile := must.M1(repoTokenizer.DownloadFile("tokenizer.json"))
	tokenizer := must.M1(dtok.FromFile(localFile))
	defer func() { _ = tokenizer.Close() }()

	for idxSentence, sentence := range robSentences {
		encoding := tokenizer.EncodeWithOptions(sentence, false,
			dtok.WithReturnTypeIDs(),
			dtok.WithReturnAttentionMask(),
		)

		// Find sequenceLen for sentence.
		sequenceLen := len(encoding.AttentionMask)
		for sequenceLen > 0 && encoding.AttentionMask[sequenceLen-1] == 0 {
			sequenceLen--
		}
		sequenceLen = 13

		results[idxSentence].Encoding[0] = padOrTrim(sequenceLen,
			sliceMap(encoding.IDs, func(id uint32) int64 { return int64(id) }),
			0)
		results[idxSentence].Encoding[1] = padOrTrim(sequenceLen,
			sliceMap(encoding.AttentionMask, func(id uint32) int64 { return int64(id) }),
			0)
		results[idxSentence].Encoding[2] = padOrTrim(sequenceLen,
			sliceMap(encoding.TypeIDs, func(id uint32) int64 { return int64(id) }),
			0)
	}

	// Replicate extra examples at the end.
	for ii := numSentences; ii < len(results); ii++ {
		results[ii] = results[ii-numSentences] // Keep repeating.
	}
	return results
}

func implParallelBenchmark[E any](
	name string,
	numWorkers, batchSize int, header bool,
	warmUpRuns int,
	inputFn func() E,
	workerFn func(workerIdx int, e E)) {
	// Parallelization:
	var wg sync.WaitGroup
	done := xsync.NewLatch()

	// Start producer of inputs:
	//   - We add some buffer, because we don't want the preparation of the inputs (producer)
	//     to be a bottleneck or even accounted for.
	examplesChan := make(chan E, numWorkers)
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Create input and output tensors.
		for {
			e := inputFn()
			// Write the example or interrupt.
			select {
			case <-done.WaitChan():
				// Finished executing, simply exit.
				return
			case examplesChan <- e:
				// Move forward to produce the next input example.
			}
		}
	}()

	// Start consumers:
	finishedCounter := make(chan struct{})
	for workerIdx := range numWorkers {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			for {
				var e E
				select {
				case <-done.WaitChan():
					return
				case e = <-examplesChan:
					// Received next input.
				}
				workerFn(workerIdx, e)
				select {
				case <-done.WaitChan():
					return
				case finishedCounter <- struct{}{}:
					// Accounted for, loop to next.
				}
			}
		}(workerIdx)
	}

	// Benchmark function is simply reading out finished
	testFn := benchmarks.NamedFunction{
		Name: name,
		Func: func() {
			<-finishedCounter
		},
	}
	benchmarks.New(testFn).
		WithWarmUps(warmUpRuns).
		WithDuration(*flagBenchDuration).
		WithHeader(header).
		WithInnerRepeats(batchSize). // Report will be "per example".
		Done()

	// done.Trigger will signal all goroutines to end.
	done.Trigger()
	wg.Wait()
}

func implBenchRobSentencesORT(parallelization, batchSize int, header bool) {
	name := fmt.Sprintf("ORT/RobSentences/Parallel=%02d/BatchSize=%02d", parallelization, batchSize)
	outputNodeName := "last_hidden_state"
	embeddingSize := 384

	// Tokenize Rob's sentences.
	examples := initializeRobSentences(batchSize)
	if len(examples) < batchSize {
		exceptions.Panicf("batchSize(%d) must be <= to the number of examples (%d)", batchSize, len(examples))
	}

	// Create session with ONNX program.
	ortInitFn()
	repoModel := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
	onnxModelPath := must.M1(repoModel.DownloadFile("model.onnx"))
	var options *ort.SessionOptions
	if ortIsCUDA {
		options = must.M1(ort.NewSessionOptions())
		cudaOptions := must.M1(ort.NewCUDAProviderOptions())
		// must.M(cudaOptions.Update(map[string]string{"device_id": "0"}))
		must.M(options.AppendExecutionProviderCUDA(cudaOptions))
	} else {
		if parallelization > 1 {
			options = must.M1(ort.NewSessionOptions())
			must.M(options.SetIntraOpNumThreads(1))
			must.M(options.SetInterOpNumThreads(1))
			must.M(options.SetCpuMemArena(false))
			must.M(options.SetMemPattern(false))
		}
	}

	// Create sessions, one per parallel run.
	sessions := make([]*ort.DynamicAdvancedSession, parallelization)
	for pIdx := range parallelization {
		sessions[pIdx] = must.M1(ort.NewDynamicAdvancedSession(
			onnxModelPath,
			[]string{"input_ids", "attention_mask", "token_type_ids"}, []string{outputNodeName},
			options))
	}
	defer func() {
		for _, session := range sessions {
			must.M(session.Destroy())
		}
	}()

	// Generating examples for sessions.
	type ExampleInput [3]*ort.Tensor[int64]
	sentenceIdx := 0
	inputFn := func() (inputTensors ExampleInput) {
		sentenceLen := len(examples[sentenceIdx].Encoding[0])
		inputShape := ort.NewShape(int64(batchSize), int64(sentenceLen))
		for ii := range inputTensors {
			inputTensors[ii] = must.M1(ort.NewEmptyTensor[int64](inputShape))
		}
		// Create batch for each input tensor.
		for inputIdx, t := range inputTensors {
			flat := t.GetData()
			for inBatchIdx := range batchSize {
				example := examples[(sentenceIdx+inBatchIdx)%len(examples)]
				copy(flat[inBatchIdx*sentenceLen:], example.Encoding[inputIdx])
			}
		}
		// Next batch.
		sentenceIdx += batchSize
		if sentenceIdx+batchSize >= len(examples) {
			sentenceIdx = 0
		}
		return
	}

	// workerFn is executed in each goroutine -- one per parallelization
	workerFn := func(workerIdx int, inputTensors ExampleInput) {
		session := sessions[workerIdx]
		sentenceLen := inputTensors[0].GetShape()[1]
		outputShape := ort.NewShape(int64(batchSize), int64(sentenceLen), int64(embeddingSize))
		outputTensor := must.M1(ort.NewEmptyTensor[float32](outputShape))
		// Execute program.
		must.M(session.Run(
			[]ort.Value{inputTensors[0], inputTensors[1], inputTensors[2]},
			[]ort.Value{outputTensor},
		))
	}

	// Benchmark function is simply reading out finished
	warmUpRuns := 10
	implParallelBenchmark(name, parallelization, batchSize, header, warmUpRuns, inputFn, workerFn)
}

const robSentencesEmbeddingsFileName = "rob_sentences_embeddings.bin"

func implBenchRobSentencesXLA(t *testing.T, parallelization, batchSize int, header bool) {
	name := fmt.Sprintf("XLA/RobSentences/Parallel=%02d/BatchSize=%02d", parallelization, batchSize)
	// Make sure to release all resources no longer in use.
	for _ = range 10 {
		runtime.GC()
	}

	// Tokenize Rob's sentences.
	examples := initializeRobSentences(batchSize)
	if len(examples) < batchSize {
		exceptions.Panicf("batchSize(%d) must be <= to the number of examples (%d)", batchSize, len(examples))
	}
	if (*flagSaveEmbeddings || *flagCheckEmbeddings) && batchSize != len(robSentences) {
		exceptions.Panicf("batchSize(%d) must be %d (all robSentences) when saving embeddings (--save_embeddings) or "+
			"checking embeddings (--check_embeddings)", batchSize, len(robSentences))
	}

	// Build model
	repoModel := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
	onnxModelPath := must.M1(repoModel.DownloadFile("model.onnx"))
	backend := graphtest.BuildTestBackend()
	model := must.M1(onnx.ReadFile(onnxModelPath))
	ctx := context.New()
	must.M(model.VariablesToContext(ctx))
	ctx = ctx.Reuse()
	exec := context.NewExec(backend, ctx, func(ctx *context.Context, tokenIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
		//fmt.Printf("Exec inputs (tokens, mask, types): %s, %s, %s\n", tokenIDs.Shape(), attentionMask.Shape(), tokenTypeIDs.Shape())
		g := tokenIDs.Graph()
		outputs := model.CallGraph(ctx, g,
			map[string]*graph.Node{
				"input_ids":      tokenIDs,
				"attention_mask": attentionMask,
				"token_type_ids": tokenTypeIDs,
			})
		if *flagPrintXLAGraph {
			fmt.Printf("Graph:\n%s\n", g)
		}
		return outputs[0]
	})
	defer exec.Finalize()

	// Load expected results.
	var referenceEmbeddings *tensors.Tensor
	if *flagCheckEmbeddings {
		var err error
		referenceEmbeddings, err = tensors.Load(robSentencesEmbeddingsFileName)
		if err != nil {
			panic(err)
		}
	}

	// Generating examples for sessions.
	type ExampleInput [3]*tensors.Tensor
	sentenceLen := 13
	inputsPool := sync.Pool{
		New: func() any {
			var inputTensors ExampleInput
			for ii := range inputTensors {
				inputTensors[ii] = tensors.FromShape(shapes.Make(dtypes.Int64, batchSize, sentenceLen))
			}
			return inputTensors
		},
	}
	nextSentenceIdx := 0
	inputFn := func() (inputTensors ExampleInput) {
		inputTensors = inputsPool.Get().(ExampleInput)
		for inputIdx := range inputTensors {
			t := inputTensors[inputIdx]
			tensors.MutableFlatData[int64](t, func(flat []int64) {
				for inBatchIdx := range batchSize {
					example := examples[(nextSentenceIdx+inBatchIdx)%len(examples)]
					copy(flat[inBatchIdx*sentenceLen:], example.Encoding[inputIdx])
				}
			})
		}
		// Next batch.
		nextSentenceIdx = (nextSentenceIdx + batchSize) % len(examples)
		return
	}

	if *flagSaveEmbeddings {
		// Run inline and save the resulting embeddings:
		fmt.Println("Generating embeddings to save:")
		inputTensors := inputFn()
		output := exec.Call(inputTensors[0], inputTensors[1], inputTensors[2])[0]
		fmt.Printf("\tSaving reference embeddings to %q - shape=%s, embedding[0, 0, 0]=%.3f, token[0, 0]=%d\n",
			robSentencesEmbeddingsFileName,
			output.Shape(),
			tensors.CopyFlatData[float32](output)[0],
			tensors.CopyFlatData[int64](inputTensors[0])[0])
		err := output.Save(robSentencesEmbeddingsFileName)
		if err != nil {
			panic(err)
		}
		output.FinalizeAll()
		return
	}

	var workerCount int
	workerFn := func(workerIdx int, inputTensors ExampleInput) {
		defer inputsPool.Put(inputTensors)
		output := exec.Call(inputTensors[0], inputTensors[1], inputTensors[2])[0]
		tensors.ConstFlatData(output, func(flat []float32) {
			// Force local copy: this should be part of the cost.
			_ = flat
		})
		if referenceEmbeddings != nil {
			requireSameTensorsFloat32(t, referenceEmbeddings, output, checkingEmbeddingsDelta)
		}
		workerCount++
		output.FinalizeAll()
	}

	// Benchmark function is simply reading out finished
	warmUpRuns := 2 * len(examples)
	if *flagCheckEmbeddings {
		warmUpRuns = 1
	}
	implParallelBenchmark(
		name, parallelization, batchSize, header, warmUpRuns, inputFn, workerFn)
}

func TestRobSentences_BenchORT(t *testing.T) {
	if testing.Short() || *flagBenchDuration == 0 {
		t.SkipNow()
	}
	count := 0
	for _, parallelism := range []int{4} { // {2, 3, 4, 6, 8} {
		for _, batchSize := range []int{256} { // 1, 2, 4, 8, 16, 32} {
			implBenchRobSentencesORT(parallelism, batchSize, count == 0)
			count++
		}
	}
}

func TestRobSentences_BenchXLA(t *testing.T) {
	if testing.Short() || *flagBenchDuration == 0 {
		t.SkipNow()
	}
	count := 0
	// Change parallelism/batchSize according to backend, see best values in the bottom
	// of the "Rob Sentences" sheet in:
	// https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=397722581#gid=397722581
	for _, parallelism := range []int{32} { // {4, 6, 8} {
		for _, batchSize := range []int{len(robSentences)} { // 1, 2, 4, 8, 16, 32} {
			implBenchRobSentencesXLA(t, parallelism, batchSize, count == 0)
			count++
		}
	}
}

func TestRobSentences_SaveEmbeddings(t *testing.T) {
	if !*flagSaveEmbeddings {
		fmt.Println("Skipping SaveEmbeddings test, --save_embeddings not set.")
		t.SkipNow()
		return
	}
	implBenchRobSentencesXLA(t, 1, len(robSentences), false)
}

const checkingEmbeddingsDelta = 1e-3

func TestRobSentences_CheckEmbeddings(t *testing.T) {
	if !*flagCheckEmbeddings {
		fmt.Println("Skipping CheckEmbeddings test, --check_embeddings not set.")
		t.SkipNow()
		return
	}
	implBenchRobSentencesXLA(t, 1, len(robSentences), false)
}
