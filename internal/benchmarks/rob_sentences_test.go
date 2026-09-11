package benchmarks

// This file is an extension of knights_sbert_test but defining the test sentences on robSentences.

import (
	"flag"
	"fmt"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	dtok "github.com/daulet/tokenizers"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	modelonnx "github.com/gomlx/gomlx/ml/model/onnx"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/gomlx/gomlx/support/xsync"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"github.com/janpfeifer/must"
	"github.com/streadway/quantile"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	flagDynamic          = flag.Bool("dynamic", true, "Enables dynamic shapes use on TestRobSentences_BenchXLA if backend supports it")
	flagBenchConcurrency []int
	flagBenchBatchSizes  []int
	flagSaveONNX         = flag.String("save_onnx", "", "If set and backend is ONNX, save the graph after building it to the file path")
	flagONNXModel        = flag.String("onnx_model", "", "Path to ONNX model to benchmark with TestRobSentences_BenchORT (defaults to downloaded HF model or save_onnx)")

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

func init() {
	// Set-up int list flags
	flag.Func("bench_concurrency", "Sets the set of concurrency execution to run during benchmarks. If not set, it uses default list.",
		sliceIntFlag(&flagBenchConcurrency))
	flag.Func("bench_batch", "Sets the set of BatchSize to use during benchmarks. If not set, it uses default list.",
		sliceIntFlag(&flagBenchBatchSizes))
}

func sliceIntFlag(storeValue *[]int) func(string) error {
	return func(value string) error {
		for _, part := range strings.Split(value, ",") {
			v, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				return err
			}
			*storeValue = append(*storeValue, v)
		}
		return nil
	}
}

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

// formatDuration formats the duration with 2 decimal places but keeping the unit suffix.
func formatDuration(d time.Duration) string {
	s := d.String()
	i := 0
	for ; i < len(s); i++ {
		if (s[i] < '0' || s[i] > '9') && s[i] != '.' {
			break
		}
	}
	// Found the time unit (the suffix)
	num := s[:i]
	unit := s[i:]
	f, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return s
	}
	return fmt.Sprintf("%.2f%s", f, unit)
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
	//   - We add some buffer because we don't want the preparation of the inputs (producer)
	//     to be a bottleneck or even accounted for.
	examplesChan := make(chan E, numWorkers)
	wg.Go(func() {
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
	})

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

	// Warm-up:
	for range warmUpRuns {
		<-finishedCounter
	}

	// Benchmark collection:
	estimates := []quantile.Estimate{
		quantile.Known(0.05, 0.001),
		quantile.Known(0.50, 0.001),
		quantile.Known(0.99, 0.001),
	}
	estimator := quantile.New(estimates...)
	var totalTime time.Duration
	var count int
	timer := time.NewTimer(*flagBenchDuration)

collection:
	for {
		select {
		case <-timer.C:
			break collection
		default:
			start := time.Now()
			<-finishedCounter
			elapsed := time.Since(start)
			estimator.Add(float64(elapsed) / float64(time.Nanosecond))
			totalTime += elapsed
			count++
		}
	}
	timer.Stop()

	// Header:
	const (
		nameColWidth = 50
		colWidth     = 10
	)
	if header {
		fmt.Printf("%-*s\t%*s\t%*s\t%*s\t%*s\t%*s\t%*s\n",
			nameColWidth, "Benchmarks:",
			colWidth+2, "Sentences/s",
			colWidth, "Mean",
			colWidth, "Median",
			colWidth, "5%-tile",
			colWidth, "99%-tile",
			colWidth+2, "Runs(xBatch)")
	}

	var (
		meanPerExample   time.Duration
		medianPerExample time.Duration
		q5PerExample     time.Duration
		q99PerExample    time.Duration
		sentencesPerSec  float64
	)
	if count > 0 && totalTime > 0 {
		meanPerExample = (totalTime / time.Duration(count)) / time.Duration(batchSize)
		medianPerExample = time.Duration(int(estimator.Get(0.50))) * time.Nanosecond / time.Duration(batchSize)
		q5PerExample = time.Duration(int(estimator.Get(0.05))) * time.Nanosecond / time.Duration(batchSize)
		q99PerExample = time.Duration(int(estimator.Get(0.99))) * time.Nanosecond / time.Duration(batchSize)
		sentencesPerSec = float64(count*batchSize) / totalTime.Seconds()
	}

	sentencesStr := fmt.Sprintf("%.1f/s", sentencesPerSec)
	runsStr := fmt.Sprintf("%d (x%d)", count, batchSize)
	fmt.Printf("%-*s\t%*s\t%*s\t%*s\t%*s\t%*s\t%*s\n",
		nameColWidth, name,
		colWidth+2, sentencesStr,
		colWidth, formatDuration(meanPerExample),
		colWidth, formatDuration(medianPerExample),
		colWidth, formatDuration(q5PerExample),
		colWidth, formatDuration(q99PerExample),
		colWidth+2, runsStr)

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
	onnxModelPath := *flagONNXModel
	if onnxModelPath == "" && *flagSaveONNX != "" {
		onnxModelPath = *flagSaveONNX
	}
	if onnxModelPath == "" {
		repoModel := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
		onnxModelPath = must.M1(repoModel.DownloadFile("model.onnx"))
	} else {
		name += fmt.Sprintf("[%s]", filepath.Base(onnxModelPath))
	}
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
			must.M(options.SetExecutionMode(ort.ExecutionModeParallel))
			must.M(options.SetGraphOptimizationLevel(99))
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
		sentenceLen := 0
		for inBatchIdx := range batchSize {
			sentenceLen = max(sentenceLen, len(examples[(sentenceIdx+inBatchIdx)%len(examples)].Encoding[0]))
		}
		inputShape := ort.NewShape(int64(batchSize), int64(sentenceLen))
		for ii := range inputTensors {
			inputTensors[ii] = must.M1(ort.NewEmptyTensor[int64](inputShape))
		}
		// Create a batch for each input tensor.
		for inputIdx, t := range inputTensors {
			flat := t.GetData()
			for inBatchIdx := range batchSize {
				example := examples[(sentenceIdx+inBatchIdx)%len(examples)]
				seq := example.Encoding[inputIdx]
				copy(flat[inBatchIdx*sentenceLen:], seq)
				for p := len(seq); p < sentenceLen; p++ {
					flat[inBatchIdx*sentenceLen+p] = 0
				}
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
	// Make sure to release all resources no longer in use.
	for range 10 {
		runtime.GC()
	}

	backend := testutil.BuildTestBackend()
	useDynamic := *flagDynamic && backend.Capabilities().HasDynamicShapes()

	name := fmt.Sprintf("XLA/RobSentences/Parallel=%02d/BatchSize=%02d", parallelization, batchSize)
	if useDynamic {
		name += "/Dynamic"
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
	onnxModel := must.M1(parser.ParseFile(onnxModelPath))

	store := model.NewStore()
	must.M(onnxModel.VariablesToScope(store.RootScope()))
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokenIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
		//fmt.Printf("Exec inputs (tokens, mask, types): %s, %s, %s\n", tokenIDs.Shape(), attentionMask.Shape(), tokenTypeIDs.Shape())
		g := tokenIDs.Graph()
		scope.SetTraining(g, false) // Inference only.
		outputs := onnxModel.CallGraph(scope, g,
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
	if useDynamic {
		exec.WithDynamicAxes(
			[]string{"batch", "seq"},
			[]string{"batch", "seq"},
			[]string{"batch", "seq"},
		)
	}
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
	maxSeqLen := 0
	for _, example := range examples {
		maxSeqLen = max(maxSeqLen, len(example.Encoding[0]))
	}

	if *flagSaveONNX != "" && modelonnx.IsONNX(backend) {
		inputShapes := []shapes.Shape{
			shapes.Make(dtypes.Int64, batchSize, maxSeqLen),
			shapes.Make(dtypes.Int64, batchSize, maxSeqLen),
			shapes.Make(dtypes.Int64, batchSize, maxSeqLen),
		}
		inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
		outputNames := []string{"last_hidden_state"}
		must.M(modelonnx.SaveToFile(backend, exec, *flagSaveONNX, inputShapes, inputNames, outputNames))
		fmt.Printf("Saved ONNX graph to %q\n", *flagSaveONNX)
	}

	var pools sync.Map
	getPool := func(seqLen int) *sync.Pool {
		if p, ok := pools.Load(seqLen); ok {
			return p.(*sync.Pool)
		}
		p := &sync.Pool{
			New: func() any {
				var inputTensors ExampleInput
				for ii := range inputTensors {
					inputTensors[ii] = tensors.FromShape(shapes.Make(dtypes.Int64, batchSize, seqLen))
				}
				return inputTensors
			},
		}
		actual, _ := pools.LoadOrStore(seqLen, p)
		return actual.(*sync.Pool)
	}

	nextSentenceIdx := 0
	inputFn := func() (inputTensors ExampleInput) {
		batchSeqLen := maxSeqLen
		if useDynamic {
			batchSeqLen = 0
			for inBatchIdx := range batchSize {
				example := examples[(nextSentenceIdx+inBatchIdx)%len(examples)]
				batchSeqLen = max(batchSeqLen, len(example.Encoding[0]))
			}
		}

		pool := getPool(batchSeqLen)
		inputTensors = pool.Get().(ExampleInput)
		for inputIdx := range inputTensors {
			t := inputTensors[inputIdx]
			tensors.MutableFlatData[int64](t, func(flat []int64) {
				for inBatchIdx := range batchSize {
					example := examples[(nextSentenceIdx+inBatchIdx)%len(examples)]
					seq := example.Encoding[inputIdx]
					copy(flat[inBatchIdx*batchSeqLen:], seq)
					for p := len(seq); p < batchSeqLen; p++ {
						flat[inBatchIdx*batchSeqLen+p] = 0
					}
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
		output := exec.MustCall1(inputTensors[0], inputTensors[1], inputTensors[2])
		fmt.Printf("\tSaving reference embeddings to %q - shape=%s, embedding[0, 0, 0]=%.3f, token[0, 0]=%d\n",
			robSentencesEmbeddingsFileName,
			output.Shape(),
			tensors.MustCopyFlatData[float32](output)[0],
			tensors.MustCopyFlatData[int64](inputTensors[0])[0])
		err := output.Save(robSentencesEmbeddingsFileName)
		if err != nil {
			panic(err)
		}
		output.FinalizeAll()
		return
	}

	var workerCount int
	workerFn := func(workerIdx int, inputTensors ExampleInput) {
		seqLen := inputTensors[0].Shape().Dim(1)
		defer getPool(seqLen).Put(inputTensors)
		output := exec.MustCall1(inputTensors[0], inputTensors[1], inputTensors[2])
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
	warmUpRuns := 2 * (len(examples) + batchSize - 1) / batchSize
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
	concurrencies := []int{16}
	if len(flagBenchConcurrency) > 0 {
		concurrencies = flagBenchConcurrency
	}
	batchSizes := []int{16}
	if len(flagBenchBatchSizes) > 0 {
		batchSizes = flagBenchBatchSizes
	}
	for _, concurrency := range concurrencies { // {4, 6, 8} {
		for _, batchSize := range batchSizes { // 1, 2, 4, 8, 16, 32} {
			implBenchRobSentencesORT(concurrency, batchSize, count == 0)
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
	concurrencies := []int{8}
	if len(flagBenchConcurrency) > 0 {
		concurrencies = flagBenchConcurrency
	}
	batchSizes := []int{128}
	if len(flagBenchBatchSizes) > 0 {
		batchSizes = flagBenchBatchSizes
	}
	for _, concurrency := range concurrencies { // {4, 6, 8} {
		for _, batchSize := range batchSizes { // 1, 2, 4, 8, 16, 32} {
			implBenchRobSentencesXLA(t, concurrency, batchSize, count == 0)
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

const checkingEmbeddingsDelta = 1e-2

func TestRobSentences_CheckEmbeddings(t *testing.T) {
	if !*flagCheckEmbeddings {
		fmt.Println("Skipping CheckEmbeddings test, --check_embeddings not set.")
		t.SkipNow()
		return
	}
	implBenchRobSentencesXLA(t, 1, len(robSentences), false)
}

func TestRobSentences_DynamicCallGraph(t *testing.T) {
	repoModel := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
	onnxModelPath := must.M1(repoModel.DownloadFile("model.onnx"))
	backend := testutil.BuildTestBackend()
	if !backend.Capabilities().HasDynamicShapes() {
		t.Skipf("Backend %q does not support dynamic shapes", backend.Name())
	}
	onnxModel := must.M1(parser.ParseFile(onnxModelPath))
	store := model.NewStore()
	must.M(onnxModel.VariablesToScope(store.RootScope()))

	exec := model.MustNewExec(backend, store, func(scope *model.Scope, tokenIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
		g := tokenIDs.Graph()
		outputs := onnxModel.CallGraph(scope, g,
			map[string]*graph.Node{
				"input_ids":      tokenIDs,
				"attention_mask": attentionMask,
				"token_type_ids": tokenTypeIDs,
			})
		return outputs[0]
	})
	exec.WithDynamicAxes(
		[]string{"batch", "seq"},
		[]string{"batch", "seq"},
		[]string{"batch", "seq"},
	)
	defer exec.Finalize()

	// Call with dynamic length based on first 2 examples:
	examples := initializeRobSentences(2)
	batchSeqLen := max(len(examples[0].Encoding[0]), len(examples[1].Encoding[0]))
	tIDs := tensors.FromShape(shapes.Make(dtypes.Int64, 2, batchSeqLen))
	tMask := tensors.FromShape(shapes.Make(dtypes.Int64, 2, batchSeqLen))
	tTypes := tensors.FromShape(shapes.Make(dtypes.Int64, 2, batchSeqLen))
	for inputIdx, tTarget := range []*tensors.Tensor{tIDs, tMask, tTypes} {
		tensors.MutableFlatData[int64](tTarget, func(flat []int64) {
			for exIdx := 0; exIdx < 2; exIdx++ {
				seq := examples[exIdx].Encoding[inputIdx]
				copy(flat[exIdx*batchSeqLen:], seq)
				for p := len(seq); p < batchSeqLen; p++ {
					flat[exIdx*batchSeqLen+p] = 0
				}
			}
		})
	}

	outDyn := exec.MustCall1(tIDs, tMask, tTypes)

	// Now run static exec with same input:
	execStatic := model.MustNewExec(backend, store, func(scope *model.Scope, tokenIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
		g := tokenIDs.Graph()
		outputs := onnxModel.CallGraph(scope, g,
			map[string]*graph.Node{
				"input_ids":      tokenIDs,
				"attention_mask": attentionMask,
				"token_type_ids": tokenTypeIDs,
			})
		return outputs[0]
	})
	defer execStatic.Finalize()
	outStat := execStatic.MustCall1(tIDs, tMask, tTypes)

	requireSameTensorsFloat32(t, outStat, outDyn, 1e-4)
}
