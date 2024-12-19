package benchmarks

// This file is an extension of knights_sbert_test but defining the test sentences on robSentences.

import (
	"fmt"
	dtok "github.com/daulet/tokenizers"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/janpfeifer/go-benchmarks"
	"github.com/janpfeifer/must"
	ort "github.com/yalue/onnxruntime_go"
	"testing"
)

var (
	robSentences = []string{
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
		"Sustainable Agriculture Group Promotes Organic Farming",
		"Cloud Based Solutions Unveils New Secure Data Services",
		"Tech Innovators Inc. to Host Annual Tech Summit This Fall",
	}
)

// initializeRobSentences tokenizes the fixed robSentences (as opposed to using FineWeb, the default),
// and trim any padding.
func initializeRobSentences() []tokenizedSentence {
	results := make([]tokenizedSentence, len(robSentences))

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

		// Find seuqenceLen for sentence.
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
	return results
}

func TestBenchRobSentencesORT(t *testing.T) {
	batchSize := 32
	name := fmt.Sprintf("ORT/RobSentences/BatchSize=%d", batchSize)
	outputNodeName := "last_hidden_state"
	embeddingSize := 384

	// Tokenize Rob's sentences.
	examples := initializeRobSentences()
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
	}

	session := must.M1(ort.NewDynamicAdvancedSession(
		onnxModelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"}, []string{outputNodeName},
		options))
	defer func() { must.M(session.Destroy()) }()

	sentenceIdx := 0
	runIdx := 0
	testFn := benchmarks.NamedFunction{
		Name: name,
		Func: func() {
			sentenceLen := len(examples[sentenceIdx].Encoding[0])

			// Create input and output tensors.
			var inputTensors [3]*ort.Tensor[int64]
			inputShape := ort.NewShape(int64(batchSize), int64(sentenceLen))
			for ii := range inputTensors {
				inputTensors[ii] = must.M1(ort.NewEmptyTensor[int64](inputShape))
			}
			outputShape := ort.NewShape(int64(batchSize), int64(sentenceLen), int64(embeddingSize))
			outputTensor := must.M1(ort.NewEmptyTensor[float32](outputShape))

			// Create batch for each input tensor.
			for inputIdx, t := range inputTensors {
				flat := t.GetData()
				for inBatchIdx := range batchSize {
					example := examples[sentenceIdx+inBatchIdx]
					copy(flat[inBatchIdx*sentenceLen:], example.Encoding[inputIdx])
				}
			}

			// Execute program.
			must.M(session.Run(
				[]ort.Value{inputTensors[0], inputTensors[1], inputTensors[2]},
				[]ort.Value{outputTensor},
			))

			// Next batch.
			sentenceIdx += batchSize
			if sentenceIdx+batchSize >= len(examples) {
				sentenceIdx = 0
			}
			runIdx++
		},
	}

	benchmarks.New(testFn).
		WithWarmUps(10).
		WithDuration(*flagBenchDuration).
		Done()
}

func TestBenchRobSentencesXLA(t *testing.T) {
	batchSize := 1
	name := fmt.Sprintf("ORT/RobSentences/BatchSize=%d", batchSize)

	// Tokenize Rob's sentences.
	examples := initializeRobSentences()
	if len(examples) < batchSize {
		exceptions.Panicf("batchSize(%d) must be <= to the number of examples (%d)", batchSize, len(examples))
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

	sentenceLen := 13
	var inputTensors [3]*tensors.Tensor
	for ii := range inputTensors {
		inputTensors[ii] = tensors.FromShape(shapes.Make(dtypes.Int64, batchSize, sentenceLen))
	}

	nextSentenceIdx := 0
	runIdx := 0
	testFn := benchmarks.NamedFunction{
		Name: name,
		Func: func() {
			sentenceLen := len(examples[nextSentenceIdx].Encoding[0])

			// Create input and output tensors.
			for inputIdx, t := range inputTensors {
				tensors.MutableFlatData[int64](t, func(flat []int64) {
					for inBatchIdx := range batchSize {
						example := examples[nextSentenceIdx+inBatchIdx]
						copy(flat[inBatchIdx*sentenceLen:], example.Encoding[inputIdx])
					}
				})
			}

			// Execute program.
			output := exec.Call(inputTensors[0], inputTensors[1], inputTensors[2])[0]
			tensors.ConstFlatData(output, func(flat []float32) {
				_ = flat
			})
			output.FinalizeAll()

			// Next batch.
			nextSentenceIdx += batchSize
			if nextSentenceIdx+batchSize >= len(examples) {
				nextSentenceIdx = 0
			}
			runIdx++
		},
	}

	benchmarks.New(testFn).
		WithWarmUps(10).
		WithDuration(*flagBenchDuration).
		WithWarmUps(3 * len(examples)).
		Done()
}
