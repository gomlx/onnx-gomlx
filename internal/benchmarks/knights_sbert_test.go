package benchmarks

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
	"github.com/janpfeifer/must"
	parquet "github.com/parquet-go/parquet-go"
	"os"
	"testing"
	"unicode/utf8"
)

var (
	// HuggingFace authentication token read from environment.
	// It can be created in https://huggingface.co
	// Some files may require it for downloading.
	hfAuthToken = os.Getenv("HF_TOKEN")

	KnightsAnalyticsSBertID = "KnightsAnalytics/all-MiniLM-L6-v2"
	FineWebID               = "HuggingFaceFW/fineweb"
	FineWebSampleFile       = "sample/10BT/000_00000.parquet"

	BatchSizes = []int{1, 16, 64}
)

// tokenizedSentence stores the tokenized input for models of a sentence.
type tokenizedSentence struct {
	Encoding [3][]int64 // IDs, Masks, tokenTypeIDs
}

// fineWebEntry: inspection of fields in parquet file done with tool in
// github.com/xitongsys/parquet-go/tool/parquet-tools.
//
// The parquet annotations are described in: https://pkg.go.dev/github.com/parquet-go/parquet-go#SchemaOf
type fineWebEntry struct {
	Text  string  `parquet:"text,snappy"`
	ID    string  `parquet:"id,snappy"`
	Dump  string  `parquet:"dump,snappy"`
	URL   string  `parquet:"url,snappy"`
	Score float64 `parquet:"language_score"`
}

// trimString returns s trimmed to at most maxLength runes. If trimmed it appends "…" at the end.
func trimString(s string, maxLength int) string {
	if utf8.RuneCountInString(s) <= maxLength {
		return s
	}
	runes := []rune(s)
	return string(runes[:maxLength-1]) + "…"
}

func padOrTrim[T any](n int, values []T, padding T) []T {
	if len(values) >= n {
		return values[:n]
	}
	newValues := make([]T, n)
	copy(newValues, values)
	for ii := len(values); ii < n; ii++ {
		newValues[ii] = padding
	}
	return newValues
}

// sampleFineWeb returns the first n tokenized sentences from a 2Gb sample of the FineWeb dataset.
//
// The modelID is used to download the tokenization model.
//
// sequenceLen is the length of each sentence in number of tokens.
// If the original sentence is longer, it is truncated.
// If it is shorter, it is padded.
func sampleFineWeb(modelID string, n, sequenceLen int) []tokenizedSentence {
	results := make([]tokenizedSentence, n)

	// Download repo file.
	repo := hub.New(FineWebID).WithType(hub.RepoTypeDataset).WithAuth(hfAuthToken)
	localSampleFile := must.M1(repo.DownloadFile(FineWebSampleFile))

	// Parquet reading using parquet-go: it's somewhat cumbersome (to open the file it needs its size!?), but it works.
	schema := parquet.SchemaOf(&fineWebEntry{})
	fSize := must.M1(os.Stat(localSampleFile)).Size()
	fReader := must.M1(os.Open(localSampleFile))
	fParquet := must.M1(parquet.OpenFile(fReader, fSize))
	reader := parquet.NewGenericReader[fineWebEntry](fParquet, schema)
	defer func() { _ = reader.Close() }()

	// Create tokenizer.
	repoTokenizer := hub.New(modelID).WithAuth(hfAuthToken)
	localFile := must.M1(repoTokenizer.DownloadFile("tokenizer.json"))
	tokenizer := must.M1(dtok.FromFile(localFile))
	defer func() { _ = tokenizer.Close() }()

	// Read a batch at a time and tokenize.
	const maxBatchSize = 32
	current := 0
	for current < n {
		batchSize := min(maxBatchSize, n-current)
		rows := make([]fineWebEntry, batchSize)
		numRead := must.M1(reader.Read(rows))
		if numRead == 0 {
			break
		}
		for _, row := range rows {
			encoding := tokenizer.EncodeWithOptions(row.Text, false,
				dtok.WithReturnTypeIDs(),
				dtok.WithReturnAttentionMask())
			results[current].Encoding[0] = padOrTrim(sequenceLen,
				sliceMap(encoding.IDs, func(id uint32) int64 { return int64(id) }),
				0)
			results[current].Encoding[1] = padOrTrim(sequenceLen,
				sliceMap(encoding.AttentionMask, func(id uint32) int64 { return int64(id) }),
				0)
			results[current].Encoding[2] = padOrTrim(sequenceLen,
				sliceMap(encoding.TypeIDs, func(id uint32) int64 { return int64(id) }),
				0)
			current++
		}
	}
	if current < n {
		exceptions.Panicf("requested %d sentences to sample, got only %d", n, current)
	}
	return results
}

func benchmarkKnightsSBertXLA(b *testing.B, name, onnxModelPath string, numSentences, sentenceLength, batchSize int) {
	if numSentences < batchSize {
		exceptions.Panicf("batchSize(%d) must be >= to the number of sentences sampled (%d)", batchSize, numSentences)
	}
	tokenizedExamples := sampleFineWeb(KnightsAnalyticsSBertID, numSentences, sentenceLength)

	// Build model:
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
		return outputs[0]
	})
	defer exec.Finalize()

	// Create input tensors:
	var inputTensors [3]*tensors.Tensor // tokenIDs, attentionMask, tokenTypeIDs
	for ii := range inputTensors {
		inputTensors[ii] = tensors.FromShape(shapes.Make(dtypes.Int64, batchSize, sentenceLength))
	}

	current := 0
	testFn := func() {
		// Create batch for each input tensor.
		for inputIdx, t := range inputTensors {
			tensors.MutableFlatData[int64](t, func(flat []int64) {
				for exampleIdx := range batchSize {
					sample := tokenizedExamples[current+exampleIdx]
					copy(flat[exampleIdx*sentenceLength:], sample.Encoding[inputIdx])
				}
			})
		}

		// Execute program.
		outputs := exec.Call(inputTensors[0], inputTensors[1], inputTensors[2])
		for _, output := range outputs {
			output.FinalizeAll()
		}

		// Next batch.
		current += numSentences
		if current+batchSize > numSentences {
			current = 0
		}
	}

	// WarmUp:
	for _ = range 10 {
		testFn()
	}

	// Benchmark:
	b.ResetTimer()

	b.Run(fmt.Sprintf("%s/BatchSize=%d:", name, batchSize),
		func(b *testing.B) {
			for _ = range b.N {
				testFn()
			}
		})
}

func BenchmarkKnightsSBertXLA(b *testing.B) {
	repo := hub.New(KnightsAnalyticsSBertID).WithAuth(hfAuthToken)
	onnxModelPath := must.M1(repo.DownloadFile("model.onnx"))
	for _, batchSize := range BatchSizes {
		benchmarkKnightsSBertXLA(b, "Full", onnxModelPath, 10000, 128, batchSize)
	}
}

func TestKnightsSBert(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	tokenizedExamples := sampleFineWeb(KnightsAnalyticsSBertID, 10, 3)
	for _, example := range tokenizedExamples {
		t.Log(example)
	}
}
