package benchmarks

import (
	dtok "github.com/daulet/tokenizers"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/go-huggingface/hub"
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
)

// tokenizedSentence stores the tokenized input for models of a sentence.
type tokenizedSentence struct {
	IDs, Masks, tokenTypeIDs []int64
}

// fineWebEntry: inspection of fields in parque file done with tool in
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
			results[current].IDs = padOrTrim(sequenceLen,
				sliceMap(encoding.IDs, func(id uint32) int64 { return int64(id) }),
				0)
			results[current].Masks = padOrTrim(sequenceLen,
				sliceMap(encoding.IDs, func(id uint32) int64 { return int64(id) }),
				0)
			results[current].tokenTypeIDs = padOrTrim(sequenceLen,
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

func TestKnightsSBert(t *testing.T) {
	tokenzinedExamples := sampleFineWeb(KnightsAnalyticsSBertID, 10, 3)
	for _, example := range tokenzinedExamples {
		t.Log(example)
	}
}
