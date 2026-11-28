// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package modeltests implements end-to-end ONNX model parity tests for onnx-gomlx,
// validating GoMLX conversion and execution against golden outputs produced directly
// by ONNX Runtime (ORT).
//
// See docs/ModelTests.md for architecture, CLI flags, and how to add new model tests.
package modeltests

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gomlx/compute/support/humanize"
)

// ModelSpec defines a full-model ONNX parity test stored in testdata/<model>/spec.json.
type ModelSpec struct {
	Name       string        `json:"name"`
	Family     string        `json:"family"` // "text", "tensor"
	RepoID     string        `json:"repo_id"`
	Revision   string        `json:"revision,omitempty"`
	SizeTier   string        `json:"size_tier"` // "small", "medium", "large"
	Tags       []string      `json:"tags,omitempty"`
	Gated      bool          `json:"gated,omitempty"`
	Provenance Provenance    `json:"provenance,omitempty"`
	Variants   []VariantSpec `json:"variants"`
}

// Provenance records how the golden ORT files were produced for reproducibility.
type Provenance struct {
	Generator      string `json:"generator,omitempty"`
	Command        string `json:"command,omitempty"`
	RuntimeVersion string `json:"runtime_version,omitempty"`
	GeneratedAt    string `json:"generated_at,omitempty"`
}

// ModelOptions holds optional overrides applied when parsing/running the ONNX model.
type ModelOptions struct {
	AllowDTypePromotion bool     `json:"allow_dtype_promotion,omitempty"`
	ApproximateGelu     bool     `json:"approximate_gelu,omitempty"`
	SkipBackends        []string `json:"skip_backends,omitempty"`
	SkipDynamic         bool     `json:"skip_dynamic,omitempty"`
}

// VariantSpec defines one testable .onnx file within a model spec (e.g., fp32 vs qint8).
type VariantSpec struct {
	Name              string        `json:"name"`
	RepoID            string        `json:"repo_id,omitempty"`
	Revision          string        `json:"revision,omitempty"`
	ONNXFile          string        `json:"onnx_file"`
	ExternalDataFiles []string      `json:"external_data_files,omitempty"`
	ModelOptions      *ModelOptions `json:"model_options,omitempty"`
	Text              *TextSpec     `json:"text,omitempty"`
	Tensor            *TensorSpec   `json:"tensor,omitempty"`
}

// TextSpec configures NLP ONNX model tests (embedders, classifiers, rerankers, NER).
type TextSpec struct {
	Task               string            `json:"task"` // "embed" | "logits"
	OutputName         string            `json:"output_name"`
	UseSharedSentences bool              `json:"use_shared_sentences,omitempty"`
	PadTokenID         int64             `json:"pad_token_id"`
	Cases              []TextCase        `json:"cases"`
	EmbeddingsFile     string            `json:"embeddings_file,omitempty"`
	BatchEmbeddingsFile string           `json:"batch_embeddings_file,omitempty"`
	EmbeddingDim       int               `json:"embedding_dim,omitempty"`
	SingleLogitsFile   string            `json:"single_logits_file,omitempty"`
	BatchLogitsFile    string            `json:"batch_logits_file,omitempty"`
	BatchLogitsShape   []int             `json:"batch_logits_shape,omitempty"`
	SkipBatchTest      bool              `json:"skip_batch_test,omitempty"`
	Tolerances         NumericTolerances `json:"tolerances,omitempty"`
}

// TextCase stores pre-tokenized inputs (exact tensors fed to ORT) and expected output shape.
type TextCase struct {
	Text          string  `json:"text"`
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask"`
	TokenTypeIDs  []int64 `json:"token_type_ids,omitempty"`
	OutputShape   []int   `json:"output_shape,omitempty"`
}

// TensorSpec configures generic tensor ONNX model tests (vision, audio, tabular, trees).
type TensorSpec struct {
	DynamicAxes map[string][]string `json:"dynamic_axes,omitempty"`
	Cases       []TensorCase        `json:"cases"`
	Tolerances  NumericTolerances   `json:"tolerances,omitempty"`
}

// TensorCase defines one seeded multi-input/multi-output test invocation.
type TensorCase struct {
	Name    string             `json:"name"`
	Seed    uint64             `json:"seed"`
	Inputs  []TensorInputSpec  `json:"inputs"`
	Outputs []TensorOutputSpec `json:"outputs"`
}

// TensorInputSpec defines one deterministic input tensor.
type TensorInputSpec struct {
	Name  string  `json:"name"`
	DType string  `json:"dtype"` // "float32"
	Shape []int   `json:"shape"`
	Dist  string  `json:"dist"` // "uniform", "uniform_signed", "normal", "iris"
	Scale float64 `json:"scale,omitempty"`
}

// TensorOutputSpec defines one golden output tensor from ORT.
type TensorOutputSpec struct {
	Name         string       `json:"name"`
	DType        string       `json:"dtype"` // "float32", "int64"
	Shape        []int        `json:"shape"`
	File         string       `json:"file"`
	SampleStride int          `json:"sample_stride,omitempty"`
	Stats        *TensorStats `json:"stats,omitempty"`
}

// TensorStats summarizes a full float32 output tensor when only strided samples are stored.
type TensorStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	L2Norm float64 `json:"l2_norm"`
}

// NumericTolerances controls numerical comparison thresholds against ORT ground truth.
type NumericTolerances struct {
	MinCosineSim   float64 `json:"min_cosine_sim,omitempty"`
	MaxAbsDiff     float64 `json:"max_abs_diff,omitempty"`
	MaxMeanAbsDiff float64 `json:"max_mean_abs_diff,omitempty"`
	CheckArgmax    bool    `json:"check_argmax,omitempty"`
}

type loadedSpec struct {
	Dir  string
	Spec ModelSpec
}

func loadModelSpecs(testdataDir string) ([]loadedSpec, error) {
	entries, err := os.ReadDir(testdataDir)
	if err != nil {
		return nil, fmt.Errorf("reading testdata dir %q: %w", testdataDir, err)
	}
	var specs []loadedSpec
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(testdataDir, e.Name())
		specPath := filepath.Join(dir, "spec.json")
		data, err := os.ReadFile(specPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("reading %q: %w", specPath, err)
		}
		var spec ModelSpec
		if err := json.Unmarshal(data, &spec); err != nil {
			return nil, fmt.Errorf("parsing %q: %w", specPath, err)
		}
		if spec.Name == "" {
			spec.Name = e.Name()
		}
		specs = append(specs, loadedSpec{Dir: dir, Spec: spec})
	}
	slices.SortFunc(specs, func(a, b loadedSpec) int {
		return strings.Compare(a.Spec.Name, b.Spec.Name)
	})
	return specs, nil
}

func readF32File(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("%q has %d bytes, not a multiple of 4", path, len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

func readI64File(path string) ([]int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("%q has %d bytes, not a multiple of 8", path, len(data))
	}
	out := make([]int64, len(data)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out, nil
}

type vectorMetrics struct {
	CosineSim   float64
	MaxAbsDiff  float64
	MeanAbsDiff float64
	RMSE        float64
}

func computeVectorMetrics(got, want []float32) vectorMetrics {
	if len(got) == 0 || len(got) != len(want) {
		return vectorMetrics{}
	}
	var dot, normGotSq, normWantSq, sumAbs, sumSq, maxAbs float64
	for i := range got {
		g := float64(got[i])
		w := float64(want[i])
		d := math.Abs(g - w)
		dot += g * w
		normGotSq += g * g
		normWantSq += w * w
		sumAbs += d
		sumSq += d * d
		if d > maxAbs {
			maxAbs = d
		}
	}
	n := float64(len(got))
	cosSim := 1.0
	if denom := math.Sqrt(normGotSq) * math.Sqrt(normWantSq); denom > 0 {
		cosSim = dot / denom
	}
	return vectorMetrics{
		CosineSim:   cosSim,
		MaxAbsDiff:  maxAbs,
		MeanAbsDiff: sumAbs / n,
		RMSE:        math.Sqrt(sumSq / n),
	}
}

func computeStatsF32(data []float32) TensorStats {
	if len(data) == 0 {
		return TensorStats{}
	}
	minV, maxV := float64(data[0]), float64(data[0])
	var sum, sumSq float64
	for _, v := range data {
		fv := float64(v)
		if fv < minV {
			minV = fv
		}
		if fv > maxV {
			maxV = fv
		}
		sum += fv
		sumSq += fv * fv
	}
	return TensorStats{
		Min:    minV,
		Max:    maxV,
		Mean:   sum / float64(len(data)),
		L2Norm: math.Sqrt(sumSq),
	}
}

func argmaxF32(vals []float32) int {
	if len(vals) == 0 {
		return -1
	}
	bestIdx := 0
	bestVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > bestVal {
			bestVal = vals[i]
			bestIdx = i
		}
	}
	return bestIdx
}

func tierRank(tier string) int {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "small", "":
		return 1
	case "medium":
		return 2
	case "large":
		return 3
	default:
		return 99
	}
}

func shapeSize(dims []int) int {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return n
}

// runReportRow records timing and peak memory for one (model, variant, backend, mode) execution.
type runReportRow struct {
	Model     string
	Variant   string
	Backend   string
	Mode      string
	Status    string
	LoadTime  time.Duration
	InferTime time.Duration
	Mem       MemoryStats
	Notes     string
}

func printSummaryTable(rows []runReportRow) {
	if len(rows) == 0 {
		return
	}
	headers := []string{"Model", "Variant", "Backend", "Mode", "Status", "Load", "Infer", "Peak RSS", "Delta RSS", "Peak Heap", "Notes"}
	rightAlign := []bool{false, false, false, false, false, true, true, true, true, true, false}

	tableRows := make([][]string, 0, len(rows))
	for _, r := range rows {
		loadStr := "-"
		if r.LoadTime > 0 {
			loadStr = humanize.Duration(r.LoadTime)
		}
		inferStr := "-"
		if r.InferTime > 0 {
			inferStr = humanize.Duration(r.InferTime)
		}
		peakRSS := "-"
		if r.Mem.PeakRSSBytes > 0 {
			peakRSS = humanize.Bytes(r.Mem.PeakRSSBytes)
		}
		deltaRSS := "-"
		if r.Mem.DeltaPeakRSSBytes > 0 {
			deltaRSS = "+" + humanize.Bytes(r.Mem.DeltaPeakRSSBytes)
		}
		peakHeap := "-"
		if r.Mem.PeakHeapAllocBytes > 0 {
			peakHeap = humanize.Bytes(r.Mem.PeakHeapAllocBytes)
		}
		tableRows = append(tableRows, []string{
			r.Model, r.Variant, r.Backend, r.Mode, r.Status,
			loadStr, inferStr, peakRSS, deltaRSS, peakHeap, r.Notes,
		})
	}

	widths := make([]int, len(headers))
	for c, h := range headers {
		widths[c] = utf8.RuneCountInString(h)
	}
	for _, row := range tableRows {
		for c, cell := range row {
			if w := utf8.RuneCountInString(cell); w > widths[c] {
				widths[c] = w
			}
		}
	}

	var b strings.Builder
	writeBorder := func(left, mid, right string) {
		b.WriteString(left)
		for c, w := range widths {
			b.WriteString(strings.Repeat("─", w+2))
			if c+1 < len(widths) {
				b.WriteString(mid)
			}
		}
		b.WriteString(right)
		b.WriteByte('\n')
	}
	writeRow := func(cells []string) {
		b.WriteString("│")
		for c, cell := range cells {
			pad := widths[c] - utf8.RuneCountInString(cell)
			b.WriteByte(' ')
			if rightAlign[c] {
				b.WriteString(strings.Repeat(" ", pad))
				b.WriteString(cell)
			} else {
				b.WriteString(cell)
				b.WriteString(strings.Repeat(" ", pad))
			}
			b.WriteString(" │")
		}
		b.WriteByte('\n')
	}

	writeBorder("┌", "┬", "┐")
	writeRow(headers)
	writeBorder("├", "┼", "┤")
	for _, row := range tableRows {
		writeRow(row)
	}
	writeBorder("└", "┴", "┘")

	fmt.Printf("\n=== ONNX-GoMLX Model Tests Summary ===\n%s", b.String())
}
