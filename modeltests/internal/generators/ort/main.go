// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Command ort generates golden reference outputs for onnx-gomlx/modeltests by running
// .onnx models directly with ONNX Runtime (ORT), without converting them through onnx-gomlx.
//
// This command lives in its own Go module so onnx-gomlx/modeltests has no runtime dependency
// on yalue/onnxruntime_go.
//
// Example usage (with GOWORK=off):
//
//	cd modeltests/internal/generators/ort
//	GOWORK=off go run . -preset all
//	GOWORK=off go run . -preset knights-all-minilm-l6-v2
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	flagORTLib      = flag.String("ort_lib", "", "Path to libonnxruntime shared library (defaults to ORT_SO_PATH or standard install locations)")
	flagTestdataDir = flag.String("testdata", "../../../testdata", "Path to modeltests/testdata directory")
	flagPreset      = flag.String("preset", "", "Comma-separated preset names to generate, or 'all'")
	flagList        = flag.Bool("list", false, "List available presets and exit")
)

// SharedSentences mirrors modeltests.SharedSentences.
type SharedSentences struct {
	Queries            []string `json:"queries"`
	Documents          []string `json:"documents"`
	TokenizerEdgeCases []string `json:"tokenizer_edge_cases"`
}

// ModelSpec mirrors modeltests.ModelSpec.
type ModelSpec struct {
	Name       string        `json:"name"`
	Family     string        `json:"family"`
	RepoID     string        `json:"repo_id"`
	Revision   string        `json:"revision,omitempty"`
	SizeTier   string        `json:"size_tier"`
	Tags       []string      `json:"tags,omitempty"`
	Gated      bool          `json:"gated,omitempty"`
	Provenance Provenance    `json:"provenance,omitempty"`
	Variants   []VariantSpec `json:"variants"`
}

type Provenance struct {
	Generator      string `json:"generator,omitempty"`
	Command        string `json:"command,omitempty"`
	RuntimeVersion string `json:"runtime_version,omitempty"`
	GeneratedAt    string `json:"generated_at,omitempty"`
}

type ModelOptions struct {
	AllowDTypePromotion bool     `json:"allow_dtype_promotion,omitempty"`
	ApproximateGelu     bool     `json:"approximate_gelu,omitempty"`
	SkipBackends        []string `json:"skip_backends,omitempty"`
	SkipDynamic         bool     `json:"skip_dynamic,omitempty"`
}

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

type TextSpec struct {
	Task               string            `json:"task"` // "embed" | "logits"
	OutputName         string            `json:"output_name"`
	UseSharedSentences bool              `json:"use_shared_sentences,omitempty"`
	PadTokenID         int64             `json:"pad_token_id"`
	Cases              []TextCase        `json:"cases"`
	EmbeddingsFile      string            `json:"embeddings_file,omitempty"`
	BatchEmbeddingsFile string            `json:"batch_embeddings_file,omitempty"`
	EmbeddingDim        int               `json:"embedding_dim,omitempty"`
	SingleLogitsFile   string            `json:"single_logits_file,omitempty"`
	BatchLogitsFile    string            `json:"batch_logits_file,omitempty"`
	BatchLogitsShape   []int             `json:"batch_logits_shape,omitempty"`
	SkipBatchTest      bool              `json:"skip_batch_test,omitempty"`
	Tolerances         NumericTolerances `json:"tolerances,omitempty"`
}

type TextCase struct {
	Text          string  `json:"text"`
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask"`
	TokenTypeIDs  []int64 `json:"token_type_ids,omitempty"`
	OutputShape   []int   `json:"output_shape,omitempty"`
}

type TensorSpec struct {
	DynamicAxes map[string][]string `json:"dynamic_axes,omitempty"`
	Cases       []TensorCase        `json:"cases"`
	Tolerances  NumericTolerances   `json:"tolerances,omitempty"`
}

type TensorCase struct {
	Name    string             `json:"name"`
	Seed    uint64             `json:"seed"`
	Inputs  []TensorInputSpec  `json:"inputs"`
	Outputs []TensorOutputSpec `json:"outputs"`
}

type TensorInputSpec struct {
	Name  string  `json:"name"`
	DType string  `json:"dtype"` // "float32", "int64"
	Shape []int   `json:"shape"`
	Dist  string  `json:"dist"` // "uniform", "uniform_signed", "normal", "iris"
	Scale float64 `json:"scale,omitempty"`
}

type TensorOutputSpec struct {
	Name         string       `json:"name"`
	DType        string       `json:"dtype"` // "float32", "int64"
	Shape        []int        `json:"shape"`
	File         string       `json:"file"`
	SampleStride int          `json:"sample_stride,omitempty"`
	Stats        *TensorStats `json:"stats,omitempty"`
}

type TensorStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	L2Norm float64 `json:"l2_norm"`
}

type NumericTolerances struct {
	MinCosineSim   float64 `json:"min_cosine_sim,omitempty"`
	MaxAbsDiff     float64 `json:"max_abs_diff,omitempty"`
	MaxMeanAbsDiff float64 `json:"max_mean_abs_diff,omitempty"`
	CheckArgmax    bool    `json:"check_argmax,omitempty"`
}

type presetVariant struct {
	Name         string
	ONNXFile     string
	ModelOptions *ModelOptions
	Tolerances   NumericTolerances
}

type presetDef struct {
	Name       string
	Family     string
	RepoID     string
	SizeTier   string
	Tags       []string
	TextTask   string // "embed" | "logits"
	TextOut    string // empty = first output
	Variants   []presetVariant
	TensorSpec *TensorSpec
}

var presets = []presetDef{
	{
		Name:     "knights-all-minilm-l6-v2",
		Family:   "text",
		RepoID:   "KnightsAnalytics/all-MiniLM-L6-v2",
		SizeTier: "small",
		Tags:     []string{"text", "embed", "bert", "fp32", "dynamic"},
		TextTask: "embed",
		TextOut:  "last_hidden_state",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-4, MaxMeanAbsDiff: 1e-4},
			},
		},
	},
	{
		Name:     "sentence-transformers-all-minilm-l6-v2",
		Family:   "text",
		RepoID:   "sentence-transformers/all-MiniLM-L6-v2",
		SizeTier: "small",
		Tags:     []string{"text", "embed", "bert", "quantized", "dynamic"},
		TextTask: "embed",
		TextOut:  "last_hidden_state",
		Variants: []presetVariant{
			{
				Name:       "fp32",
				ONNXFile:   "onnx/model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-4, MaxMeanAbsDiff: 1e-4},
			},
			{
				Name:       "qint8_avx512",
				ONNXFile:   "onnx/model_qint8_avx512.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99, MaxAbsDiff: 3e-2, MaxMeanAbsDiff: 1e-2},
			},
			{
				Name:       "quint8_avx2",
				ONNXFile:   "onnx/model_quint8_avx2.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99, MaxAbsDiff: 3e-2, MaxMeanAbsDiff: 1e-2},
			},
		},
	},
	{
		Name:     "all-mpnet-base-v2",
		Family:   "text",
		RepoID:   "sentence-transformers/all-mpnet-base-v2",
		SizeTier: "medium",
		Tags:     []string{"text", "embed", "mpnet", "fp32", "dynamic"},
		TextTask: "embed",
		TextOut:  "last_hidden_state",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "onnx/model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-4, MaxMeanAbsDiff: 1e-4},
			},
		},
	},
	{
		Name:     "distilbert-sst2",
		Family:   "text",
		RepoID:   "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english",
		SizeTier: "small",
		Tags:     []string{"text", "classify", "distilbert", "fp32", "dynamic"},
		TextTask: "logits",
		TextOut:  "logits",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3, CheckArgmax: true},
			},
		},
	},
	{
		Name:     "roberta-go-emotions",
		Family:   "text",
		RepoID:   "SamLowe/roberta-base-go_emotions-onnx",
		SizeTier: "medium",
		Tags:     []string{"text", "classify", "roberta", "fp32", "dynamic"},
		TextTask: "logits",
		TextOut:  "logits",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "onnx/model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3, CheckArgmax: true},
			},
		},
	},
	{
		Name:     "distilbert-ner",
		Family:   "text",
		RepoID:   "KnightsAnalytics/distilbert-NER",
		SizeTier: "small",
		Tags:     []string{"text", "ner", "distilbert", "fp32", "dynamic"},
		TextTask: "logits",
		TextOut:  "logits",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3},
			},
		},
	},
	{
		Name:     "jina-reranker-v1-tiny-en",
		Family:   "text",
		RepoID:   "jinaai/jina-reranker-v1-tiny-en",
		SizeTier: "small",
		Tags:     []string{"text", "reranker", "alibi", "fp32", "dynamic"},
		TextTask: "logits",
		TextOut:  "logits",
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "onnx/model.onnx",
				Tolerances: NumericTolerances{MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3},
			},
		},
	},
	{
		Name:     "deberta-v3-zeroshot",
		Family:   "text",
		RepoID:   "KnightsAnalytics/deberta-v3-base-zeroshot-v1",
		SizeTier: "medium",
		Tags:     []string{"text", "classify", "deberta", "fp32"},
		TextTask: "logits",
		TextOut:  "logits",
		Variants: []presetVariant{
			{
				Name:         "default",
				ONNXFile:     "model.onnx",
				ModelOptions: &ModelOptions{SkipDynamic: true},
				Tolerances:   NumericTolerances{MinCosineSim: 0.9999, MaxAbsDiff: 1e-2, MaxMeanAbsDiff: 2e-3, CheckArgmax: true},
			},
		},
	},
	{
		Name:     "iris-decision-tree",
		Family:   "tensor",
		RepoID:   "KnightsAnalytics/iris-decision-tree",
		SizeTier: "small",
		Tags:     []string{"tabular", "tree", "onnxml", "dynamic"},
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "model.onnx",
				Tolerances: NumericTolerances{MaxAbsDiff: 1e-5, MaxMeanAbsDiff: 1e-5},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"input": {"batch", ""},
			},
			Cases: []TensorCase{
				{
					Name: "batch1",
					Seed: 42,
					Inputs: []TensorInputSpec{
						{Name: "input", DType: "float32", Shape: []int{1, 4}, Dist: "iris"},
					},
				},
				{
					Name: "batch6",
					Seed: 43,
					Inputs: []TensorInputSpec{
						{Name: "input", DType: "float32", Shape: []int{6, 4}, Dist: "iris"},
					},
				},
			},
		},
	},
	{
		Name:     "resnet50",
		Family:   "tensor",
		RepoID:   "KnightsAnalytics/resnet50",
		SizeTier: "small",
		Tags:     []string{"vision", "cnn", "nchw", "fp32", "dynamic"},
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "resnet50-v1-12.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3, CheckArgmax: true},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"data": {"batch", "", "", ""},
			},
			Cases: []TensorCase{
				{
					Name: "batch1",
					Seed: 101,
					Inputs: []TensorInputSpec{
						{Name: "data", DType: "float32", Shape: []int{1, 3, 224, 224}, Dist: "normal", Scale: 1.0},
					},
				},
				{
					Name: "batch2",
					Seed: 102,
					Inputs: []TensorInputSpec{
						{Name: "data", DType: "float32", Shape: []int{2, 3, 224, 224}, Dist: "normal", Scale: 1.0},
					},
				},
			},
		},
	},
	{
		Name:     "inception-v3",
		Family:   "tensor",
		RepoID:   "recursionerr/nsfw_01",
		SizeTier: "small",
		Tags:     []string{"vision", "cnn", "nhwc", "fp32", "dynamic"},
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "inception_v3.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3, CheckArgmax: true},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"input_1:0": {"batch", "", "", ""},
			},
			Cases: []TensorCase{
				{
					Name: "batch1",
					Seed: 201,
					Inputs: []TensorInputSpec{
						{Name: "input_1:0", DType: "float32", Shape: []int{1, 299, 299, 3}, Dist: "uniform", Scale: 1.0},
					},
				},
				{
					Name: "batch2",
					Seed: 202,
					Inputs: []TensorInputSpec{
						{Name: "input_1:0", DType: "float32", Shape: []int{2, 299, 299, 3}, Dist: "uniform", Scale: 1.0},
					},
				},
			},
		},
	},
	{
		Name:     "dino-vits16",
		Family:   "tensor",
		RepoID:   "Xenova/dino-vits16",
		SizeTier: "small",
		Tags:     []string{"vision", "vit", "fp32", "dynamic"},
		Variants: []presetVariant{
			{
				Name:       "default",
				ONNXFile:   "onnx/model.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"pixel_values": {"batch", "", "", ""},
			},
			Cases: []TensorCase{
				{
					Name: "batch1_224",
					Seed: 301,
					Inputs: []TensorInputSpec{
						{Name: "pixel_values", DType: "float32", Shape: []int{1, 3, 224, 224}, Dist: "normal", Scale: 1.0},
					},
				},
				{
					Name: "batch2_224",
					Seed: 302,
					Inputs: []TensorInputSpec{
						{Name: "pixel_values", DType: "float32", Shape: []int{2, 3, 224, 224}, Dist: "normal", Scale: 1.0},
					},
				},
			},
		},
	},
	{
		Name:     "segformer-b0",
		Family:   "tensor",
		RepoID:   "Xenova/segformer-b0-finetuned-ade-512-512",
		SizeTier: "small",
		Tags:     []string{"vision", "segmentation", "fp32"},
		Variants: []presetVariant{
			{
				Name:         "default",
				ONNXFile:     "onnx/model.onnx",
				ModelOptions: &ModelOptions{SkipDynamic: true},
				Tolerances:   NumericTolerances{MinCosineSim: 0.99999, MaxAbsDiff: 5e-3, MaxMeanAbsDiff: 1e-3},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"pixel_values": {"batch", "", "", ""},
			},
			Cases: []TensorCase{
				{
					Name: "batch1_128",
					Seed: 401,
					Inputs: []TensorInputSpec{
						{Name: "pixel_values", DType: "float32", Shape: []int{1, 3, 128, 128}, Dist: "normal", Scale: 1.0},
					},
				},
				{
					Name: "batch2_128",
					Seed: 402,
					Inputs: []TensorInputSpec{
						{Name: "pixel_values", DType: "float32", Shape: []int{2, 3, 128, 128}, Dist: "normal", Scale: 1.0},
					},
				},
			},
		},
	},
	{
		Name:     "wav2vec2-base-q4",
		Family:   "tensor",
		RepoID:   "Xenova/wav2vec2-base-960h",
		SizeTier: "small",
		Tags:     []string{"audio", "speech", "q4", "matmulnbits", "dynamic"},
		Variants: []presetVariant{
			{
				Name:       "q4",
				ONNXFile:   "onnx/model_q4.onnx",
				Tolerances: NumericTolerances{MinCosineSim: 0.999, MaxAbsDiff: 5e-2, MaxMeanAbsDiff: 5e-3},
			},
		},
		TensorSpec: &TensorSpec{
			DynamicAxes: map[string][]string{
				"input_values": {"batch", "samples"},
			},
			Cases: []TensorCase{
				{
					Name: "batch1_16k",
					Seed: 501,
					Inputs: []TensorInputSpec{
						{Name: "input_values", DType: "float32", Shape: []int{1, 16000}, Dist: "uniform_signed", Scale: 0.5},
					},
				},
				{
					Name: "batch2_8k",
					Seed: 502,
					Inputs: []TensorInputSpec{
						{Name: "input_values", DType: "float32", Shape: []int{2, 8000}, Dist: "uniform_signed", Scale: 0.5},
					},
				},
			},
		},
	},
}

func resolveORTLibPath(flagVal string) (string, error) {
	if flagVal != "" {
		return flagVal, nil
	}
	if env := os.Getenv("ORT_SO_PATH"); env != "" {
		return env, nil
	}
	home, _ := os.UserHomeDir()
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			filepath.Join(home, ".local/lib/libonnxruntime.dylib"),
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
	default:
		candidates = []string{
			filepath.Join(home, ".local/lib/libonnxruntime.so"),
			filepath.Join(home, ".local/lib/onnxruntime/libonnxruntime.so"),
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
		}
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("could not find libonnxruntime; pass -ort_lib or set ORT_SO_PATH")
}

func main() {
	flag.Parse()

	if *flagList {
		for _, p := range presets {
			fmt.Printf("%-38s %-7s %-7s %s (%d variants)\n", p.Name, p.Family, p.SizeTier, p.RepoID, len(p.Variants))
		}
		return
	}

	if *flagPreset == "" {
		fmt.Fprintln(os.Stderr, "Specify -preset=<name>[,<name>...] or -preset=all (use -list to see presets)")
		os.Exit(2)
	}

	ortLib, err := resolveORTLibPath(*flagORTLib)
	if err != nil {
		fatalf("%v", err)
	}
	ort.SetSharedLibraryPath(ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		fatalf("ort.InitializeEnvironment(%q): %v", ortLib, err)
	}
	defer func() { _ = ort.DestroyEnvironment() }()
	ortVersion := ort.GetVersion()
	fmt.Printf("Using ONNX Runtime %s (%s)\n", ortVersion, ortLib)

	sharedPath := filepath.Join(*flagTestdataDir, "reference_sentences.json")
	sharedData, err := os.ReadFile(sharedPath)
	if err != nil {
		fatalf("reading %s: %v", sharedPath, err)
	}
	var shared SharedSentences
	if err := json.Unmarshal(sharedData, &shared); err != nil {
		fatalf("parsing %s: %v", sharedPath, err)
	}

	selected := selectPresets(*flagPreset)
	if len(selected) == 0 {
		fatalf("no presets matched %q", *flagPreset)
	}

	for _, p := range selected {
		fmt.Printf("\n=== Generating %s (%s) ===\n", p.Name, p.RepoID)
		if err := generatePreset(p, &shared, ortVersion); err != nil {
			fatalf("preset %s failed: %v", p.Name, err)
		}
	}
}

func selectPresets(raw string) []presetDef {
	if strings.EqualFold(strings.TrimSpace(raw), "all") {
		return presets
	}
	want := strings.Split(raw, ",")
	for i := range want {
		want[i] = strings.TrimSpace(want[i])
	}
	var out []presetDef
	for _, p := range presets {
		if slices.Contains(want, p.Name) {
			out = append(out, p)
		}
	}
	return out
}

func generatePreset(p presetDef, shared *SharedSentences, ortVersion string) error {
	repo := hub.New(p.RepoID)
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		repo.WithAuth(tok)
	}
	if err := repo.DownloadInfo(false); err != nil {
		return fmt.Errorf("DownloadInfo(%s): %w", p.RepoID, err)
	}
	commitHash := ""
	if info := repo.Info(); info != nil {
		commitHash = info.CommitHash
	}

	outDir := filepath.Join(*flagTestdataDir, p.Name)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	spec := ModelSpec{
		Name:     p.Name,
		Family:   p.Family,
		RepoID:   p.RepoID,
		Revision: commitHash,
		SizeTier: p.SizeTier,
		Tags:     p.Tags,
		Provenance: Provenance{
			Generator:      "modeltests/internal/generators/ort",
			Command:        fmt.Sprintf("GOWORK=off go run ./modeltests/internal/generators/ort -preset=%s", p.Name),
			RuntimeVersion: fmt.Sprintf("onnxruntime %s (CPU)", ortVersion),
			GeneratedAt:    time.Now().UTC().Format(time.RFC3339),
		},
	}

	for _, pv := range p.Variants {
		onnxPath, err := repo.DownloadFile(pv.ONNXFile)
		if err != nil {
			return fmt.Errorf("DownloadFile(%s): %w", pv.ONNXFile, err)
		}
		fmt.Printf("  variant=%s file=%s\n", pv.Name, pv.ONNXFile)

		inInfos, outInfos, err := ort.GetInputOutputInfo(onnxPath)
		if err != nil {
			return fmt.Errorf("GetInputOutputInfo(%s): %w", onnxPath, err)
		}
		inNames := make([]string, len(inInfos))
		for i, info := range inInfos {
			inNames[i] = info.Name
			fmt.Printf("    IN : %s %s %v\n", info.Name, info.DataType, info.Dimensions)
		}
		outNames := make([]string, len(outInfos))
		for i, info := range outInfos {
			outNames[i] = info.Name
			fmt.Printf("    OUT: %s %s %v\n", info.Name, info.DataType, info.Dimensions)
		}

		vSpec := VariantSpec{
			Name:         pv.Name,
			ONNXFile:     pv.ONNXFile,
			ModelOptions: pv.ModelOptions,
		}

		switch p.Family {
		case "text":
			ts, err := generateTextVariant(repo, onnxPath, outDir, pv, p, inNames, outNames, shared)
			if err != nil {
				return err
			}
			vSpec.Text = ts
		case "tensor":
			ts, err := generateTensorVariant(onnxPath, outDir, pv, p, inInfos, outInfos)
			if err != nil {
				return err
			}
			vSpec.Tensor = ts
		default:
			return fmt.Errorf("unsupported family %q", p.Family)
		}
		spec.Variants = append(spec.Variants, vSpec)
	}

	specBytes, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	specBytes = append(specBytes, '\n')
	specPath := filepath.Join(outDir, "spec.json")
	if err := os.WriteFile(specPath, specBytes, 0o644); err != nil {
		return err
	}
	fmt.Printf("  wrote %s\n", specPath)
	return nil
}

func generateTextVariant(
	repo *hub.Repo,
	onnxPath, outDir string,
	pv presetVariant,
	p presetDef,
	inNames, outNames []string,
	shared *SharedSentences,
) (*TextSpec, error) {
	tok, err := tokenizers.New(repo)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer for %s: %w", p.RepoID, err)
	}
	padID := int64(0)
	if id, err := tok.SpecialTokenID(api.TokPad); err == nil {
		padID = int64(id)
	}

	targetOut := p.TextOut
	if targetOut == "" {
		targetOut = outNames[0]
	}
	if !slices.Contains(outNames, targetOut) {
		return nil, fmt.Errorf("output %q not found in model outputs %v", targetOut, outNames)
	}

	hasTokenTypes := slices.Contains(inNames, "token_type_ids")
	session, err := ort.NewDynamicAdvancedSession(onnxPath, inNames, []string{targetOut}, nil)
	if err != nil {
		return nil, fmt.Errorf("NewDynamicAdvancedSession: %w", err)
	}
	defer func() { _ = session.Destroy() }()

	sentences := make([]string, 0, len(shared.Queries)+len(shared.Documents))
	sentences = append(sentences, shared.Queries...)
	sentences = append(sentences, shared.Documents...)

	cases := make([]TextCase, len(sentences))
	maxLen := 0
	for i, s := range sentences {
		idsInt := tok.Encode(s)
		ids := make([]int64, len(idsInt))
		mask := make([]int64, len(idsInt))
		var types []int64
		if hasTokenTypes {
			types = make([]int64, len(idsInt))
		}
		for j, id := range idsInt {
			ids[j] = int64(id)
			mask[j] = 1
		}
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
		cases[i] = TextCase{
			Text:          s,
			InputIDs:      ids,
			AttentionMask: mask,
			TokenTypeIDs:  types,
		}
	}

	suffix := ""
	if pv.Name != "" && pv.Name != "default" {
		suffix = "_" + pv.Name
	}

	ts := &TextSpec{
		Task:               p.TextTask,
		OutputName:         targetOut,
		UseSharedSentences: true,
		PadTokenID:         padID,
		Tolerances:         pv.Tolerances,
	}

	// 1. Run single-sentence inference for every sentence.
	var allEmbeddings []float32
	var allSingleLogits []float32
	for i := range cases {
		c := &cases[i]
		seqLen := int64(len(c.InputIDs))
		outData, outShape, err := runTextORT(session, inNames, 1, seqLen, c.InputIDs, c.AttentionMask, c.TokenTypeIDs)
		if err != nil {
			return nil, fmt.Errorf("single sentence %d: %w", i, err)
		}
		switch p.TextTask {
		case "embed":
			if len(outShape) != 3 {
				return nil, fmt.Errorf("embed output %q expected rank 3, got %v", targetOut, outShape)
			}
			dim := outShape[2]
			ts.EmbeddingDim = dim
			emb := meanPoolAndL2Normalize(outData, c.AttentionMask, 1, int(seqLen), dim)
			allEmbeddings = append(allEmbeddings, emb...)
		case "logits":
			c.OutputShape = outShape
			allSingleLogits = append(allSingleLogits, outData...)
		default:
			return nil, fmt.Errorf("unknown text task %q", p.TextTask)
		}
	}
	ts.Cases = cases

	if p.TextTask == "embed" {
		embFile := fmt.Sprintf("embeddings%s.f32", suffix)
		if err := writeF32File(filepath.Join(outDir, embFile), allEmbeddings); err != nil {
			return nil, err
		}
		ts.EmbeddingsFile = embFile
		fmt.Printf("    wrote %s (%d sentences x %d dims)\n", embFile, len(cases), ts.EmbeddingDim)
	} else {
		singleFile := fmt.Sprintf("logits_single%s.f32", suffix)
		if err := writeF32File(filepath.Join(outDir, singleFile), allSingleLogits); err != nil {
			return nil, err
		}
		ts.SingleLogitsFile = singleFile
		fmt.Printf("    wrote %s (%d floats)\n", singleFile, len(allSingleLogits))
	}

	// 2. Run padded batch inference across all sentences.
	batchSize := len(cases)
	bIDs := make([]int64, batchSize*maxLen)
	bMask := make([]int64, batchSize*maxLen)
	bTypes := make([]int64, batchSize*maxLen)
	for i := range bIDs {
		bIDs[i] = padID
	}
	for i, c := range cases {
		copy(bIDs[i*maxLen:], c.InputIDs)
		copy(bMask[i*maxLen:], c.AttentionMask)
		if hasTokenTypes {
			copy(bTypes[i*maxLen:], c.TokenTypeIDs)
		}
	}
	batchData, batchShape, err := runTextORT(session, inNames, int64(batchSize), int64(maxLen), bIDs, bMask, bTypes)
	if err != nil {
		return nil, fmt.Errorf("batch inference: %w", err)
	}
	if p.TextTask == "embed" {
		// Save ORT's own batched embeddings as ground truth for embed_batch (important for
		// DynamicQuantizeLinear models where per-tensor min/max differs between B=1 and B=N).
		batchEmb := meanPoolAndL2Normalize(batchData, bMask, batchSize, maxLen, ts.EmbeddingDim)
		maxDiff := 0.0
		for i := range batchEmb {
			if d := math.Abs(float64(batchEmb[i] - allEmbeddings[i])); d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-5 {
			batchEmbFile := fmt.Sprintf("embeddings_batch%s.f32", suffix)
			if err := writeF32File(filepath.Join(outDir, batchEmbFile), batchEmb); err != nil {
				return nil, err
			}
			ts.BatchEmbeddingsFile = batchEmbFile
			fmt.Printf("    wrote %s (ORT single-vs-batch max_abs_diff=%.3e)\n", batchEmbFile, maxDiff)
		} else {
			fmt.Printf("    ORT single-vs-batch embedding max_abs_diff=%.3e\n", maxDiff)
		}
	} else {
		batchFile := fmt.Sprintf("logits_batch%s.f32", suffix)
		if err := writeF32File(filepath.Join(outDir, batchFile), batchData); err != nil {
			return nil, err
		}
		ts.BatchLogitsFile = batchFile
		ts.BatchLogitsShape = batchShape
		fmt.Printf("    wrote %s shape=%v (%d floats)\n", batchFile, batchShape, len(batchData))
	}
	return ts, nil
}

func runTextORT(
	session *ort.DynamicAdvancedSession,
	inNames []string,
	batch, seqLen int64,
	ids, mask, types []int64,
) ([]float32, []int, error) {
	shape := ort.NewShape(batch, seqLen)
	inVals := make([]ort.Value, len(inNames))
	defer func() {
		for _, v := range inVals {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()
	for i, name := range inNames {
		var data []int64
		switch name {
		case "input_ids":
			data = ids
		case "attention_mask":
			data = mask
		case "token_type_ids":
			data = types
		default:
			return nil, nil, fmt.Errorf("unexpected text model input %q", name)
		}
		t, err := ort.NewTensor(shape, data)
		if err != nil {
			return nil, nil, fmt.Errorf("NewTensor(%s): %w", name, err)
		}
		inVals[i] = t
	}
	outVals := make([]ort.Value, 1)
	if err := session.Run(inVals, outVals); err != nil {
		return nil, nil, fmt.Errorf("session.Run: %w", err)
	}
	defer func() {
		if outVals[0] != nil {
			_ = outVals[0].Destroy()
		}
	}()
	outT, ok := outVals[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("expected float32 output tensor, got %T", outVals[0])
	}
	raw := outT.GetData()
	cp := make([]float32, len(raw))
	copy(cp, raw)
	s64 := outT.GetShape()
	outShape := make([]int, len(s64))
	for i, d := range s64 {
		outShape[i] = int(d)
	}
	return cp, outShape, nil
}

func meanPoolAndL2Normalize(hidden []float32, mask []int64, batch, seqLen, dim int) []float32 {
	out := make([]float32, batch*dim)
	for b := range batch {
		var valid float64
		sum := make([]float64, dim)
		for s := range seqLen {
			if mask[b*seqLen+s] == 0 {
				continue
			}
			valid++
			off := (b*seqLen + s) * dim
			for d := range dim {
				sum[d] += float64(hidden[off+d])
			}
		}
		if valid == 0 {
			valid = 1
		}
		var normSq float64
		for d := range dim {
			sum[d] /= valid
			normSq += sum[d] * sum[d]
		}
		norm := math.Sqrt(normSq)
		if norm < 1e-12 {
			norm = 1e-12
		}
		for d := range dim {
			out[b*dim+d] = float32(sum[d] / norm)
		}
	}
	return out
}

const maxFullOutputElements = 65536

func generateTensorVariant(
	onnxPath, outDir string,
	pv presetVariant,
	p presetDef,
	inInfos, outInfos []ort.InputOutputInfo,
) (*TensorSpec, error) {
	inNames := make([]string, len(inInfos))
	for i, info := range inInfos {
		inNames[i] = info.Name
	}
	outNames := make([]string, len(outInfos))
	for i, info := range outInfos {
		outNames[i] = info.Name
	}

	session, err := ort.NewDynamicAdvancedSession(onnxPath, inNames, outNames, nil)
	if err != nil {
		return nil, fmt.Errorf("NewDynamicAdvancedSession: %w", err)
	}
	defer func() { _ = session.Destroy() }()

	suffix := ""
	if pv.Name != "" && pv.Name != "default" {
		suffix = "_" + pv.Name
	}

	ts := &TensorSpec{
		DynamicAxes: p.TensorSpec.DynamicAxes,
		Tolerances:  pv.Tolerances,
	}

	for _, cDef := range p.TensorSpec.Cases {
		inVals := make([]ort.Value, len(inNames))
		inputByName := make(map[string]TensorInputSpec, len(cDef.Inputs))
		for _, inp := range cDef.Inputs {
			inputByName[inp.Name] = inp
		}
		rng := rand.New(rand.NewPCG(cDef.Seed, 0))
		for i, name := range inNames {
			inp, ok := inputByName[name]
			if !ok {
				return nil, fmt.Errorf("case %q missing input %q", cDef.Name, name)
			}
			val, err := makeORTInputTensor(inp, rng)
			if err != nil {
				return nil, fmt.Errorf("case %q input %q: %w", cDef.Name, name, err)
			}
			inVals[i] = val
		}

		outVals := make([]ort.Value, len(outNames))
		runErr := session.Run(inVals, outVals)
		for _, v := range inVals {
			if v != nil {
				_ = v.Destroy()
			}
		}
		if runErr != nil {
			return nil, fmt.Errorf("case %q session.Run: %w", cDef.Name, runErr)
		}

		tc := TensorCase{
			Name:   cDef.Name,
			Seed:   cDef.Seed,
			Inputs: cDef.Inputs,
		}
		for oi, ov := range outVals {
			s64 := ov.GetShape()
			shape := make([]int, len(s64))
			for j, d := range s64 {
				shape[j] = int(d)
			}
			switch t := ov.(type) {
			case *ort.Tensor[float32]:
				data := t.GetData()
				stats := computeStatsF32(data)
				stride := 1
				saved := data
				if len(data) > maxFullOutputElements {
					stride = (len(data) + 4095) / 4096
					saved = make([]float32, 0, (len(data)+stride-1)/stride)
					for idx := 0; idx < len(data); idx += stride {
						saved = append(saved, data[idx])
					}
				}
				file := fmt.Sprintf("%s_out%d%s.f32", cDef.Name, oi, suffix)
				if err := writeF32File(filepath.Join(outDir, file), saved); err != nil {
					_ = ov.Destroy()
					return nil, err
				}
				oSpec := TensorOutputSpec{
					Name:  outNames[oi],
					DType: "float32",
					Shape: shape,
					File:  file,
					Stats: &stats,
				}
				if stride > 1 {
					oSpec.SampleStride = stride
				}
				tc.Outputs = append(tc.Outputs, oSpec)
				fmt.Printf("    case=%s out[%d]=%s float32 shape=%v saved=%d (stride=%d)\n",
					cDef.Name, oi, outNames[oi], shape, len(saved), stride)
			case *ort.Tensor[int64]:
				data := t.GetData()
				file := fmt.Sprintf("%s_out%d%s.i64", cDef.Name, oi, suffix)
				if err := writeI64File(filepath.Join(outDir, file), data); err != nil {
					_ = ov.Destroy()
					return nil, err
				}
				tc.Outputs = append(tc.Outputs, TensorOutputSpec{
					Name:  outNames[oi],
					DType: "int64",
					Shape: shape,
					File:  file,
				})
				fmt.Printf("    case=%s out[%d]=%s int64 shape=%v vals=%v\n",
					cDef.Name, oi, outNames[oi], shape, data)
			default:
				_ = ov.Destroy()
				return nil, fmt.Errorf("case %q out[%d]=%s unsupported ORT tensor type %T", cDef.Name, oi, outNames[oi], ov)
			}
			_ = ov.Destroy()
		}
		ts.Cases = append(ts.Cases, tc)
	}
	return ts, nil
}

func makeORTInputTensor(spec TensorInputSpec, rng *rand.Rand) (ort.Value, error) {
	shape64 := make(ort.Shape, len(spec.Shape))
	size := 1
	for i, d := range spec.Shape {
		shape64[i] = int64(d)
		size *= d
	}
	scale := spec.Scale
	if scale == 0 {
		scale = 1.0
	}
	switch spec.DType {
	case "float32":
		data := make([]float32, size)
		fillFloat32(data, spec.Dist, scale, rng)
		return ort.NewTensor(shape64, data)
	default:
		return nil, fmt.Errorf("unsupported input dtype %q", spec.DType)
	}
}

// fillFloat32 MUST stay bit-identical to modeltests/tensor_test.go's fillFloat32.
func fillFloat32(data []float32, dist string, scale float64, rng *rand.Rand) {
	switch dist {
	case "uniform":
		for i := range data {
			data[i] = float32(rng.Float64() * scale)
		}
	case "uniform_signed":
		for i := range data {
			data[i] = float32((rng.Float64()*2.0 - 1.0) * scale)
		}
	case "normal":
		for i := range data {
			data[i] = float32(rng.NormFloat64() * scale)
		}
	case "iris":
		// Realistic Iris dataset rows spanning all 3 classes (setosa, versicolor, virginica).
		samples := [][4]float32{
			{5.1, 3.5, 1.4, 0.2}, // setosa
			{6.0, 2.7, 4.5, 1.5}, // versicolor
			{6.7, 3.1, 5.6, 2.4}, // virginica
			{4.9, 3.0, 1.4, 0.2}, // setosa
			{5.5, 2.4, 3.8, 1.1}, // versicolor
			{7.2, 3.6, 6.1, 2.5}, // virginica
		}
		for r := 0; r*4+3 < len(data); r++ {
			s := samples[r%len(samples)]
			copy(data[r*4:], s[:])
		}
	default:
		panic(fmt.Sprintf("unknown input distribution %q", dist))
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

func writeF32File(path string, data []float32) error {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return os.WriteFile(path, buf, 0o644)
}

func writeI64File(path string, data []int64) error {
	buf := make([]byte, len(data)*8)
	for i, v := range data {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(v))
	}
	return os.WriteFile(path, buf, 0o644)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "ERROR: "+format+"\n", args...)
	os.Exit(1)
}
