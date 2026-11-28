# ONNX-GoMLX Full Model Tests (`modeltests/`)

This guide documents the end-to-end ONNX model parity test suite under [`modeltests/`](../modeltests/), how to run it across backends and shape modes, and how to add new ONNX models.

---

## 1. Architecture & Philosophy

- **First-class top-level package**: Lives in [`modeltests/`](../modeltests/) so every covered ONNX model, variant, and golden file is immediately visible.
- **Direct ONNX Runtime (ORT) as ground truth**: All golden outputs in `modeltests/testdata/<model>/` are produced by running the raw `.onnx` file directly with ONNX Runtime (via [`modeltests/internal/generators/ort/`](../modeltests/internal/generators/ort/)), completely bypassing `onnx-gomlx` conversion.
- **Isolated generator module**: `modeltests/internal/generators/ort/` has its own `go.mod` (`github.com/yalue/onnxruntime_go`), so running `modeltests` itself has zero runtime dependency on `yalue/onnxruntime_go`.
- **Disabled by default**: Running `go test ./...` skips `modeltests` cleanly unless `-modeltests` (or `GOMLX_MODELTESTS=1`) is set.
- **Tests both Static and Dynamic shapes**: By default (`-mode=both`), every model runs with concrete static shapes on all backends, AND runs a single compiled dynamic-axis graph (`exec.WithDynamicAxes(...)`) across multiple batch sizes / sequence lengths on every backend that reports `Capabilities().HasDynamicShapes()` (such as `go` and `onnx:cpu`).
- **Multi-backend matrix**: Test one or more backend configurations in a single invocation with `-backends="go;go:ops_sequential;xla:cpu;onnx:cpu"`.
- **Per-model peak memory & latency summary**: Reports load time, inference time, peak RSS, delta RSS (via `/proc/self/clear_refs` `VmHWM` on Linux + 20ms RSS sampler, and `compute/support.MaxRSS()` on macOS), peak Go heap, and accuracy metrics in a summary table at the end of the run.
- **Shared reference corpus**: Text models share [`modeltests/testdata/reference_sentences.json`](../modeltests/testdata/reference_sentences.json) (Dolly, MS MARCO, and multilingual `"Why is the sky blue?"` in EN/DE/FR/ZH/JA), matching `go-huggingface/modeltests/`.

---

## 2. Running Model Tests

Always pass `-p=1` so tests run sequentially for accurate peak RSS measurements and clean GPU/backend initialization.

```bash
# Default go test skips modeltests:
go test ./modeltests

# Run all 14 ONNX models (16 variants) on the default backend (GOMLX_BACKEND):
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests

# Multi-backend matrix (add -tags=onnx to include the compute-onnx ORT backend):
GOEXPERIMENT=simd go test -v -p=1 -tags=onnx ./modeltests -modeltests \
  -backends="go;go:ops_sequential;xla:cpu;onnx:cpu"

# Filter by size tier, tag, cache presence, mode, or subtest regex:
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -max_tier=small -backends="go"
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -tag=quantized -backends="go"
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -mode=dynamic -backends="go"
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -cached_only -backends="go"
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -run TestModels/knights-all-minilm-l6-v2
```

### Supported Flags

| Flag | Default | Description |
|---|---|---|
| `-modeltests` | `false` | Required to run model tests (or set `GOMLX_MODELTESTS=1`). |
| `-backends` | `""` | Semicolon-separated GoMLX backend configs (e.g. `"go;go:ops_sequential;xla:cpu;onnx:cpu"`). Empty uses `compute.New()` (`GOMLX_BACKEND`). |
| `-mode` | `"both"` | Shape compilation mode: `"static"`, `"dynamic"`, or `"both"` (runs dynamic only on backends with `HasDynamicShapes()`). |
| `-max_tier` | `""` | Maximum size tier: `small`, `medium`, or `large` (empty runs all). |
| `-tag` | `""` | Only run models carrying the specified tag (e.g. `embed`, `vision`, `audio`, `quantized`, `tree`, `dynamic`). |
| `-cached_only` | `false` | Skip models whose `.onnx` file is not already in the local HuggingFace cache. |
| `-strict` | `false` | Fail instead of skipping when a gated/unreachable model cannot be fetched. |
| `-set` | `""` | Semicolon-separated GoMLX store hyperparameter overrides (e.g. `"quantized_matmul=false"`). |

---

## 3. Current Model Roster

All golden files were generated directly with ONNX Runtime (`libonnxruntime` CPU).

| Model | Variant(s) | Family | Size Tier | Dynamic Mode | What It Covers |
|---|---|---|---|---|---|
| `knights-all-minilm-l6-v2` | `default` | `text` (embed) | `small` | Yes | BERT/MiniLM FP32, SDPA + sentinel-to-bool attention mask, 3 inputs |
| `sentence-transformers-all-minilm-l6-v2` | `fp32`, `qint8_avx512`, `quint8_avx2` | `text` (embed) | `small` | Yes | FP32 + INT8/UINT8 `DynamicQuantizeLinear` + `MatMulInteger` quantized dense/QKV fusion |
| `all-mpnet-base-v2` | `default` | `text` (embed) | `medium` | Yes | MPNet relative position bias (additive float SDPA bias), 768-dim |
| `distilbert-sst2` | `default` | `text` (logits) | `small` | Yes | DistilBERT 2-class sequence classification + argmax check |
| `roberta-go-emotions` | `default` | `text` (logits) | `medium` | Yes | RoBERTa 28-class classification, BPE tokenizer |
| `distilbert-ner` | `default` | `text` (logits) | `small` | Yes | DistilBERT 3D token classification (`[B, S, 9]`) |
| `jina-reranker-v1-tiny-en` | `default` | `text` (logits) | `small` | Yes | Jina BERT with ALiBi positional bias (`[B, 1]`) |
| `deberta-v3-zeroshot` | `default` | `text` (logits) | `medium` | Static only (`Tile` repeats) | DeBERTa-v3 disentangled relative attention (`GatherElements`) |
| `iris-decision-tree` | `default` | `tensor` | `small` | Yes | ONNX-ML `TreeEnsembleClassifier`, multi-output (`Int64` labels + `Float32` probabilities) |
| `resnet50` | `default` | `tensor` | `small` | Yes | NCHW 2D CNN (`Conv`, `BatchNormalization`, `MaxPool`, `GlobalAveragePool`, `Gemm`) |
| `inception-v3` | `default` | `tensor` | `small` | Yes | NHWC 2D CNN (`[B, 299, 299, 3] -> [B, 5]`) |
| `dino-vits16` | `default` | `tensor` | `small` | Yes | DINO Vision Transformer (`Conv` patch embed, `Resize` pos-embed, LayerNorm, Attention) |
| `segformer-b0` | `default` | `tensor` | `small` | Static only (`Resize` `sizes`) | SegFormer hierarchical ViT + MLP segmentation decoder |
| `wav2vec2-base-q4` | `q4` | `tensor` | `small` | Yes | 4-bit quantized speech recognition (`com.microsoft::MatMulNBits` + 1D `Conv` + GroupNorm) |

---

## 4. Directory & Schema Overview

```
modeltests/
├── modeltests_test.go                   # CLI flags, TestMain summary hook, TestModels dispatcher
├── runner.go                            # ModelSpec structs, .f32/.i64 readers, metrics, summary table
├── memory.go                            # Per-test RSS & Go heap tracker
├── memory_linux.go                      # /proc/self/status + /proc/self/clear_refs
├── memory_other.go                      # Portable fallback (uses compute/support.MaxRSS())
├── text_test.go                         # Text ONNX harness (embed single+batch, logits single+batch)
├── tensor_test.go                       # Generic tensor ONNX harness (float32 + int64 multi-output)
├── internal/generators/ort/             # Isolated Go module running ORT directly
│   ├── go.mod
│   ├── go.sum
│   └── main.go
└── testdata/
    ├── reference_sentences.json         # Shared text corpus (Dolly + MS MARCO + multilingual)
    └── <model-name>/
        ├── spec.json                    # ModelSpec + provenance + cases + tolerances
        └── *.f32 / *.i64                # Compact little-endian ORT golden outputs
```

### Handling Large Outputs Compactly
For models whose output tensors exceed 65,536 elements (such as `dino-vits16` `[2, 197, 384]` or `segformer-b0` `[2, 150, 32, 32]`), the ORT generator automatically stores:
1. A deterministic strided sample (`sample_stride` chosen so `<= 4,096` floats / `<= 16 KiB` are stored in `.f32`), **plus**
2. Full-tensor global summary statistics (`min`, `max`, `mean`, `l2_norm`) in `spec.json`, verified against every element produced by GoMLX.

This keeps all 14 models' golden `testdata/` under ~600 KiB total while preserving strict numerical verification.

---

## 5. Adding a New ONNX Model Test

1. Add a `presetDef` entry in [`modeltests/internal/generators/ort/main.go`](../modeltests/internal/generators/ort/main.go) (`Family: "text"` or `Family: "tensor"`).
2. Generate the ORT ground truth (from `modeltests/internal/generators/ort/` with `GOWORK=off` so the generator uses its isolated `go.mod`):
   ```bash
   cd modeltests/internal/generators/ort
   GOWORK=off go run . -preset=<your-model-name>
   ```
   Pass `-ort_lib=/path/to/libonnxruntime.so` (or set `ORT_SO_PATH`) if ORT is installed in a non-standard location.
3. Run your new test across backends:
   ```bash
   GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests \
     -run TestModels/<your-model-name> -backends="go;xla:cpu"
   ```
4. Commit `spec.json`, the compact `.f32`/`.i64` golden files, the generator preset update, and update the roster table in this document.
