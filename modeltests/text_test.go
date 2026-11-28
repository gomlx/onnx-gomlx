// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx"
)

func runTextModel(t *testing.T, loaded loadedSpec, backendConfigs []string) {
	spec := loaded.Spec
	for _, variant := range spec.Variants {
		t.Run(variant.Name, func(t *testing.T) {
			if variant.Text == nil {
				t.Fatalf("variant %q has nil text spec", variant.Name)
			}
			repoID := spec.RepoID
			if variant.RepoID != "" {
				repoID = variant.RepoID
			}
			revision := spec.Revision
			if variant.Revision != "" {
				revision = variant.Revision
			}
			repo, ok := openRepoForSpec(t, repoID, revision, spec.Gated, variant.ONNXFile)
			if !ok {
				return
			}

			var skipBackends []string
			skipDynamic := false
			if variant.ModelOptions != nil {
				skipBackends = variant.ModelOptions.SkipBackends
				skipDynamic = variant.ModelOptions.SkipDynamic
			}

			for _, cfg := range backendConfigs {
				b, label, err := createBackend(cfg)
				if err != nil {
					t.Errorf("Failed to initialize backend %q: %+v", cfg, err)
					recordReport(runReportRow{
						Model: spec.Name, Variant: variant.Name, Backend: cfg, Mode: "-",
						Status: "FAIL", Notes: fmt.Sprintf("init err: %v", err),
					})
					continue
				}
				if shouldSkipBackend(skipBackends, cfg, b.Name()) {
					b.Finalize()
					t.Run(formatBackendSubtestName(cfg, label), func(t *testing.T) {
						t.Skipf("Backend %q skipped by spec model_options.skip_backends", label)
					})
					continue
				}
				modes := resolveModesForBackend(b, skipDynamic)
				b.Finalize()
				if len(modes) == 0 {
					t.Run(formatBackendSubtestName(cfg, label), func(t *testing.T) {
						t.Skipf("Mode %q not supported by backend %q", *flagMode, label)
					})
					continue
				}

				t.Run(formatBackendSubtestName(cfg, label), func(t *testing.T) {
					for _, mode := range modes {
						t.Run("mode="+mode, func(t *testing.T) {
							runTextVariantOnBackend(t, loaded, repo, variant, cfg, label, mode == "dynamic")
						})
					}
				})
			}
		})
	}
}

func runTextVariantOnBackend(
	t *testing.T,
	loaded loadedSpec,
	repo *hub.Repo,
	variant VariantSpec,
	backendCfg, backendLabel string,
	dynamic bool,
) {
	spec := loaded.Spec
	ts := variant.Text
	modeName := "static"
	if dynamic {
		modeName = "dynamic"
	}

	cleanupMemory()
	memTracker := startMemoryTracker()
	var loadTime, inferTime time.Duration
	var notes string

	defer func() {
		cleanupMemory()
		memStats := memTracker.Stop()
		t.Logf("[%s/%s/%s/%s] load=%s infer=%s %s",
			spec.Name, variant.Name, backendLabel, modeName,
			loadTime.Round(100*time.Microsecond), inferTime.Round(100*time.Microsecond), memStats.String())
		recordReport(runReportRow{
			Model:     spec.Name,
			Variant:   variant.Name,
			Backend:   backendLabel,
			Mode:      modeName,
			Status:    testStatus(t),
			LoadTime:  loadTime,
			InferTime: inferTime,
			Mem:       memStats,
			Notes:     notes,
		})
	}()

	b, _, err := createBackend(backendCfg)
	if err != nil {
		t.Fatalf("createBackend(%q): %+v", backendCfg, err)
	}
	defer b.Finalize()

	loadStart := time.Now()
	onnxModel, store, err := loadONNXVariant(t, repo, variant)
	if err != nil {
		t.Fatalf("loadONNXVariant: %+v", err)
	}
	defer func() { _ = onnxModel.Close() }()
	loadTime = time.Since(loadStart)
	t.Logf("Loaded %s (%s) on %s (%s) in %s", spec.RepoID, variant.ONNXFile, b.Description(), modeName, loadTime.Round(100*time.Microsecond))

	inNames, _ := onnxModel.Inputs()
	targetOut := ts.OutputName
	if targetOut == "" {
		outNames, _ := onnxModel.Outputs()
		targetOut = outNames[0]
	}

	exec, err := buildTextExec(b, store, onnxModel, inNames, targetOut, dynamic)
	if err != nil {
		t.Fatalf("buildTextExec: %+v", err)
	}
	defer exec.Finalize()

	inferStart := time.Now()
	switch ts.Task {
	case "embed":
		notes = verifyTextEmbeddings(t, loaded.Dir, ts, inNames, exec)
	case "logits":
		notes = verifyTextLogits(t, loaded.Dir, ts, inNames, exec)
	default:
		t.Fatalf("unknown text task %q", ts.Task)
	}
	inferTime = time.Since(inferStart)
}

func buildTextExec(
	b compute.Backend,
	store *model.Store,
	onnxModel onnx.Model,
	inNames []string,
	targetOut string,
	dynamic bool,
) (*model.Exec, error) {
	exec, err := model.NewExec(b, store, func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
		g := inputs[0].Graph()
		inMap := make(map[string]*graph.Node, len(inNames))
		for i, name := range inNames {
			inMap[name] = inputs[i]
		}
		return onnxModel.CallGraph(scope, g, inMap, targetOut)
	})
	if err != nil {
		return nil, err
	}
	if dynamic {
		axisSpecs := make([][]string, len(inNames))
		for i := range inNames {
			axisSpecs[i] = []string{"batch", "seq"}
		}
		exec.WithDynamicAxes(axisSpecs...)
	}
	return exec, nil
}

func runTextExecOnce(
	t *testing.T,
	exec *model.Exec,
	inNames []string,
	batch, seqLen int,
	ids, mask, types []int64,
) ([]float32, []int) {
	t.Helper()
	sh := shapes.Make(dtypes.Int64, batch, seqLen)
	args := make([]any, len(inNames))
	created := make([]*tensors.Tensor, 0, len(inNames))
	for i, name := range inNames {
		var src []int64
		switch name {
		case "input_ids":
			src = ids
		case "attention_mask":
			src = mask
		case "token_type_ids":
			src = types
		default:
			t.Fatalf("unexpected text model input %q", name)
		}
		tens := tensors.FromShape(sh)
		tensors.MutableFlatData[int64](tens, func(flat []int64) {
			copy(flat, src)
		})
		args[i] = tens
		created = append(created, tens)
	}
	defer func() {
		for _, tens := range created {
			tens.FinalizeAll()
		}
	}()

	outs, err := exec.Call(args...)
	if err != nil {
		t.Fatalf("exec.Call(batch=%d, seqLen=%d): %+v", batch, seqLen, err)
	}
	defer func() {
		for _, o := range outs {
			o.FinalizeAll()
		}
	}()
	if len(outs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outs))
	}
	flat := tensors.MustCopyFlatData[float32](outs[0])
	dims := append([]int(nil), outs[0].Shape().Dimensions...)
	return flat, dims
}

func verifyTextEmbeddings(t *testing.T, specDir string, ts *TextSpec, inNames []string, exec *model.Exec) string {
	refPath := filepath.Join(specDir, ts.EmbeddingsFile)
	refEmb, err := readF32File(refPath)
	if err != nil {
		t.Fatalf("readF32File(%s): %v", refPath, err)
	}
	dim := ts.EmbeddingDim
	if len(refEmb) != len(ts.Cases)*dim {
		t.Fatalf("expected %d floats in %s, got %d", len(ts.Cases)*dim, refPath, len(refEmb))
	}

	tol := ts.Tolerances
	if tol.MinCosineSim == 0 {
		tol.MinCosineSim = 0.999
	}
	if tol.MaxAbsDiff == 0 {
		tol.MaxAbsDiff = 5e-3
	}
	if tol.MaxMeanAbsDiff == 0 {
		tol.MaxMeanAbsDiff = 1e-3
	}

	worstCos := 1.0
	worstMaxAbs := 0.0

	t.Run("embed_single", func(t *testing.T) {
		for i, c := range ts.Cases {
			seqLen := len(c.InputIDs)
			raw, dims := runTextExecOnce(t, exec, inNames, 1, seqLen, c.InputIDs, c.AttentionMask, c.TokenTypeIDs)
			if len(dims) != 3 || dims[0] != 1 || dims[1] != seqLen || dims[2] != dim {
				t.Fatalf("sentence %d: got shape %v, want [1 %d %d]", i, dims, seqLen, dim)
			}
			got := meanPoolAndL2Normalize(raw, c.AttentionMask, 1, seqLen, dim)
			want := refEmb[i*dim : (i+1)*dim]
			m := computeVectorMetrics(got, want)
			if m.CosineSim < worstCos {
				worstCos = m.CosineSim
			}
			if m.MaxAbsDiff > worstMaxAbs {
				worstMaxAbs = m.MaxAbsDiff
			}
			t.Logf("sentence %d (%d toks): cos=%.7f max_abs=%.3e mean_abs=%.3e", i, seqLen, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
			if m.CosineSim < tol.MinCosineSim {
				t.Errorf("sentence %d cosine sim %.7f < %.7f", i, m.CosineSim, tol.MinCosineSim)
			}
			if m.MaxAbsDiff > tol.MaxAbsDiff {
				t.Errorf("sentence %d max abs diff %.3e > %.3e", i, m.MaxAbsDiff, tol.MaxAbsDiff)
			}
			if m.MeanAbsDiff > tol.MaxMeanAbsDiff {
				t.Errorf("sentence %d mean abs diff %.3e > %.3e", i, m.MeanAbsDiff, tol.MaxMeanAbsDiff)
			}
		}
	})

	if !ts.SkipBatchTest && len(ts.Cases) > 1 {
		t.Run("embed_batch", func(t *testing.T) {
			refBatchEmb := refEmb
			if ts.BatchEmbeddingsFile != "" {
				var err error
				refBatchEmb, err = readF32File(filepath.Join(specDir, ts.BatchEmbeddingsFile))
				if err != nil {
					t.Fatalf("readF32File(%s): %v", ts.BatchEmbeddingsFile, err)
				}
				if len(refBatchEmb) != len(ts.Cases)*dim {
					t.Fatalf("expected %d floats in %s, got %d", len(ts.Cases)*dim, ts.BatchEmbeddingsFile, len(refBatchEmb))
				}
			}
			batchSize := len(ts.Cases)
			maxLen := 0
			for _, c := range ts.Cases {
				if len(c.InputIDs) > maxLen {
					maxLen = len(c.InputIDs)
				}
			}
			bIDs := make([]int64, batchSize*maxLen)
			bMask := make([]int64, batchSize*maxLen)
			bTypes := make([]int64, batchSize*maxLen)
			for i := range bIDs {
				bIDs[i] = ts.PadTokenID
			}
			for i, c := range ts.Cases {
				copy(bIDs[i*maxLen:], c.InputIDs)
				copy(bMask[i*maxLen:], c.AttentionMask)
				if len(c.TokenTypeIDs) > 0 {
					copy(bTypes[i*maxLen:], c.TokenTypeIDs)
				}
			}
			raw, dims := runTextExecOnce(t, exec, inNames, batchSize, maxLen, bIDs, bMask, bTypes)
			if len(dims) != 3 || dims[0] != batchSize || dims[1] != maxLen || dims[2] != dim {
				t.Fatalf("batch: got shape %v, want [%d %d %d]", dims, batchSize, maxLen, dim)
			}
			gotAll := meanPoolAndL2Normalize(raw, bMask, batchSize, maxLen, dim)
			for i := range ts.Cases {
				got := gotAll[i*dim : (i+1)*dim]
				want := refBatchEmb[i*dim : (i+1)*dim]
				m := computeVectorMetrics(got, want)
				if m.CosineSim < worstCos {
					worstCos = m.CosineSim
				}
				if m.MaxAbsDiff > worstMaxAbs {
					worstMaxAbs = m.MaxAbsDiff
				}
				t.Logf("batch[%d] (len=%d/%d): cos=%.7f max_abs=%.3e mean_abs=%.3e",
					i, len(ts.Cases[i].InputIDs), maxLen, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
				if m.CosineSim < tol.MinCosineSim {
					t.Errorf("batch[%d] cosine sim %.7f < %.7f", i, m.CosineSim, tol.MinCosineSim)
				}
				if m.MaxAbsDiff > tol.MaxAbsDiff {
					t.Errorf("batch[%d] max abs diff %.3e > %.3e", i, m.MaxAbsDiff, tol.MaxAbsDiff)
				}
				if m.MeanAbsDiff > tol.MaxMeanAbsDiff {
					t.Errorf("batch[%d] mean abs diff %.3e > %.3e", i, m.MeanAbsDiff, tol.MaxMeanAbsDiff)
				}
			}
		})
	}

	return fmt.Sprintf("min_cos=%.5f max_diff=%.2e", worstCos, worstMaxAbs)
}

func verifyTextLogits(t *testing.T, specDir string, ts *TextSpec, inNames []string, exec *model.Exec) string {
	refSinglePath := filepath.Join(specDir, ts.SingleLogitsFile)
	refSingle, err := readF32File(refSinglePath)
	if err != nil {
		t.Fatalf("readF32File(%s): %v", refSinglePath, err)
	}

	tol := ts.Tolerances
	if tol.MaxAbsDiff == 0 {
		tol.MaxAbsDiff = 5e-3
	}
	if tol.MaxMeanAbsDiff == 0 {
		tol.MaxMeanAbsDiff = 1e-3
	}

	worstCos := 1.0
	worstMaxAbs := 0.0

	t.Run("logits_single", func(t *testing.T) {
		offset := 0
		for i, c := range ts.Cases {
			seqLen := len(c.InputIDs)
			got, dims := runTextExecOnce(t, exec, inNames, 1, seqLen, c.InputIDs, c.AttentionMask, c.TokenTypeIDs)
			if !slices.Equal(dims, c.OutputShape) {
				t.Fatalf("sentence %d: got shape %v, want %v", i, dims, c.OutputShape)
			}
			n := shapeSize(c.OutputShape)
			if offset+n > len(refSingle) {
				t.Fatalf("sentence %d: reference file %s truncated", i, refSinglePath)
			}
			want := refSingle[offset : offset+n]
			offset += n

			m := computeVectorMetrics(got, want)
			if m.CosineSim < worstCos {
				worstCos = m.CosineSim
			}
			if m.MaxAbsDiff > worstMaxAbs {
				worstMaxAbs = m.MaxAbsDiff
			}
			t.Logf("sentence %d (%d toks) shape=%v: cos=%.7f max_abs=%.3e mean_abs=%.3e",
				i, seqLen, dims, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
			if tol.MinCosineSim > 0 && m.CosineSim < tol.MinCosineSim {
				t.Errorf("sentence %d cosine sim %.7f < %.7f", i, m.CosineSim, tol.MinCosineSim)
			}
			if m.MaxAbsDiff > tol.MaxAbsDiff {
				t.Errorf("sentence %d max abs diff %.3e > %.3e", i, m.MaxAbsDiff, tol.MaxAbsDiff)
			}
			if m.MeanAbsDiff > tol.MaxMeanAbsDiff {
				t.Errorf("sentence %d mean abs diff %.3e > %.3e", i, m.MeanAbsDiff, tol.MaxMeanAbsDiff)
			}
			if tol.CheckArgmax && len(dims) == 2 && dims[0] == 1 {
				if gArg, wArg := argmaxF32(got), argmaxF32(want); gArg != wArg {
					t.Errorf("sentence %d argmax=%d, want %d", i, gArg, wArg)
				}
			}
		}
	})

	if !ts.SkipBatchTest && ts.BatchLogitsFile != "" && len(ts.Cases) > 1 {
		t.Run("logits_batch", func(t *testing.T) {
			refBatchPath := filepath.Join(specDir, ts.BatchLogitsFile)
			refBatch, err := readF32File(refBatchPath)
			if err != nil {
				t.Fatalf("readF32File(%s): %v", refBatchPath, err)
			}
			batchSize := len(ts.Cases)
			maxLen := 0
			for _, c := range ts.Cases {
				if len(c.InputIDs) > maxLen {
					maxLen = len(c.InputIDs)
				}
			}
			bIDs := make([]int64, batchSize*maxLen)
			bMask := make([]int64, batchSize*maxLen)
			bTypes := make([]int64, batchSize*maxLen)
			for i := range bIDs {
				bIDs[i] = ts.PadTokenID
			}
			for i, c := range ts.Cases {
				copy(bIDs[i*maxLen:], c.InputIDs)
				copy(bMask[i*maxLen:], c.AttentionMask)
				if len(c.TokenTypeIDs) > 0 {
					copy(bTypes[i*maxLen:], c.TokenTypeIDs)
				}
			}
			got, dims := runTextExecOnce(t, exec, inNames, batchSize, maxLen, bIDs, bMask, bTypes)
			if !slices.Equal(dims, ts.BatchLogitsShape) {
				t.Fatalf("batch: got shape %v, want %v", dims, ts.BatchLogitsShape)
			}
			// For 3D token-level logits [B, S, C], only compare unpadded positions (mask == 1),
			// since padded positions are ignored downstream.
			gotCmp, wantCmp := got, refBatch
			if len(dims) == 3 {
				cols := dims[2]
				gotCmp = make([]float32, 0, len(got))
				wantCmp = make([]float32, 0, len(refBatch))
				for bIdx := range batchSize {
					for sIdx := range maxLen {
						if bMask[bIdx*maxLen+sIdx] == 0 {
							continue
						}
						off := (bIdx*maxLen + sIdx) * cols
						gotCmp = append(gotCmp, got[off:off+cols]...)
						wantCmp = append(wantCmp, refBatch[off:off+cols]...)
					}
				}
			}
			m := computeVectorMetrics(gotCmp, wantCmp)
			if m.CosineSim < worstCos {
				worstCos = m.CosineSim
			}
			if m.MaxAbsDiff > worstMaxAbs {
				worstMaxAbs = m.MaxAbsDiff
			}
			t.Logf("batch shape=%v: cos=%.7f max_abs=%.3e mean_abs=%.3e", dims, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
			if tol.MinCosineSim > 0 && m.CosineSim < tol.MinCosineSim {
				t.Errorf("batch cosine sim %.7f < %.7f", m.CosineSim, tol.MinCosineSim)
			}
			if m.MaxAbsDiff > tol.MaxAbsDiff {
				t.Errorf("batch max abs diff %.3e > %.3e", m.MaxAbsDiff, tol.MaxAbsDiff)
			}
			if m.MeanAbsDiff > tol.MaxMeanAbsDiff {
				t.Errorf("batch mean abs diff %.3e > %.3e", m.MeanAbsDiff, tol.MaxMeanAbsDiff)
			}
			if tol.CheckArgmax && len(dims) == 2 {
				rows, cols := dims[0], dims[1]
				for r := range rows {
					gArg := argmaxF32(got[r*cols : (r+1)*cols])
					wArg := argmaxF32(refBatch[r*cols : (r+1)*cols])
					if gArg != wArg {
						t.Errorf("batch row %d argmax=%d, want %d", r, gArg, wArg)
					}
				}
			}
		})
	}

	if tol.MinCosineSim > 0 {
		return fmt.Sprintf("min_cos=%.5f max_diff=%.2e", worstCos, worstMaxAbs)
	}
	return fmt.Sprintf("max_diff=%.2e", worstMaxAbs)
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
