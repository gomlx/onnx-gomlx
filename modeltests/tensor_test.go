// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"fmt"
	"math"
	"math/rand/v2"
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

func runTensorModel(t *testing.T, loaded loadedSpec, backendConfigs []string) {
	spec := loaded.Spec
	for _, variant := range spec.Variants {
		t.Run(variant.Name, func(t *testing.T) {
			if variant.Tensor == nil {
				t.Fatalf("variant %q has nil tensor spec", variant.Name)
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
			skipDynamic := len(variant.Tensor.DynamicAxes) == 0
			if variant.ModelOptions != nil {
				skipBackends = variant.ModelOptions.SkipBackends
				if variant.ModelOptions.SkipDynamic {
					skipDynamic = true
				}
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
							runTensorVariantOnBackend(t, loaded, repo, variant, cfg, label, mode == "dynamic")
						})
					}
				})
			}
		})
	}
}

func runTensorVariantOnBackend(
	t *testing.T,
	loaded loadedSpec,
	repo *hub.Repo,
	variant VariantSpec,
	backendCfg, backendLabel string,
	dynamic bool,
) {
	spec := loaded.Spec
	ts := variant.Tensor
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
	outNames, _ := onnxModel.Outputs()

	exec, err := buildTensorExec(b, store, onnxModel, inNames, outNames, ts.DynamicAxes, dynamic)
	if err != nil {
		t.Fatalf("buildTensorExec: %+v", err)
	}
	defer exec.Finalize()

	tol := ts.Tolerances
	if tol.MaxAbsDiff == 0 {
		tol.MaxAbsDiff = 5e-3
	}
	if tol.MaxMeanAbsDiff == 0 {
		tol.MaxMeanAbsDiff = 1e-3
	}

	inferStart := time.Now()
	worstCos := 1.0
	worstMaxAbs := 0.0
	hasFloatOut := false
	int64Checked := 0

	for _, tc := range ts.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			inByName := make(map[string]TensorInputSpec, len(tc.Inputs))
			for _, inp := range tc.Inputs {
				inByName[inp.Name] = inp
			}
			rng := rand.New(rand.NewPCG(tc.Seed, 0))
			args := make([]any, len(inNames))
			createdTensors := make([]*tensors.Tensor, 0, len(inNames))
			for i, name := range inNames {
				inp, ok := inByName[name]
				if !ok {
					t.Fatalf("case %q missing input %q", tc.Name, name)
				}
				tens := makeGoMLXInputTensor(t, inp, rng)
				args[i] = tens
				createdTensors = append(createdTensors, tens)
			}
			defer func() {
				for _, tens := range createdTensors {
					tens.FinalizeAll()
				}
			}()

			outs, err := exec.Call(args...)
			if err != nil {
				t.Fatalf("exec.Call(%s): %+v", tc.Name, err)
			}
			defer func() {
				for _, o := range outs {
					o.FinalizeAll()
				}
			}()

			if len(outs) != len(tc.Outputs) {
				t.Fatalf("case %q: got %d outputs, want %d", tc.Name, len(outs), len(tc.Outputs))
			}

			for oi, wantSpec := range tc.Outputs {
				gotT := outs[oi]
				if !slices.Equal(gotT.Shape().Dimensions, wantSpec.Shape) {
					t.Errorf("case %q out[%d]=%s: got shape %v, want %v",
						tc.Name, oi, wantSpec.Name, gotT.Shape().Dimensions, wantSpec.Shape)
					continue
				}
				refPath := filepath.Join(loaded.Dir, wantSpec.File)
				switch wantSpec.DType {
				case "float32":
					hasFloatOut = true
					wantVals, err := readF32File(refPath)
					if err != nil {
						t.Fatalf("readF32File(%s): %v", refPath, err)
					}
					gotFull := tensors.MustCopyFlatData[float32](gotT)
					gotCmp := gotFull
					if wantSpec.SampleStride > 1 {
						gotCmp = make([]float32, 0, len(wantVals))
						for idx := 0; idx < len(gotFull); idx += wantSpec.SampleStride {
							gotCmp = append(gotCmp, gotFull[idx])
						}
					}
					if len(gotCmp) != len(wantVals) {
						t.Fatalf("case %q out[%d]: got %d values (stride=%d), want %d",
							tc.Name, oi, len(gotCmp), wantSpec.SampleStride, len(wantVals))
					}
					m := computeVectorMetrics(gotCmp, wantVals)
					if m.CosineSim < worstCos {
						worstCos = m.CosineSim
					}
					if m.MaxAbsDiff > worstMaxAbs {
						worstMaxAbs = m.MaxAbsDiff
					}
					t.Logf("case %s out[%d]=%s shape=%v: cos=%.7f max_abs=%.3e mean_abs=%.3e",
						tc.Name, oi, wantSpec.Name, wantSpec.Shape, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)

					if tol.MinCosineSim > 0 && m.CosineSim < tol.MinCosineSim {
						t.Errorf("case %s out[%d] cosine sim %.7f < %.7f", tc.Name, oi, m.CosineSim, tol.MinCosineSim)
					}
					if m.MaxAbsDiff > tol.MaxAbsDiff {
						t.Errorf("case %s out[%d] max abs diff %.3e > %.3e", tc.Name, oi, m.MaxAbsDiff, tol.MaxAbsDiff)
					}
					if m.MeanAbsDiff > tol.MaxMeanAbsDiff {
						t.Errorf("case %s out[%d] mean abs diff %.3e > %.3e", tc.Name, oi, m.MeanAbsDiff, tol.MaxMeanAbsDiff)
					}
					if wantSpec.Stats != nil {
						gotStats := computeStatsF32(gotFull)
						if d := math.Abs(gotStats.Min - wantSpec.Stats.Min); d > tol.MaxAbsDiff {
							t.Errorf("case %s out[%d] global min diff %.3e > %.3e (got=%g want=%g)",
								tc.Name, oi, d, tol.MaxAbsDiff, gotStats.Min, wantSpec.Stats.Min)
						}
						if d := math.Abs(gotStats.Max - wantSpec.Stats.Max); d > tol.MaxAbsDiff {
							t.Errorf("case %s out[%d] global max diff %.3e > %.3e (got=%g want=%g)",
								tc.Name, oi, d, tol.MaxAbsDiff, gotStats.Max, wantSpec.Stats.Max)
						}
						if d := math.Abs(gotStats.Mean - wantSpec.Stats.Mean); d > tol.MaxMeanAbsDiff {
							t.Errorf("case %s out[%d] global mean diff %.3e > %.3e (got=%g want=%g)",
								tc.Name, oi, d, tol.MaxMeanAbsDiff, gotStats.Mean, wantSpec.Stats.Mean)
						}
					}
					if tol.CheckArgmax && len(wantSpec.Shape) == 2 && wantSpec.SampleStride <= 1 {
						rows, cols := wantSpec.Shape[0], wantSpec.Shape[1]
						for r := range rows {
							gArg := argmaxF32(gotFull[r*cols : (r+1)*cols])
							wArg := argmaxF32(wantVals[r*cols : (r+1)*cols])
							if gArg != wArg {
								t.Errorf("case %s out[%d] row %d argmax=%d, want %d", tc.Name, oi, r, gArg, wArg)
							}
						}
					}
				case "int64":
					wantVals, err := readI64File(refPath)
					if err != nil {
						t.Fatalf("readI64File(%s): %v", refPath, err)
					}
					gotVals := tensors.MustCopyFlatData[int64](gotT)
					if !slices.Equal(gotVals, wantVals) {
						t.Errorf("case %s out[%d]=%s int64 mismatch:\n got=%v\nwant=%v",
							tc.Name, oi, wantSpec.Name, gotVals, wantVals)
					}
					int64Checked += len(gotVals)
				default:
					t.Fatalf("unsupported output dtype %q", wantSpec.DType)
				}
			}
		})
	}
	inferTime = time.Since(inferStart)

	switch {
	case hasFloatOut && int64Checked > 0:
		notes = fmt.Sprintf("max_diff=%.2e i64=%d", worstMaxAbs, int64Checked)
	case hasFloatOut:
		notes = fmt.Sprintf("min_cos=%.5f max_diff=%.2e", worstCos, worstMaxAbs)
	case int64Checked > 0:
		notes = fmt.Sprintf("i64_exact=%d", int64Checked)
	}
}

func buildTensorExec(
	b compute.Backend,
	store *model.Store,
	onnxModel onnx.Model,
	inNames, outNames []string,
	dynamicAxes map[string][]string,
	dynamic bool,
) (*model.Exec, error) {
	exec, err := model.NewExec(b, store, func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
		g := inputs[0].Graph()
		inMap := make(map[string]*graph.Node, len(inNames))
		for i, name := range inNames {
			inMap[name] = inputs[i]
		}
		return onnxModel.CallGraph(scope, g, inMap, outNames...)
	})
	if err != nil {
		return nil, err
	}
	if dynamic {
		axisSpecs := make([][]string, len(inNames))
		for i, name := range inNames {
			axisSpecs[i] = dynamicAxes[name]
		}
		exec.WithDynamicAxes(axisSpecs...)
	}
	return exec, nil
}

func makeGoMLXInputTensor(t *testing.T, spec TensorInputSpec, rng *rand.Rand) *tensors.Tensor {
	t.Helper()
	scale := spec.Scale
	if scale == 0 {
		scale = 1.0
	}
	switch spec.DType {
	case "float32":
		tens := tensors.FromShape(shapes.Make(dtypes.Float32, spec.Shape...))
		tensors.MutableFlatData[float32](tens, func(flat []float32) {
			fillFloat32(flat, spec.Dist, scale, rng)
		})
		return tens
	default:
		t.Fatalf("unsupported input dtype %q", spec.DType)
		return nil
	}
}

// fillFloat32 MUST stay bit-identical to modeltests/internal/generators/ort/main.go's fillFloat32.
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
		samples := [][4]float32{
			{5.1, 3.5, 1.4, 0.2},
			{6.0, 2.7, 4.5, 1.5},
			{6.7, 3.1, 5.6, 2.4},
			{4.9, 3.0, 1.4, 0.2},
			{5.5, 2.4, 3.8, 1.1},
			{7.2, 3.6, 6.1, 2.5},
		}
		for r := 0; r*4+3 < len(data); r++ {
			s := samples[r%len(samples)]
			copy(data[r*4:], s[:])
		}
	default:
		panic(fmt.Sprintf("unknown input distribution %q", dist))
	}
}
