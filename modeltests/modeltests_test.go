// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagModelTests = flag.Bool("modeltests", false,
		"Enable full model tests, which are slow and may require downloading large models.")
	flagBackends = flag.String("backends", "",
		`Semicolon-separated GoMLX backend configs to test (e.g. "go;go:ops_sequential;xla:cpu;onnx:cpu"). `+
			"When empty, uses the default backend (GOMLX_BACKEND).")
	flagMode = flag.String("mode", "both",
		`Shape compilation mode to test: "static", "dynamic", or "both" (runs dynamic only on backends with HasDynamicShapes).`)
	flagMaxTier = flag.String("max_tier", "",
		"Maximum model size tier to run: small, medium, or large (empty runs all).")
	flagCachedOnly = flag.Bool("cached_only", false,
		"Skip models whose files are not already present in the local HuggingFace cache.")
	flagTag = flag.String("tag", "",
		"Only run models carrying this tag (e.g. text, embed, vision, audio, quantized, tree).")
	flagStrict = flag.Bool("strict", false,
		"Fail instead of skipping when a gated model lacks HF_TOKEN or cannot be fetched.")
	flagSettings = flag.String("set", "",
		`Semicolon-separated GoMLX store hyperparameter overrides (e.g. "quantized_matmul=false").`)
)

const skipModelTestsMsg = "Set -modeltests to enable testing model tests, which are slow, and may require downloading of large models to run."

var (
	reportMu   sync.Mutex
	reportRows []runReportRow
)

func recordReport(row runReportRow) {
	reportMu.Lock()
	reportRows = append(reportRows, row)
	reportMu.Unlock()
}

func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	flag.Parse()

	code := m.Run()

	reportMu.Lock()
	rows := append([]runReportRow(nil), reportRows...)
	reportMu.Unlock()
	if len(rows) > 0 {
		printSummaryTable(rows)
	}
	os.Exit(code)
}

func parseBackendConfigs() []string {
	raw := strings.TrimSpace(*flagBackends)
	if raw == "" {
		return []string{""}
	}
	parts := strings.Split(raw, ";")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func resolveModesForBackend(b compute.Backend, skipDynamic bool) []string {
	mode := strings.ToLower(strings.TrimSpace(*flagMode))
	hasDyn := b.Capabilities().HasDynamicShapes() && !skipDynamic
	switch mode {
	case "static":
		return []string{"static"}
	case "dynamic":
		if !hasDyn {
			return nil
		}
		return []string{"dynamic"}
	default: // "both"
		if hasDyn {
			return []string{"static", "dynamic"}
		}
		return []string{"static"}
	}
}

func createBackend(cfg string) (compute.Backend, string, error) {
	if cfg == "" {
		b, err := compute.New()
		if err != nil {
			return nil, "", err
		}
		label := os.Getenv("GOMLX_BACKEND")
		if label == "" {
			label = b.Name()
		}
		return b, label, nil
	}
	b, err := compute.NewWithConfig(cfg)
	if err != nil {
		return nil, cfg, err
	}
	return b, cfg, nil
}

func shouldSkipBackend(skipList []string, backendCfg, backendName string) bool {
	for _, s := range skipList {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if s == backendCfg || s == backendName || strings.HasPrefix(backendCfg, s+":") {
			return true
		}
	}
	return false
}

func openRepoForSpec(t *testing.T, repoID, revision string, gated bool, requiredFile string) (*hub.Repo, bool) {
	t.Helper()
	repo := hub.New(repoID)
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		repo.WithAuth(tok)
	}
	if revision != "" {
		repo.WithRevision(revision)
	}

	if *flagCachedOnly && !isRepoCached(repo, requiredFile) {
		t.Skipf("Skipping %s because -cached_only is set and %q is not present in local cache", repoID, requiredFile)
		return nil, false
	}

	if err := repo.DownloadInfo(false); err != nil {
		if gated && os.Getenv("HF_TOKEN") == "" && !*flagStrict {
			t.Skipf("Skipping gated model %s (set HF_TOKEN or -strict): %v", repoID, err)
			return nil, false
		}
		if !*flagStrict {
			t.Skipf("Skipping %s because repo info could not be fetched (use -strict to fail): %v", repoID, err)
			return nil, false
		}
		t.Fatalf("Failed to fetch repo info for %s: %v", repoID, err)
	}
	return repo, true
}

func isRepoCached(repo *hub.Repo, requiredFile string) bool {
	cacheDir, err := repo.CacheDir()
	if err != nil {
		return false
	}
	snapsDir := filepath.Join(cacheDir, "snapshots")
	entries, err := os.ReadDir(snapsDir)
	if err != nil || len(entries) == 0 {
		return false
	}
	if requiredFile == "" {
		return true
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		candidate := filepath.Join(snapsDir, e.Name(), requiredFile)
		if _, err := os.Stat(candidate); err == nil {
			return true
		}
	}
	return false
}

func loadONNXVariant(t *testing.T, repo *hub.Repo, v VariantSpec) (onnx.Model, *model.Store, error) {
	t.Helper()
	onnxPath, err := repo.DownloadFile(v.ONNXFile)
	if err != nil {
		return nil, nil, fmt.Errorf("DownloadFile(%q): %w", v.ONNXFile, err)
	}
	for _, ext := range v.ExternalDataFiles {
		if _, err := repo.DownloadFile(ext); err != nil {
			return nil, nil, fmt.Errorf("DownloadFile external data %q: %w", ext, err)
		}
	}

	m, err := parser.ParseFile(onnxPath)
	if err != nil {
		return nil, nil, fmt.Errorf("parser.ParseFile(%q): %w", onnxPath, err)
	}
	if v.ModelOptions != nil {
		if v.ModelOptions.AllowDTypePromotion {
			m.AllowDTypePromotion()
		}
		if v.ModelOptions.ApproximateGelu {
			m.ForceApproximateGelu(true)
		}
	}

	store := model.NewStore()
	store.SetParam(layers.ParamQuantizedMatmul, true)
	if *flagSettings != "" {
		paramsSet, err := commandline.ParseSettings(store, *flagSettings)
		if err != nil {
			_ = m.Close()
			return nil, nil, fmt.Errorf("parsing -set=%q: %w", *flagSettings, err)
		}
		if len(paramsSet) > 0 {
			t.Logf("Modified settings:\n%s", commandline.SprintModifiedSettings(store, paramsSet))
		}
	}
	if err := m.VariablesToScope(store.RootScope()); err != nil {
		_ = m.Close()
		return nil, nil, fmt.Errorf("VariablesToScope: %w", err)
	}
	return m, store, nil
}

func cleanupMemory() {
	for range 2 {
		runtime.GC()
	}
	debug.FreeOSMemory()
}

func TestModels(t *testing.T) {
	if !*flagModelTests && os.Getenv("GOMLX_MODELTESTS") == "" {
		t.Skip(skipModelTestsMsg)
	}

	testdataDir := "testdata"
	specs, err := loadModelSpecs(testdataDir)
	if err != nil {
		t.Fatalf("Failed to load model specs from %q: %v", testdataDir, err)
	}
	if len(specs) == 0 {
		t.Fatalf("No model specs found under %q", testdataDir)
	}

	backendConfigs := parseBackendConfigs()
	maxTierRank := 99
	if *flagMaxTier != "" {
		maxTierRank = tierRank(*flagMaxTier)
	}
	filterTag := strings.ToLower(strings.TrimSpace(*flagTag))

	for _, loaded := range specs {
		spec := loaded.Spec
		t.Run(spec.Name, func(t *testing.T) {
			if tierRank(spec.SizeTier) > maxTierRank {
				t.Skipf("Skipping %s (tier=%q exceeds -max_tier=%q)", spec.Name, spec.SizeTier, *flagMaxTier)
			}
			if filterTag != "" && !slices.ContainsFunc(spec.Tags, func(tag string) bool {
				return strings.EqualFold(tag, filterTag)
			}) {
				t.Skipf("Skipping %s (does not carry tag %q)", spec.Name, filterTag)
			}

			switch spec.Family {
			case "text":
				runTextModel(t, loaded, backendConfigs)
			case "tensor":
				runTensorModel(t, loaded, backendConfigs)
			default:
				t.Fatalf("Unknown model family %q in %s", spec.Family, loaded.Dir)
			}
		})
	}
}

func testStatus(t *testing.T) string {
	if t.Failed() {
		return "FAIL"
	}
	if t.Skipped() {
		return "SKIP"
	}
	return "PASS"
}

func formatBackendSubtestName(cfg, resolvedLabel string) string {
	name := cfg
	if name == "" {
		name = resolvedLabel
	}
	if name == "" {
		return "default"
	}
	return fmt.Sprintf("backend=%s", name)
}
