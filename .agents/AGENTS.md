# ONNX-GoMLX Agent Guide

This document is intended for AI coding agents and developers working on or using the `onnx-gomlx` codebase.
It provides instructions on using the library, understanding its internal architecture, implementing new operations, materializing constant subexpressions, handling fused operations, and debugging.

---

## 1. Using ONNX-GoMLX

`onnx-gomlx` translates ONNX computational graphs (`.onnx` protobuf models) into [GoMLX](https://github.com/gomlx/gomlx) execution graphs, enabling high-performance ML inference, fine-tuning, and graph composition in Go.

### 1.1 Standard Inference Workflow

```go
package main

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default" // Registers default backend (e.g. XLA, Go, or ONNX)
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx"
)

func main() {
	// 1. Load and parse the ONNX model file.
	onnxModel, err := onnx.ReadFile("path/to/model.onnx")
	if err != nil {
		panic(err)
	}

	// 2. Extract weights/initializers into a GoMLX model.Store.
	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		panic(err)
	}

	// 3. Create an execution function.
	backend := backends.New()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, inputIDs, attentionMask *graph.Node) *graph.Node {
		g := inputIDs.Graph()
		outputs := onnxModel.CallGraph(scope, g, map[string]*graph.Node{
			"input_ids":      inputIDs,
			"attention_mask": attentionMask,
		})
		return outputs[0]
	})
	defer exec.Finalize()

	// 4. Run execution with tensors.
	// ... construct inputTensors ...
	// out := exec.MustCall1(tIDs, tMask)
}
```

### 1.2 Dynamic Shapes Support

Some backends (such as the Go portable backend `GOMLX_BACKEND=go` and the ONNX Runtime backend `GOMLX_BACKEND=onnx:cpu`) natively support dynamic dimensions (unknown batch size, variable sequence lengths) without requiring graph recompilation for every shape variation.

To enable dynamic shapes, configure `exec.WithDynamicAxes`:

```go
exec := model.MustNewExec(backend, store, func(scope *model.Scope, inputIDs, attentionMask, tokenTypeIDs *graph.Node) *graph.Node {
    g := inputIDs.Graph()
    outputs := onnxModel.CallGraph(scope, g, map[string]*graph.Node{
        "input_ids":      inputIDs,
        "attention_mask": attentionMask,
        "token_type_ids": tokenTypeIDs,
    })
    return outputs[0]
})

// Specify dynamic axis names for each parameter:
exec.WithDynamicAxes(
    []string{"batch", "seq"}, // input_ids: shape [?, ?]
    []string{"batch", "seq"}, // attention_mask: shape [?, ?]
    []string{"batch", "seq"}, // token_type_ids: shape [?, ?]
)
defer exec.Finalize()

// Subsequent calls with different batch sizes or sequence lengths will run
// on the same compiled graph without triggering recompilation:
outBatch1 := exec.MustCall1(tIDsBatch1, tMaskBatch1, tTypesBatch1)
outBatch2 := exec.MustCall1(tIDsBatch2, tMaskBatch2, tTypesBatch2)
```

Backends that do not support dynamic shapes (like static XLA PJRT) will pad inputs to fixed static dimensions.

---

## 2. Architecture & Codebase Layout

- **`onnx/`**: Public API package (`Model` interface) and **`onnx/parser/`** (`Parse`, `ParseFile`, `ParseFromReader`).
- **`internal/onnxgomlx/`**: Core conversion engine.
  - **`model.go` & `graph.go`**: Model state, graph translation lifecycle, and operator dispatch (`convertNode`).
  - **`ops.go` & `ops_*.go`**: Operator converters (`ops_reshape.go`, `ops_slice.go`, `ops_pad.go`, `ops_pool.go`, `ops_scatter.go`, `ops_conv_integer.go`, `ops_conv_transpose.go`, `ops_grid_sample.go`, `ops_matmul_nbits.go`, `ops_norm_onehot.go`, `ops_random.go`, `ops_tree_ensemble.go`, `ops_einsum.go`, `ops_runtime_broadcast.go`, etc.).
  - **`dynamicshape.go` & `shapes.go`**: Symbolic/unnamed dynamic axis tracking (`shapes.UniqueAxisName("#onnx")`) and intermediate ONNX shape inference.
  - **`materialize.go`**: Constant subexpression evaluation engine. Evaluates ONNX nodes that compute tensor parameters (shapes, slices, broadcasting dimensions) at graph building time, while letting dynamic dependencies fall back to runtime graph ops.
  - **`fusion/`**: Fused operation matchers (`sdpa.go`, `qkvdense.go`, `dense_gelu.go`, `quantized_dense.go`, `quantized_qkv.go`). Detects multi-op subgraphs and emits GoMLX `nn.` / `attention.` fused operations.
- **`internal/onnxgraph/`**: Topological ordering, dependency tracking, and graph representation of ONNX protobuf nodes.
- **`internal/benchmarks/`**: Parity tests, benchmarks vs. ONNX Runtime, and integration tests across backends.

---

## 3. Developing and Extending ONNX-GoMLX

### 3.1 Implementing New Operator Converters

When adding a new ONNX operator:
1. Implement the converter in a focused `internal/onnxgomlx/ops_<category>.go` file (keeping per-op files under ~200 lines) or in `ops.go` for small helpers:
   ```go
   func convertMyOp(node *protos.NodeProto, inputs []*graph.Node) *graph.Node
   ```
   Prefer delegating to GoMLX `nn.` / `attention.` / `activation.` layers (e.g., `nn.QuantizedDense`, `nn.LayerNorm`, `graph.Softmax`, `graph.TopK`) so hardware-accelerated fused kernels are used automatically when supported.
2. Wire the case into `convertNode` in `internal/onnxgomlx/graph.go` (and `inferShapeForNameDepth` in `shapes.go` when downstream `Shape` materialization benefits from knowing output rank/dims).
3. **Pay close attention to broadcasting & dynamic axes**:
   - ONNX follows NumPy-style right-aligned multidirectional broadcasting, including runtime size-1 expansion across distinct dynamic axes.
   - Use `runtimeBroadcast` / `broadcastWithSpecs` / `runtimeExpand` (`ops_runtime_broadcast.go`) so static and already-aligned dynamic shapes take a zero-overhead fast path while distinct dynamic axes broadcast and validate compatibility at runtime.
4. **Extracting node attributes**:
   Use helper functions in `internal/onnxgomlx/` to read attributes from `node.Attribute` (e.g. `GetIntAttrOr`, `GetIntsAttrOr`, `GetFloatAttrOr`, `GetStringAttrOr`, `GetStringsAttrOr`).

---

### 3.2 Materializing Constant Subexpressions

#### Why Materialization is Required
In ONNX, operators frequently receive metadata (such as target shapes for `Reshape` and `Expand`, slice limits for `Slice`, tile repetitions for `Tile`) as **tensor inputs** rather than node attributes. These tensor inputs are often produced by small subgraphs of operators:
`Shape -> Gather -> Unsqueeze -> Concat -> Reshape`

Because GoMLX graph building requires static shapes, rank, or dimension parameters for many graph operations (or needs constants to determine if a dimension is 1), `onnx-gomlx` includes a constant materialization engine in `internal/onnxgomlx/materialize.go`.

#### How to Use `materializeConstantExpression`
```go
// Attempt to evaluate a tensor input as a static GoMLX Tensor:
tensor, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
if err == nil {
    dims := tensorToInts(tensor)
    // Use static dims...
}
```

#### Handling Dynamic Shapes in Materialization
When dynamic shapes are active, an input's dimension may be `-1` (`shapes.DynamicDim`):
- `materializeConstantExpression` on `Shape` nodes will return `-1` for dynamic dimensions.
- **Always handle dynamic cases**:
  - Do not panic if a dimension is `-1`.
  - In operators like `convertExpand` and `convertReshape`, check whether any target dimension is dynamic or cannot be statically materialized.
  - If dynamic, fall back to GoMLX dynamic operations:
    - Use `graph.DynamicReshape(operand, shapeNode)` when available.
    - Use `graph.BroadcastToDims` or graph-level operations rather than assuming all dimensions are concrete integers $> 0$.

---

### 3.3 Handling and Preserving Fused Operations

GoMLX and its backends provide hardware-accelerated fused operations:
- **`FusedScaledDotProductAttention`** (`compute.OpTypeFusedScaledDotProductAttention` via `attention.Core`)
- **`FusedAttentionQKVProjection`** (`compute.OpTypeFusedAttentionQKVProjection`)
- **`FusedDense`** (`compute.OpTypeFusedDense`)
- **`FusedQuantizedDense`** (`compute.OpTypeFusedQuantizedDense` via `nn.QuantizedDense` with `QuantizedDenseConfig`)
- **`FusedLayerNorm`** (`compute.OpTypeFusedLayerNorm`)
- **`FusedSoftmax`** (`compute.OpTypeFusedSoftmax`)

#### How Fusion Works
1. Before node-by-node conversion, `fusion.FindCandidates(model, onnxGraph)` scans the graph for specific patterns (e.g., attention blocks matching `Q @ K.T -> Scale -> Mask/Bias -> Softmax -> @ V`).
2. Each matched candidate registers the ONNX nodes it subsumes so they are skipped during regular conversion.
3. The candidate emits the fused GoMLX call (e.g., `attention.Core(q, k, v, ...)`).

#### The Fallback Contract (Crucial Rule)
- **NEVER special-case backend names in `onnx-gomlx`** (e.g., do NOT write `if strings.Contains(backend.Name(), "go") { disableFusion() }`).
- `onnx-gomlx` must emit the standard GoMLX layer call (`attention.Core`, etc.) uniformly across all backends.
- If a backend does not support a specific configuration of a fused op (e.g. broadcasting masks across sequence length, dynamic dims, unsupported dtypes), the **backend's graph builder** must return `errors.Wrapf(compute.ErrNotImplemented, ...)`.
- GoMLX's `InternalFusedOpCaller` catches `compute.ErrNotImplemented` and automatically falls back to the decomposed equivalent graph.
- This ensures full portability and keeps backend-specific capability knowledge inside the backend.

---

## 4. Debugging & Verification

### 4.1 Inspecting ONNX Models with `onnx_printer`

The repository `github.com/gomlx/compute-onnx/cmd/onnx_printer` provides a CLI tool to inspect the structure of `.onnx` models, displaying node types, input/output tensor names, shapes, initializers, and attributes.

To run it:
```bash
# From the gomlx workspace:
go run ../compute-onnx/cmd/onnx_printer <path-to-model.onnx>

# Or directly:
go run github.com/gomlx/compute-onnx/cmd/onnx_printer@latest <path-to-model.onnx>
```

Useful flags:
- `-n <int>` or `-max_items <int>`: Number of initial constant elements to display for weights/initializers (default: 10).
- `-show_doc`: Print docstrings embedded in the ONNX model.
- Pipe support: `cat model.onnx | go run github.com/gomlx/compute-onnx/cmd/onnx_printer`

### 4.2 Inspecting Graph Construction
When an operator conversion fails:
- Print node shapes: `fmt.Printf("node %s shape: %s\n", node.Name, operand.Shape())`
- Inspect the GoMLX graph: `fmt.Println(g.String())`
- Check whether an op is returning dynamic shapes: `node.Shape().IsDynamic()`

### 4.3 Isolating Fused Op vs. Decomposed Discrepancies
If a model produces different numerical results between static and dynamic runs or between backends:
1. Run with fusion disabled:
   ```bash
   GOMLX_FUSION=false go test ./...
   ```
   If the test passes with `GOMLX_FUSION=false`, the issue is in the backend's fused kernel or fused pattern matcher.
2. Compare GoMLX output directly against ONNX Runtime:
   Use `ort.NewDynamicAdvancedSession` with the same input tensors and assert closeness (`1e-4` tolerance for float32).
3. Check SIMD kernel operand ordering:
   In `simd/archsimd`, remember that:
   ```go
   x.MulAdd(y, z) // computes (x * y) + z
   ```
   To accumulate $(w \times v) + \text{out}$, write `w.MulAdd(v, out)`, NOT `out.MulAdd(w, v)`.

### 4.4 Running Benchmarks and Verification
```bash
# Go portable backend benchmark:
GOMLX_BACKEND=go go test -tags=onnx ./internal/benchmarks/ -test.run TestRobSentences_BenchXLA -test.v -bench_duration=10s

# ONNX Runtime backend benchmark:
GOMLX_BACKEND=onnx:cpu go test -tags=onnx ./internal/benchmarks/ -test.run TestRobSentences_BenchXLA -test.v -bench_duration=10s

# Dynamic call graph parity test:
GOMLX_BACKEND=go go test -tags=onnx ./internal/benchmarks/ -test.run TestRobSentences_DynamicCallGraph -test.v
```

---

## Related Repositories (including this one)

- **[`github.com/gomlx/gomlx`](https://github.com/gomlx/gomlx)**: High-level ML framework in Go (tensors, layers, automatic differentiation, optimizers, model training loops). GoMLX delegates lower-level graph compilation and execution to a `compute.Backend`.
- **[`github.com/gomlx/compute`](https://github.com/gomlx/compute)**: Foundational execution and backend abstraction interface (`compute.Backend`, `compute.Builder`, `compute.Function`, `compute.Executable`, `compute.Buffer`), shared `dtypes`, `shapes`, `shapeinference`, and backend compliance tests (`compute/support/backendtest`).
  - It also includes the "go" backend: written in pure Go, portable (including WASM), optionally supporting SIMD.
- **`github.com/gomlx/go-xla`**: Depends **only** on `github.com/gomlx/compute` (never on `github.com/gomlx/gomlx`, to avoid circular dependencies).
- **`github.com/gomlx/compute-onnx`**: Implements a ONNX Runtime (ORT) based backend. Supports Dynamic Shapes. Depends **only** on `github.com/gomlx/compute`
- **`github.com/gomlx/go-darwinml`**: Implements the "mps" (Apple's MPSGraph based) and "mlx" (Apple's MLX based) backends. Depends **only** on `github.com/gomlx/compute`.
- **`github.com/gomlx/onnx-gomlx`**: Converts `.onnx` models to GoMLX models. 
- **`github.com/gomlx/go-huggingface`**: Hugging Face models tools: download repo files, iterate over datasets, tokenizer implementations, model implementations (using GoMLX). Currently it supports only a subset of the models ( mainly LLMs for text generation and instruction following) but is expandable.

When navigating the codebase of any of the related projects, read their .agents/AGENTS.md file to get more information about
the structure of the codebase and their specific conventions.

## Documentation & Git Workflow

- Record user-visible or API changes in `docs/CHANGELOG.md` (with `docs/CHANGELOG-pjrt.md` and `docs/CHANGELOG-stablehlo.md` kept as historical archives).
- Keep `.agents/AGENTS.md` (symlinked from `GEMINI.md`) updated whenever architecture, conventions, or non-obvious behavior changes.
- **Never `git commit` or `git push`** without explicit user instruction; leave changes uncommitted in the working tree for review.
- When changing code, if not explicitly asked, always wait for the operator to review the changes before committing them.

## Coding Style & Dependencies

- **Minimal External Dependencies**: This is a library, we want to avoid adding dependencies to our users.
- **Error Handling**: Wrap errors with stack traces using `github.com/pkg/errors` (`errors.Errorf`, `errors.Wrapf`, `errors.WithMessagef`) and format with `"%+v"` when logging stack traces. Follow GoMLX conventions: when building graphs, we throw (panic) errors. Outside of graph building functions, or in generic/helper functions that perhaps can be used in both contexts, we return errors instead of panicking.
- **Follow Existing Patterns**: Before writing new code, read neighboring files in the same package to match established conventions .
- **Avoid Large Files (Split Per-Op Implementations > 200 Lines)**: Keep files focused so directory listings stay scannable. Place complex op families (> ~200 lines, e.g., `ops_dotgeneral.go`, `ops_reduce.go`, `ops_scatter.go`, `flash.go`) in dedicated files rather than growing a monolith.
- **Copyright Notes**: Normal source files start with:
  ```go
  // Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0
  ```
  Auto-generated files (`gen_*`) omit the copyright header and instead carry a comment identifying the generator tool.
