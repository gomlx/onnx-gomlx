# ONNX-GoMLX from ONNX to GoMLX and back

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gomlx/onnx-gomlx?tab=doc)

## 📖 Overview
ONNX-GoMLX converts [ONNX models](https://onnx.ai/) (`.onnx` suffix) to 
[GoMLX (an accelerated machine learning framework for Go](https://github.com/gomlx/gomlx) and optionally back to ONNX.

The main use cases so far are:

1. **Fine-tuning**: import an inference only ONNX model to GoMLX, and use its auto-differentiation and training loop to
   fine-tune models. It allows saving the fine-tuned model as a GoMLX checkpoint or export the fine-tuned weights
   back to the ONNX model. This can also be used to expand / combine models.
2. **Inference**: use an ONNX file using Go and not having to include [ONNX Runtime](https://onnxruntime.ai/) (or Python)
   -- at the cost of including XLA/PJRT (the current only backend for GoMLX). It also allows one to extend the
   model with extra ML pre/post-processing using GoMLX (image transformations, normalization, combining models,
   building ensembles, etc.). This may be interesting for large/expensive models, or large throughput on large
   batches.
    * **TODO**: Benchmark XLA/PJRT (Google) vs ONNX Runtime (Microsoft) -- both are sophisticated, well maintained
      numerical computation engines. It should be interesting to evaluate the performance in various hardwares: 
      amd64, arm64, GPU.
    * Notice if you want to simply get a pure Go inference of ONNX models, see 
      [github.com/AdvancedClimateSystems/gonnx](https://github.com/AdvancedClimateSystems/gonnx) or
      [github.com/oramasearch/onnx-go](https://github.com/oramasearch/onnx-go). They will be slower (~8x based on a SentenceEncoder model, BERT based using `gonnx` vs `ONNXRuntime`) than 
      the XLA inference (or `onnxruntime`) for large projects, but for many use cases it doesn't matter, and they
      are a much smaller pure Go dependency. Only for CPU (no GPU support).

## 🚧 **EXPERIMENTAL and UNDER-DEVELOPMENT**

It's "fresh from the oven", so it may not be working as intended, and there are no guarantees of any type.

There are 10 or so models that are working so far:

* [Sentence Enconding all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
has been working perfectly, see example below.
* [ONNX-GoMLX demo/development notebook](https://github.com/gomlx/onnx-gomlx/blob/main/onnx-go.ipynb): both serves as a functional test and to demo what it can do.

But not all operations ("ops") are converted yet. If you try it and find some that is not, please let us know (create an "issue")
we will be happy to try to convert them -- generally, all the required scaffolding and tooling is already there, and
converting ops has been very easy.

We'll leave this "experimental" note here for now. If we don't bump into any issues we'll remove the note.

## 🎓 Example

We download (and cache) the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
using [github.com/gomlx/go-huggingface](https://github.com/gomlx/go-huggingface).

The tokens were for now hardcoded -- eventually [github.com/gomlx/go-huggingface](https://github.com/gomlx/go-huggingface) should also
do the tokenization for various models.

```go
import (
	"github.com/gomlx/onnx-gomlx/onnx"
    "github.com/gomlx/go-huggingface/hub"
)

...

// Download and cache ONNX model from HuggingFace.
hfAuthToken := os.Getenv("HF_TOKEN")
hfModelID := "sentence-transformers/all-MiniLM-L6-v2"
repo := hub.New(modelID).WithAuth(hfAuthToken)
modelPath := must.M1(repo.DownloadFile("onnx/model.onnx"))

// Parse ONNX model.
model := must.M1(onnx.ReadFile(modelPath))

// Convert ONNX variables (model weights) to GoMLX Context -- which stores variables and can be checkpointed (saved):
ctx := context.New()
must.M(model.VariablesToContext(ctx))

// Execute it with GoMLX/XLA:
sentences := []string{
    "This is an example sentence",
    "Each sentence is converted"}
//... tokenize ...
inputIDs := [][]int64{
    {101, 2023, 2003, 2019, 2742, 6251,  102},
    { 101, 2169, 6251, 2003, 4991,  102,    0}}
tokenTypeIDs := [][]int64{
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0}}
attentionMask := [][]int64{
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0}}
var embeddings []*tensors.Tensor
embeddings = context.ExecOnceN( // Execute a GoMLX computation graph with a context
	backends.New(),  // GoMLX backend to use (defaults to XLA) 
	ctx, // Context store the model variables/weights and optional hyperparameters.
	func (ctx *context.Context, inputs []*Node) []*Node {
		// Convert ONNX model (in `model`) to a GoMLX computation graph. It returns a slice of values (with only one for this model)
		return model.CallGraph(ctx, inputs[0].Graph(), map[string]*Node{
			"input_ids": inputs[0],
			"attention_mask": inputs[1],
			"token_type_ids": inputs[2]}, targetOutputs...)
	}, 
	inputIDs, attentionMask, tokenTypeIDs)  // Inputs to the GoMLX function.
fmt.Printf("Embeddings: %s", embeddings)
```

The output looks like:

```
Embeddings: [2][7][384]float32{
 {{-0.0886, -0.0368, 0.0180, ..., 0.0261, 0.0912, -0.0152},
  {-0.0200, -0.0014, -0.0177, ..., 0.0204, 0.0522, 0.1991},
  {-0.0196, -0.0336, -0.0319, ..., 0.0203, 0.0709, 0.0644},
  ...,
  {-0.0253, 0.0408, 0.0125, ..., -0.0270, 0.0377, 0.1133},
  {-0.0140, -0.0275, 0.0796, ..., -0.0748, 0.0774, -0.0657},
  {0.0318, -0.0032, -0.0210, ..., 0.0387, 0.0191, -0.0059}},
 {{-0.0886, -0.0368, 0.0180, ..., 0.0261, 0.0912, -0.0152},
  {0.0304, 0.0531, -0.0238, ..., -0.1011, 0.0218, 0.0473},
  {-0.0027, -0.0508, 0.0805, ..., -0.0777, 0.0881, -0.0560},
  ...,
  {0.0928, 0.0165, -0.0976, ..., 0.0449, 0.0390, -0.0182},
  {0.0231, 0.0090, -0.0213, ..., 0.0232, 0.0191, -0.0066},
  {-0.0213, 0.0019, 0.0043, ..., 0.0561, 0.0170, 0.0256}}}
```

## Fine-Tuning

1. Extract the ONNX model's weight to GoMLX `Context`: see `Model.VariablesToContext()`.
2. Use `Model.CallGraph()` in your GoMLX model function (see example just above).
3. Train model as usual in GoMLX.
4. Depending how you are going to use the model:
   1. Save the model as a GoMLX checkpoint, as usual.
   2. Save the model by updating the ONNX model: after training use `Model.ContextToONNX()` to copy the update variable  
      values from GoMLX `Context` back to the ONNX model (in-memory), and then use `Model.Write()` or 
      `Model.SaveToFile()` to save the updated ONNX model to disk.
   
## 🤝 Collaborating

Collaboration is very welcome: either in the form of code, or simply with ideas with real applicability. Don't
hesitate starting a discussion or issue in the repository.

If you are interested, we have two notebooks we use to compare results. They are a good starting point for anyone curious:

* [Go Version](https://github.com/gomlx/onnx-gomlx/blob/main/onnx-go.ipynb)
* [ONNX Python Version](https://github.com/gomlx/onnx-gomlx)

## 🥳 Thanks

* This project was born from brainstorming with the talented folks at [KnightAnalytics](https://www.knightsanalytics.com/).
  Without their insights and enthusiasm this wouldn't have gotten off the ground.
* [ONNX models](https://onnx.ai/) is such a nice open source standard to communicate models across different implementations.
* [OpenXLA/XLA](https://github.com/openxla/xla) the open-source backend engine by Google that powers this implementation.
* Sources of inspiration:
  * https://github.com/knights-analytics/onnx-gomlx
  * https://github.com/oramasearch/onnx-go
