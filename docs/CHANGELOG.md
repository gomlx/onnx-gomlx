# v0.3.3: Basic Quantization support (@ajroetker); Updated for GoMLX 0.25.0

- Updates for GoMLX 0.25.0
- MatMulInteger quantization operation for integer matrix multiplication on quantized values. (by @ajroetker)
  - Y = (A - a_zero_point) * (B - b_zero_point)
- Clean ups (by @siherrmann)

# v0.3.2: New ops (2025/11/18) 

- Thanks to @timakey11 for this release the contribution!
- New Operators: 
  - `LayerNormalization`: Added support for the LayerNormalization operator, a key component in
    many modern neural network architectures.
  - `Split`: Implemented the Split operator, allowing tensors to be split into multiple outputs.
  - `If`: Added support for the If control flow operator, enabling conditional execution within the
    graph.
- Subgraph support: used by the `If` operator.
- New Models that we can now run :
  - https://huggingface.co/mirth/chonky_modernbert_base_1
  - https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1

# v0.3.1: Updated test dependencies

* Updated test dependencies: including go-huggingface and github.com/daulet/tokenizers to their latest versions.
* Added InceptionV3 model benchmark.
* Added support for `Pad` and `AveragePool` ONNX ops.

# v0.3.0: Updated to GoMLX v0.24.0

* Updated dependencies to GoMLX v0.24.0, and its improved directory organization.
  Since GoMLX changed the API (package directories changed), we bump the version here as well.
* Updates to README.md.

# v0.2.5 2025/08/22

* Updated dependencies to GoMLX v0.22.1
* Added Conv, MaxPool, BatchNormalization, and AverageGlobalPool operations.
* Added `Sin` and `Cos`
* Added `ScatterND` and `Trilu`
* Updated `Slice` to fully match ONNX spec

# v0.2.4 2025/06/12 

* Added support for other ONNX dtypes that require conversion during reading.
  Also added conversion when saving values back to the ONNX proto.
* Updated dependencies to GoMLX.
* Added `onnxtests.py` to help test/explore individual ONNX ops using ONNXRuntime. 
* New ops: `DequantizeLinear`, `DynamicQuantizeLinear`.

# v0.2.3 2025/05/31

* Added Save/Check values of outputs for internal/benchmarks: allows it to be
  used as a functional test during the development of GoMLX SimpleGo backend.
* Updated dependencies to latest GoMLX v0.19.5

# v0.2.2 2025/05/22

* Added Min and Max operators.
* Updated dependency to GoMLX v0.19.3.

# v0.2.1 2025/05/01

* Updated to GoMLX v0.19.1
* Included default GoMLX backends by default.

# v0.2.0 2025/02/02

* Updated to GoMLX v0.17.0
* Added bitwise operators.
* Added parallel benchmarks.
* Added benchmarks documentation.

# v0.1.5 🎄 2024/12/19 🎄

* Added `internal/bechmarks` package: See progress in https://docs.google.com/spreadsheets/d/1ikpJH6rVVHq8ES-IA8U4lkKH4XsTSpRyZewXwGTgits/edit?gid=1753191050#gid=1753191050
  * Benchmark ONNX models with XLA, ONNX Runtime (ORT), CPU and GPU
  * Very simple models
  * KnightsAnalytics/all-MiniLM-L6-v2
  * Slices (parts of) KnightsAnalytics/all-MiniLM-L6-v2
* Updated dependencies to GoMLX 0.16.1 with lots of accelerations.

# v0.1.4 - 2024/11/28

* Added Flatten op support.

# v0.1.3 - 2024/11/21

* Added ContextToONNX to save variables back to the ONNX model (in memory).
* Refactored internal/togomlx to inside onnx/ subdirectory.
* Added Model.Write and Model.SaveToFile.

# v0.1.2 - 2024/11/17

* Added LSTM op support, with a small example. 

# v0.1.1 - 2024/11/15

* Assume some variables are constant during constant-expression evaluation.
* Improved pretty-printing of attributes: include their values for small values.
* New ops: Range, Tile, CumSum, Not, Tanh, GatherElements, several standard unary and binary operators.
* Fixed ops: Where.

# v0.1.0

* First working version – for a few models.
* Constant-expression evaluation during a model build: needed for parameters that are fed dynamically 
  to ONNX but require static values in GoMLX/XLA.
