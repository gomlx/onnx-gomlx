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

* Added ContextToONNX to save variables back to ONNX model (in memory).
* Refactored internal/togomlx to inside onnx/ subdir.
* Added Model.Write and Model.SaveToFile.

# v0.1.2 - 2024/11/17

* Added LSTM op support, with small example. 

# v0.1.1 - 2024/11/15

* Assume some variables are constant during constant-expression evaluation.
* Improved pretty-printing of attributes: include their values for small values.
* New ops: Range, Tile, CumSum, Not, Tanh, GatherElements, several standard unary and binary operators.
* Fixed ops: Where.

# v0.1.0

* First working version -- for a few models.
* Constant-expression evaluation during model build: needed for parameters that are fed dynamically 
  to ONNX, but require static values in GoMLX/XLA.