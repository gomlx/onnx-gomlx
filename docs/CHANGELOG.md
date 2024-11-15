# v0.1.1 - 2024/11/15

* Assume some variables are constant during constant-expression evaluation.
* Improved pretty-printing of attributes: include their values for small values.
* New ops: Range, Tile, CumSum, Not, Tanh, GatherElements, several standard unary and binary operators.
* Fixed ops: Where.

# v0.1.0

* First working version -- for a few models.
* Constant-expression evaluation during model build: needed for parameters that are fed dynamically 
  to ONNX, but require static values in GoMLX/XLA.