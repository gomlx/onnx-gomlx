# Protobuf Generated Files

All files in this directory are generated using `onnx-gomlx/internal/cmd/protoc_onnx_protos` tool, except
the `protos.go` file, which includes the one `//go:generate go run ../cmd/protoc_onnx_protos` line.

Notice there are two variants of ONNX protos: `onnx.proto` and `onnx-ml.proto`, and one can't include both,
since they redefine each other. Why would they do that :sad: !? (and not document it in the proto files ...) 

Anyway, this project takes the `onnx-ml.proto`, because according to the [IR mention](https://github.com/onnx/onnx/blob/main/docs/IR.md)
it seems to be more complete, even though `onnx-gomlx` not necessarily supports all its operations.

