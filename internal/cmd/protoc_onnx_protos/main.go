// protoc_onnx_protos compiles the .proto from the https://github.com/onnx/onnx sources to subpackages of
// onnx-gomlx/internal/protos.
//
// It uses the standard `protoc` tool. Remember to update your go protoc plugin with
// `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`
//
// It should be executed under the onnx-gomlx/internal/protos directory -- suggested as a go:generate --
// and it requires ONNX_SRC to be set to a cloned github.com/onnx/onnx repository.
//
// It copies over the proto files from "${ONNX_SRC}" to the current directory, editing its contents to
// onnx-gomlx repository. Then it executes `protoc` to generate the Go code.
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

const (
	onnxSrcEnvVar   = "ONNX_SRC"
	goProtosPackage = "github.com/janpfeifer/onnx-gomlx/internal/protos"
)

var protoFiles = []string{
	// We use the "onnx-ml.proto3" version of this file. See brief mention here:
	// https://github.com/onnx/onnx/blob/main/docs/IR.md
	"onnx-ml.proto",
	"onnx-operators-ml.proto",
	"onnx-data.proto",
}

// must log.Fatalf if err is not nil.
func must(err error) {
	if err != nil {
		log.Fatalf("Error:\n%+v\n", err)
	}
}

// must1 is like must, but returns the output if there was no error.
func must1[T any](t T, err error) T {
	must(err)
	return t
}

func main() {
	onnxSrc := os.Getenv(onnxSrcEnvVar)
	if onnxSrc == "" {
		fmt.Fprintf(os.Stderr, "Please set %s to the directory containing the locally cloned github.com/onnx/onnx repository.\n", onnxSrcEnvVar)
		os.Exit(1)
	}

	// Generate the --go_opt=M... flags
	goOpts := make([]string, len(protoFiles))
	for ii, protoFile := range protoFiles {
		goOpts[ii] = fmt.Sprintf("--go_opt=M%s=%s", protoFile, goProtosPackage)
	}

	// Read file from $ONNX_SRC, remove any go_package options, rewrite the package to "protoFiles", and write to currnet
	// directory.
	for _, protoFile := range protoFiles {
		// Fix file and write to current directory.
		protoPath := filepath.Join(onnxSrc, "onnx", protoFile) + "3" // Use the version 3 proto file.
		protoContents := must1(os.ReadFile(protoPath))
		protoContents = removeGoPackageOption(protoContents)
		protoContents = fixPackageName(protoContents)
		protoContents = fixImports(protoContents)
		must(os.WriteFile(protoFile, protoContents, 0644))
	}

	// Compile each of the proto files.
	for _, protoFile := range protoFiles {
		// Construct the protoc command
		args := []string{
			"--go_out=.",
			"-I=.",
			fmt.Sprintf("-I=%s/onnx", onnxSrc),
			fmt.Sprintf("--go_opt=module=%s", goProtosPackage),
		}
		args = append(args, goOpts...)
		args = append(args, protoFile)
		cmd := exec.Command("protoc", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "Error executing protoc for %s: %v\n", protoFile, err)
			_, _ = fmt.Fprintf(os.Stderr, "Command:\n%s\n", cmd)
			os.Exit(1)
		}
	}
}

var reRemoveGoPackageOption = regexp.MustCompile(`option\s+go_package\s*=\s*"[^"]*?";`)

func removeGoPackageOption(content []byte) []byte {
	return reRemoveGoPackageOption.ReplaceAll(content, []byte{})
}

var rePackageName = regexp.MustCompile(`package onnx;`)

func fixPackageName(content []byte) []byte {
	return rePackageName.ReplaceAll(content, []byte(`package protos;`))
}

var reImports = regexp.MustCompile(`import\s+"onnx/(.*)3";`)

func fixImports(content []byte) []byte {
	return reImports.ReplaceAll(content, []byte(`import "$1";`))
}
