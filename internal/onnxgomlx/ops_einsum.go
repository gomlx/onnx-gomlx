// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package onnxgomlx

import (
	"sort"
	"strings"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/core/graph"
)

// expandEinsum resolves ellipses into unused axis labels and aligns shared axes
// (including singleton broadcasts and dynamic axis names) before GoMLX Einsum.
func expandEinsum(equation string, inputs []*Node) (string, []*Node) {
	equation = strings.ReplaceAll(equation, " ", "")
	parts := strings.Split(equation, "->")
	if len(parts) > 2 {
		exceptions.Panicf("Einsum: invalid equation %q", equation)
	}
	operands := strings.Split(parts[0], ",")
	if len(operands) != len(inputs) {
		exceptions.Panicf("Einsum: input count mismatch for equation %q (%d operands vs %d inputs)", equation, len(operands), len(inputs))
	}
	lengths := make([]int, len(inputs))
	maxLength := 0
	for i, operand := range operands {
		if strings.Contains(operand, "...") {
			lengths[i] = inputs[i].Rank() - len(strings.ReplaceAll(operand, "...", ""))
			if lengths[i] < 0 {
				exceptions.Panicf("Einsum: rank mismatch for %q with shape %s", operand, inputs[i].Shape())
			}
			maxLength = max(maxLength, lengths[i])
		}
	}
	labels := ""
	for _, r := range "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" {
		if !strings.ContainsRune(equation, r) && len(labels) < maxLength {
			labels += string(r)
		}
	}
	if len(labels) < maxLength {
		exceptions.Panicf("Einsum: too many axes in %q", equation)
	}
	for i, operand := range operands {
		operands[i] = strings.ReplaceAll(operand, "...", labels[maxLength-lengths[i]:])
		if len(operands[i]) != inputs[i].Rank() {
			exceptions.Panicf("Einsum: rank mismatch for %q with shape %s", operand, inputs[i].Shape())
		}
	}
	output := ""
	if len(parts) == 2 {
		output = strings.ReplaceAll(parts[1], "...", labels)
	} else {
		counts := make(map[rune]int)
		for _, operand := range operands {
			for _, r := range operand {
				counts[r]++
			}
		}
		var single []string
		for r, count := range counts {
			if count == 1 && !strings.ContainsRune(labels, r) {
				single = append(single, string(r))
			}
		}
		sort.Strings(single)
		output = labels + strings.Join(single, "")
	}

	// Broadcast singleton shared axes and align shared dynamic axis names across operands.
	inputs = append([]*Node(nil), inputs...)
	if len(inputs) == 2 {
		specs0 := DimensionSpecsFor(inputs[0])
		specs1 := DimensionSpecsFor(inputs[1])
		bcast0, bcast1 := false, false
		rename0, rename1 := false, false
		for ax0, r := range operands[0] {
			ax1 := strings.IndexRune(operands[1], r)
			if ax1 < 0 {
				continue
			}
			d0, d1 := inputs[0].Shape().Dim(ax0), inputs[1].Shape().Dim(ax1)
			switch {
			case d0 == 1 && d1 != 1:
				specs0[ax0] = specs1[ax1]
				bcast0 = true
			case d1 == 1 && d0 != 1:
				specs1[ax1] = specs0[ax0]
				bcast1 = true
			case d0 == shapes.DynamicDim && d1 == shapes.DynamicDim:
				n0, n1 := inputs[0].Shape().AxisName(ax0), inputs[1].Shape().AxisName(ax1)
				if n0 != n1 {
					shared := shapes.UniqueAxisName("#onnx_einsum")
					specs0[ax0] = NamedDynamicDim(shared, DimensionSize(inputs[0], ax0))
					specs1[ax1] = NamedDynamicDim(shared, DimensionSize(inputs[1], ax1))
					rename0, rename1 = true, true
				}
			}
		}
		if rename0 {
			inputs[0] = DynamicReshape(inputs[0], specs0...)
		}
		if rename1 {
			inputs[1] = DynamicReshape(inputs[1], specs1...)
		}
		if bcast0 {
			axes := make([]int, inputs[0].Rank())
			for i := range axes {
				axes[i] = i
			}
			inputs[0] = DynamicBroadcastInDim(inputs[0], axes, specs0...)
		}
		if bcast1 {
			axes := make([]int, inputs[1].Rank())
			for i := range axes {
				axes[i] = i
			}
			inputs[1] = DynamicBroadcastInDim(inputs[1], axes, specs1...)
		}
	}

	return strings.Join(operands, ",") + "->" + output, inputs
}
