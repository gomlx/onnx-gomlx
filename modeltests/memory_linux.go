// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build linux

package modeltests

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func resetKernelPeakRSS() bool {
	return os.WriteFile("/proc/self/clear_refs", []byte("5\n"), 0) == nil
}

func currentRSSBytes() int64 {
	return readProcStatusKiB("VmRSS:") * 1024
}

func kernelPeakRSSBytes() int64 {
	return readProcStatusKiB("VmHWM:") * 1024
}

func readProcStatusKiB(prefix string) int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer func() { _ = f.Close() }()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		fields := strings.Fields(line[len(prefix):])
		if len(fields) == 0 {
			return 0
		}
		v, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil {
			return 0
		}
		return v
	}
	return 0
}
