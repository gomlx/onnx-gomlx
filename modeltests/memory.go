// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"time"

	"github.com/gomlx/compute/support"
	"github.com/gomlx/compute/support/humanize"
)

// MemoryStats reports per-test memory usage metrics.
type MemoryStats struct {
	BaselineRSSBytes   int64
	PeakRSSBytes       int64
	EndRSSBytes        int64
	DeltaPeakRSSBytes  int64
	ProcessMaxRSSBytes int64
	PeakHeapAllocBytes int64
	ClearRefsUsed      bool
}

// String formats a concise single-line summary for test logs.
func (m MemoryStats) String() string {
	return fmt.Sprintf("peak_rss=%s (delta=+%s, base=%s), peak_go_heap=%s, proc_max_rss=%s",
		humanize.Bytes(m.PeakRSSBytes),
		humanize.Bytes(max(m.DeltaPeakRSSBytes, 0)),
		humanize.Bytes(m.BaselineRSSBytes),
		humanize.Bytes(m.PeakHeapAllocBytes),
		humanize.Bytes(m.ProcessMaxRSSBytes),
	)
}

// memoryTracker measures peak RSS and Go heap during a single sequential model test.
type memoryTracker struct {
	clearRefsOK bool
	baselineRSS int64

	mu       sync.Mutex
	peakRSS  int64
	peakHeap int64
	stopCh   chan struct{}
	doneCh   chan struct{}
}

func startMemoryTracker() *memoryTracker {
	for range 2 {
		runtime.GC()
	}
	debug.FreeOSMemory()

	clearOK := resetKernelPeakRSS()
	baseRSS := currentRSSBytes()

	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	mt := &memoryTracker{
		clearRefsOK: clearOK,
		baselineRSS: baseRSS,
		peakRSS:     baseRSS,
		peakHeap:    int64(ms.HeapAlloc),
		stopCh:      make(chan struct{}),
		doneCh:      make(chan struct{}),
	}

	go func() {
		defer close(mt.doneCh)
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-mt.stopCh:
				return
			case <-ticker.C:
				mt.sample()
			}
		}
	}()
	return mt
}

func (mt *memoryTracker) sample() {
	rss := currentRSSBytes()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	heap := int64(ms.HeapAlloc)

	mt.mu.Lock()
	if rss > mt.peakRSS {
		mt.peakRSS = rss
	}
	if heap > mt.peakHeap {
		mt.peakHeap = heap
	}
	mt.mu.Unlock()
}

func (mt *memoryTracker) Stop() MemoryStats {
	mt.sample()
	close(mt.stopCh)
	<-mt.doneCh

	endRSS := currentRSSBytes()
	kernelPeak := kernelPeakRSSBytes()
	procMax := support.MaxRSS()

	mt.mu.Lock()
	peakRSS := mt.peakRSS
	peakHeap := mt.peakHeap
	mt.mu.Unlock()

	if endRSS > peakRSS {
		peakRSS = endRSS
	}
	if mt.clearRefsOK && kernelPeak > peakRSS {
		peakRSS = kernelPeak
	}
	if peakRSS == 0 {
		peakRSS = procMax
	}

	delta := peakRSS - mt.baselineRSS
	if delta < 0 {
		delta = 0
	}
	return MemoryStats{
		BaselineRSSBytes:   mt.baselineRSS,
		PeakRSSBytes:       peakRSS,
		EndRSSBytes:        endRSS,
		DeltaPeakRSSBytes:  delta,
		ProcessMaxRSSBytes: procMax,
		PeakHeapAllocBytes: peakHeap,
		ClearRefsUsed:      mt.clearRefsOK,
	}
}
