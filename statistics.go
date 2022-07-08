// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
)

// Statistics captures statistics
type Statistics struct {
	Sum        float64
	SumSquared float64
	Count      int
}

// Add adds a statistic
func (s *Statistics) Add(value float64) {
	s.Sum += value
	s.SumSquared += value * value
	s.Count++
}

// StandardDeviation calculates the standard deviation
func (s Statistics) StandardDeviation() float64 {
	sum, count := s.Sum, float64(s.Count)
	return math.Sqrt((s.SumSquared - sum*sum/count) / count)
}

// Average calculates the average
func (s Statistics) Average() float64 {
	return s.Sum / float64(s.Count)
}

// String returns the statistics as a string`
func (s Statistics) String() string {
	return fmt.Sprintf("%f +- %f", s.Average(), s.StandardDeviation())
}

// ComplexStatistics captures statistics for complex numbers
type ComplexStatistics struct {
	Sum        complex128
	SumSquared complex128
	Count      int
}

// Add adds a statistic
func (s *ComplexStatistics) Add(value complex128) {
	s.Sum += value
	s.SumSquared += value * value
	s.Count++
}

// StandardDeviation calculates the standard deviation
func (s ComplexStatistics) StandardDeviation() complex128 {
	sum, count := s.Sum, complex(float64(s.Count), 0)
	return cmplx.Sqrt((s.SumSquared - sum*sum/count) / count)
}

// Average calculates the average
func (s ComplexStatistics) Average() complex128 {
	return s.Sum / complex(float64(s.Count), 0)
}

// String returns the statistics as a string`
func (s ComplexStatistics) String() string {
	return fmt.Sprintf("%f +- %f", s.Average(), s.StandardDeviation())
}
