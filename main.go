// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
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

func main() {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	_ = rnd

	iris := datum.Fisher
	others := tf32.NewSet()
	others.Add("input", 4, len(iris))
	others.Add("output", 4, len(iris))

	stats := [4]Statistics{}
	for _, w := range others.Weights {
		for _, data := range iris {
			for i, measure := range data.Measures {
				stats[i].Add(measure)
				w.X = append(w.X, float32(measure))
			}
		}
	}
}
