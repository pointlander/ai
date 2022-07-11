// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
	"math/rand"
	"sort"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
)

// PositionEncoding add position encoding to vector
func PositionEncoding(input *tf32.V) {
	length, d, t := len(input.X), input.S[0], 0.0
	for i := 0; i < length; i += d {
		k := 0.0
		for j := 0; j < d; j++ {
			if j&1 == 0 {
				input.X[i+j] += float32(math.Sin(math.Pow(10000, -2*k/float64(d)) * t))
			} else {
				input.X[i+j] += float32(math.Cos(math.Pow(10000, -2*k/float64(d)) * t))
				k++
			}
		}
		t++
	}
}

// SelectedPositionEncoding add position encoding to vector for positions
func SelectedPositionEncoding(positions []int, input *tf32.V) {
	length, d, t := len(input.X), input.S[0], 0
	for i := 0; i < length; i += d {
		k := 0.0
		for j := 0; j < d; j++ {
			position := float64(positions[t]) / 4096
			if j&1 == 0 {
				input.X[i+j] += float32(math.Sin(math.Pow(10000, -2*k/float64(d)) * position))
			} else {
				input.X[i+j] += float32(math.Cos(math.Pow(10000, -2*k/float64(d)) * position))
				k++
			}
		}
		t++
	}
}

// ComplexPositionEncoding add position encoding to a complex vector
func ComplexPositionEncoding(input *tc128.V) {
	length, d, t := len(input.X), input.S[0], 0.0
	for i := 0; i < length; i += d {
		k := 0.0
		for j := 0; j < d; j++ {
			if j&1 == 0 {
				input.X[i+j] += complex(0, math.Sin(math.Pow(10000, -2*k/float64(d))*t))
			} else {
				input.X[i+j] += complex(0, math.Cos(math.Pow(10000, -2*k/float64(d))*t))
				k++
			}
		}
		t++
	}
}

// Quadratic computes the quadratic cost of two tensors
func Quadratic(k tf32.Continuation, a, b *tf32.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sum := tf32.NewV(1), float32(0.0)
	for i, ax := range a.X {
		p := (ax - b.X[i])
		sum += p * p
	}
	c.X = append(c.X, .5*sum)
	if k(&c) {
		return true
	}
	d := c.D[0]
	for i, ax := range a.X {
		a.D[i] += (ax - b.X[i]) * d
		b.D[i] += (b.X[i] - ax) * d
	}
	return false
}

// ComplexQuadratic computes the quadratic cost of two complex tensors
func ComplexQuadratic(k tc128.Continuation, a, b *tc128.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[0] != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sum := tc128.NewV(1), complex(0, 0)
	for i, ax := range a.X {
		p := (ax - b.X[i])
		sum += p * p
	}
	c.X = append(c.X, .5*sum)
	if k(&c) {
		return true
	}
	d := c.D[0]
	for i, ax := range a.X {
		a.D[i] += (ax - b.X[i]) * d
		b.D[i] += (b.X[i] - ax) * d
	}
	return false
}

// ComplexSigmoid computes the sigmoid of a complex tensor
func ComplexSigmoid(k tc128.Continuation, a *tc128.V) bool {
	c := tc128.NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, complex(1+math.Cos(cmplx.Phase(j)), 0)*j/2)
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		ax := a.X[i]
		a.D[i] += j * complex((1+math.Cos(cmplx.Phase(ax)))/2, 0)
	}
	return false
}

func exp(a float32) float32 {
	return float32(math.Exp(float64(a)))
}

// Softmax is the softmax function
func Softmax(k tf32.Continuation, a *tf32.V) bool {
	c, size, sum := tf32.NewV(a.S...), len(a.X), float32(0.0)
	for i := 0; i < size; i++ {
		e := exp(a.X[i])
		sum += e
		c.X = append(c.X, e)
	}
	for i, cx := range c.X {
		c.X[i] = cx / sum
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

// SelectPositions selects the positions of input data
func SelectPositions(rnd *rand.Rand, max int, positions []int) {
	for i := range positions {
		positions[i] = rnd.Intn(max)
	}
	sort.Ints(positions)
}
