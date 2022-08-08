// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
	"math/rand"

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
func SelectedPositionEncoding(positions []Position, input *tf32.V) {
	length, d, t := len(input.X), input.S[0], 0
	for i := 0; i < length; i += d {
		k := 0.0
		for j := 0; j < d; j++ {
			position := float64(positions[t].Positions[j])
			if j&1 == 0 {
				input.X[i+j] += float32(math.Sin(math.Pow(8*4096, -2*k/float64(d)) * position))
			} else {
				input.X[i+j] += float32(math.Cos(math.Pow(8*4096, -2*k/float64(d)) * position))
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

// Concat concats two tensors
func Concat(k tf32.Continuation, a, b *tf32.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c := tf32.NewV(a.S[0]+b.S[0], a.S[1])
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			c.X = append(c.X, a.X[i*a.S[0]+j])
		}
		for j := 0; j < b.S[0]; j++ {
			c.X = append(c.X, b.X[i*b.S[0]+j])
		}
	}
	if k(&c) {
		return true
	}
	for i := 0; i < a.S[1]; i++ {
		for j := 0; j < a.S[0]; j++ {
			a.D[i*a.S[0]+j] += c.D[i*c.S[0]+j]
		}
		for j := 0; j < b.S[0]; j++ {
			b.D[i*b.S[0]+j] += c.D[i*c.S[0]+j+a.S[0]]
		}
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

// ReLu is the rectified linear unit function
func ReLu(k tf32.Continuation, a *tf32.V) bool {
	c := tf32.NewV(a.S[0], a.S[1])
	for _, j := range a.X {
		max := j
		if max < 0 {
			max = 0
		}
		c.X = append(c.X, max)
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		if c.X[i] != 0 {
			a.D[i>>1] += j
		}
	}
	return false
}

// Position is a position
type Position struct {
	X         int
	Y         int
	Crossed   bool
	Positions []int
}

// SelectPositions selects the positions of input data
func SelectPositions(rnd *rand.Rand, width, height int, positions []Position) {
	w, h := width/4, height/4
	s := 0
	for k := 0; k < height; k += h {
		for j := 0; j < width; j += w {
			set, index := positions[s], 0
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					x := (j + x + width) % width
					y := (k + y + height) % height
					set.Positions[index] = x + y*width
					index++
				}
			}
			s++
		}
	}
	/*pixel := 0
	for _, set := range positions {
		index := 0
		for j := 0; j < width; j += w {
			for k := 0; k < height; k += h {
				x := j + (pixel % w)
				y := k + (pixel / w)
				//x := j + rnd.Intn(w)
				//y := k + rnd.Intn(h)

				set.Positions[index] = x + y*width
				index++
			}
		}
		pixel++
		//sort.Ints(positions[i].Positions)
	}*/
	/*for s, set := range positions {
		x, y := rnd.Intn(width), rnd.Intn(height)
		for i := range set.Positions {
			xx := (x + int(rnd.NormFloat64()*float64(width/8))) % width
			yy := (y + int(rnd.NormFloat64()*float64(height/8))) % height
			if xx < 0 {
				xx = -xx
			}
			if yy < 0 {
				yy = -yy
			}
			set.Positions[i] = yy*width + xx
		}
		sort.Ints(set.Positions)
		positions[s].X = x
		positions[s].X = y
	}
	for i, a := range positions {
		if a.Crossed {
			continue
		}
		min, closest := math.MaxFloat64, 0
		for j, b := range positions {
			if i == j || b.Crossed {
				continue
			}
			x1, x2, y1, y2 := float64(a.X), float64(b.X), float64(a.Y), float64(b.Y)
			distance := math.Sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
			if distance < min {
				min, closest = distance, j
			}
		}
		positions[i].Crossed = true
		positions[closest].Crossed = true
		for k, x := range positions[i].Positions {
			if k&1 == 1 {
				positions[i].Positions[k], positions[closest].Positions[k] = positions[closest].Positions[k], x
			}
		}
		sort.Ints(positions[i].Positions)
		sort.Ints(positions[closest].Positions)
	}*/
}

// PositionEncodingLayer add position encoding to vector
func PositionEncodingLayer(k tf32.Continuation, a *tf32.V) bool {
	c := tf32.NewV(a.S...)
	length, width, t := len(a.X), a.S[0], 0.0
	for i := 0; i < length; i += width {
		k := 0.0
		for j := 0; j < width; j++ {
			if j&1 == 0 {
				c.X = append(c.X, a.X[i+j]+float32(math.Sin(math.Pow(10000, -2*k/float64(width))*t)))
			} else {
				c.X = append(c.X, a.X[i+j]+float32(math.Cos(math.Pow(10000, -2*k/float64(width))*t)))
				k++
			}
		}
		t++
	}
	if k(&c) {
		return true
	}
	for i := 0; i < length; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[i+j]
		}
	}
	return false
}

// Mask masks the input data
func Mask(k tf32.Continuation, a *tf32.V) bool {
	c := tf32.NewV(128+128, 1)
	for i := 0; i < 128+128; i++ {
		c.X = append(c.X, a.X[i])
	}
	if k(&c) {
		return true
	}
	for i := 0; i < 128+128; i++ {
		a.D[i] += c.D[i]
	}
	return false
}

// AverageRows averages the rows of a tensor
func AverageRows(k tf32.Continuation, a *tf32.V) bool {
	size, width, n := len(a.X), a.S[0], float32(a.S[1])
	c := tf32.NewV(width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
		}
	}
	for i := 0; i < width; i++ {
		c.X[i] /= n
	}
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[j] / n
		}
	}
	return false
}

// Normalize normalizes the input data
// https://math.stackexchange.com/questions/4046499/partial-derivative-of-sample-standard-deviation-w-r-t-individual-data-points
// (x - u)/s
// (1 - 1/n)/s + (x - u)(2/n)(x/n - u)/(-2*s^3)
// (1 - 1/n)/s - (x - u)(x/n - u)/(n*s^3)
// (n^2 s^2 - n s^2 - n u^2 + n u x + u x - x^2)/(n^2 s^3)
func Normalize(k tf32.Continuation, a *tf32.V) bool {
	size, width, n := len(a.X), a.S[0], float32(a.S[1])
	c, mean := tf32.NewV(a.S...), make([]float32, width)
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			mean[j] += ax
		}
	}
	for i := 0; i < width; i++ {
		mean[i] /= n
	}
	deviation := make([]float32, width)
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			diff := (ax - mean[j])
			deviation[j] += diff * diff
		}
	}
	for i := 0; i < width; i++ {
		deviation[i] = float32(math.Sqrt(float64(deviation[i] / n)))
		if deviation[i] == 0 {
			deviation[i] = 0.001
		}
	}
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X = append(c.X, (ax-mean[j])/deviation[j])
		}
	}
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			u, s, x := mean[j], deviation[j], a.X[i+j]
			a.D[i+j] += c.D[i+j] * (n*n*s*s - n*s*s - n*u*u + n*u*x + u*x - x*x) / (n * n * s * s * s)
		}
	}
	return false
}
