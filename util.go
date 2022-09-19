// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand"

	"github.com/ALTree/bigfloat"
	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-9
)

// Functions are functions for neural networks
type Functions struct {
	tf32.Context
	Rnd          *rand.Rand
	FConcat      func(a, b tf32.Meta) tf32.Meta
	FAverageRows func(a tf32.Meta) tf32.Meta
	FAbs         func(a tf32.Meta) tf32.Meta
	FAvg         func(a tf32.Meta) tf32.Meta
	FAdd         func(a, b tf32.Meta) tf32.Meta
	FMul         func(a, b tf32.Meta) tf32.Meta
	FSum         func(a tf32.Meta) tf32.Meta
	FQuadratic   func(a, b tf32.Meta) tf32.Meta
	FHadamard    func(a, b tf32.Meta) tf32.Meta
	FSigmoid     func(a tf32.Meta) tf32.Meta
	FSumRows     func(a tf32.Meta) tf32.Meta
	FSoftmax     func(a tf32.Meta) tf32.Meta
	FT           func(a tf32.Meta) tf32.Meta
	FSoftmax0    func(a tf32.Meta) tf32.Meta
	FRelu        func(a tf32.Meta) tf32.Meta
	FNorm        func(a tf32.Meta) tf32.Meta
	FHadamard0   func(a, b tf32.Meta) tf32.Meta
	FDropout     func(a tf32.Meta) tf32.Meta
}

// CreateFunctions creates functions
func CreateFunctions(dummy bool) *Functions {
	f := &Functions{
		Rnd: rand.New(rand.NewSource(1)),
	}
	f.FConcat = tf32.B(f.Concat)
	f.FAverageRows = tf32.U(f.AverageRows)
	f.FAbs = tf32.U(f.Abs)
	f.FAvg = tf32.U(f.Avg)
	f.FAdd = tf32.B(f.Add)
	f.FMul = tf32.B(f.Mul)
	f.FSum = tf32.U(f.Sum)
	f.FQuadratic = tf32.B(f.Quadratic)
	f.FHadamard = tf32.B(f.Hadamard)
	f.FSigmoid = tf32.U(f.Sigmoid)
	f.FSumRows = tf32.U(f.SumRows)
	f.FSoftmax = tf32.U(f.Softmax1Big)
	f.FT = tf32.U(f.T)
	f.FSoftmax0 = tf32.U(f.Softmax0)
	f.FRelu = tf32.U(f.ReLu)
	f.FNorm = tf32.U(f.Normalize)
	f.FHadamard0 = tf32.B(f.Hadamard0)
	if dummy {
		f.FDropout = tf32.U(func(k tf32.Continuation, node int, a *tf32.V) bool {
			return k(a)
		})
	} else {
		f.FDropout = tf32.U(f.Dropout)
	}
	return f
}

// RegularAttention implements the attention mechanism described in
// https://arxiv.org/abs/1706.03762?amp=1
func RegularAttention(f *Functions, query, key, value, dk tf32.Meta) tf32.Meta {
	return f.FT(f.FMul(f.FSoftmax0(f.FHadamard(f.FMul(query, key), dk)), value))
}

// SimpleAttention implements the attention mechanism described in
// https://openreview.net/forum?id=pW--cu2FCHY
func SimpleAttention(f *Functions, query, key, value, dk tf32.Meta) tf32.Meta {
	return f.FHadamard(f.FSigmoid(query), f.FSumRows(f.FHadamard(f.FSoftmax(key), value)))
}

// IdentityAttention implements an identity attention
func IdentityAttention(f *Functions, query, key, value, dk tf32.Meta) tf32.Meta {
	return value
}

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
func Quadratic(k tf32.Continuation, node int, a, b *tf32.V) bool {
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
func ComplexQuadratic(k tc128.Continuation, node int, a, b *tc128.V) bool {
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
func (f *Functions) Concat(k tf32.Continuation, node int, a, b *tf32.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c := tf32.NewV(a.S[0]+b.S[0], a.S[1])
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i := 0; i < a.S[1]; i++ {
			for j := 0; j < a.S[0]; j++ {
				c.X = append(c.X, a.X[i*a.S[0]+j])
			}
			for j := 0; j < b.S[0]; j++ {
				c.X = append(c.X, b.X[i*b.S[0]+j])
			}
		}
	}
	f.Set(node, c.X)
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
func ComplexSigmoid(k tc128.Continuation, node int, a *tc128.V) bool {
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
	/*if a > 88 {
		a = 88
	}*/
	return float32(math.Exp(float64(a)))
}

// Softmax0 is the softmax function
func (f *Functions) Softmax0(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size, sum := tf32.NewV(a.S...), len(a.X), 0.0
	values := make([]float64, size)
	for i := 0; i < size; i++ {
		e := math.Exp(float64(a.X[i]))
		if math.IsNaN(e) || math.IsInf(e, 0) {
			panic(fmt.Errorf("%f is not a valid exponent", a.X[i]))
		}
		sum += e
		values[i] = e
	}
	for i, v := range values {
		values[i] /= sum
		c.X = append(c.X, float32(v/sum))
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := values[i]
		a.D[i] += d * float32(cx-cx*cx)
	}
	return false
}

// Softmax1 is the softmax function
func (f *Functions) Softmax1(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		max := float32(0)
		for _, v := range a.X {
			if v > max {
				max = v
			}
		}
		max *= S
		for i := 0; i < size; i += width {
			sum := float32(0.0)
			for _, ax := range a.X[i : i+width] {
				e := exp(ax - max)
				sum += e
				c.X = append(c.X, e)
			}
			for j, cx := range c.X[i : i+width] {
				c.X[i+j] = cx / sum
			}
		}
	}
	f.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

// Softmax1Big is the softmax function for big numbers
func (f *Functions) Softmax1Big(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		max := float32(0)
		for _, v := range a.X {
			if v > max {
				max = v
			}
		}
		s := float64(max) * S
		values := make([]float64, width)
		for i := 0; i < size; i += width {
			sum := 0.0
			for j, ax := range a.X[i : i+width] {
				values[j] = math.Exp(float64(ax) - s)
				sum += values[j]
			}
			for _, cx := range values {
				c.X = append(c.X, float32(cx/sum))
			}
		}
	}
	f.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

// Clamp is a clamp activation function
func Clamp(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size := tf32.NewV(a.S...), len(a.X)
	for i := 0; i < size; i++ {
		if ax := a.X[i]; ax > 709 {
			c.X = append(c.X, 709)
		} else {
			c.X = append(c.X, ax)
		}
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		if ax := a.X[i]; ax > 709 {
			a.D[i] += 0
		} else {
			a.D[i] += d
		}
	}
	return false
}

// Hadamard computes the hadamard product of two tensors
func (f *Functions) Hadamard0(k tf32.Continuation, node int, a, b *tf32.V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	if a.S[1] != b.S[1] && b.S[1] != 1 {
		panic("dimensions are not the same")
	}
	c := tf32.NewV(a.S...)
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i, j := range a.X {
			c.X = append(c.X, j*b.X[i%length])
		}
	}
	f.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * b.X[i%length]
		b.D[i%length] += j * a.X[i]
	}
	return false
}

// Dropout is a dropout regularization function
func (f *Functions) Dropout(k tf32.Continuation, node int, a *tf32.V) bool {
	size, width := len(a.X), a.S[0]
	c, drops, factor := tf32.NewV(a.S...), make([]int, width), float32(1)/(1-.1)
	for i := range drops {
		if f.Rnd.Float64() > .1 {
			drops[i] = 1
		}
	}
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
	}
	f.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			if drops[j] == 1 {
				a.D[i+j] += c.D[i+j]
			}
		}
	}
	return false
}

// SoftmaxBig is the softmax function implemented with big float
func SoftmaxBig(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size, sum := tf32.NewV(a.S...), len(a.X), big.NewFloat(0.0)
	values, done := make([]big.Float, size), make(chan bool, 8)
	process := func(i int) {
		v := big.NewFloat(float64(a.X[i]))
		v.SetPrec(128)
		e := bigfloat.Exp(v)
		if e.IsInf() {
			panic(fmt.Errorf("%f is not a valid exponent", a.X[i]))
		}
		values[i] = *e
		done <- true
	}
	for i := 0; i < size; i++ {
		go process(i)
	}
	for i := 0; i < size; i++ {
		<-done
	}
	sum.SetPrec(128)
	for _, v := range values {
		sum.Add(sum, &v)
	}
	for i := range values {
		values[i].Quo(&values[i], sum)
		value, _ := values[i].Float32()
		c.X = append(c.X, value)
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
func (f *Functions) ReLu(k tf32.Continuation, node int, a *tf32.V) bool {
	c := tf32.NewV(a.S[0], a.S[1])
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			max := j
			if max < 0 {
				max = 0
			}
			c.X = append(c.X, max)
		}
	}
	f.Set(node, c.X)
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
	w, h := width/7, height/7
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
func PositionEncodingLayer(k tf32.Continuation, node int, a *tf32.V) bool {
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
func Mask(k tf32.Continuation, node int, a *tf32.V) bool {
	width := a.S[0]
	c := tf32.NewV(width, 1)
	for i := 0; i < width; i++ {
		c.X = append(c.X, a.X[i])
	}
	if k(&c) {
		return true
	}
	for i := 0; i < width; i++ {
		a.D[i] += c.D[i]
	}
	return false
}

// AverageRows averages the rows of a tensor
func (f *Functions) AverageRows(k tf32.Continuation, node int, a *tf32.V) bool {
	size, width, n := len(a.X), a.S[0], float32(a.S[1])
	c := tf32.NewV(width)
	cached := f.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				c.X[j] += ax
			}
		}
		for i := 0; i < width; i++ {
			c.X[i] /= n
		}
	}
	f.Set(node, c.X)
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
func (f *Functions) Normalize(k tf32.Continuation, node int, a *tf32.V) bool {
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
	}
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			if deviation[j] == 0 {
				c.X = append(c.X, ax-mean[j])
			} else {
				c.X = append(c.X, (ax-mean[j])/deviation[j])
			}
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
