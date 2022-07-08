// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"strings"

	"github.com/mjibson/go-dsp/fft"
	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Iris is the iris dataset
func Iris(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

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
	//PositionEncoding(others.Weights[0])

	set := tf32.NewSet()
	set.Add("embed", 4, hiddenSize)
	set.Add("position", hiddenSize, len(iris))
	set.Add("query", hiddenSize, hiddenSize)
	set.Add("key", hiddenSize, hiddenSize)
	set.Add("value", hiddenSize, hiddenSize)
	set.Add("project", hiddenSize, 4)

	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	quadratic := tf32.B(Quadratic)

	input := tf32.Sigmoid(tf32.Add(set.Get("position"), tf32.Mul(set.Get("embed"), others.Get("input"))))
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	transformer := tf32.Mul(set.Get("project"),
		tf32.Hadamard(tf32.Sigmoid(query),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key))), value))))
	cost := quadratic(transformer, others.Get("output"))

	alpha, eta, iterations := float32(.001), float32(.001), 8*2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()
		others.Zero()

		/*if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
			for j := range d {
				d[j] /= 10
			}
		}

		index := 0
		for _, data := range iris {
			for i, measure := range data.Measures {
				if d[i] == 0 {
					inputs.X[index] = float32(measure)
				} else {
					inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
				}
				index++
			}
		}*/

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		/*if total < .1 {
			break
		}*/
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	/*position := set.Weights[4]
	for i := 0; i < len(iris); i++ {
		for j := 0; j < 4; j++ {
			fmt.Printf("%f ", position.X[i*4+j])
		}
		fmt.Println()
	}*/
}

// ComplexIris is the iris dataset using complex numbers
func ComplexIris(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	iris := datum.Fisher
	others := tc128.NewSet()
	others.Add("input", 4, len(iris))
	others.Add("output", 4, len(iris))

	stats := [4]ComplexStatistics{}
	for _, w := range others.Weights {
		for _, data := range iris {
			for i, measure := range data.Measures {
				stats[i].Add(complex(measure, 0))
				w.X = append(w.X, complex(measure, 0))
			}
		}
	}
	//ComplexPositionEncoding(others.Weights[0])

	set := tc128.NewSet()
	set.Add("embed", 4, hiddenSize)
	set.Add("position", hiddenSize, len(iris))
	set.Add("query", hiddenSize, hiddenSize)
	set.Add("key", hiddenSize, hiddenSize)
	set.Add("value", hiddenSize, hiddenSize)
	set.Add("project", hiddenSize, 4)

	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]complex128, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]complex128, len(p.X)))
	}

	quadratic := tc128.B(ComplexQuadratic)

	input := tc128.Sigmoid(tc128.Add(set.Get("position"), tc128.Mul(set.Get("embed"), others.Get("input"))))
	query := tc128.Mul(set.Get("query"), input)
	key := tc128.Mul(set.Get("key"), input)
	value := tc128.Mul(set.Get("value"), input)
	transformer := tc128.Mul(set.Get("project"),
		tc128.Hadamard(tc128.Sigmoid(query),
			tc128.SumRows(tc128.Hadamard(tc128.T(tc128.Softmax(tc128.T(key))), value))))
	cost := quadratic(transformer, others.Get("output"))

	alpha, eta, iterations := complex128(.5+.5i), complex128(.5+.5i), 32*2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0.0)
		set.Zero()
		others.Zero()

		/*if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
			for j := range d {
				d[j] /= 10
			}
		}

		index := 0
		for _, data := range iris {
			for i, measure := range data.Measures {
				if d[i] == 0 {
					inputs.X[index] = float32(measure)
				} else {
					inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
				}
				index++
			}
		}*/

		total += tc128.Gradient(cost).X[0]
		sum := complex128(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := cmplx.Abs(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*complex(scaling, 0)
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		fmt.Println(i, total)
		/*if total < .1 {
			break
		}*/
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "complex_cost.png")
	if err != nil {
		panic(err)
	}
}

// IrisFFT is the iris dataset processing using FFT
func IrisFFT(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)

	}
	iris := datum.Fisher
	others := tf32.NewSet()
	others.Add("input", 4, len(iris))
	others.Add("output", 4, len(iris))

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
		for i := 0; i < 4; i++ {
			r := make([]float64, len(iris))
			for j, data := range iris {
				r[j] = data.Measures[i]
			}
			f := fft.FFTReal(r)
			max := 0.0
			for _, value := range f {
				if cmplx.Abs(value) > max {
					max = cmplx.Abs(value)
				}
			}
			for j, value := range f {
				w.X[j*4+i] = float32(cmplx.Abs(value))
			}
		}
		fmt.Println(w.X)
	}

	set := tf32.NewSet()
	set.Add("l1", 4, hiddenSize)
	set.Add("b1", hiddenSize, len(iris))
	set.Add("l2", 2*hiddenSize, 4)
	set.Add("b2", 4, len(iris))

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	quadratic := tf32.B(Quadratic)

	l1 := tf32.Everett(tf32.Add(set.Get("b1"), tf32.Mul(set.Get("l1"), others.Get("input"))))
	l2 := tf32.Add(set.Get("b2"), tf32.Mul(set.Get("l2"), l1))
	cost := quadratic(l2, others.Get("output"))

	alpha, eta, iterations := float32(.001), float32(.001), 4*1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()
		others.Zero()

		/*if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
			for j := range d {
				d[j] /= 10
			}
		}

		index := 0
		for _, data := range iris {
			for i, measure := range data.Measures {
				if d[i] == 0 {
					inputs.X[index] = float32(measure)
				} else {
					inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
				}
				index++
			}
		}*/

		total += tf32.Gradient(cost).X[0]
		if math.IsInf(float64(total), 0) {
			fmt.Println("inf")
			break
		} else if math.IsNaN(float64(total)) {
			fmt.Println("nan")
			break
		}
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		/*if total < .1 {
			break
		}*/
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "fft_cost.png")
	if err != nil {
		panic(err)
	}
}

// ComplexIrisFFT is the iris dataset processing using complex FFT
func ComplexIrisFFT(textMode bool, hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)

	}
	iris := datum.Fisher

	text := []byte("hello world!")

	width, size := 4, len(iris)
	if textMode {
		width, size = 256, len(text)
	}

	others := tc128.NewSet()
	others.Add("input", width, size)
	others.Add("output", width, size)

	if textMode {
		length := complex(float64(size), 0)
		for _, w := range others.Weights {
			w.X = w.X[:cap(w.X)]
			for i := 0; i < width; i++ {
				r := make([]complex128, size)
				for j, data := range text {
					if int(data) == i {
						r[j] = complex(1, 0)
					}
				}
				f := fft.FFT(r)
				for j, value := range f {
					w.X[j*width+i] = value / length
				}
			}
		}
	} else {
		max := 0.0
		for i := 0; i < width; i++ {
			r := make([]complex128, size)
			for j, data := range iris {
				r[j] = complex(data.Measures[i], 0)
			}
			f := fft.FFT(r)
			for _, value := range f {
				if cmplx.Abs(value) > max {
					max = cmplx.Abs(value)
				}
			}
		}

		for _, w := range others.Weights {
			w.X = w.X[:cap(w.X)]
			for i := 0; i < width; i++ {
				r := make([]complex128, size)
				for j, data := range iris {
					r[j] = complex(data.Measures[i], 0)
				}
				f := fft.FFT(r)
				for j, value := range f {
					w.X[j*width+i] = value / complex(max, 0)
				}
			}
		}
	}

	set := tc128.NewSet()
	set.Add("l1", width, hiddenSize)
	set.Add("b1", hiddenSize, size)
	set.Add("l2", hiddenSize, width)
	set.Add("b2", width, size)

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]complex128, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]complex128, len(p.X)))
	}

	quadratic := tc128.B(ComplexQuadratic)
	activation := tc128.U(ComplexSigmoid)

	l1 := activation(tc128.Add(set.Get("b1"), tc128.Mul(set.Get("l1"), others.Get("input"))))
	l2 := tc128.Add(set.Get("b2"), tc128.Mul(set.Get("l2"), l1))
	cost := quadratic(l2, others.Get("output"))

	alpha, eta, iterations := complex128(.01+.01i), complex128(.01+.01i), 8*2048
	if textMode {
		alpha, eta, iterations = complex128(.01+.01i), complex128(.01+.01i), 16*2048
	}
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0.0)
		set.Zero()
		others.Zero()

		/*if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
			for j := range d {
				d[j] /= 10
			}
		}

		index := 0
		for _, data := range iris {
			for i, measure := range data.Measures {
				if d[i] == 0 {
					inputs.X[index] = float32(measure)
				} else {
					inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
				}
				index++
			}
		}*/

		total += tc128.Gradient(cost).X[0]
		sum := complex128(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := cmplx.Abs(sum)
		scaling := complex(1.0, 0)
		if norm > 1 {
			scaling = 1 / complex(math.Sqrt(norm), 0)
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				//deltas[j][k] = alpha*deltas[j][k] - eta*d*complex(scaling, 0)
				//set.Weights[j].X[k] += deltas[j][k]
				_ = alpha
				set.Weights[j].X[k] -= eta * d * scaling
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		fmt.Println(i, total)
		if cmplx.Abs(total) < .001 {
			break
		}
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "complex_fft_cost.png")
	if err != nil {
		panic(err)
	}
}

// ComplexTransformerIrisFFT is the iris dataset transformer processing using complex FFT
func ComplexTransformerIrisFFT(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)

	}
	iris := datum.Fisher
	others := tc128.NewSet()
	others.Add("input", 4, len(iris))
	others.Add("output", 4, len(iris))

	max := 0.0
	for i := 0; i < 4; i++ {
		r := make([]complex128, len(iris))
		for j, data := range iris {
			r[j] = complex(data.Measures[i], 0)
		}
		f := fft.FFT(r)
		for _, value := range f {
			if cmplx.Abs(value) > max {
				max = cmplx.Abs(value)
			}
		}
	}

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
		for i := 0; i < 4; i++ {
			r := make([]complex128, len(iris))
			for j, data := range iris {
				r[j] = complex(data.Measures[i], 0)
			}
			f := fft.FFT(r)
			for j, value := range f {
				w.X[j*4+i] = value / complex(max, 0)
			}
		}
	}

	set := tc128.NewSet()
	set.Add("query", 4, hiddenSize)
	set.Add("key", 4, hiddenSize)
	set.Add("value", 4, hiddenSize)
	set.Add("project", hiddenSize, 4)

	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]complex128, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]complex128, len(p.X)))
	}

	quadratic := tc128.B(ComplexQuadratic)

	query := tc128.Mul(set.Get("query"), others.Get("input"))
	key := tc128.Mul(set.Get("key"), others.Get("input"))
	value := tc128.Mul(set.Get("value"), others.Get("input"))
	l1 := tc128.Mul(tc128.T(value), tc128.Mul(query, key))
	l2 := tc128.Mul(set.Get("project"), l1)
	cost := quadratic(l2, others.Get("output"))

	alpha, eta, iterations := complex128(.1+.1i), complex128(.1+.1i), 4*2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0.0)
		set.Zero()
		others.Zero()

		/*if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
			for j := range d {
				d[j] /= 10
			}
		}

		index := 0
		for _, data := range iris {
			for i, measure := range data.Measures {
				if d[i] == 0 {
					inputs.X[index] = float32(measure)
				} else {
					inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
				}
				index++
			}
		}*/

		total += tc128.Gradient(cost).X[0]
		sum := complex128(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := cmplx.Abs(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				//deltas[j][k] = alpha*deltas[j][k] - eta*d*complex(scaling, 0)
				//set.Weights[j].X[k] += deltas[j][k]
				_ = alpha
				set.Weights[j].X[k] -= eta * d * complex(scaling, 0)
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		fmt.Println(i, total)
		if cmplx.Abs(total) < .001 {
			break
		}
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "complex_transformer_fft_cost.png")
	if err != nil {
		panic(err)
	}
}
