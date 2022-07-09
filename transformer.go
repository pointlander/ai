// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Transformer is a transformer
func Transformer(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	text := []byte("hello world!")
	width, size := 256, len(text)
	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", width, size)

	for _, w := range others.Weights {
		for _, symbol := range text {
			data := make([]float32, width)
			data[symbol] = 1
			w.X = append(w.X, data...)
		}
	}

	set := tf32.NewSet()
	set.Add("encoding", width, size)
	set.Add("query", width, hiddenSize)
	set.Add("key", width, hiddenSize)
	set.Add("value", width, hiddenSize)
	set.Add("project", hiddenSize, width)

	for _, w := range set.Weights {
		if w.N == "encoding" {
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

	input := tf32.Add(others.Get("input"), set.Get("encoding"))
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(tf32.Softmax(tf32.Mul(query, key)), tf32.T(value)))
	l2 := tf32.Mul(set.Get("project"), l1)
	cost := quadratic(l2, others.Get("output"))

	alpha, eta, iterations := float32(.1), float32(.1), 4*2048
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
		norm := math.Sqrt(float64(sum))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / float32(norm)
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				//deltas[j][k] = alpha*deltas[j][k] - eta*d*complex(scaling, 0)
				//set.Weights[j].X[k] += deltas[j][k]
				_ = alpha
				set.Weights[j].X[k] -= eta * d * scaling
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		if total < .001 {
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "transformer_cost.png")
	if err != nil {
		panic(err)
	}

	l2(func(a *tf32.V) bool {
		output := ""
		for i := 0; i < size; i++ {
			max, index := float32(0.0), 0
			for j := 0; j < width; j++ {
				if a.X[i*width+j] > max {
					max = a.X[i*width+j]
					index = j
				}
			}
			output += string(rune(index))
		}
		fmt.Println(output)
		return true
	})

}
