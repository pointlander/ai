// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"syscall"

	"github.com/pointlander/datum/mnist"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// ProbabilisticTransformer is a probabilistic transformer
func ProbabilisticTransformer(hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 1, 256
	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", 10, size)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs, outputs := others.ByName["input"], others.ByName["output"]

	set := tf32.NewSet()
	set.Add("query", width, hiddenSize)
	set.Add("key", width, hiddenSize)
	set.Add("value", width, hiddenSize)
	set.Add("project", hiddenSize, 10)

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

	input := others.Get("input")
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(tf32.Softmax(tf32.Mul(query, key)), tf32.T(value)))
	l2 := tf32.Mul(set.Get("project"), l1)
	cost := quadratic(l2, others.Get("output"))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.0001), float32(.0001), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	selections := make([]int, size)
	symbols := images.Train.Width * images.Train.Height
	for i, image := range images.Train.Images {
		total := float32(0.0)
		set.Zero()
		others.Zero()
		for j := range inputs.X {
			inputs.X[j] = 0
		}
		for j := range outputs.X {
			outputs.X[j] = 0
		}

		SelectPositions(rnd, symbols, selections)
		for j, selection := range selections {
			inputs.X[j] = float32(image[selection])
			outputs.X[j*10+int(images.Train.Labels[selection])] = 1
		}
		SelectedPositionEncoding(selections, inputs)

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
		if halt {
			break
		}
		if i%1000 == 0 {
			set.Save(fmt.Sprintf("%d_set.w", i), total, i)
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "probabilistic_transformer_cost.png")
	if err != nil {
		panic(err)
	}

	set.Save("set.w", 0, 0)
}
