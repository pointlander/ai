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
	rnd := rand.New(rand.NewSource(2))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 2, 256
	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", 10, size)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs, outputs := others.ByName["input"], others.ByName["output"]

	set := tf32.NewSet()
	set.Add("positions", width, size)
	set.Add("query", width, hiddenSize)
	set.Add("key", width, hiddenSize)
	set.Add("value", width, hiddenSize)
	set.Add("project", hiddenSize, 10)
	set.Add("bias", 10, 1)

	for _, w := range set.Weights {
		if w.N == "bias" || w.N == "positions" {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}
	positions := set.ByName["positions"]

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	quadratic := tf32.B(Quadratic)
	softmax := tf32.U(Softmax)

	input := tf32.Add(others.Get("input"), set.Get("positions"))
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(softmax(tf32.Mul(query, key)), tf32.T(value)))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias")))

	cost := quadratic(l2, others.Get("output"))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.001), float32(.001), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	selections := make([]int, size)
	symbols := images.Train.Width * images.Train.Height
	encodings := make([]float32, 2*symbols)
	for i := range encodings {
		encodings[i] = float32(math.Abs(rnd.NormFloat64()))
	}
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

		if i > 0 {
			index := 0
			for _, selection := range selections {
				encodings[2*selection] = positions.X[index]
				index++
				encodings[2*selection+1] = positions.X[index]
				index++
			}
		}
		SelectPositions(rnd, symbols, selections)
		index, index1 := 0, 0
		for j, selection := range selections {
			inputs.X[index] = float32(image[selection])
			index++
			inputs.X[index] = float32(image[(selection+1)%symbols])
			index++
			outputs.X[j*10+int(images.Train.Labels[i])] = 1
			positions.X[index1] = encodings[2*selection]
			index1++
			positions.X[index1] = encodings[2*selection+1]
			index1++
		}

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

// InferenceProbabilisticTransformer is a probabilistic transformer inference
func InferenceProbabilisticTransformer(test int, name string, hiddenSize int) {
	rnd := rand.New(rand.NewSource(1))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 2, 256
	others := tf32.NewSet()
	others.Add("input", width, size)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs := others.ByName["input"]

	set := tf32.NewSet()
	_, _, err = set.Open(name)
	if err != nil {
		panic(err)
	}

	input := others.Get("input")
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(tf32.Softmax(tf32.Mul(query, key)), tf32.T(value)))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias")))

	image := images.Test.Images[test]
	fmt.Println(images.Test.Labels[test])
	selections := make([]int, size)
	symbols := images.Train.Width * images.Train.Height
	histogram := make([]float32, 10)
	for shot := 0; shot < 10; shot++ {
		for j := range inputs.X {
			inputs.X[j] = 0
		}
		SelectPositions(rnd, symbols, selections)
		index := 0
		for _, selection := range selections {
			inputs.X[index] = float32(image[selection])
			index++
			inputs.X[index] = float32(image[(selection+1)%symbols])
			index++
		}
		SelectedPositionEncoding(selections, inputs)

		l2(func(a *tf32.V) bool {

			for i := 0; i < size; i += 20 {
				index := 0
				for j := 0; j < 20; j++ {
					if j&1 == 1 {
						histogram[index] += a.X[i+j]
						index++
					}
				}
			}
			return true
		})
	}
	fmt.Println(histogram)
}
