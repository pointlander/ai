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
	selections := make([]int, size)
	symbols := images.Train.Width * images.Train.Height
	SelectPositions(rnd, symbols, selections)

	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", 10, 1)
	others.Add("dk", size, 1)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs, outputs, dk := others.ByName["input"], others.ByName["output"], others.ByName["dk"]
	for i := range dk.X {
		dk.X[i] = 1 / float32(size)
	}

	set := tf32.NewSet()
	set.Add("embedding", width, hiddenSize)
	set.Add("bias1", hiddenSize, 1)
	set.Add("query", hiddenSize, hiddenSize)
	set.Add("key", hiddenSize, hiddenSize)
	set.Add("value", hiddenSize, hiddenSize)
	set.Add("project", hiddenSize, 10)
	set.Add("bias", 10, 1)

	for _, w := range set.Weights {
		if w.N == "bias" || w.N == "bias1" {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(size*size))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	//quadratic := tf32.B(Quadratic)
	softmax := tf32.U(Softmax)
	mask := tf32.U(Mask)

	input := tf32.Add(tf32.Mul(set.Get("embedding"), others.Get("input")), set.Get("bias1"))
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value)))
	l2 := mask(tf32.Softmax(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias"))))

	cost := tf32.Sum(tf32.CrossEntropy(l2, others.Get("output")))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.05), float32(.05), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	encodings := make([]float32, 2*symbols)
	for i := range encodings {
		encodings[i] = float32(math.Abs(rnd.NormFloat64()))
	}
	i := 0
	for i < 5*len(images.Train.Images) {
		index := rand.Intn(len(images.Train.Images))
		image := images.Train.Images[index]
		total := float32(0.0)
		set.Zero()
		others.Zero()
		for j := range inputs.X {
			inputs.X[j] = 0
		}
		for j := range outputs.X {
			outputs.X[j] = 0
		}

		for j, selection := range selections {
			inputs.X[j] = float32(image[selection])
		}
		outputs.X[int(images.Train.Labels[index])] = 1
		PositionEncoding(inputs)

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
		if halt || math.IsNaN(float64(total)) {
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
	width, size := 1, 256
	selections := make([]int, size)
	symbols := images.Train.Width * images.Train.Height
	SelectPositions(rnd, symbols, selections)
	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("dk", size, 1)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs, dk := others.ByName["input"], others.ByName["dk"]
	for i := range dk.X {
		dk.X[i] = 1 / float32(size)
	}

	set := tf32.NewSet()
	_, _, err = set.Open(name)
	if err != nil {
		panic(err)
	}

	//quadratic := tf32.B(Quadratic)
	softmax := tf32.U(Softmax)
	mask := tf32.U(Mask)

	input := tf32.Add(tf32.Mul(set.Get("embedding"), others.Get("input")), set.Get("bias1"))
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value)))
	l2 := mask(tf32.Softmax(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias"))))

	image := images.Test.Images[test]
	fmt.Println(images.Test.Labels[test])

	histogram := make([]float32, 10)
	for j := range inputs.X {
		inputs.X[j] = 0
	}

	for j, selection := range selections {
		inputs.X[j] = float32(image[selection])
	}
	PositionEncoding(inputs)

	l2(func(a *tf32.V) bool {
		for j := 0; j < 10; j++ {
			histogram[j] += a.X[j]
		}
		return true
	})
	fmt.Println(histogram)
}
