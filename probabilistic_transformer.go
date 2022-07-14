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
	"sort"
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
	width, size := hiddenSize, 256
	selections := make([][]int, size)
	for i := range selections {
		selections[i] = make([]int, width)
	}
	SelectPositions(rnd, images.Train.Width, images.Train.Height, selections)

	others := tf32.NewSet()
	others.Add("input", width, size)
	others.Add("output", 10, 1)
	others.Add("dk", size, 1)
	others.Add("alpha", 1, 1)

	for _, w := range others.Weights {
		w.X = w.X[:cap(w.X)]
	}
	inputs, outputs, dk := others.ByName["input"], others.ByName["output"], others.ByName["dk"]
	for i := range dk.X {
		dk.X[i] = 1 / float32(size)
	}
	others.ByName["alpha"].X[0] = 0.1

	set := tf32.NewSet()
	set.Add("query", width, hiddenSize)
	set.Add("key", width, hiddenSize)
	set.Add("value", width, width)
	set.Add("project", width, width)
	set.Add("bias", width, 1)
	set.Add("query1", width, hiddenSize)
	set.Add("key1", width, hiddenSize)
	set.Add("value1", width, width)
	set.Add("project1", width, 10)
	set.Add("bias1", 10, 1)

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

	input := others.Get("input")
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value))), input)
	l2 := tf32.Add(tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias"))), l1)
	query1 := tf32.Mul(set.Get("query1"), l2)
	key1 := tf32.Mul(set.Get("key1"), l2)
	value1 := tf32.Mul(set.Get("value1"), l2)
	l3 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query1, key1), others.Get("dk"))), tf32.T(value1))), l2)
	l4 := mask(tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project1"), l3), set.Get("bias1"))))

	regularization := tf32.Add(tf32.Sum(tf32.Abs(set.Get("query"))), tf32.Sum(tf32.Abs(set.Get("key"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("value"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("project"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("bias"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("query1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("key1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("value1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("project1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("bias1"))))
	regularization = tf32.Hadamard(regularization, others.Get("alpha"))
	cost := tf32.Add(tf32.Sum(tf32.Quadratic(l4, others.Get("output"))), regularization)

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.00001), float32(.00001), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < 5*len(images.Train.Images) {
		index := rnd.Intn(len(images.Train.Images))
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

		for j, set := range selections {
			for i, value := range set {
				inputs.X[j*width+i] =
					float32(image[value])
			}
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
	width, size := hiddenSize, 256
	selections := make([][]int, size)
	for i := range selections {
		selections[i] = make([]int, width)
	}
	SelectPositions(rnd, images.Train.Width, images.Train.Height, selections)
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

	input := others.Get("input")
	query := tf32.Mul(set.Get("query"), input)
	key := tf32.Mul(set.Get("key"), input)
	value := tf32.Mul(set.Get("value"), input)
	l1 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value))), input)
	l2 := tf32.Add(tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"), l1), set.Get("bias"))), l1)
	query1 := tf32.Mul(set.Get("query1"), l2)
	key1 := tf32.Mul(set.Get("key1"), l2)
	value1 := tf32.Mul(set.Get("value1"), l2)
	l3 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query1, key1), others.Get("dk"))), tf32.T(value1))), l2)
	l4 := mask(tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project1"), l3), set.Get("bias1"))))

	image := images.Test.Images[test]
	fmt.Println(images.Test.Labels[test])

	type Result struct {
		Probability float32
		Index       int
	}
	histogram := make([]Result, 10)
	for j := range inputs.X {
		inputs.X[j] = 0
	}

	for j, set := range selections {
		for i, value := range set {
			inputs.X[j*width+i] = float32(image[value])
		}
	}
	PositionEncoding(inputs)

	l4(func(a *tf32.V) bool {
		for j := 0; j < 10; j++ {
			histogram[j].Probability += a.X[j]
			histogram[j].Index = j
		}
		return true
	})
	sort.Slice(histogram, func(i, j int) bool {
		return histogram[i].Probability > histogram[j].Probability
	})
	fmt.Println(histogram)
}
