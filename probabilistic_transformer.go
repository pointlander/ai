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
	"strings"
	"syscall"

	"github.com/pointlander/datum/mnist"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// ProbabilisticTransformer is a probabilistic transformer
func ProbabilisticTransformer(head int, hiddenSize int) {
	// 5329 10000 SelectedPositionEncoding
	// 5108 10000 PositionEncoding
	// 7092 10000 Normalization
	// 7099 10000 SelectedPositionEncoding
	// 5377 10000 SelectedPositionEncoding per image
	// 5825 10000 Embeddings
	rnd := rand.New(rand.NewSource(int64(head + 1)))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 49, 16
	selections := make([]Position, size)
	for i := range selections {
		selections[i].Positions = make([]int, width)
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
		dk.X[i] = 1 / float32(hiddenSize)
	}
	others.ByName["alpha"].X[0] = 0.01

	set := tf32.NewSet()
	set.Add("encode", width, hiddenSize)
	set.Add("biasEncode", hiddenSize, size)
	set.Add("n1_1", hiddenSize, 1)
	set.Add("bn1_1", hiddenSize, 1)
	set.Add("query", hiddenSize, hiddenSize)
	set.Add("key", hiddenSize, hiddenSize)
	set.Add("value", hiddenSize, hiddenSize)
	set.Add("n1_2", hiddenSize, 1)
	set.Add("bn1_2", hiddenSize, 1)
	set.Add("W1_1", hiddenSize, hiddenSize)
	set.Add("b1_1", hiddenSize, 1)
	set.Add("W1_2", hiddenSize, hiddenSize)
	set.Add("b1_2", hiddenSize, 1)
	set.Add("n2_1", hiddenSize, 1)
	set.Add("bn2_1", hiddenSize, 1)
	set.Add("query1", hiddenSize, hiddenSize)
	set.Add("key1", hiddenSize, hiddenSize)
	set.Add("value1", hiddenSize, hiddenSize)
	set.Add("n2_2", hiddenSize, 1)
	set.Add("bn2_2", hiddenSize, 1)
	set.Add("W2_1", hiddenSize, hiddenSize)
	set.Add("b2_1", hiddenSize, 1)
	set.Add("W2_2", hiddenSize, hiddenSize)
	set.Add("b2_2", hiddenSize, 1)
	set.Add("project", hiddenSize, 10)
	set.Add("bias", 10, 1)

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(hiddenSize*hiddenSize))
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
	relu := tf32.U(ReLu)
	mask := tf32.U(Mask)
	norm := tf32.U(Normalize)
	average := tf32.U(AverageRows)
	//encode := tf32.U(PositionEncodingLayer)

	input := tf32.Add(tf32.Mul(set.Get("encode"), others.Get("input")), set.Get("biasEncode"))
	norm_input := tf32.Add(tf32.Hadamard(norm(input), set.Get("n1_1")), set.Get("bn1_1"))
	query := tf32.Mul(set.Get("query"), norm_input)
	key := tf32.Mul(set.Get("key"), norm_input)
	value := tf32.Mul(set.Get("value"), norm_input)
	l1 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value))), input)
	norm_l1 := tf32.Add(tf32.Hadamard(norm(l1), set.Get("n1_2")), set.Get("bn1_2"))
	l2 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W1_2"), relu(tf32.Add(tf32.Mul(set.Get("W1_1"), norm_l1), set.Get("b1_1")))), set.Get("b1_2")), l1)
	norm_l2 := tf32.Add(tf32.Hadamard(norm(l2), set.Get("n2_1")), set.Get("bn2_1"))
	query1 := tf32.Mul(set.Get("query1"), norm_l2)
	key1 := tf32.Mul(set.Get("key1"), norm_l2)
	value1 := tf32.Mul(set.Get("value1"), norm_l2)
	l3 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query1, key1), others.Get("dk"))), tf32.T(value1))), l2)
	norm_l3 := tf32.Add(tf32.Hadamard(norm(l3), set.Get("n2_2")), set.Get("bn2_2"))
	l4 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W2_2"), relu(tf32.Add(tf32.Mul(set.Get("W2_1"), norm_l3), set.Get("b2_1")))), set.Get("b2_2")), l3)
	output := mask(average(tf32.Add(tf32.Mul(set.Get("project"), l4), set.Get("bias"))))

	/*regularization := tf32.Add(tf32.Sum(tf32.Abs(set.Get("query"))), tf32.Sum(tf32.Abs(set.Get("key"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("value"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("W1_1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("b1_1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("W1_2"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("b1_2"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("query1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("key1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("value1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("W2_1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("b2_1"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("W2_2"))))
	regularization = tf32.Add(regularization, tf32.Sum(tf32.Abs(set.Get("b2_2"))))
	regularization = tf32.Hadamard(regularization, others.Get("alpha"))
	cost := tf32.Add(tf32.Sum(tf32.Quadratic(l4, others.Get("output"))), regularization)*/
	cost := tf32.Sum(tf32.Quadratic(output, others.Get("output")))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.001), float32(.001), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	i := 0
	total := float32(0.0)
	for i < len(images.Train.Images) {
		index := rnd.Intn(len(images.Train.Images))
		image := images.Train.Images[index]

		for j := range inputs.X {
			inputs.X[j] = 0
		}
		for j := range outputs.X {
			outputs.X[j] = 0
		}

		//SelectPositions(rnd, images.Train.Width, images.Train.Height, selections)
		for j, set := range selections {
			for i, value := range set.Positions {
				inputs.X[j*width+i] =
					float32(image[value])
			}
		}
		outputs.X[int(images.Train.Labels[index])] = 1
		//SelectedPositionEncoding(selections, inputs)
		//PositionEncoding(inputs)
		/*statistics := make([]Statistics, width)
		for j := 0; j < len(inputs.X); j += width {
			for k := 0; k < width; k++ {
				statistics[k].Add(float64(inputs.X[j+k]))
			}
		}
		for j := 0; j < len(inputs.X); j += width {
			for k := 0; k < width; k++ {
				inputs.X[j+k] = float32((float64(inputs.X[j+k]) - statistics[k].Average()) / statistics[k].StandardDeviation())
			}
		}*/

		total += tf32.Gradient(cost).X[0]
		if i > 0 && i%100 == 0 {
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
					//deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
					//set.Weights[j].X[k] += deltas[j][k]
					_ = alpha
					set.Weights[j].X[k] -= eta * d * scaling
				}
			}

			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
			fmt.Println(head, i, total)
			set.Zero()
			others.Zero()
			total = 0
		}

		if halt || math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}
		if i%1000 == 0 {
			set.Save(fmt.Sprintf("%d_%d_set.w", head, i), total, i)
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_probabilistic_transformer_cost.png", head))
	if err != nil {
		panic(err)
	}

	set.Save(fmt.Sprintf("%d_set.w", head), 0, 0)
}

// InferenceProbabilisticTransformer is a probabilistic transformer inference
func InferenceProbabilisticTransformer(h, test int, name string, hiddenSize int) {
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 49, 16
	type Head struct {
		Head       tf32.Meta
		Inputs     *tf32.V
		Selections []Position
	}
	heads := make([]Head, h)
	for i := range heads {
		rnd := rand.New(rand.NewSource(int64(i + 1)))
		selections := make([]Position, size)
		for i := range selections {
			selections[i].Positions = make([]int, width)
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
		_, _, err = set.Open(fmt.Sprintf("%d_%s", i, name))
		if err != nil {
			panic(err)
		}

		//quadratic := tf32.B(Quadratic)
		softmax := tf32.U(Softmax)
		relu := tf32.U(ReLu)
		mask := tf32.U(Mask)
		norm := tf32.U(Normalize)
		average := tf32.U(AverageRows)
		//encode := tf32.U(PositionEncodingLayer)

		input := tf32.Add(tf32.Mul(set.Get("encode"), others.Get("input")), set.Get("biasEncode"))
		norm_input := tf32.Add(tf32.Hadamard(norm(input), set.Get("n1_1")), set.Get("bn1_1"))
		query := tf32.Mul(set.Get("query"), norm_input)
		key := tf32.Mul(set.Get("key"), norm_input)
		value := tf32.Mul(set.Get("value"), norm_input)
		l1 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query, key), others.Get("dk"))), tf32.T(value))), input)
		norm_l1 := tf32.Add(tf32.Hadamard(norm(l1), set.Get("n1_2")), set.Get("bn1_2"))
		l2 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W1_2"), relu(tf32.Add(tf32.Mul(set.Get("W1_1"), norm_l1), set.Get("b1_1")))), set.Get("b1_2")), l1)
		norm_l2 := tf32.Add(tf32.Hadamard(norm(l2), set.Get("n2_1")), set.Get("bn2_1"))
		query1 := tf32.Mul(set.Get("query1"), norm_l2)
		key1 := tf32.Mul(set.Get("key1"), norm_l2)
		value1 := tf32.Mul(set.Get("value1"), norm_l2)
		l3 := tf32.Add(tf32.T(tf32.Mul(softmax(tf32.Hadamard(tf32.Mul(query1, key1), others.Get("dk"))), tf32.T(value1))), l2)
		norm_l3 := tf32.Add(tf32.Hadamard(norm(l3), set.Get("n2_2")), set.Get("bn2_2"))
		l4 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W2_2"), relu(tf32.Add(tf32.Mul(set.Get("W2_1"), norm_l3), set.Get("b2_1")))), set.Get("b2_2")), l3)
		output := mask(average(tf32.Add(tf32.Mul(set.Get("project"), l4), set.Get("bias"))))
		heads[i] = Head{
			Head:       output,
			Inputs:     inputs,
			Selections: selections,
		}
	}

	type Result struct {
		Probability float32
		Index       int
	}

	process := func(head Head, test int) []Result {
		histogram := make([]Result, 10)
		image := images.Test.Images[test]

		//for i := 0; i < 32; i++ {
		//SelectPositions(rnd, images.Train.Width, images.Train.Height, selections)
		for j := range head.Inputs.X {
			head.Inputs.X[j] = 0
		}
		for j, set := range head.Selections {
			for i, value := range set.Positions {
				head.Inputs.X[j*width+i] = float32(image[value])
			}
		}
		//SelectedPositionEncoding(selections, inputs)
		//PositionEncoding(inputs)
		/*statistics := make([]Statistics, width)
		for j := 0; j < len(inputs.X); j += width {
			for k := 0; k < width; k++ {
				statistics[k].Add(float64(inputs.X[j+k]))
			}
		}
		for j := 0; j < len(inputs.X); j += width {
			for k := 0; k < width; k++ {
				inputs.X[j+k] = float32((float64(inputs.X[j+k]) - statistics[k].Average()) / statistics[k].StandardDeviation())
			}
		}*/

		head.Head(func(a *tf32.V) bool {
			for j := 0; j < 10; j++ {
				histogram[j].Probability += a.X[j]
				histogram[j].Index = j
			}
			return true
		})
		//}
		return histogram
	}

	if test == -1 {
		correct := 0
		for i := range images.Test.Images {
			var votes [10]int
			for _, head := range heads {
				histogram := process(head, i)
				sort.Slice(histogram, func(i, j int) bool {
					return histogram[i].Probability > histogram[j].Probability
				})
				votes[histogram[0].Index]++
			}
			max, x := 0, 0
			for key, value := range votes {
				if value > max {
					max = value
					x = key
				}
			}
			if x == int(images.Test.Labels[i]) {
				correct++
			}
		}
		fmt.Println(correct, len(images.Test.Images))
	} else {
		for _, head := range heads {
			fmt.Println(images.Test.Labels[test])
			histogram := process(head, test)
			sort.Slice(histogram, func(i, j int) bool {
				return histogram[i].Probability > histogram[j].Probability
			})
			fmt.Println(histogram)
		}
	}
}
