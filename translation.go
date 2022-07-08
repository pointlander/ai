// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// TranslateToGerman translates english to german
func TranslateToGerman(name string, size int, english []byte) {
	others := tf32.NewSet()
	others.Add("input", 256, size)
	input := others.Weights[0]
	input.X = input.X[:cap(input.X)]

	set := tf32.NewSet()
	_, _, err := set.Open(name)
	if err != nil {
		panic(err)
	}

	in := tf32.Sigmoid(tf32.Add(set.Get("position"), tf32.Mul(set.Get("embed"), others.Get("input"))))
	query := tf32.Mul(set.Get("query"), in)
	key := tf32.Mul(set.Get("key"), in)
	value := tf32.Mul(set.Get("value"), in)
	transformer := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"),
		tf32.Hadamard(tf32.Sigmoid(query),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key))), value)))), set.Get("bias")))

	query1 := tf32.Mul(set.Get("query1"), transformer)
	key1 := tf32.Mul(set.Get("key1"), transformer)
	value1 := tf32.Mul(set.Get("value1"), transformer)
	transformer1 := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("project1"),
		tf32.Hadamard(tf32.Sigmoid(query1),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key1))), value1)))), set.Get("bias1")))
	for j := range input.X {
		input.X[j] = 0
	}
	j := 0
	for _, value := range english {
		input.X[256*j+int(value)] = 1
		j++
	}
	//PositionEncoding(input)

	transformer1(func(a *tf32.V) bool {
		output := make([]byte, 0, size)
		for i := 0; i < size; i++ {
			max, symbol := float32(0.0), 0
			for j := 0; j < 256; j++ {
				if s := a.X[256*i+j]; s > max {
					max, symbol = s, j
				}
			}
			fmt.Println(max, symbol)
			output = append(output, byte(symbol))
		}
		fmt.Println(string(output))
		return true
	})
}

// TrainingData is the english and german training data
type TrainingData struct {
	English    [][]byte
	MaxEnglish int
	German     [][]byte
	MaxGerman  int
}

// LoadTrainingData loads the training data
func LoadTrainingData(size int) TrainingData {
	englishIn, err := os.Open("europarl-v7.de-en.en")
	if err != nil {
		panic(err)
	}
	defer englishIn.Close()
	englishReader := bufio.NewReader(englishIn)
	english, maxEnglish := make([][]byte, 0, 8), 0
	for {
		line, err := englishReader.ReadString('\n')
		if err != nil {
			break
		}
		data := []byte(strings.TrimSpace(line))
		if length := len(data); length > maxEnglish {
			maxEnglish = length
		}
		if len(data) > size {
			data = data[:size]
		}
		english = append(english, data)
	}

	germanIn, err := os.Open("europarl-v7.de-en.de")
	if err != nil {
		panic(err)
	}
	defer germanIn.Close()
	germanReader := bufio.NewReader(germanIn)
	german, maxGerman := make([][]byte, 0, 8), 0
	for {
		line, err := germanReader.ReadString('\n')
		if err != nil {
			break
		}
		data := []byte(strings.TrimSpace(line))
		if length := len(data); length > maxGerman {
			maxGerman = length
		}
		if len(data) > size {
			data = data[:size]
		}
		german = append(german, data)
	}

	if len(english) != len(german) {
		panic("unequal length")
	}

	return TrainingData{
		English:    english,
		MaxEnglish: maxEnglish,
		German:     german,
		MaxGerman:  maxGerman,
	}
}

// LearnToTranslate learns to translates english to german
func LearnToTranslate(size, hiddenSize int) {
	data := LoadTrainingData(size)
	english, german := data.English, data.German

	rnd := rand.New(rand.NewSource(1))

	others := tf32.NewSet()
	others.Add("input", 256, size)
	others.Add("output", 256, size)
	input, output := others.Weights[0], others.Weights[1]
	input.X = input.X[:cap(input.X)]
	output.X = output.X[:cap(output.X)]

	set := tf32.NewSet()
	set.Add("embed", 256, hiddenSize)
	set.Add("position", hiddenSize, size)
	set.Add("query", hiddenSize, hiddenSize)
	set.Add("key", hiddenSize, hiddenSize)
	set.Add("value", hiddenSize, hiddenSize)
	set.Add("project", hiddenSize, hiddenSize)
	set.Add("bias", hiddenSize, size)
	set.Add("query1", hiddenSize, hiddenSize)
	set.Add("key1", hiddenSize, hiddenSize)
	set.Add("value1", hiddenSize, hiddenSize)
	set.Add("project1", hiddenSize, 256)
	set.Add("bias1", 256, size)

	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	/*deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}*/

	in := tf32.Sigmoid(tf32.Add(set.Get("position"), tf32.Mul(set.Get("embed"), others.Get("input"))))
	query := tf32.Mul(set.Get("query"), in)
	key := tf32.Mul(set.Get("key"), in)
	value := tf32.Mul(set.Get("value"), in)
	transformer := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project"),
		tf32.Hadamard(tf32.Sigmoid(query),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key))), value)))), set.Get("bias")))

	query1 := tf32.Mul(set.Get("query1"), transformer)
	key1 := tf32.Mul(set.Get("key1"), transformer)
	value1 := tf32.Mul(set.Get("value1"), transformer)
	transformer1 := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("project1"),
		tf32.Hadamard(tf32.Sigmoid(query1),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key1))), value1)))), set.Get("bias1")))

	cost := tf32.Sum(tf32.CrossEntropy(transformer1, others.Get("output")))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.01), float32(.01), 2048
	points := make(plotter.XYs, 0, iterations)
	{
		in := []byte("hello world!")
		out := in
		for j := range input.X {
			input.X[j] = 0
		}
		for j := range output.X {
			output.X[j] = 0
		}
		j := 0
		for _, value := range in {
			input.X[256*j+int(value)] = 1
			j++
		}
		j = 0
		for _, value := range out {
			output.X[256*j+int(value)] = 1
			j++
		}
	}
	for i := 0; i < iterations; i++ {
		_, _ = english, german
		/*for i, in := range english {
		out := german[i]
		for j := range input.X {
			input.X[j] = 0
		}
		for j := range output.X {
			output.X[j] = 0
		}
		j := 0
		for _, value := range in {
			input.X[256*j+int(value)] = 1
			j++
		}
		j = 0
		for _, value := range out {
			output.X[256*j+int(value)] = 1
			j++
		}*/
		//PositionEncoding(input)

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
				/*deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]*/
				_ = eta
				set.Weights[j].X[k] -= alpha * d * scaling
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		/*if total < .1 {
			break
		}*/
		if halt {
			break
		}
		if i%1000 == 0 {
			set.Save(fmt.Sprintf("%d_set.w", i), total, i)
		}
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "translate_cost.png")
	if err != nil {
		panic(err)
	}

	set.Save("set.w", 0, 0)
}
