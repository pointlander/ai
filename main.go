// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var (
	// FlagIris is a flag to run the Iris dataset
	FlagIris = flag.Bool("iris", false, "Iris mode")
	// FlagTranslate is a flag to run the translation for english to german
	FlagLearn = flag.Bool("learn", false, "learn to translate mode")
	// FlagGerman translate english to german
	FlagGerman = flag.String("german", "", "translate english to german")
	// FlagName is a flag to run the named weight set
	FlagName = flag.String("name", "set.w", "name of the weight set")
	// FlagComplex is a flag to run the complex mode
	FlagComplex = flag.Bool("complex", false, "Complex mode")
)

// Statistics captures statistics
type Statistics struct {
	Sum        float64
	SumSquared float64
	Count      int
}

// Add adds a statistic
func (s *Statistics) Add(value float64) {
	s.Sum += value
	s.SumSquared += value * value
	s.Count++
}

// StandardDeviation calculates the standard deviation
func (s Statistics) StandardDeviation() float64 {
	sum, count := s.Sum, float64(s.Count)
	return math.Sqrt((s.SumSquared - sum*sum/count) / count)
}

// Average calculates the average
func (s Statistics) Average() float64 {
	return s.Sum / float64(s.Count)
}

// String returns the statistics as a string`
func (s Statistics) String() string {
	return fmt.Sprintf("%f +- %f", s.Average(), s.StandardDeviation())
}

// ComplexStatistics captures statistics for complex numbers
type ComplexStatistics struct {
	Sum        complex128
	SumSquared complex128
	Count      int
}

// Add adds a statistic
func (s *ComplexStatistics) Add(value complex128) {
	s.Sum += value
	s.SumSquared += value * value
	s.Count++
}

// StandardDeviation calculates the standard deviation
func (s ComplexStatistics) StandardDeviation() complex128 {
	sum, count := s.Sum, complex(float64(s.Count), 0)
	return cmplx.Sqrt((s.SumSquared - sum*sum/count) / count)
}

// Average calculates the average
func (s ComplexStatistics) Average() complex128 {
	return s.Sum / complex(float64(s.Count), 0)
}

// String returns the statistics as a string`
func (s ComplexStatistics) String() string {
	return fmt.Sprintf("%f +- %f", s.Average(), s.StandardDeviation())
}

func main() {
	flag.Parse()

	if *FlagIris {
		if *FlagComplex {
			ComplexIris(32)
		} else {
			Iris(64)
		}
		return
	} else if *FlagLearn {
		LearnToTranslate(64, 2048)
		return
	} else if *FlagGerman != "" {
		TranslateToGerman(*FlagName, 64, []byte(*FlagGerman))
		return
	}
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
	transformer1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project1"),
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

// LearnToTranslate learns to translates english to german
func LearnToTranslate(size, hiddenSize int) {
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

	fmt.Println(maxEnglish, maxGerman)

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
	transformer1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("project1"),
		tf32.Hadamard(tf32.Sigmoid(query1),
			tf32.SumRows(tf32.Hadamard(tf32.T(tf32.Softmax(tf32.T(key1))), value1)))), set.Get("bias1")))

	cost := tf32.Sum(tf32.Quadratic(transformer1, others.Get("output")))

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
	set.Add("query", 4, hiddenSize)
	set.Add("key", 4, hiddenSize)
	set.Add("value", 4, hiddenSize)
	set.Add("project", hiddenSize, 4)
	set.Add("position", 4, len(iris))

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

	input := tc128.Add(set.Get("position"), others.Get("input"))
	query := tc128.Mul(set.Get("query"), input)
	key := tc128.Mul(set.Get("key"), input)
	value := tc128.Mul(set.Get("value"), input)
	transformer := tc128.Mul(set.Get("project"),
		tc128.Hadamard(tc128.Sigmoid(query),
			tc128.SumRows(tc128.Hadamard(tc128.Softmax(key), value))))
	cost := quadratic(transformer, others.Get("output"))

	alpha, eta, iterations := complex128(.1+.1i), complex128(.1+.1i), 16*2048
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
