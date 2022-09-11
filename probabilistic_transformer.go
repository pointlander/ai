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
	"time"

	"github.com/pointlander/datum/mnist"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// BatchSize is the size of a batch
	BatchSize = 100
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// Heads is the number of heads
	Heads = 8
	// Layers is the number of layers
	Layers = 2
)

type (
	// Attention is an attention function
	Attention func(f *Functions, query, key, value, dk tf32.Meta) tf32.Meta
	// HeadType is the type of head
	HeadType int
	// Configuration is the configuation for the model
	Configuration struct {
		HeadType
		Head       int
		HiddenSize int
		Attention  Attention
		Swap       bool
	}
	// Adam is an adam state
	Adam struct {
		M float32
		V float32
	}
)

const (
	// HeadTypeRegular is the regular head type
	HeadTypeRegular HeadType = iota
	// HeadTypeReZero is the ReZero head type
	HeadTypeReZero
)

// RegularHead implements the regular head
func (t Configuration) RegularHead(f *Functions, l, h int, input tf32.Meta, set, others tf32.Set) tf32.Meta {
	norm_input := f.FAdd(f.FHadamard(f.FNorm(input), set.Get(fmt.Sprintf("n%d_%d_1", l, h))), set.Get(fmt.Sprintf("bn%d_%d_1", l, h)))
	query := f.FMul(set.Get(fmt.Sprintf("query%d_%d", l, h)), norm_input)
	key := f.FMul(set.Get(fmt.Sprintf("key%d_%d", l, h)), norm_input)
	value := f.FMul(set.Get(fmt.Sprintf("value%d_%d", l, h)), norm_input)
	if t.Swap {
		value = f.FMul(norm_input, set.Get(fmt.Sprintf("value%d_%d", l, h)))
	}
	l1 := f.FAdd(f.FDropout(t.Attention(f, query, key, value, others.Get("dk"))), input)
	norm_l1 := f.FAdd(f.FHadamard(f.FNorm(l1), set.Get(fmt.Sprintf("n%d_%d_2", l, h))), set.Get(fmt.Sprintf("bn%d_%d_2", l, h)))
	return f.FAdd(f.FDropout(f.FAdd(f.FMul(set.Get(fmt.Sprintf("W%d_%d_2", l, h)),
		f.FRelu(f.FAdd(f.FMul(set.Get(fmt.Sprintf("W%d_%d_1", l, h)), norm_l1), set.Get(fmt.Sprintf("b%d_%d_1", l, h))))),
		set.Get(fmt.Sprintf("b%d_%d_2", l, h)))), l1)
}

// ReZeroHead implements the ReZero head
func (t Configuration) ReZeroHead(f *Functions, l, h int, input tf32.Meta, set, others tf32.Set) tf32.Meta {
	query := f.FMul(set.Get(fmt.Sprintf("query%d_%d", l, h)), input)
	key := f.FMul(set.Get(fmt.Sprintf("key%d_%d", l, h)), input)
	value := f.FMul(set.Get(fmt.Sprintf("value%d_%d", l, h)), input)
	if t.Swap {
		value = f.FMul(input, set.Get(fmt.Sprintf("value%d_%d", l, h)))
	}
	l1 := f.FAdd(f.FHadamard0(f.FDropout(t.Attention(f, query, key, value, others.Get("dk"))), set.Get(fmt.Sprintf("a%d_%d_1", l, h))), input)
	return f.FAdd(f.FHadamard0(f.FDropout(f.FAdd(f.FMul(set.Get(fmt.Sprintf("W%d_%d_2", l, h)),
		f.FRelu(f.FAdd(f.FMul(set.Get(fmt.Sprintf("W%d_%d_1", l, h)), l1), set.Get(fmt.Sprintf("b%d_%d_1", l, h))))),
		set.Get(fmt.Sprintf("b%d_%d_2", l, h)))), set.Get(fmt.Sprintf("a%d_%d_2", l, h))), l1)
}

func (t Configuration) Circuit(f *Functions, set, others tf32.Set) tf32.Meta {
	next := f.FAdd(f.FConcat(set.Get("position"), f.FMul(set.Get("encode"), others.Get("input"))), set.Get("biasEncode"))
	var heads [Layers][Heads]tf32.Meta
	for l := range heads {
		for h := range heads[l] {
			if t.HeadType == HeadTypeRegular {
				heads[l][h] = t.RegularHead(f, l, h, next, set, others)
			} else if t.HeadType == HeadTypeReZero {
				heads[l][h] = t.ReZeroHead(f, l, h, next, set, others)
			} else {
				panic(fmt.Errorf("%d invalid head type", t.HeadType))
			}
		}
		cat := f.FConcat(f.FAverageRows(heads[l][0]), f.FAverageRows(heads[l][1]))
		for i := 2; i < Heads; i++ {
			cat = f.FConcat(cat, f.FAverageRows(heads[l][i]))
		}
		if l < Layers-1 {
			cat = f.FConcat(heads[l][0], heads[l][1])
			for i := 2; i < Heads; i++ {
				cat = f.FConcat(cat, heads[l][i])
			}
		}
		next = f.FAdd(f.FMul(set.Get(fmt.Sprintf("project%d", l)), cat), set.Get(fmt.Sprintf("bias%d", l)))
	}
	return next
}

// ProbabilisticTransformer is a probabilistic transformer
func ProbabilisticTransformer(head int, hiddenSize int, attention Attention) {
	// 5329 10000 SelectedPositionEncoding
	// 5108 10000 PositionEncoding
	// 7092 10000 Normalization
	// 7099 10000 SelectedPositionEncoding
	// 5377 10000 SelectedPositionEncoding per image
	// 5825 10000 Embeddings
	// 8001 10000 SimpleAttention
	// 8001 10000 RegularAttention
	// 8001 10000 IdentityAttention
	// 2003 10000 SimpleAttention without sort
	// 2946 10000 RegularAttention with sort
	// 1957 10000 RegularAttention without sort
	rnd := rand.New(rand.NewSource(int64(head + 1)))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 49, 16+1
	selections := make([]Position, size-1)
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
	set.Add("position", 8, size)
	hiddenSize += 8
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
	set.Add("project1", 10, 10)
	set.Add("bias1", 10, 1)

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		}
		factor := math.Sqrt(2.0 / float64(hiddenSize))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	f := CreateFunctions(false)
	//quadratic := tf32.B(Quadratic)
	relu := f.FRelu
	mask := tf32.U(Mask)
	norm := f.FNorm
	//average := tf32.U(AverageRows)
	//encode := tf32.U(PositionEncodingLayer)
	concat := f.FConcat

	input := tf32.Add(concat(set.Get("position"), tf32.Mul(set.Get("encode"), others.Get("input"))), set.Get("biasEncode"))
	norm_input := tf32.Add(tf32.Hadamard(norm(input), set.Get("n1_1")), set.Get("bn1_1"))
	query := tf32.Mul(set.Get("query"), norm_input)
	key := tf32.Mul(set.Get("key"), norm_input)
	value := tf32.Mul(set.Get("value"), norm_input)
	l1 := tf32.Add(attention(nil, query, key, value, others.Get("dk")), input)
	norm_l1 := tf32.Add(tf32.Hadamard(norm(l1), set.Get("n1_2")), set.Get("bn1_2"))
	l2 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W1_2"), relu(tf32.Add(tf32.Mul(set.Get("W1_1"), norm_l1), set.Get("b1_1")))), set.Get("b1_2")), l1)
	norm_l2 := tf32.Add(tf32.Hadamard(norm(l2), set.Get("n2_1")), set.Get("bn2_1"))
	query1 := tf32.Mul(set.Get("query1"), norm_l2)
	key1 := tf32.Mul(set.Get("key1"), norm_l2)
	value1 := tf32.Mul(set.Get("value1"), norm_l2)
	l3 := tf32.Add(attention(nil, query1, key1, value1, others.Get("dk")), l2)
	norm_l3 := tf32.Add(tf32.Hadamard(norm(l3), set.Get("n2_2")), set.Get("bn2_2"))
	l4 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W2_2"), relu(tf32.Add(tf32.Mul(set.Get("W2_1"), norm_l3), set.Get("b2_1")))), set.Get("b2_2")), l3)
	output := tf32.Add(tf32.Mul(set.Get("project1"), relu(tf32.Add(tf32.Mul(set.Get("project"), mask(l4)), set.Get("bias")))), set.Get("bias1"))

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
	cost := tf32.Quadratic(output, others.Get("output"))

	c, halt := make(chan os.Signal), false
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		halt = true
	}()

	alpha, eta, iterations := float32(.01), float32(.01), len(images.Train.Images)
	points := make(plotter.XYs, 0, iterations)
	i := 0
	total := float32(0.0)
	reduced := false
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
		for j := 0; j < width; j++ {
			inputs.X[j] = 1
		}
		for j, set := range selections {
			for i, value := range set.Positions {
				inputs.X[(j+1)*width+i] =
					float32(image[value]) / 255
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

		f.Clear()
		total += tf32.Gradient(cost).X[0]
		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] += d
			}
		}
		set.Zero()

		if i > 0 && i%100 == 0 {
			sum := float32(0.0)
			/*for _, p := range set.Weights {
				for _, d := range p.D {
					sum += d * d
				}
			}*/
			for _, p := range deltas {
				for _, d := range p {
					d /= 100
					sum += d * d
				}
			}
			norm := math.Sqrt(float64(sum))
			scaling := float32(1.0)
			if norm > 1 {
				scaling = 1 / float32(norm)
			}

			for j, w := range set.Weights {
				for k := range w.D {
					//deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
					//set.Weights[j].X[k] += deltas[j][k]
					_ = alpha
					set.Weights[j].X[k] -= eta * deltas[j][k] / 100 * scaling
					deltas[j][k] = 0
				}
			}
			total /= 100
			if !reduced && total < .5 {
				reduced = true
				eta *= .1
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
			fmt.Println(head, i, total)
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

// ProbabilisticTransformerParallel is a probabilistic transformer
func (t Configuration) ProbabilisticTransformerParallel() {
	// 6783 10000 Removed short circuit
	// 7624 10000 Simple
	// 3796 10000 Regular
	// 958 10000 Identity
	// 7144 10000 Simple row based softmax
	// 8872 10000 Simple with 64 float wide
	// 8919 10000 Simple with 8 heads
	// 9520 10000 Simple with 8 heads and 2 layers
	rnd := rand.New(rand.NewSource(int64(t.Head + 1)))
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 16, 49+1
	selections := make([]Position, size-1)
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
		dk.X[i] = 1 / float32(math.Sqrt(float64(size)))
	}
	others.ByName["alpha"].X[0] = 0.01

	set := tf32.NewSet()
	set.Add("encode", width, t.HiddenSize)
	set.Add("position", t.HiddenSize, size)
	t.HiddenSize *= 2
	set.Add("biasEncode", t.HiddenSize, size)

	for l := 0; l < Layers; l++ {
		for h := 0; h < Heads; h++ {
			set.Add(fmt.Sprintf("a%d_%d_1", l, h), 1, 1)
			set.Add(fmt.Sprintf("a%d_%d_2", l, h), 1, 1)
			if t.HeadType == HeadTypeRegular {
				set.Add(fmt.Sprintf("n%d_%d_1", l, h), t.HiddenSize, 1)
				set.Add(fmt.Sprintf("bn%d_%d_1", l, h), t.HiddenSize, 1)
			}
			set.Add(fmt.Sprintf("query%d_%d", l, h), t.HiddenSize, t.HiddenSize)
			set.Add(fmt.Sprintf("key%d_%d", l, h), t.HiddenSize, t.HiddenSize)
			set.Add(fmt.Sprintf("value%d_%d", l, h), t.HiddenSize, t.HiddenSize)
			if t.HeadType == HeadTypeRegular {
				set.Add(fmt.Sprintf("n%d_%d_2", l, h), t.HiddenSize, 1)
				set.Add(fmt.Sprintf("bn%d_%d_2", l, h), t.HiddenSize, 1)
			}
			set.Add(fmt.Sprintf("W%d_%d_1", l, h), t.HiddenSize, t.HiddenSize)
			set.Add(fmt.Sprintf("b%d_%d_1", l, h), t.HiddenSize, 1)
			set.Add(fmt.Sprintf("W%d_%d_2", l, h), t.HiddenSize, t.HiddenSize)
			set.Add(fmt.Sprintf("b%d_%d_2", l, h), t.HiddenSize, 1)
		}
		if l < Layers-1 {
			set.Add(fmt.Sprintf("project%d", l), Heads*t.HiddenSize, t.HiddenSize)
			set.Add(fmt.Sprintf("bias%d", l), t.HiddenSize, 1)
		} else {
			set.Add(fmt.Sprintf("project%d", l), Heads*t.HiddenSize, 10)
			set.Add(fmt.Sprintf("bias%d", l), 10, 1)
		}
		//set.Add("project1", t.HiddenSize, t.HiddenSize)
		//set.Add("bias1", t.HiddenSize, 1)
		//set.Add("project2", t.HiddenSize, 10)
		//set.Add("bias2", 10, 1)
	}

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "a") || strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			continue
		} else if strings.HasPrefix(w.N, "n") {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 1)
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	gradients := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		gradients = append(gradients, make([]float32, len(p.X)))
	}

	adam := make([][]Adam, 0, 8)
	for _, p := range set.Weights {
		adam = append(adam, make([]Adam, len(p.X)))
	}

	f := CreateFunctions(false)

	circuit := t.Circuit(f, set, others)

	regularization := f.FConcat(f.FAvg(f.FAbs(set.Get("encode"))), f.FAvg(f.FAbs(set.Get("position"))))
	regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get("biasEncode"))))
	for l := 0; l < Layers; l++ {
		for h := 0; h < Heads; h++ {
			regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("query%d_%d", l, h)))))
			regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("key%d_%d", l, h)))))
			regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("value%d_%d", l, h)))))
			if l < Layers-1 {
				regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("W%d_%d_1", l, h)))))
				regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("b%d_%d_1", l, h)))))
				regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("W%d_%d_2", l, h)))))
				regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("b%d_%d_2", l, h)))))
			}
		}
		if l < Layers-1 {
			regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("project%d", l)))))
			regularization = f.FConcat(regularization, f.FAvg(f.FAbs(set.Get(fmt.Sprintf("bias%d", l)))))
		}
	}
	regularization = f.FAvg(regularization)

	cost := f.FAdd(f.FSum(f.FQuadratic(circuit, others.Get("output"))), regularization)

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
	u := 0.0
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), u)
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	pinf := float32(2/(1-B2) - 1)
	fmt.Println(pinf)
	reduced := false
	start := time.Now()
	for i < 10*len(images.Train.Images) {
		index := rnd.Intn(len(images.Train.Images))
		image := images.Train.Images[index]

		for j := range inputs.X {
			inputs.X[j] = 0
		}
		for j := range outputs.X {
			outputs.X[j] = 0
		}

		/*for j := 0; j < width; j++ {
			inputs.X[j] = 1
		}*/
		for j, set := range selections {
			for i, value := range set.Positions {
				inputs.X[(j+1)*width+i] =
					float32(image[value]) / 255
			}
		}
		outputs.X[int(images.Train.Labels[index])] = 1

		f.Clear()
		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range gradients {
			for _, d := range p {
				sum += d * d
			}
		}
		norm := math.Sqrt(float64(sum))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / float32(norm)
		}
		_ = scaling
		for j, w := range set.Weights {
			for k, d := range w.D {
				gradients[j][k] += d //* scaling
			}
		}
		set.Zero()
		others.Zero()

		if i > 0 && i%BatchSize == 0 {
			sum := float32(0.0)
			/*for _, p := range set.Weights {
				for _, d := range p.D {
					sum += d * d
				}
			}*/
			for _, p := range gradients {
				for _, d := range p {
					d /= BatchSize
					sum += d * d
				}
			}
			norm := math.Sqrt(float64(sum))
			scaling := float32(1.0)
			if norm > 1 {
				scaling = 1 / float32(norm)
			}

			u++
			b1, b2 := pow(B1), pow(B2)
			for j, w := range set.Weights {
				for k := range w.D {
					//deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
					//set.Weights[j].X[k] += deltas[j][k]
					_ = alpha
					//set.Weights[j].X[k] -= eta * gradients[j][k] / BatchSize * scaling
					_ = scaling
					g := gradients[j][k] / BatchSize
					m := B1*adam[j][k].M + (1-B1)*g
					v := B2*adam[j][k].V + (1-B2)*g*g
					adam[j][k].M = m
					adam[j][k].V = v
					gradients[j][k] = 0
					mhat := m / (1 - b1)
					/*pt := pinf - 2*float32(u)*b2/(1-b2)
					if pt > 4 {
						if v == 0 {
							v = 1e-8
						}
						lt := float32(math.Sqrt(float64((1 - b2) / v)))
						rt := float32(math.Sqrt(float64(((pt - 4) * (pt - 2) * pinf) / ((pinf - 4) * (pinf - 2) * pt))))
						if math.IsNaN(float64(lt)) || math.IsInf(float64(lt), 0) {
							panic(fmt.Errorf("lt is nan %f %f", lt, v))
						}
						if math.IsNaN(float64(rt)) || math.IsInf(float64(rt), 0) {
							panic(fmt.Errorf("rt is nan %f", rt))
						}
						set.Weights[j].X[k] -= eta * rt * mhat * lt
					} else {
						set.Weights[j].X[k] -= eta * mhat
					}*/
					vhat := v / (1 - b2)
					set.Weights[j].X[k] -= eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
				}
			}
			total /= BatchSize
			if !reduced && total < .5 {
				reduced = true
				//eta *= .1
			}
			end := time.Since(start)
			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
			fmt.Println(t.Head, i, total, end)
			others.Zero()
			total = 0
			start = time.Now()
		}

		if halt || math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}
		if i%1000 == 0 {
			set.Save(fmt.Sprintf("%d_%d_set.w", t.Head, i), total, i)
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_probabilistic_transformer_cost.png", t.Head))
	if err != nil {
		panic(err)
	}

	set.Save(fmt.Sprintf("%d_set.w", t.Head), 0, 0)
}

// InferenceProbabilisticTransformer is a probabilistic transformer inference
func InferenceProbabilisticTransformer(h, test int, name string, hiddenSize int, attention Attention) {
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 49, 16+1
	type Head struct {
		Head       tf32.Meta
		Inputs     *tf32.V
		Selections []Position
	}
	f := CreateFunctions(true)
	heads := make([]Head, h)
	for i := range heads {
		rnd := rand.New(rand.NewSource(int64(i + 1)))
		selections := make([]Position, size-1)
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
		relu := f.FRelu
		mask := tf32.U(Mask)
		norm := f.FNorm
		//average := tf32.U(AverageRows)
		//encode := tf32.U(PositionEncodingLayer)
		concat := f.FConcat

		input := tf32.Add(concat(set.Get("position"), tf32.Mul(set.Get("encode"), others.Get("input"))), set.Get("biasEncode"))
		norm_input := tf32.Add(tf32.Hadamard(norm(input), set.Get("n1_1")), set.Get("bn1_1"))
		query := tf32.Mul(set.Get("query"), norm_input)
		key := tf32.Mul(set.Get("key"), norm_input)
		value := tf32.Mul(set.Get("value"), norm_input)
		l1 := tf32.Add(attention(nil, query, key, value, others.Get("dk")), input)
		norm_l1 := tf32.Add(tf32.Hadamard(norm(l1), set.Get("n1_2")), set.Get("bn1_2"))
		l2 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W1_2"), relu(tf32.Add(tf32.Mul(set.Get("W1_1"), norm_l1), set.Get("b1_1")))), set.Get("b1_2")), l1)
		norm_l2 := tf32.Add(tf32.Hadamard(norm(l2), set.Get("n2_1")), set.Get("bn2_1"))
		query1 := tf32.Mul(set.Get("query1"), norm_l2)
		key1 := tf32.Mul(set.Get("key1"), norm_l2)
		value1 := tf32.Mul(set.Get("value1"), norm_l2)
		l3 := tf32.Add(attention(nil, query1, key1, value1, others.Get("dk")), l2)
		norm_l3 := tf32.Add(tf32.Hadamard(norm(l3), set.Get("n2_2")), set.Get("bn2_2"))
		l4 := tf32.Add(tf32.Add(tf32.Mul(set.Get("W2_2"), relu(tf32.Add(tf32.Mul(set.Get("W2_1"), norm_l3), set.Get("b2_1")))), set.Get("b2_2")), l3)
		output := tf32.Add(tf32.Mul(set.Get("project1"), relu(tf32.Add(tf32.Mul(set.Get("project"), mask(l4)), set.Get("bias")))), set.Get("bias1"))
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
		for j := 0; j < width; j++ {
			head.Inputs.X[j] = 1
		}
		for j, set := range head.Selections {
			for i, value := range set.Positions {
				head.Inputs.X[(j+1)*width+i] = float32(image[value]) / 255
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

		f.Clear()
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

// InferenceProbabilisticTransformerParallel is a probabilistic transformer inference
func (t Configuration) InferenceProbabilisticTransformerParallel(h, test int, name string) {
	images, err := mnist.Load()
	if err != nil {
		panic(err)
	}
	width, size := 16, 49+1
	type Voter struct {
		Head       tf32.Meta
		Inputs     *tf32.V
		Selections []Position
	}
	f := CreateFunctions(true)
	voters := make([]Voter, h)
	for i := range voters {
		rnd := rand.New(rand.NewSource(int64(i + 1)))
		selections := make([]Position, size-1)
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
			dk.X[i] = 1 / float32(math.Sqrt(float64(size)))
		}

		set := tf32.NewSet()
		_, _, err = set.Open(fmt.Sprintf("%d_%s", i, name))
		if err != nil {
			panic(err)
		}

		circuit := t.Circuit(f, set, others)

		voters[i] = Voter{
			Head:       circuit,
			Inputs:     inputs,
			Selections: selections,
		}
	}

	type Result struct {
		Probability float32
		Index       int
	}

	process := func(voter Voter, test int) []Result {
		histogram := make([]Result, 10)
		image := images.Test.Images[test]

		for j := range voter.Inputs.X {
			voter.Inputs.X[j] = 0
		}
		/*for j := 0; j < width; j++ {
			head.Inputs.X[j] = 1
		}*/
		for j, set := range voter.Selections {
			for i, value := range set.Positions {
				voter.Inputs.X[(j+1)*width+i] = float32(image[value]) / 255
			}
		}

		f.Clear()
		voter.Head(func(a *tf32.V) bool {
			for j := 0; j < 10; j++ {
				histogram[j].Probability += a.X[j]
				histogram[j].Index = j
			}
			return true
		})
		return histogram
	}

	if test == -1 {
		correct := 0
		for i := range images.Test.Images {
			var votes [10]int
			for _, voter := range voters {
				histogram := process(voter, i)
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
		for _, voter := range voters {
			fmt.Println(images.Test.Labels[test])
			histogram := process(voter, test)
			sort.Slice(histogram, func(i, j int) bool {
				return histogram[i].Probability > histogram[j].Probability
			})
			fmt.Println(histogram)
		}
	}
}
