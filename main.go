// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/big"

	"github.com/ALTree/bigfloat"
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
	// FlagFFT is a flag to run the FFT mode
	FlagFFT = flag.Bool("fft", false, "FFT mode")
	// FlagTransformer is a flag to run the transformer mode
	FlagTransformer = flag.Bool("transformer", false, "transformer mode")
	// FlagTest is a flag to run the test mode
	FlagTest = flag.Int("test", -1, "test mode")
	// FlagHeads is a flag to set the number of heads
	FlagHeads = flag.Int("heads", 1, "number of heads")
)

func main() {
	flag.Parse()

	if *FlagIris {
		if *FlagFFT {
			if *FlagComplex {
				if *FlagTransformer {
					ComplexTransformerIrisFFT(32)
				} else {
					ComplexIrisFFT(true, 32)
				}
			} else {
				IrisFFT(32)
			}
		} else if *FlagComplex {
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

	//Transformer(32)
	if *FlagTransformer {
		for i := 0; i < 256; i++ {
			y := float32(i)
			x := float32(math.Exp(float64(y)))
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				fmt.Println(x, i)
				break
			}
		}
		for i := 0; i < 1024; i++ {
			x := math.Exp(float64(i))
			if math.IsNaN(x) || math.IsInf(x, 0) {
				fmt.Println(x, i)
				break
			}
		}
		for i := 0; i < 1024; i++ {
			r := bigfloat.Exp(big.NewFloat(float64(i)))
			if r.IsInf() {
				fmt.Println(r, i)
				break
			}
		}
		r := bigfloat.Exp(big.NewFloat(float64(1024 * 1024)))
		fmt.Println(r.String())
		for i := 0; i < *FlagHeads; i++ {
			t := Configuration{
				Head:       i,
				HiddenSize: 32,
				Attention:  RegularAttention,
				Swap:       true,
			}
			t.ProbabilisticTransformerParallel()
		}
	} else if *FlagName != "" {
		t := Configuration{
			HiddenSize: 32,
			Attention:  RegularAttention,
			Swap:       true,
		}
		t.InferenceProbabilisticTransformerParallel(*FlagHeads, *FlagTest, *FlagName)
	}
}
