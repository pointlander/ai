// Copyright 2022 The AI Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
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
		ProbabilisticTransformer(128)
	} else if *FlagName != "" {
		InferenceProbabilisticTransformer(0, *FlagName, 128)
	}
}
