package main

import (
	"fmt"
	"math"
	"time"
)

func normalisePixelValue(val uint8) float64 {
	return float64(val) / 255.0
}

func toInput(pixels []uint8) []*Value {
	var result []*Value
	for i := range pixels {
		dataVal := normalisePixelValue(pixels[i])
		result = append(result, NewValue(dataVal))
	}
	return result
}

func toOutput(data uint8) *Value {
	return NewValue(float64(data))
}

func buildTrainingExamples(imgs [][]uint8, lbls []uint8) []TrainingExample {
	var examples []TrainingExample
	for i := range imgs {
		examples = append(examples, TrainingExample{
			input:  toInput(imgs[i]),
			output: toOutput(lbls[i]),
		})
	}
	return examples
}

func exponentiate(input *Value) *Value {
	e := NewValue(math.E)
	return e.Pow(input)
}

func Softmax(inputs []*Value) []*Value {
	var exponentiatedInputs []*Value
	var sumExponentiatedInputs *Value
	for i := range inputs {
		ei := exponentiate(inputs[i])
		exponentiatedInputs = append(exponentiatedInputs, ei)
		if sumExponentiatedInputs == nil {
			sumExponentiatedInputs = ei
		} else {
			sumExponentiatedInputs = sumExponentiatedInputs.Add(ei)
		}
	}

	var output []*Value
	for i := range exponentiatedInputs {
		output = append(output, exponentiatedInputs[i].Div(sumExponentiatedInputs))
	}
	return output
}

// assuming single correct classification
func LossCrossEntropy(value *Value) *Value {
	return value.Ln().Mul(NewValue(-1.0))
}

func LossMnist(output []*Value, expectedDigitValue *Value) *Value {
	expectedDigit := int(expectedDigitValue.Data)
	if expectedDigit < 0 || expectedDigit > 9 || expectedDigit > len(output) {
		panic("expectedDigit out of bounds")
	}
	return LossCrossEntropy(output[expectedDigit])
}

func Milestone1() {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	trainImgs, trainLbls, _, _, _ := loader.LoadData()

	network := NewRandomNetwork([]LayerDims{
		{numWeights: 28 * 28, numNeurons: 128},
		{numWeights: 128, numNeurons: 10}, // output layer
	})

	examples := buildTrainingExamples(trainImgs, trainLbls)

	loss := NewValue(1.0)
	learningRate := 0.1

	sampleExamples := examples[:1000]
	for _, example := range sampleExamples {
		output := network.Forward(example.input)
		normalisedOutput := Softmax(output)
		expectedLabel := example.output.Data
		loss = LossMnist(normalisedOutput, example.output)
		for i, o := range normalisedOutput {
			fmt.Printf("i: %d, o: %.2f\t ", i, o.Data)
		}

		fmt.Printf("expectedLabel: %d, probability: %.4f, loss: %.4f\n", int(expectedLabel), normalisedOutput[int(expectedLabel)].Data, loss.Data)

		loss.Gradient = 1
		loss.Backward()
		network.Tune(learningRate)
	}

	network.Store(fmt.Sprintf("./mnist_network_run_%s.json", time.Now().Format(time.DateTime)))
	fmt.Printf("\n\n\n final loss: %.16f\n", loss.Data)
}

func Milestone1Verify() {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	_, _, verificationImgs, verificationLbls, _ := loader.LoadData()

	network, err := LoadNetworkFromFile("mnist_network_run_2025-12-23 15:33:27.json")
	if err != nil {
		panic("could not load network from file")
	}

	examples := buildTrainingExamples(verificationImgs, verificationLbls)

	sampleExamples := examples[:100]
	for exampleIdx, example := range sampleExamples {
		output := network.Forward(example.input)
		normalisedOutput := Softmax(output)

		highestProbabilityIdx := -1
		curMaxProbability := 0.0
		for i, o := range normalisedOutput {
			// fmt.Printf("i: %d, o: %.4f\t ", i, o.Data)
			if o.Data > curMaxProbability {
				curMaxProbability = o.Data
				highestProbabilityIdx = i
			}
		}
		expectedLabel := example.output.Data
		fmt.Printf("example idx: %d, guessedLabel: %d, expectedLabel: %d, probability: %.4f\n", exampleIdx, highestProbabilityIdx, int(expectedLabel), normalisedOutput[int(expectedLabel)].Data)
	}
}

// new row in the matrix is a new example in the batch
func toTrackedMatrixInput(imgs [][]uint8) *TrackedMatrix {
	result := NewTrackedMatrix(len(imgs), 28*28)
	for i := range imgs {
		for pixelIdx := range imgs[i] {
			dataVal := normalisePixelValue(imgs[i][pixelIdx])
			result.Matrix.Set(i, pixelIdx, dataVal)
		}
	}
	return result
}

// new row in the matrix is a new example in the batch
func toTrackedMatrixOutput(lbls []uint8) *TrackedMatrix {
	result := NewTrackedMatrix(len(lbls), 1)
	for i := range lbls {
		dataVal := float64(lbls[i])
		result.Matrix.Set(i, 0, dataVal)
	}
	return result
}

var batchSize = 32

func buildMNetworkTrainingExamples(imgs [][]uint8, lbls []uint8) []MNetworkTrainingExample {
	var examples []MNetworkTrainingExample
	for i := 0; i < len(imgs); i = i + batchSize {
		if i+batchSize > len(imgs) {
			examples = append(examples, MNetworkTrainingExample{
				input:  toTrackedMatrixInput(imgs[i:]),
				output: toTrackedMatrixOutput(lbls[i:]),
			})
		} else {
			examples = append(examples, MNetworkTrainingExample{
				input:  toTrackedMatrixInput(imgs[i : i+batchSize]),
				output: toTrackedMatrixOutput(lbls[i : i+batchSize]),
			})
		}
	}
	return examples
}

func MnistMNetworkTraining() {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	trainImgs, trainLbls, _, _, _ := loader.LoadData()

	network := NewRandomMNetwork([]LayerDims{
		{numWeights: 28 * 28, numNeurons: 128},
		{numWeights: 128, numNeurons: 10}, // output layer
	})

	examples := buildMNetworkTrainingExamples(trainImgs, trainLbls)

	var loss *TrackedMatrix
	learningRate := 0.1

	epochs := 4

	for range epochs {
		for _, example := range examples {
			output := network.Forward(example.input)
			loss = output.SoftMaxCrossEntropy2(example.output)
			fmt.Printf("loss: %s\n", loss.Matrix.String())
			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)
		}
	}

	network.Store(fmt.Sprintf("./mnist_network_run_%s.json", time.Now().Format(time.DateTime)))
}

const (
	pass = "✓"
	fail = "✗"
)

func MnistMNetworkVerify() {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	_, _, verificationImgs, verificationLbls, _ := loader.LoadData()

	network, err := LoadMNetworkFromFile("mnist_network_run_2025-12-28 13:27:28.json")
	if err != nil {
		panic("could not load network from file")
	}

	examples := buildMNetworkTrainingExamples(verificationImgs, verificationLbls)

	var countIncorrect int
	for _, example := range examples {
		output := network.Forward(example.input)
		targetVals := example.output.Values

		for i := range targetVals {
			guessedVal := 0
			maxVal := output.Matrix.Row(i)[guessedVal]
			for j, val := range output.Matrix.Row(i) {
				if val > maxVal {
					maxVal = val
					guessedVal = j
				}
			}

			// var resultStr string
			if int(targetVals[i]) == guessedVal {
				// resultStr = pass
			} else {
				// resultStr = fail
				countIncorrect++
			}
			// fmt.Printf("guessedVal: %d, target: %.2f, correct?: %s\n", guessedVal, targetVals[i], resultStr)
		}
	}

	fmt.Printf("------\nRESULTS\n------\n\n")
	fmt.Printf("countCorrect: %d, countIncorrect: %d, accuracy: %.4f\n\n", len(examples)*batchSize, countIncorrect, 1.0-float64(countIncorrect)/float64(len(examples)*batchSize))
}
