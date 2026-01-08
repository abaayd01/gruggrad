package mnist

import (
	"fmt"
	"math"
	"time"

	gruggrad "abaayd01/gruggrad/internal/gruggrad"
)

func normalisePixelValue(val uint8) float64 {
	return float64(val) / 255.0
}

func toInput(pixels []uint8) []*gruggrad.Value {
	var result []*gruggrad.Value
	for i := range pixels {
		dataVal := normalisePixelValue(pixels[i])
		result = append(result, gruggrad.NewValue(dataVal))
	}
	return result
}

func toOutput(data uint8) *gruggrad.Value {
	return gruggrad.NewValue(float64(data))
}

func buildTrainingExamples(imgs [][]uint8, lbls []uint8) []gruggrad.TrainingExample {
	var examples []gruggrad.TrainingExample
	for i := range imgs {
		examples = append(examples, gruggrad.TrainingExample{
			Input:  toInput(imgs[i]),
			Output: toOutput(lbls[i]),
		})
	}
	return examples
}

func exponentiate(input *gruggrad.Value) *gruggrad.Value {
	e := gruggrad.NewValue(math.E)
	return e.Pow(input)
}

func Softmax(inputs []*gruggrad.Value) []*gruggrad.Value {
	var exponentiatedInputs []*gruggrad.Value
	var sumExponentiatedInputs *gruggrad.Value
	for i := range inputs {
		ei := exponentiate(inputs[i])
		exponentiatedInputs = append(exponentiatedInputs, ei)
		if sumExponentiatedInputs == nil {
			sumExponentiatedInputs = ei
		} else {
			sumExponentiatedInputs = sumExponentiatedInputs.Add(ei)
		}
	}

	var output []*gruggrad.Value
	for i := range exponentiatedInputs {
		output = append(output, exponentiatedInputs[i].Div(sumExponentiatedInputs))
	}
	return output
}

// assuming single correct classification
func LossCrossEntropy(value *gruggrad.Value) *gruggrad.Value {
	return value.Ln().Mul(gruggrad.NewValue(-1.0))
}

func LossMnist(output []*gruggrad.Value, expectedDigitValue *gruggrad.Value) *gruggrad.Value {
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

	network := gruggrad.NewRandomNetwork([]gruggrad.LayerDims{
		{NumWeights: 28 * 28, NumNeurons: 128},
		{NumWeights: 128, NumNeurons: 10}, // output layer
	})

	examples := buildTrainingExamples(trainImgs, trainLbls)

	loss := gruggrad.NewValue(1.0)
	learningRate := 0.1

	sampleExamples := examples[:1000]
	for _, example := range sampleExamples {
		output := network.Forward(example.Input)
		normalisedOutput := Softmax(output)
		expectedLabel := example.Output.Data
		loss = LossMnist(normalisedOutput, example.Output)
		for i, o := range normalisedOutput {
			fmt.Printf("i: %d, o: %.2f\t ", i, o.Data)
		}

		fmt.Printf("expectedLabel: %d, probability: %.4f, loss: %.4f\n", int(expectedLabel), normalisedOutput[int(expectedLabel)].Data, loss.Data)

		loss.Gradient = 1
		loss.Backward()
		network.Tune(learningRate)
	}

	network.Store(fmt.Sprintf("./network_runs/mnist_network_run_%s.json", time.Now().Format(time.DateTime)))
	fmt.Printf("\n\n\n final loss: %.16f\n", loss.Data)
}

func Milestone1Verify() {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	_, _, verificationImgs, verificationLbls, _ := loader.LoadData()

	network, err := gruggrad.LoadNetworkFromFile("mnist_network_run_2025-12-23 15:33:27.json")
	if err != nil {
		panic("could not load network from file")
	}

	examples := buildTrainingExamples(verificationImgs, verificationLbls)

	sampleExamples := examples[:100]
	for exampleIdx, example := range sampleExamples {
		output := network.Forward(example.Input)
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
		expectedLabel := example.Output.Data
		fmt.Printf("example idx: %d, guessedLabel: %d, expectedLabel: %d, probability: %.4f\n", exampleIdx, highestProbabilityIdx, int(expectedLabel), normalisedOutput[int(expectedLabel)].Data)
	}
}

// new row in the matrix is a new example in the batch
func toTrackedMatrixInput(imgs [][]uint8) *gruggrad.TrackedMatrix {
	result := gruggrad.NewTrackedMatrix(len(imgs), 28*28)
	for i := range imgs {
		for pixelIdx := range imgs[i] {
			dataVal := normalisePixelValue(imgs[i][pixelIdx])
			result.Matrix.Set(i, pixelIdx, dataVal)
		}
	}
	return result
}

// new row in the matrix is a new example in the batch
func toTrackedMatrixOutput(lbls []uint8) *gruggrad.TrackedMatrix {
	result := gruggrad.NewTrackedMatrix(len(lbls), 1)
	for i := range lbls {
		dataVal := float64(lbls[i])
		result.Matrix.Set(i, 0, dataVal)
	}
	return result
}

var BatchSize = 32

func buildMNetworkTrainingExamples(imgs [][]uint8, lbls []uint8) []gruggrad.MNetworkTrainingExample {
	var examples []gruggrad.MNetworkTrainingExample
	for i := 0; i < len(imgs); i = i + BatchSize {
		if i+BatchSize > len(imgs) {
			examples = append(examples, gruggrad.MNetworkTrainingExample{
				Input:  toTrackedMatrixInput(imgs[i:]),
				Output: toTrackedMatrixOutput(lbls[i:]),
			})
		} else {
			examples = append(examples, gruggrad.MNetworkTrainingExample{
				Input:  toTrackedMatrixInput(imgs[i : i+BatchSize]),
				Output: toTrackedMatrixOutput(lbls[i : i+BatchSize]),
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

	network := gruggrad.NewRandomMNetwork([]gruggrad.LayerDims{
		{NumWeights: 28 * 28, NumNeurons: 128},
		{NumWeights: 128, NumNeurons: 10}, // output layer
	})

	examples := buildMNetworkTrainingExamples(trainImgs, trainLbls)

	var loss *gruggrad.TrackedMatrix
	learningRate := 0.1

	epochs := 4

	for range epochs {
		for _, example := range examples {
			output := network.Forward(example.Input)
			loss = output.SoftMaxCrossEntropy2(example.Output)
			fmt.Printf("loss: %s\n", loss.Matrix.String())
			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)
		}
	}

	network.Store(fmt.Sprintf("./network_runs/mnist_network_run_%s.json", time.Now().Format(time.DateTime)))
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

	network, err := gruggrad.LoadMNetworkFromFile("mnist_network_run_2025-12-28 13:27:28.json")
	if err != nil {
		panic("could not load network from file")
	}

	examples := buildMNetworkTrainingExamples(verificationImgs, verificationLbls)

	var countIncorrect int
	for _, example := range examples {
		output := network.Forward(example.Input)
		targetVals := example.Output.Values

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
	fmt.Printf("countCorrect: %d, countIncorrect: %d, accuracy: %.4f\n\n", len(examples)*BatchSize, countIncorrect, 1.0-float64(countIncorrect)/float64(len(examples)*BatchSize))
}
