package main

import (
	"fmt"
	"testing"
	"time"
)

func TestMNetworkForward(t *testing.T) {
	network, err := LoadMNetworkFromFile("mnetwork_run_2025-12-27.json")
	if err != nil {
		t.Errorf("failed to load network from file: %s", err)
	}

	examples := []MNetworkTrainingExample{
		{
			input:  NewTrackedMatrixWithValues(1, 2, []float64{0, 0}),
			output: NewTrackedMatrixWithValues(1, 1, []float64{0}),
		},
		{
			input:  NewTrackedMatrixWithValues(1, 2, []float64{1, 0}),
			output: NewTrackedMatrixWithValues(1, 1, []float64{1}),
		},
		{
			input:  NewTrackedMatrixWithValues(1, 2, []float64{0, 1}),
			output: NewTrackedMatrixWithValues(1, 1, []float64{1}),
		},
		{
			input:  NewTrackedMatrixWithValues(1, 2, []float64{1, 1}),
			output: NewTrackedMatrixWithValues(1, 1, []float64{0}),
		},
	}

	for _, example := range examples {
		output := network.Forward(example.input)
		fmt.Printf("\ninput:\n%s\n\noutput:\n%s\n\n========\n", example.input.String(), output.String())
	}
}

func TestMNetworkTraining(t *testing.T) {
	learningRate := 0.01

	network := NewRandomMNetwork([]LayerDims{
		{numWeights: 2, numNeurons: 16},
		{numWeights: 16, numNeurons: 1}, // output layer
	})

	examples := []MNetworkTrainingExample{
		{
			input: NewTrackedMatrixWithValues(
				4, 2,
				[]float64{
					0, 0,
					1, 0,
					0, 1,
					1, 1,
				},
			),
			output: NewTrackedMatrixWithValues(
				4, 1,
				[]float64{
					0,
					1,
					1,
					0,
				},
			),
		},
	}

	var loss *TrackedMatrix

	epochs := 1000
	for range epochs {
		for _, example := range examples {
			output := network.Forward(example.input)
			expected := example.output

			loss = MLossXOR(expected, output)

			fmt.Printf("output: %s, loss: %s\n", output.Matrix.String(), loss.Matrix.String())
			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)
		}
	}
	network.Store(fmt.Sprintf("./mnetwork_run_%s.json", time.Now().Format(time.DateOnly)))
	fmt.Printf("\n\n\n final loss: %s\n", loss.Matrix.String())
}

func TestMLossXOR(t *testing.T) {
	input := NewTrackedMatrixWithValues(2, 1, []float64{5, 6})
	target := NewTrackedMatrixWithValues(2, 1, []float64{0, 0})

	output := MLossXOR(target, input)

	fmt.Printf("output:\n%s\n\n", output.Matrix.String())
}
