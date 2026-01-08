package gruggrad

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
			Input: NewTrackedMatrixWithValues(1, 2, []float64{0, 0}),
			Output: NewTrackedMatrixWithValues(1, 1, []float64{0}),
		},
		{
			Input: NewTrackedMatrixWithValues(1, 2, []float64{1, 0}),
			Output: NewTrackedMatrixWithValues(1, 1, []float64{1}),
		},
		{
			Input: NewTrackedMatrixWithValues(1, 2, []float64{0, 1}),
			Output: NewTrackedMatrixWithValues(1, 1, []float64{1}),
		},
		{
			Input: NewTrackedMatrixWithValues(1, 2, []float64{1, 1}),
			Output: NewTrackedMatrixWithValues(1, 1, []float64{0}),
		},
	}

	for _, example := range examples {
		output := network.Forward(example.Input)
		fmt.Printf("\ninput:\n%s\n\noutput:\n%s\n\n========\n", example.Input.String(), output.String())
	}
}

func TestMNetworkTraining(t *testing.T) {
	learningRate := 0.01

	network := NewRandomMNetwork([]LayerDims{
		{NumWeights: 2, NumNeurons: 16},
		{NumWeights: 16, NumNeurons: 1}, // output layer
	})

	examples := []MNetworkTrainingExample{
		{
			Input: NewTrackedMatrixWithValues(
				4, 2,
				[]float64{
					0, 0,
					1, 0,
					0, 1,
					1, 1,
				},
			),
			Output: NewTrackedMatrixWithValues(
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
			output := network.Forward(example.Input)
			expected := example.Output

			loss = MLossXOR(expected, output)

			fmt.Printf("Output: %s, loss: %s\n", output.Matrix.String(), loss.Matrix.String())
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

	fmt.Printf("Output: \n%s\n\n", output.Matrix.String())
}
