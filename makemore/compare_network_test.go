package makemore

import (
	"abaayd01/gruggrad/internal/gruggrad"
	"fmt"
	"strconv"
	"testing"
)

func TestCompareNetworkToBigrams(t *testing.T) {
	// Train using mini-batch gradient descent
	numEpochs := 50
	batchSize := 16
	fmt.Println("Training network with mini-batch GD for " + strconv.Itoa(numEpochs) + " epochs...")

	learningRate := 0.01

	network := gruggrad.NewRandomMNetwork([]gruggrad.LayerDims{
		{NumWeights: VocabSize, NumNeurons: VocabSize},
	})

	examples := buildTrainingExamples("./names.txt")

	// Train with mini-batches
	for epoch := 0; epoch < numEpochs; epoch++ {
		totalLoss := 0.0

		for i := 0; i < len(examples); i += batchSize {
			end := i + batchSize
			if end > len(examples) {
				end = len(examples)
			}
			currentBatchSize := end - i

			// Stack examples into batch matrices
			batchInput := gruggrad.NewTrackedMatrix(currentBatchSize, VocabSize)
			batchTarget := gruggrad.NewTrackedMatrix(currentBatchSize, 1)

			for j := 0; j < currentBatchSize; j++ {
				example := examples[i+j]
				copy(batchInput.Values[j*VocabSize:(j+1)*VocabSize], example.Input.Values)
				batchTarget.Values[j] = example.Output.Values[0]
			}

			output := network.Forward(batchInput)
			loss := output.SoftMaxCrossEntropy2(batchTarget)
			totalLoss += loss.Values[0] * float64(currentBatchSize)

			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)
		}

		avgLoss := totalLoss / float64(len(examples))
		if epoch%10 == 0 || epoch == numEpochs-1 {
			fmt.Printf("Epoch %3d: Loss = %.6f\n", epoch+1, avgLoss)
		}
	}

	// Now compare
	fmt.Println("\n============================================================")
	compareNetworkToBigrams(network)
}

func compareNetworkToBigrams(network *gruggrad.MNetwork) {
	expectedProbs := GenerateProbabilityMatrixFromFile("./names.txt")

	fmt.Println("Comparing network weights to bigram probabilities:")
	fmt.Println()

	testChars := []struct {
		char  rune
		index int
		name  string
	}{
		{BoundaryRune, BoundaryIdx, "BOUNDARY"},
		{'a', 0, "'a'"},
		{'e', 4, "'e'"},
	}

	for _, tc := range testChars {
		fmt.Printf("Character %s (index %d):\n", tc.name, tc.index)

		weights := network.Layers[0].Weights
		logits := make([]float64, VocabSize)
		biases := network.Layers[0].Biases.Values

		for j := 0; j < VocabSize; j++ {
			logits[j] = weights.Values[tc.index*VocabSize+j] + biases[j]
		}

		networkProbs := softmax(logits)

		type charProb struct {
			char     rune
			expected float64
			actual   float64
		}

		var probs []charProb
		for j := 0; j < VocabSize; j++ {
			char := IdxToRune(j)
			probs = append(probs, charProb{
				char:     char,
				expected: expectedProbs.Get(tc.index, j),
				actual:   networkProbs[j],
			})
		}

		// Sort by expected probability
		for i := 0; i < len(probs)-1; i++ {
			for j := i + 1; j < len(probs); j++ {
				if probs[j].expected > probs[i].expected {
					probs[i], probs[j] = probs[j], probs[i]
				}
			}
		}

		fmt.Println("  Top 5 by expected probability:")
		for i := 0; i < 5; i++ {
			p := probs[i]
			charStr := fmt.Sprintf("'%c'", p.char)
			if p.char == BoundaryRune {
				charStr = "BOUNDARY"
			}

			fmt.Printf("    %s: expected=%.6f, actual=%.6f, diff=%+.6f\n",
				charStr, p.expected, p.actual, p.actual-p.expected)
		}
		fmt.Println()
	}
}
