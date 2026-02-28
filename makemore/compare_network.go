package makemore

import (
	"abaayd01/gruggrad/internal/gruggrad"
	"fmt"
	"math"
)

// CompareToBigramProbabilities loads a trained network and compares its learned
// probabilities (after softmax) to the expected bigram probabilities
func CompareToBigramProbabilities(networkPath string) {
	// Load trained network
	network, err := gruggrad.LoadMNetworkFromFile(networkPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load network: %v", err))
	}

	// Load expected bigram probabilities
	expectedProbs := GenerateProbabilityMatrixFromFile("./makemore/names.txt")

	// For each character (input), compare network output to expected probabilities
	fmt.Println("Comparing network weights to bigram probabilities:")
	fmt.Println()

	// Check a few interesting characters
	testChars := []struct {
		char  rune
		index int
		name  string
	}{
		{BoundaryRune, BoundaryIdx, "BOUNDARY"},
		{'a', 0, "'a'"},
		{'e', 4, "'e'"},
		{'m', 12, "'m'"},
	}

	for _, tc := range testChars {
		fmt.Printf("Character %s (index %d):\n", tc.name, tc.index)

		// Get network weights for this input character (row in weight matrix)
		// Weights are stored as (VocabSize, VocabSize) where row=input, col=output
		weights := network.Layers[0].Weights
		logits := make([]float64, VocabSize)
		biases := network.Layers[0].Biases.Values

		for j := 0; j < VocabSize; j++ {
			logits[j] = weights.Values[tc.index*VocabSize+j] + biases[j]
		}

		// Apply softmax to get probabilities
		networkProbs := softmax(logits)

		// Show top 5 most probable next characters
		type charProb struct {
			char     rune
			index    int
			expected float64
			actual   float64
			diff     float64
		}

		var probs []charProb
		for j := 0; j < VocabSize; j++ {
			char := IdxToRune(j)
			expected := expectedProbs.Get(tc.index, j)
			actual := networkProbs[j]
			probs = append(probs, charProb{
				char:     char,
				index:    j,
				expected: expected,
				actual:   actual,
				diff:     actual - expected,
			})
		}

		// Sort by expected probability (descending)
		for i := 0; i < len(probs)-1; i++ {
			for j := i + 1; j < len(probs); j++ {
				if probs[j].expected > probs[i].expected {
					probs[i], probs[j] = probs[j], probs[i]
				}
			}
		}

		// Show top 5
		fmt.Println("  Top 5 by expected probability:")
		for i := 0; i < 5 && i < len(probs); i++ {
			p := probs[i]
			charStr := fmt.Sprintf("'%c'", p.char)
			if p.char == BoundaryRune {
				charStr = "BOUNDARY"
			}

			fmt.Printf("    %s: expected=%.6f, actual=%.6f, diff=%+.6f\n",
				charStr, p.expected, p.actual, p.diff)
		}
		fmt.Println()
	}
}

func softmax(logits []float64) []float64 {
	// Find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Compute exp(logit - max)
	expLogits := make([]float64, len(logits))
	sumExp := 0.0
	for i, l := range logits {
		expLogits[i] = math.Exp(l - maxLogit)
		sumExp += expLogits[i]
	}

	// Normalize
	probs := make([]float64, len(logits))
	for i := range expLogits {
		probs[i] = expLogits[i] / sumExp
	}
	return probs
}
