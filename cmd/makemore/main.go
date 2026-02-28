package main

import (
	"abaayd01/gruggrad/internal/gruggrad"
	"abaayd01/gruggrad/makemore"
	"fmt"
	"os"
	"sort"
	"strings"
)

func main() {
	args := os.Args
	if len(args) < 2 {
		fmt.Printf("did not provide a cmd\n")
		os.Exit(1)
	}

	cmd := args[1]

	switch cmd {
	case "generate-probabilities":
		generateBigramProbabilities()
	case "train":
		train()
	case "sample":
		for range 5 {
			sampleFromNetwork()
		}
	default:
		fmt.Fprintf(os.Stderr, "Error: unknown command '%s'\n\n", cmd)
		fmt.Fprintf(os.Stderr, "Available commands:\n")
		fmt.Fprintf(os.Stderr, "  generate-probabilities  Generate bigram probability matrix from names.txt\n")
		fmt.Fprintf(os.Stderr, "  train                   Train neural network on names.txt\n")
		fmt.Fprintf(os.Stderr, "  sample                  Generate 5 samples from trained network\n")
		fmt.Fprintf(os.Stderr, "\nUsage: go run cmd/makemore/main.go <command>\n")
		os.Exit(1)
	}
}

func generateBigramProbabilities() {
	probabilityMatrix := makemore.GenerateProbabilityMatrixFromFile("./makemore/names.txt")
	bigramResults := make(map[string]float64)

	for i := range probabilityMatrix.NumRows {
		curRune := idxToString(i)
		for j := range probabilityMatrix.NumCols {
			nextRune := idxToString(j)
			bigramResults[curRune+nextRune] = probabilityMatrix.Get(i, j)
		}
	}

	tbl := stringifyMap(bigramResults)
	err := os.WriteFile("./makemore/names_bigram_probabilities.txt", []byte(tbl), 0644)
	if err != nil {
		panic(err)
	}
}

func sampleFromNetwork() {
	filename := makemore.LatestNetworkRunPath
	network, err := gruggrad.LoadMNetworkFromFile(filename)
	if err != nil {
		panic(err)
	}

	var name string
	currentRune := makemore.BoundaryRune

	// Sample character by character until boundary rune
	for {
		currentRuneIdx := makemore.RuneToIdx(currentRune)
		inputMatrix := gruggrad.NewTrackedMatrix(1, makemore.VocabSize)
		inputMatrix.Set(0, currentRuneIdx, 1.0)

		output := network.Forward(inputMatrix)
		probs := output.SoftMax()

		// Sample from the distribution
		sampledIdx := probs.Multinomial()

		// Check if we hit end of string
		if sampledIdx == makemore.BoundaryIdx {
			break
		}

		// Convert index to character and append
		sampledRune := makemore.IdxToRune(sampledIdx)
		name += string(sampledRune)
		currentRune = sampledRune
	}

	fmt.Printf("Neural network generated name: '%s' (length: %d)\n", name, len(name))
}

func train() {
	fmt.Println("training...")
	makemore.TrainBatched()
}

func stringifyMap(m map[string]float64) string {
	var sb strings.Builder
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		sb.WriteString(fmt.Sprintf("%-4s | %8.8f\n", k, m[k]))
	}
	return sb.String()
}

func idxToString(idx int) string {
	if idx < 0 || idx >= makemore.VocabSize {
		panic("idx out of range for rune conversion")
	}

	if idx == makemore.BoundaryIdx {
		return string(makemore.BoundaryRune)
	}

	return string(makemore.IdxToRune(idx))
}
