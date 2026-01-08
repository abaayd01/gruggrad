package main

import (
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
	default:
		fmt.Printf("unknown cmd '%s'\n", cmd)
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
	if idx < 0 || idx > 27 {
		panic("idx out of range for rune conversion")
	}

	if idx == 26 {
		return "S"
	}

	if idx == 27 {
		return "E"
	}

	val := idx + int('a')
	return string(val)
}
