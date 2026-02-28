package makemore

import (
	"abaayd01/gruggrad/internal/gruggrad"
	"fmt"
	"os"
	"strings"
	"time"
)

const (
	AlphabetSize      = 26
	VocabSize         = AlphabetSize + 1
	BoundaryIdx       = AlphabetSize
	BoundaryRune rune = '.'

	LatestNetworkRunPath = "./network_runs/makemore_network_run_latest.json"
)

func GenerateProbabilityMatrixFromFile(filename string) *gruggrad.Matrix {
	bts, err := os.ReadFile(filename)
	if err != nil {
		panic("could not read file")
	}
	strs := strings.Split(string(bts), "\n")
	return GenerateProbabilityMatrix(strs)
}

func GenerateProbabilityMatrix(strs []string) *gruggrad.Matrix {
	frequencyMatrix := gruggrad.NewMatrix(VocabSize, VocabSize)

	for _, str := range strs {
		if len(str) == 0 {
			continue
		}
		framedName := frameString(str)
		runePairs := convertToRunePairs(framedName)
		for _, pair := range runePairs {
			currentCharIdx, nextCharIdx := RuneToIdx(pair[0]), RuneToIdx(pair[1])
			curVal := frequencyMatrix.Get(currentCharIdx, nextCharIdx)
			frequencyMatrix.Set(currentCharIdx, nextCharIdx, curVal+1.0)
		}
	}

	// Normalize row-wise to get conditional probabilities P(next | current)
	return frequencyMatrix.RowMap(func(row []float64) []float64 {
		// Calculate row sum
		rowSum := 0.0
		for _, val := range row {
			rowSum += val
		}

		// If row sum is 0, return zeros (avoid division by zero)
		if rowSum == 0 {
			return row
		}

		// Normalize each element by the row sum
		normalized := make([]float64, len(row))
		for i, val := range row {
			normalized[i] = val / rowSum
		}
		return normalized
	})
}

func frameString(str string) string {
	return string(BoundaryRune) + str + string(BoundaryRune)
}

func convertToRunePairs(framedString string) [][2]rune {
	runes := []rune(framedString)
	result := make([][2]rune, 0, len(runes))
	for i := 0; i < len(runes)-1; i++ {
		result = append(result, [2]rune{runes[i], runes[i+1]})
	}
	return result
}

func RuneToIdx(r rune) int {
	if r == BoundaryRune {
		return BoundaryIdx
	}

	val := int(r) - int('a')
	if val < 0 || val >= AlphabetSize {
		panic(fmt.Sprintf("invalid rune: %c", r))
	}
	return val
}

func IdxToRune(idx int) rune {
	if idx == BoundaryIdx {
		return BoundaryRune
	}

	if idx < 0 || idx >= AlphabetSize {
		panic(fmt.Sprintf("invalid idx: %d", idx))
	}

	return rune('a' + idx)
}

func buildTrainingExamples(filename string) []gruggrad.MNetworkTrainingExample {
	bts, err := os.ReadFile(filename)
	if err != nil {
		panic("could not read file")
	}
	strs := strings.Split(string(bts), "\n")

	var examples []gruggrad.MNetworkTrainingExample
	for _, str := range strs {
		if len(str) == 0 {
			continue
		}
		framedName := frameString(str)
		runePairs := convertToRunePairs(framedName)
		for _, pair := range runePairs {
			currentCharIdx, nextCharIdx := RuneToIdx(pair[0]), RuneToIdx(pair[1])
			inputMatrix := gruggrad.NewTrackedMatrix(1, VocabSize)
			inputMatrix.Set(0, currentCharIdx, 1.0)
			outputMatrix := gruggrad.NewTrackedMatrix(1, 1)
			outputMatrix.Set(0, 0, float64(nextCharIdx))
			examples = append(examples, gruggrad.MNetworkTrainingExample{
				Input:  inputMatrix,
				Output: outputMatrix,
			})
		}
	}

	return examples
}

func Train() {
	learningRate := 0.01

	network := gruggrad.NewRandomMNetwork([]gruggrad.LayerDims{
		{NumWeights: VocabSize, NumNeurons: VocabSize},
	})

	examples := buildTrainingExamples("./makemore/names.txt")

	epochs := 3
	for epoch := range epochs {
		totalLoss := 0.0
		for _, example := range examples {
			output := network.Forward(example.Input)
			loss := output.SoftMaxCrossEntropy2(example.Output)
			totalLoss += loss.Matrix.Values[0]
			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)
		}
		avgLoss := totalLoss / float64(len(examples))
		fmt.Printf("Epoch %d/%d: Average Loss = %.6f\n", epoch+1, epochs, avgLoss)
	}

	network.Store(fmt.Sprintf("./network_runs/makemore_network_run_%s.json", time.Now().Format(time.DateTime)))
	network.Store(LatestNetworkRunPath)
	fmt.Println("Training complete! Network saved.")
}

func TrainBatched() {
	learningRate := 0.01
	batchSize := 16

	network := gruggrad.NewRandomMNetwork([]gruggrad.LayerDims{
		{NumWeights: VocabSize, NumNeurons: VocabSize},
	})

	examples := buildTrainingExamples("./makemore/names.txt")

	epochs := 25
	for epoch := range epochs {
		totalLoss := 0.0
		numBatches := 0

		// Process examples in batches
		for i := 0; i < len(examples); i += batchSize {
			// Determine batch size (last batch may be smaller)
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
				// Copy input values
				copy(batchInput.Values[j*VocabSize:(j+1)*VocabSize], example.Input.Values)
				// Copy target value
				batchTarget.Values[j] = example.Output.Values[0]
			}

			// Forward, backward, update
			output := network.Forward(batchInput)
			loss := output.SoftMaxCrossEntropy2(batchTarget)
			totalLoss += loss.Matrix.Values[0] * float64(currentBatchSize)

			loss.Gradients.Values = []float64{1}
			loss.Backward()
			network.Tune(learningRate)

			numBatches++
		}

		avgLoss := totalLoss / float64(len(examples))
		fmt.Printf("Epoch %d/%d: Average Loss = %.6f (%d batches)\n", epoch+1, epochs, avgLoss, numBatches)
	}

	network.Store(fmt.Sprintf("./network_runs/makemore_network_run_batched_%s.json", time.Now().Format(time.DateTime)))
	network.Store(LatestNetworkRunPath)
	fmt.Println("Batched training complete! Network saved.")
}

// TrainFullBatch uses the entire dataset for each gradient update
// This should converge perfectly for this simple problem
func TrainFullBatch() {
	learningRate := 1.0 // Can use higher LR with full batch

	network := gruggrad.NewRandomMNetwork([]gruggrad.LayerDims{
		{NumWeights: VocabSize, NumNeurons: VocabSize},
	})

	examples := buildTrainingExamples("./makemore/names.txt")

	// Stack ALL examples into one big batch
	batchInput := gruggrad.NewTrackedMatrix(len(examples), VocabSize)
	batchTarget := gruggrad.NewTrackedMatrix(len(examples), 1)

	for i, example := range examples {
		copy(batchInput.Values[i*VocabSize:(i+1)*VocabSize], example.Input.Values)
		batchTarget.Values[i] = example.Output.Values[0]
	}

	epochs := 100
	for epoch := 0; epoch < epochs; epoch++ {
		output := network.Forward(batchInput)
		loss := output.SoftMaxCrossEntropy2(batchTarget)

		loss.Gradients.Values = []float64{1}
		loss.Backward()
		network.Tune(learningRate)

		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("Epoch %3d: Loss = %.6f\n", epoch+1, loss.Values[0])
		}
	}

	network.Store(fmt.Sprintf("./network_runs/makemore_network_run_fullbatch_%s.json", time.Now().Format(time.DateTime)))
	network.Store(LatestNetworkRunPath)
	fmt.Println("Full-batch training complete! Network saved.")
}

func ConvertToNGramString(str string, n int) string {
	return string(BoundaryRune)
}
