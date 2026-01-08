package main

import (
	"encoding/json"
	"fmt"
	"os"

	gruggrad "abaayd01/gruggrad/internal/gruggrad"
)

func convertNetworkToMNetwork(inputFile, outputFile string) error {
	// Read the Network JSON file
	jsonData, err := os.ReadFile(inputFile)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", inputFile, err)
	}

	// Unmarshal into SerializedNetwork
	var network gruggrad.SerializedNetwork
	err = json.Unmarshal(jsonData, &network)
	if err != nil {
		return fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Convert to SerializedMNetwork
	var mNetwork gruggrad.SerializedMNetwork
	for _, layer := range network.Layers {
		if len(layer.Neurons) == 0 {
			continue
		}

		// Determine dimensions
		numNeurons := len(layer.Neurons)
		numWeights := len(layer.Neurons[0].Weights)

		// Create weight matrix (numNeurons × numWeights)
		// Each row is one neuron's weights
		weightValues := make([]float64, numNeurons*numWeights)
		biasValues := make([]float64, numNeurons)

		for i, neuron := range layer.Neurons {
			// Copy weights for this neuron (one row in the weight matrix)
			for j, weight := range neuron.Weights {
				weightValues[i*numWeights+j] = weight
			}
			// Copy bias
			biasValues[i] = neuron.Bias
		}

		mLayer := gruggrad.SerializedMLayer{
			Weights: gruggrad.SerializedMatrix{
				Values:  weightValues,
				NumRows: numNeurons,
				NumCols: numWeights,
			},
			Biases: gruggrad.SerializedMatrix{
				Values:  biasValues,
				NumRows: numNeurons,
				NumCols: 1,
			},
		}

		mNetwork.Layers = append(mNetwork.Layers, mLayer)
	}

	// Marshal to JSON
	jsonData, err = json.MarshalIndent(mNetwork, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal MNetwork: %w", err)
	}

	// Write to file
	err = os.WriteFile(outputFile, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write file %s: %w", outputFile, err)
	}

	fmt.Printf("Successfully converted %s to %s\n", inputFile, outputFile)
	fmt.Printf("Layers converted: %d\n", len(mNetwork.Layers))
	for i, layer := range mNetwork.Layers {
		fmt.Printf("  Layer %d: Weights %dx%d, Biases %dx%d\n",
			i+1,
			layer.Weights.NumRows, layer.Weights.NumCols,
			layer.Biases.NumRows, layer.Biases.NumCols)
	}
	return nil
}

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: convert_network <input_network.json> <output_mnetwork.json>")
		fmt.Println("\nConverts a Network JSON file to MNetwork JSON format")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]

	err := convertNetworkToMNetwork(inputFile, outputFile)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
