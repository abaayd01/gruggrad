package gruggrad

import (
	"encoding/json"
	"fmt"
	"os"
)

type MLayer struct {
	Weights       *TrackedMatrix
	Biases        *TrackedMatrix
	IsOutputLayer bool // If true, don't apply ReLU
}

func NewRandomMLayer(numRows, numCols int) *MLayer {
	weights := NewRandomTrackedMatrix(numRows, numCols)
	return &MLayer{
		Weights:       weights,
		Biases:        NewTrackedMatrix(1, numCols),
		IsOutputLayer: false,
	}
}

func (l *MLayer) Forward(input *TrackedMatrix) *TrackedMatrix {
	preactivation := input.MatMul(l.Weights).Add(l.Biases)

	// Don't apply activation on output layer (logits should be raw)
	if l.IsOutputLayer {
		return preactivation
	}

	return preactivation.ReLU()
}

func (l *MLayer) Tune(learningRate float64) {
	// TODO: support gradient clipping
	l.Weights.Matrix = l.Weights.Matrix.Subtract(l.Weights.Gradients.MultiplyScalar(learningRate))
	l.Weights.ZeroGradients()

	l.Biases.Matrix = l.Biases.Matrix.Subtract(l.Biases.Gradients.MultiplyScalar(learningRate))
	l.Biases.ZeroGradients()
}

type MNetwork struct {
	Layers []*MLayer
}

func NewRandomMNetwork(layerDims []LayerDims) *MNetwork {
	var layers []*MLayer
	for i, layerDim := range layerDims {
		layer := NewRandomMLayer(layerDim.NumWeights, layerDim.NumNeurons)

		// Mark the last layer as output layer (no ReLU)
		if i == len(layerDims)-1 {
			layer.IsOutputLayer = true
		}

		layers = append(layers, layer)
	}
	return &MNetwork{
		Layers: layers,
	}
}

func (n *MNetwork) Forward(input *TrackedMatrix) *TrackedMatrix {
	currentInput := input
	for i := range n.Layers {
		currentInput = n.Layers[i].Forward(currentInput)
	}
	return currentInput
}

func (n *MNetwork) Tune(learningRate float64) {
	for i := range n.Layers {
		n.Layers[i].Tune(learningRate)
	}
}

func MLossXOR(expected *TrackedMatrix, received *TrackedMatrix) *TrackedMatrix {
	diff := expected.Subtract(received)
	diffT := diff.Transpose()
	return diffT.MatMul(diff)
}

type SerializedMatrix struct {
	Values  []float64 `json:"values"`
	NumRows int       `json:"numRows"`
	NumCols int       `json:"numCols"`
}

type SerializedMLayer struct {
	Weights       SerializedMatrix `json:"weights"`
	Biases        SerializedMatrix `json:"biases"`
	IsOutputLayer bool             `json:"isOutputLayer"`
}

type SerializedMNetwork struct {
	Layers []SerializedMLayer `json:"layers"`
}

func LoadMNetworkFromFile(filename string) (*MNetwork, error) {
	// Read file
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filename, err)
	}

	// Unmarshal JSON
	var serialized SerializedMNetwork
	err = json.Unmarshal(jsonData, &serialized)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Reconstruct MNetwork
	var layers []*MLayer
	for _, sLayer := range serialized.Layers {
		// Reconstruct Weights matrix
		weights := NewMatrix(sLayer.Weights.NumRows, sLayer.Weights.NumCols)
		copy(weights.Values, sLayer.Weights.Values)
		trackedWeightMatrix := NewTrackedMatrixWithValues(weights.NumRows, weights.NumCols, weights.Values)

		// Reconstruct Biases matrix
		biases := NewMatrix(sLayer.Biases.NumRows, sLayer.Biases.NumCols)
		copy(biases.Values, sLayer.Biases.Values)
		trackedBiasMatrix := NewTrackedMatrixWithValues(biases.NumRows, biases.NumCols, biases.Values)

		layers = append(layers, &MLayer{
			Weights:       trackedWeightMatrix,
			Biases:        trackedBiasMatrix,
			IsOutputLayer: sLayer.IsOutputLayer,
		})
	}

	return &MNetwork{Layers: layers}, nil
}

func (n *MNetwork) Store(filename string) error {
	// Extract parameters into SerializedMNetwork
	var serialized SerializedMNetwork
	for _, layer := range n.Layers {
		sLayer := SerializedMLayer{
			Weights: SerializedMatrix{
				Values:  layer.Weights.Values,
				NumRows: layer.Weights.NumRows,
				NumCols: layer.Weights.NumCols,
			},
			Biases: SerializedMatrix{
				Values:  layer.Biases.Values,
				NumRows: layer.Biases.NumRows,
				NumCols: layer.Biases.NumCols,
			},
			IsOutputLayer: layer.IsOutputLayer,
		}
		serialized.Layers = append(serialized.Layers, sLayer)
	}

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(serialized, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal network: %w", err)
	}

	// Write to file
	err = os.WriteFile(filename, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("failed to write file %s: %w", filename, err)
	}

	return nil
}

type MNetworkTrainingExample struct {
	Input  *TrackedMatrix
	Output *TrackedMatrix
}
