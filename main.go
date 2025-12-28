package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
)

func main() {
	// Test harness - uncomment to run specific tests
	// Milestone1()
}

type Operation string

const (
	OpNoop                Operation = ""
	OpAdd                 Operation = "ADD"
	OpSub                 Operation = "SUBTRACT"
	OpMul                 Operation = "MULTIPLY"
	OpDiv                 Operation = "DIVIDE"
	OpPow                 Operation = "POW"
	OpLn                  Operation = "LN"
	OpRelu                Operation = "ReLU"
	OpSoftMax             Operation = "SoftMax"
	OpSoftMaxCrossEntropy Operation = "SoftMaxCrossEntropy"
	OpLeakyRelu           Operation = "LeakyReLU"
	OpTranspose           Operation = "TRANSPOSE"
)

type Value struct {
	Data        float64
	LeftParent  *Value
	RightParent *Value
	Operation   Operation
	Gradient    float64
}

func NewValue(val float64) *Value {
	return &Value{
		Data:        val,
		LeftParent:  nil,
		RightParent: nil,
		Operation:   OpNoop,
		Gradient:    0,
	}
}

func (v *Value) Add(b *Value) *Value {
	return &Value{
		Data:        v.Data + b.Data,
		LeftParent:  v,
		RightParent: b,
		Operation:   OpAdd,
		Gradient:    0,
	}
}

func (v *Value) Sub(b *Value) *Value {
	return &Value{
		Data:        v.Data - b.Data,
		LeftParent:  v,
		RightParent: b,
		Operation:   OpSub,
		Gradient:    0,
	}
}

func (v *Value) Mul(b *Value) *Value {
	return &Value{
		Data:        v.Data * b.Data,
		LeftParent:  v,
		RightParent: b,
		Operation:   OpMul,
		Gradient:    0,
	}
}

func (v *Value) Div(b *Value) *Value {
	return &Value{
		Data:        v.Data / b.Data,
		LeftParent:  v,
		RightParent: b,
		Operation:   OpDiv,
		Gradient:    0,
	}
}

func (v *Value) Pow(b *Value) *Value {
	return &Value{
		Data:        math.Pow(v.Data, b.Data),
		LeftParent:  v,
		RightParent: b,
		Operation:   OpPow,
		Gradient:    0,
	}
}

func (v *Value) Ln() *Value {
	return &Value{
		Data:        math.Log(v.Data),
		LeftParent:  v,
		RightParent: nil,
		Operation:   OpLn,
		Gradient:    0,
	}
}

func (v *Value) ReLU() *Value {
	data := v.Data
	if v.Data <= 0 {
		data = 0
	}
	return &Value{
		Data:        data,
		LeftParent:  v,
		RightParent: nil,
		Operation:   OpRelu,
		Gradient:    0,
	}
}

func (v *Value) LeakyReLU(alpha float64) *Value {
	data := v.Data
	if v.Data <= 0 {
		data = alpha * v.Data
	}
	return &Value{
		Data:        data,
		LeftParent:  v,
		RightParent: NewValue(alpha), // Store alpha for backward pass
		Operation:   OpLeakyRelu,
		Gradient:    0,
	}
}

// buildTopologicalOrder builds a reverse topological ordering of the computation graph
func buildTopologicalOrder(v *Value) []*Value {
	visited := make(map[*Value]bool)
	var topo []*Value

	var build func(*Value)
	build = func(node *Value) {
		if node == nil || visited[node] {
			return
		}
		visited[node] = true

		// Visit children first (post-order traversal)
		build(node.LeftParent)
		build(node.RightParent)

		// Add to list after visiting children
		topo = append(topo, node)
	}

	build(v)
	return topo
}

// Backward performs backpropagation using topological ordering to visit each node exactly once
func (v *Value) Backward() {
	// Build topological order
	topo := buildTopologicalOrder(v)

	// Process in reverse order (from output to inputs)
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]

		// Compute local gradients based on operation
		switch node.Operation {
		case OpNoop:
			// Leaf node, no parents to propagate to
			continue
		case OpAdd:
			node.LeftParent.Gradient += 1 * node.Gradient
			node.RightParent.Gradient += 1 * node.Gradient
		case OpSub:
			node.LeftParent.Gradient += 1 * node.Gradient
			node.RightParent.Gradient += -1 * node.Gradient
		case OpMul:
			node.LeftParent.Gradient += node.RightParent.Data * node.Gradient
			node.RightParent.Gradient += node.LeftParent.Data * node.Gradient
		case OpDiv:
			node.LeftParent.Gradient += (1 / node.RightParent.Data) * node.Gradient
			node.RightParent.Gradient += (-node.LeftParent.Data / (node.RightParent.Data * node.RightParent.Data)) * node.Gradient
		case OpPow:
			node.LeftParent.Gradient += (node.RightParent.Data * math.Pow(node.LeftParent.Data, node.RightParent.Data-1.0)) * node.Gradient
			node.RightParent.Gradient += (math.Pow(node.LeftParent.Data, node.RightParent.Data) * math.Log(node.LeftParent.Data)) * node.Gradient
		case OpLn:
			if node.LeftParent.Data < 0 {
				panic("derivative undefined for ln(y) where y < 0")
			}
			node.LeftParent.Gradient += 1 / node.LeftParent.Data * node.Gradient
		case OpRelu:
			if node.Data <= 0 {
				// Gradient is 0 when input <= 0
				node.LeftParent.Gradient += 0
			} else {
				// Gradient is 1 when input > 0
				node.LeftParent.Gradient += 1 * node.Gradient
			}
		case OpLeakyRelu:
			alpha := node.RightParent.Data
			if node.Data <= 0 {
				// Negative input: gradient is alpha
				node.LeftParent.Gradient += alpha * node.Gradient
			} else {
				// Positive input: gradient is 1
				node.LeftParent.Gradient += 1 * node.Gradient
			}
		default:
			panic("unknown op " + node.Operation)
		}
	}
}

func WalkRecur(val *Value, depth int) {
	if val == nil {
		return
	}

	PrintValue(val, depth)
	WalkRecur(val.LeftParent, depth+1)
	fmt.Println("")
	fmt.Print(strings.Repeat("\t", depth))
	WalkRecur(val.RightParent, depth+1)
}

func PrintValue(val *Value, depth int) {
	var str strings.Builder
	fmt.Fprintf(&str, "Data=%.4f, Grad=%.4f", val.Data, val.Gradient)
	if val.Operation != OpNoop {
		str.WriteString(" ->\n")
		str.WriteString(strings.Repeat("\t", depth))
		str.WriteString(string(val.Operation))
		str.WriteString("\n")
		str.WriteString(strings.Repeat("\t", depth))
	}

	fmt.Printf("%s", str.String())
}

type Neuron struct {
	Weights []*Value
	Bias    *Value
	Value   *Value
}

func NewNeuron(weights []*Value, bias *Value) *Neuron {
	return &Neuron{
		Weights: weights,
		Bias:    bias,
	}
}

func (n *Neuron) Forward(inputs []*Value) *Value {
	if len(inputs) != len(n.Weights) {
		panic("mismatched dims in Neuron!")
	}
	result := NewValue(0.0)
	for i := range inputs {
		w := n.Weights[i]
		input := inputs[i]
		result = result.Add(input.Mul(w))
	}
	result = result.Add(n.Bias)
	result = result.ReLU()
	// result = result.LeakyReLU(0.01) // Use LeakyReLU with alpha=0.01
	return result
}

func clipGradient(gradient, maxGrad float64) float64 {
	if gradient > maxGrad {
		return maxGrad
	}
	if gradient < -maxGrad {
		return -maxGrad
	}
	return gradient
}

func (n *Neuron) Tune(learningRate float64) {
	maxGrad := 1.0 // Clip gradients to [-1.0, 1.0]

	for i := range n.Weights {
		clippedGrad := clipGradient(n.Weights[i].Gradient, maxGrad)
		n.Weights[i].Data = n.Weights[i].Data - learningRate*clippedGrad
		n.Weights[i].Gradient = 0
	}

	clippedBiasGrad := clipGradient(n.Bias.Gradient, maxGrad)
	n.Bias.Data = n.Bias.Data - learningRate*clippedBiasGrad
	n.Bias.Gradient = 0
}

type Layer struct {
	Neurons []Neuron
}

func NewLayer(neurons []Neuron) *Layer {
	return &Layer{
		Neurons: neurons,
	}
}

func NewRandomLayer(countWeights, countNeurons int) *Layer {
	var neurons []Neuron
	for range countNeurons {
		var weights []*Value
		for range countWeights {

			weightVal := rand.NormFloat64() * math.Sqrt(2.0/float64(countWeights))
			weights = append(weights, NewValue(weightVal))
		}
		neurons = append(neurons, *NewNeuron(weights, NewValue(0.0)))
	}
	return NewLayer(neurons)
}

func (l *Layer) Forward(input []*Value) []*Value {
	var result []*Value
	for i := range l.Neurons {
		result = append(result, l.Neurons[i].Forward(input))
	}
	return result
}

func (l *Layer) Tune(learningRate float64) {
	for i := range l.Neurons {
		l.Neurons[i].Tune(learningRate)
	}
}

type Network struct {
	Layers []*Layer
}

func NewNetwork(layers []*Layer) *Network {
	return &Network{
		Layers: layers,
	}
}

type LayerDims struct {
	numWeights int
	numNeurons int
}

type SerializedNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type SerializedLayer struct {
	Neurons []SerializedNeuron `json:"neurons"`
}

type SerializedNetwork struct {
	Layers []SerializedLayer `json:"layers"`
}

func LoadNetworkFromFile(filename string) (*Network, error) {
	// Read file
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filename, err)
	}

	// Unmarshal JSON
	var serialized SerializedNetwork
	err = json.Unmarshal(jsonData, &serialized)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Reconstruct Network
	var layers []*Layer
	for _, sLayer := range serialized.Layers {
		var neurons []Neuron
		for _, sNeuron := range sLayer.Neurons {
			var weights []*Value
			for _, w := range sNeuron.Weights {
				weights = append(weights, NewValue(w))
			}
			bias := NewValue(sNeuron.Bias)
			neurons = append(neurons, *NewNeuron(weights, bias))
		}
		layers = append(layers, NewLayer(neurons))
	}

	return NewNetwork(layers), nil
}

func NewRandomNetwork(layerDims []LayerDims) *Network {
	var layers []*Layer
	for _, layerDim := range layerDims {
		layers = append(layers, NewRandomLayer(layerDim.numWeights, layerDim.numNeurons))
	}
	return NewNetwork(layers)
}

func (n *Network) Forward(input []*Value) []*Value {
	currentInput := input
	for i := range n.Layers {
		currentInput = n.Layers[i].Forward(currentInput)
	}
	return currentInput
}

func (n *Network) Tune(learningRate float64) {
	// somehow the loss value needs to work backwards into the network layers? -> this is done by using pointers everywhere
	// weight = weight - learningRate * gradient
	// bias = bias - learningRate * gradient

	for i := range n.Layers {
		n.Layers[i].Tune(learningRate)
	}
}

func (n *Network) Store(filename string) error {
	// Extract parameters into SerializedNetwork
	var serialized SerializedNetwork
	for _, layer := range n.Layers {
		var sNeurons []SerializedNeuron
		for _, neuron := range layer.Neurons {
			var weights []float64
			for _, weight := range neuron.Weights {
				weights = append(weights, weight.Data)
			}
			sNeurons = append(sNeurons, SerializedNeuron{
				Weights: weights,
				Bias:    neuron.Bias.Data,
			})
		}
		serialized.Layers = append(serialized.Layers, SerializedLayer{
			Neurons: sNeurons,
		})
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

func LossXOR(expected *Value, actual *Value) *Value {
	result := expected.Sub(actual).Pow(NewValue(2))
	return result
}

type TrainingExample struct {
	input  []*Value
	output *Value
}
