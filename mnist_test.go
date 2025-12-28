package main

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	// Test with simple inputs
	inputs := []*Value{
		NewValue(1.0),
		NewValue(2.0),
		NewValue(3.0),
	}

	outputs := Softmax(inputs)

	// Verify we got the right number of outputs
	if len(outputs) != len(inputs) {
		t.Fatalf("expected %d outputs, got %d", len(inputs), len(outputs))
	}

	// Verify all outputs are between 0 and 1
	for i, output := range outputs {
		if output.Data < 0 || output.Data > 1 {
			t.Errorf("output[%d] = %f, expected value between 0 and 1", i, output.Data)
		}
	}

	// Verify outputs sum to 1.0 (valid probability distribution)
	sum := 0.0
	for _, output := range outputs {
		sum += output.Data
	}
	if math.Abs(sum-1.0) > 1e-7 {
		t.Errorf("softmax outputs sum to %f, expected 1.0", sum)
	}

	// Verify that larger inputs produce larger probabilities
	// inputs are [1.0, 2.0, 3.0], so outputs should be increasing
	if outputs[0].Data >= outputs[1].Data {
		t.Errorf("expected output[1] > output[0], got %f <= %f", outputs[1].Data, outputs[0].Data)
	}
	if outputs[1].Data >= outputs[2].Data {
		t.Errorf("expected output[2] > output[1], got %f <= %f", outputs[2].Data, outputs[1].Data)
	}
}

func TestNeuronBackwardPropagation(t *testing.T) {
	// Test if gradients propagate through a single neuron
	weight1 := NewValue(0.5)
	weight2 := NewValue(0.3)
	bias := NewValue(0.1)

	neuron := NewNeuron([]*Value{weight1, weight2}, bias)

	input1 := NewValue(1.0)
	input2 := NewValue(2.0)

	output := neuron.Forward([]*Value{input1, input2})

	// Set gradient on output and backpropagate
	output.Gradient = 1.0
	output.Backward()

	// Check if gradients reached the weights
	t.Logf("weight1.Gradient = %.6f (expected non-zero)", weight1.Gradient)
	t.Logf("weight2.Gradient = %.6f (expected non-zero)", weight2.Gradient)
	t.Logf("bias.Gradient = %.6f (expected non-zero)", bias.Gradient)

	if weight1.Gradient == 0 {
		t.Errorf("weight1.Gradient is 0 - gradients not propagating!")
	}
	if weight2.Gradient == 0 {
		t.Errorf("weight2.Gradient is 0 - gradients not propagating!")
	}
}

func TestSimplifiedMNISTPattern(t *testing.T) {
	// Minimal test using MNIST pattern (Softmax + loss) to isolate the bug
	network := NewRandomNetwork([]LayerDims{
		{numWeights: 2, numNeurons: 4},
		{numWeights: 4, numNeurons: 3}, // 3 outputs for softmax
	})

	input := []*Value{NewValue(0.5), NewValue(0.8)}

	// Forward pass
	output := network.Forward(input)
	t.Logf("Output before softmax: [%.4f, %.4f, %.4f]", output[0].Data, output[1].Data, output[2].Data)
	t.Logf("Output[0] after ReLU: %.4f (if ≤0, gradients won't flow!)", output[0].Data)

	// Softmax
	normalisedOutput := Softmax(output)
	t.Logf("After softmax: [%.4f, %.4f, %.4f]", normalisedOutput[0].Data, normalisedOutput[1].Data, normalisedOutput[2].Data)

	// Loss on first output
	loss := LossCrossEntropy(normalisedOutput[0])
	t.Logf("Loss: %.4f", loss.Data)

	// Backward
	firstWeight := network.Layers[0].Neurons[0].Weights[0]
	t.Logf("Before backward - weight gradient: %.6f", firstWeight.Gradient)

	loss.Gradient = 1.0
	loss.Backward()

	// Trace gradient flow
	t.Logf("After backward - loss.Gradient: %.6f", loss.Gradient)
	t.Logf("After backward - normalisedOutput[0].Gradient: %.6f", normalisedOutput[0].Gradient)
	t.Logf("After backward - output[0].Gradient: %.6f", output[0].Gradient)
	if output[0].LeftParent != nil {
		preReLU := output[0].LeftParent
		t.Logf("After backward - output[0].LeftParent (preReLU) Op=%s, Gradient: %.6f", preReLU.Operation, preReLU.Gradient)

		// Walk deeper - preReLU should be an Add operation
		if preReLU.LeftParent != nil {
			t.Logf("  -> preReLU.LeftParent Op=%s, Gradient: %.6f", preReLU.LeftParent.Operation, preReLU.LeftParent.Gradient)

			// Keep walking to find a Mul operation (should contain weight)
			if preReLU.LeftParent.LeftParent != nil {
				t.Logf("    -> LeftParent.LeftParent Op=%s, Gradient: %.6f",
					preReLU.LeftParent.LeftParent.Operation, preReLU.LeftParent.LeftParent.Gradient)
			}
		}
		if preReLU.RightParent != nil {
			t.Logf("  -> preReLU.RightParent (bias) Op=%s, Gradient: %.6f", preReLU.RightParent.Operation, preReLU.RightParent.Gradient)
		}

		// The LeftParent chain should contain Mul operations with weights
		// Let me search for a Mul operation
		curr := preReLU.LeftParent
		mulFound := false
		for depth := 0; curr != nil && depth < 10; depth++ {
			if curr.Operation == "MULTIPLY" {
				t.Logf("Found MUL at depth %d, Gradient: %.6f", depth, curr.Gradient)
				if curr.LeftParent != nil {
					t.Logf("  MUL.LeftParent Op=%s, Gradient: %.6f", curr.LeftParent.Operation, curr.LeftParent.Gradient)
				}
				if curr.RightParent != nil {
					t.Logf("  MUL.RightParent Op=%s, Gradient: %.6f, Addr: %p", curr.RightParent.Operation, curr.RightParent.Gradient, curr.RightParent)
					t.Logf("  FirstWeight addr: %p (same as MUL.RightParent? %v)", firstWeight, curr.RightParent == firstWeight)
				}
				mulFound = true
				break
			}
			curr = curr.LeftParent
		}
		if !mulFound {
			t.Logf("No MUL operation found in LeftParent chain!")
		}
	}
	t.Logf("After backward - weight gradient: %.6f", firstWeight.Gradient)

	if firstWeight.Gradient == 0 {
		t.Errorf("Weight gradient is 0 - gradients not propagating!")
	}
}

func TestMilestone1(t *testing.T) {
	Milestone1()
}

func TestMilestone1Verify(t *testing.T) {
	Milestone1Verify()
}

func TestMnistMNetworkTraining(t *testing.T) {
	MnistMNetworkTraining()
}

func TestMnistMNetworkVerify(t *testing.T) {
	MnistMNetworkVerify()
}
