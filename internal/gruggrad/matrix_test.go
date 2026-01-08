package gruggrad

import (
	"fmt"
	"math"
	"testing"
)

// TestMatMulBackward_SimpleCase tests basic matrix multiplication backprop
// A (2x2) * B (2x2) = C (2x2)
func TestMatMulBackward_SimpleCase(t *testing.T) {
	// A = [[1, 2],
	//      [3, 4]]
	a := NewTrackedMatrixWithValues(2, 2, []float64{1, 2, 3, 4})

	// B = [[5, 6],
	//      [7, 8]]
	b := NewTrackedMatrixWithValues(2, 2, []float64{5, 6, 7, 8})

	// C = A * B = [[19, 22],
	//              [43, 50]]
	c := a.MatMul(b)

	// Set upstream gradient dL/dC = [[1, 1],
	//                                 [1, 1]]
	c.Gradients.Values = []float64{1, 1, 1, 1}

	// Run backprop
	c.Backward()

	// Expected gradients:
	// dL/dA = dL/dC * B^T = [[1, 1],   * [[5, 7],   = [[11, 15],
	//                        [1, 1]]      [6, 8]]      [11, 15]]
	expectedGradA := []float64{11, 15, 11, 15}

	// dL/dB = A^T * dL/dC = [[1, 3],   * [[1, 1],   = [[4, 4],
	//                        [2, 4]]      [1, 1]]      [6, 6]]
	expectedGradB := []float64{4, 4, 6, 6}

	// Verify gradients for A
	for i := range expectedGradA {
		if !FloatEq(a.Gradients.Values[i], expectedGradA[i]) {
			t.Errorf("Gradient for A at index %d: expected %.2f, got %.2f",
				i, expectedGradA[i], a.Gradients.Values[i])
		}
	}

	// Verify gradients for B
	for i := range expectedGradB {
		if !FloatEq(b.Gradients.Values[i], expectedGradB[i]) {
			t.Errorf("Gradient for B at index %d: expected %.2f, got %.2f",
				i, expectedGradB[i], b.Gradients.Values[i])
		}
	}
}

// TestMatMulBackward_NonSquareMatrices tests backprop with non-square matrices
// A (2x3) * B (3x2) = C (2x2)
func TestMatMulBackward_NonSquareMatrices(t *testing.T) {
	// A = [[1, 2, 3],
	//      [4, 5, 6]]  (2x3)
	a := NewTrackedMatrixWithValues(2, 3, []float64{1, 2, 3, 4, 5, 6})

	// B = [[1, 2],
	//      [3, 4],
	//      [5, 6]]  (3x2)
	b := NewTrackedMatrixWithValues(3, 2, []float64{1, 2, 3, 4, 5, 6})

	// C = A * B (2x2)
	c := a.MatMul(b)

	// Set upstream gradient dL/dC = [[1, 0],
	//                                 [0, 1]]
	c.Gradients.Values = []float64{1, 0, 0, 1}

	// Run backprop
	c.Backward()

	// Expected gradients:
	// dL/dA = dL/dC * B^T = [[1, 0],   * [[1, 3, 5],   = [[1, 3, 5],
	//                        [0, 1]]      [2, 4, 6]]      [2, 4, 6]]
	expectedGradA := []float64{1, 3, 5, 2, 4, 6}

	// dL/dB = A^T * dL/dC = [[1, 4],   * [[1, 0],   = [[1, 4],
	//                        [2, 5],      [0, 1]]      [2, 5],
	//                        [3, 6]]                    [3, 6]]
	expectedGradB := []float64{1, 4, 2, 5, 3, 6}

	// Verify gradients for A
	for i := range expectedGradA {
		if !FloatEq(a.Gradients.Values[i], expectedGradA[i]) {
			t.Errorf("Gradient for A at index %d: expected %.2f, got %.2f",
				i, expectedGradA[i], a.Gradients.Values[i])
		}
	}

	// Verify gradients for B
	for i := range expectedGradB {
		if !FloatEq(b.Gradients.Values[i], expectedGradB[i]) {
			t.Errorf("Gradient for B at index %d: expected %.2f, got %.2f",
				i, expectedGradB[i], b.Gradients.Values[i])
		}
	}
}

// TestMatMulBackward_VectorMatrix tests backprop with vector-matrix multiplication
// A (1x3) * B (3x2) = C (1x2)
func TestMatMulBackward_VectorMatrix(t *testing.T) {
	// A = [[2, 3, 4]]  (1x3 row vector)
	a := NewTrackedMatrixWithValues(1, 3, []float64{2, 3, 4})

	// B = [[1, 2],
	//      [3, 4],
	//      [5, 6]]  (3x2)
	b := NewTrackedMatrixWithValues(3, 2, []float64{1, 2, 3, 4, 5, 6})

	// C = A * B = [[31, 40]]  (1x2)
	c := a.MatMul(b)

	// Set upstream gradient dL/dC = [[2, 3]]
	c.Gradients.Values = []float64{2, 3}

	// Run backprop
	c.Backward()

	// Expected gradients:
	// dL/dA = dL/dC * B^T = [[2, 3]] * [[1, 3, 5],   = [[8, 18, 28]]
	//                                   [2, 4, 6]]
	expectedGradA := []float64{8, 18, 28}

	// dL/dB = A^T * dL/dC = [[2],   * [[2, 3]] = [[4, 6],
	//                        [3],                  [6, 9],
	//                        [4]]                  [8, 12]]
	expectedGradB := []float64{4, 6, 6, 9, 8, 12}

	// Verify gradients for A
	for i := range expectedGradA {
		if !FloatEq(a.Gradients.Values[i], expectedGradA[i]) {
			t.Errorf("Gradient for A at index %d: expected %.2f, got %.2f",
				i, expectedGradA[i], a.Gradients.Values[i])
		}
	}

	// Verify gradients for B
	for i := range expectedGradB {
		if !FloatEq(b.Gradients.Values[i], expectedGradB[i]) {
			t.Errorf("Gradient for B at index %d: expected %.2f, got %.2f",
				i, expectedGradB[i], b.Gradients.Values[i])
		}
	}
}

// TestMatMulBackward_ChainedOperations tests backprop through multiple matmuls
// (A * B) * C
func TestMatMulBackward_ChainedOperations(t *testing.T) {
	// A = [[1, 2]]  (1x2)
	a := NewTrackedMatrixWithValues(1, 2, []float64{1, 2})

	// B = [[3, 4],
	//      [5, 6]]  (2x2)
	b := NewTrackedMatrixWithValues(2, 2, []float64{3, 4, 5, 6})

	// C = [[1],
	//      [2]]  (2x1)
	c := NewTrackedMatrixWithValues(2, 1, []float64{1, 2})

	// D = (A * B) * C
	// A * B = [[13, 16]]  (1x2)
	// D = [[13, 16]] * [[1], [2]] = [[45]]  (1x1)
	d := a.MatMul(b).MatMul(c)

	// Set upstream gradient dL/dD = [[1]]
	d.Gradients.Values = []float64{1}

	// Run backprop
	d.Backward()

	// Expected gradients:
	// Working backwards through the chain:
	// Let E = A * B, so D = E * C
	// dL/dE = dL/dD * C^T = [[1]] * [[1, 2]] = [[1, 2]]
	// dL/dC = E^T * dL/dD = [[13], [16]] * [[1]] = [[13], [16]]
	expectedGradC := []float64{13, 16}

	// Now for A and B:
	// dL/dA = dL/dE * B^T = [[1, 2]] * [[3, 5], [4, 6]] = [[11, 17]]
	expectedGradA := []float64{11, 17}

	// dL/dB = A^T * dL/dE = [[1], [2]] * [[1, 2]] = [[1, 2], [2, 4]]
	expectedGradB := []float64{1, 2, 2, 4}

	// Verify gradients for A
	for i := range expectedGradA {
		if !FloatEq(a.Gradients.Values[i], expectedGradA[i]) {
			t.Errorf("Gradient for A at index %d: expected %.2f, got %.2f",
				i, expectedGradA[i], a.Gradients.Values[i])
		}
	}

	// Verify gradients for B
	for i := range expectedGradB {
		if !FloatEq(b.Gradients.Values[i], expectedGradB[i]) {
			t.Errorf("Gradient for B at index %d: expected %.2f, got %.2f",
				i, expectedGradB[i], b.Gradients.Values[i])
		}
	}

	// Verify gradients for C
	for i := range expectedGradC {
		if !FloatEq(c.Gradients.Values[i], expectedGradC[i]) {
			t.Errorf("Gradient for C at index %d: expected %.2f, got %.2f",
				i, expectedGradC[i], c.Gradients.Values[i])
		}
	}
}

// TestMatMulBackward_DifferentUpstreamGradients tests with various gradient flows
func TestMatMulBackward_DifferentUpstreamGradients(t *testing.T) {
	// A = [[1, 2]]  (1x2)
	a := NewTrackedMatrixWithValues(1, 2, []float64{1, 2})

	// B = [[3, 4],
	//      [5, 6]]  (2x2)
	b := NewTrackedMatrixWithValues(2, 2, []float64{3, 4, 5, 6})

	// C = A * B = [[13, 16]]  (1x2)
	c := a.MatMul(b)

	// Set upstream gradient dL/dC = [[0.5, 1.5]]
	c.Gradients.Values = []float64{0.5, 1.5}

	// Run backprop
	c.Backward()

	// Expected gradients:
	// dL/dA = dL/dC * B^T = [[0.5, 1.5]] * [[3, 5], [4, 6]] = [[7.5, 11.5]]
	expectedGradA := []float64{7.5, 11.5}

	// dL/dB = A^T * dL/dC = [[1], [2]] * [[0.5, 1.5]] = [[0.5, 1.5], [1.0, 3.0]]
	expectedGradB := []float64{0.5, 1.5, 1.0, 3.0}

	// Verify gradients for A
	for i := range expectedGradA {
		if !FloatEq(a.Gradients.Values[i], expectedGradA[i]) {
			t.Errorf("Gradient for A at index %d: expected %.2f, got %.2f",
				i, expectedGradA[i], a.Gradients.Values[i])
		}
	}

	// Verify gradients for B
	for i := range expectedGradB {
		if !FloatEq(b.Gradients.Values[i], expectedGradB[i]) {
			t.Errorf("Gradient for B at index %d: expected %.2f, got %.2f",
				i, expectedGradB[i], b.Gradients.Values[i])
		}
	}
}

func TestMatrixSet(t *testing.T) {
	a := NewMatrix(2, 2)
	fmt.Printf("a:\n%s\n", a.String())
	a.Set(0, 1, 999)
	fmt.Printf("a:\n%s\n", a.String())
}

func TestAdd(t *testing.T) {
	a := NewTrackedMatrixWithValues(2, 2, []float64{1, 2, 3, 4})
	b := NewTrackedMatrixWithValues(2, 2, []float64{4, 5, 6, 7})
	output := a.Add(b)

	fmt.Printf("a:\n%s\n\nb:\n%s\n\noutput:\n%s\n\n", a.String(), b.String(), output.String())
	output.Gradients.Values = []float64{1, 1, 1, 1}
	output.Backward()

	fmt.Printf("gradA:\n%s\n\ngradB:\n%s\n\ngradOutput:\n%s\n\n", a.Gradients.String(), b.Gradients.String(), output.Gradients.String())
}

func TestMatrixReLU(t *testing.T) {
	a := NewTrackedMatrixWithValues(2, 2, []float64{1, 2, 3, 4})
	b := NewTrackedMatrixWithValues(2, 2, []float64{-4, 5, -6, 7})
	output := a.Add(b).ReLU()

	fmt.Printf("a:\n%s\n\nb:\n%s\n\noutput:\n%s\n\n", a.String(), b.String(), output.String())
	output.Gradients.Values = []float64{1, 1, 1, 1}
	output.Backward()

	fmt.Printf("gradA:\n%s\n\ngradB:\n%s\n\ngradOutput:\n%s\n\n", a.Gradients.String(), b.Gradients.String(), output.Gradients.String())
}

// TestSoftMaxBackward_SingleRow tests softmax backprop with a single row
func TestSoftMaxBackward_SingleRow(t *testing.T) {
	// Input: [[1.0, 2.0, 3.0]]
	input := NewTrackedMatrixWithValues(1, 3, []float64{1.0, 2.0, 3.0})

	// Apply softmax
	output := input.SoftMax()

	// Set upstream gradient (gradient flowing from loss)
	// Using [1, 0, 0] to test non-uniform gradient flow
	output.Gradients.Values = []float64{1.0, 0.0, 0.0}

	// Run backprop
	output.Backward()

	// Verify using numerical gradient checking
	h := 1e-5 // small perturbation
	tolerance := 1e-4

	for i := 0; i < 3; i++ {
		// Compute numerical gradient for input.Values[i]
		originalVal := input.Values[i]

		// f(x + h)
		input.Values[i] = originalVal + h
		outputPlus := input.Matrix.RowMap(SoftMaxFn)
		lossPlus := outputPlus.Values[0]*1.0 + outputPlus.Values[1]*0.0 + outputPlus.Values[2]*0.0

		// f(x - h)
		input.Values[i] = originalVal - h
		outputMinus := input.Matrix.RowMap(SoftMaxFn)
		lossMinus := outputMinus.Values[0]*1.0 + outputMinus.Values[1]*0.0 + outputMinus.Values[2]*0.0

		// Restore original value
		input.Values[i] = originalVal

		// Numerical gradient: (f(x+h) - f(x-h)) / (2h)
		numericalGrad := (lossPlus - lossMinus) / (2 * h)
		analyticalGrad := input.Gradients.Values[i]

		diff := numericalGrad - analyticalGrad
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			t.Errorf("Gradient mismatch at index %d: numerical=%.6f, analytical=%.6f, diff=%.6f",
				i, numericalGrad, analyticalGrad, diff)
		} else {
			t.Logf("✓ Gradient at index %d: numerical=%.6f, analytical=%.6f (diff=%.8f)",
				i, numericalGrad, analyticalGrad, diff)
		}
	}
}

// TestSoftMaxBackward_MultipleRows tests softmax backprop with batched inputs
func TestSoftMaxBackward_MultipleRows(t *testing.T) {
	// Input: [[1.0, 2.0, 3.0],
	//         [0.5, 1.5, 2.5]]
	input := NewTrackedMatrixWithValues(2, 3, []float64{1.0, 2.0, 3.0, 0.5, 1.5, 2.5})

	// Apply softmax (row-wise)
	output := input.SoftMax()

	// Set upstream gradient
	// Row 0: [1, 0, 0], Row 1: [0, 1, 0]
	output.Gradients.Values = []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}

	// Run backprop
	output.Backward()

	// Verify using numerical gradient checking
	h := 1e-5
	tolerance := 1e-4

	for i := 0; i < 6; i++ {
		row := i / 3
		col := i % 3

		// Compute numerical gradient for input.Values[i]
		originalVal := input.Values[i]

		// f(x + h)
		input.Values[i] = originalVal + h
		outputPlus := input.Matrix.RowMap(SoftMaxFn)
		lossPlus := outputPlus.Values[0]*1.0 + outputPlus.Values[1]*0.0 + outputPlus.Values[2]*0.0 +
			outputPlus.Values[3]*0.0 + outputPlus.Values[4]*1.0 + outputPlus.Values[5]*0.0

		// f(x - h)
		input.Values[i] = originalVal - h
		outputMinus := input.Matrix.RowMap(SoftMaxFn)
		lossMinus := outputMinus.Values[0]*1.0 + outputMinus.Values[1]*0.0 + outputMinus.Values[2]*0.0 +
			outputMinus.Values[3]*0.0 + outputMinus.Values[4]*1.0 + outputMinus.Values[5]*0.0

		// Restore original value
		input.Values[i] = originalVal

		// Numerical gradient
		numericalGrad := (lossPlus - lossMinus) / (2 * h)
		analyticalGrad := input.Gradients.Values[i]

		diff := numericalGrad - analyticalGrad
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			t.Errorf("Gradient mismatch at [%d,%d] (index %d): numerical=%.6f, analytical=%.6f, diff=%.6f",
				row, col, i, numericalGrad, analyticalGrad, diff)
		} else {
			t.Logf("✓ Gradient at [%d,%d]: numerical=%.6f, analytical=%.6f (diff=%.8f)",
				row, col, numericalGrad, analyticalGrad, diff)
		}
	}
}

// TestSoftMaxBackward_UniformUpstream tests with uniform upstream gradients (all 1s)
func TestSoftMaxBackward_UniformUpstream(t *testing.T) {
	// Input: [[2.0, 1.0, 0.1]]
	input := NewTrackedMatrixWithValues(1, 3, []float64{2.0, 1.0, 0.1})

	// Apply softmax
	output := input.SoftMax()

	// Set uniform upstream gradient [1, 1, 1]
	output.Gradients.Values = []float64{1.0, 1.0, 1.0}

	// Run backprop
	output.Backward()

	// With uniform upstream gradients on softmax, the input gradients should be close to zero
	// because softmax outputs sum to 1, and uniform pull doesn't change the distribution
	// Specifically: grad[i] = output[i] * (1 - sum(output)) = output[i] * (1 - 1) = 0

	tolerance := 1e-6
	for i := 0; i < 3; i++ {
		if input.Gradients.Values[i] > tolerance || input.Gradients.Values[i] < -tolerance {
			t.Errorf("Expected near-zero gradient with uniform upstream at index %d, got %.8f",
				i, input.Gradients.Values[i])
		} else {
			t.Logf("✓ Gradient at index %d is near-zero: %.10f", i, input.Gradients.Values[i])
		}
	}
}

// TestSoftMaxCrossEntropy_SingleRow tests the combined operation with one example
func TestSoftMaxCrossEntropy_SingleRow(t *testing.T) {
	// Input logits: [[2.0, 1.0, 0.1]] (3 classes)
	// Target: class 0
	input := NewTrackedMatrixWithValues(1, 3, []float64{2.0, 1.0, 0.1})
	target := []int{0}

	// Apply combined softmax + cross-entropy
	loss := input.SoftMaxCrossEntropy(target)

	// Verify loss is positive and reasonable
	if loss.Values[0] <= 0 {
		t.Errorf("Expected positive loss, got %.6f", loss.Values[0])
	}
	t.Logf("Loss: %.6f", loss.Values[0])

	// Set upstream gradient (typically 1.0 for final loss)
	loss.Gradients.Values[0] = 1.0

	// Run backprop
	loss.Backward()

	// Verify gradients using numerical gradient checking
	h := 1e-5
	tolerance := 1e-4

	for i := 0; i < 3; i++ {
		originalVal := input.Values[i]

		// f(x + h)
		input.Values[i] = originalVal + h
		lossPlus := input.SoftMaxCrossEntropy(target)

		// f(x - h)
		input.Values[i] = originalVal - h
		lossMinus := input.SoftMaxCrossEntropy(target)

		// Restore
		input.Values[i] = originalVal

		numericalGrad := (lossPlus.Values[0] - lossMinus.Values[0]) / (2 * h)
		analyticalGrad := input.Gradients.Values[i]

		diff := numericalGrad - analyticalGrad
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			t.Errorf("Gradient mismatch at index %d: numerical=%.6f, analytical=%.6f, diff=%.6f",
				i, numericalGrad, analyticalGrad, diff)
		} else {
			t.Logf("✓ Gradient at index %d: numerical=%.6f, analytical=%.6f (diff=%.8f)",
				i, numericalGrad, analyticalGrad, diff)
		}
	}
}

// TestSoftMaxCrossEntropy_Batch tests with multiple examples
func TestSoftMaxCrossEntropy_Batch(t *testing.T) {
	// Input logits: [[2.0, 1.0, 0.1],    (target: 0)
	//                [0.5, 2.5, 1.0]]    (target: 1)
	input := NewTrackedMatrixWithValues(2, 3, []float64{2.0, 1.0, 0.1, 0.5, 2.5, 1.0})
	targets := []int{0, 1}

	// Apply combined softmax + cross-entropy
	loss := input.SoftMaxCrossEntropy(targets)

	// Verify loss is positive
	if loss.Values[0] <= 0 {
		t.Errorf("Expected positive loss, got %.6f", loss.Values[0])
	}
	t.Logf("Batch loss: %.6f", loss.Values[0])

	// Set upstream gradient
	loss.Gradients.Values[0] = 1.0

	// Run backprop
	loss.Backward()

	// Verify gradients using numerical gradient checking
	h := 1e-5
	tolerance := 1e-4

	for i := 0; i < 6; i++ {
		row := i / 3
		col := i % 3

		originalVal := input.Values[i]

		// f(x + h)
		input.Values[i] = originalVal + h
		lossPlus := input.SoftMaxCrossEntropy(targets)

		// f(x - h)
		input.Values[i] = originalVal - h
		lossMinus := input.SoftMaxCrossEntropy(targets)

		// Restore
		input.Values[i] = originalVal

		numericalGrad := (lossPlus.Values[0] - lossMinus.Values[0]) / (2 * h)
		analyticalGrad := input.Gradients.Values[i]

		diff := numericalGrad - analyticalGrad
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			t.Errorf("Gradient mismatch at [%d,%d]: numerical=%.6f, analytical=%.6f, diff=%.6f",
				row, col, numericalGrad, analyticalGrad, diff)
		} else {
			t.Logf("✓ Gradient at [%d,%d]: numerical=%.6f, analytical=%.6f (diff=%.8f)",
				row, col, numericalGrad, analyticalGrad, diff)
		}
	}
}

// TestSoftMaxCrossEntropy_GradientFormula verifies the simple gradient formula
func TestSoftMaxCrossEntropy_GradientFormula(t *testing.T) {
	// Simple test to verify gradient = softmax - one_hot
	input := NewTrackedMatrixWithValues(1, 3, []float64{1.0, 2.0, 3.0})
	targets := []int{2} // Target is class 2

	loss := input.SoftMaxCrossEntropy(targets)
	loss.Gradients.Values[0] = 1.0
	loss.Backward()

	// Manually compute softmax
	row := input.Values
	maxVal := 3.0
	exp0 := math.Exp(row[0] - maxVal)
	exp1 := math.Exp(row[1] - maxVal)
	exp2 := math.Exp(row[2] - maxVal)
	sumExp := exp0 + exp1 + exp2

	softmax0 := exp0 / sumExp
	softmax1 := exp1 / sumExp
	softmax2 := exp2 / sumExp

	// Expected gradients (averaged by batch size = 1)
	expectedGrad0 := softmax0 / 1.0         // softmax - 0
	expectedGrad1 := softmax1 / 1.0         // softmax - 0
	expectedGrad2 := (softmax2 - 1.0) / 1.0 // softmax - 1

	tolerance := 1e-6
	if !FloatEq(input.Gradients.Values[0], expectedGrad0) {
		t.Errorf("Grad[0]: expected %.6f, got %.6f", expectedGrad0, input.Gradients.Values[0])
	}
	if !FloatEq(input.Gradients.Values[1], expectedGrad1) {
		t.Errorf("Grad[1]: expected %.6f, got %.6f", expectedGrad1, input.Gradients.Values[1])
	}
	if !FloatEq(input.Gradients.Values[2], expectedGrad2) {
		t.Errorf("Grad[2]: expected %.6f, got %.6f", expectedGrad2, input.Gradients.Values[2])
	}

	// Verify gradient sum is close to 0 (due to softmax constraint)
	gradSum := input.Gradients.Values[0] + input.Gradients.Values[1] + input.Gradients.Values[2]
	if gradSum > tolerance || gradSum < -tolerance {
		t.Logf("Note: Gradient sum is %.10f (expected ~0 due to softmax constraint)", gradSum)
	}

	t.Logf("✓ Gradients match formula: [%.6f, %.6f, %.6f]",
		input.Gradients.Values[0], input.Gradients.Values[1], input.Gradients.Values[2])
}

// TestMatrixLeakyReLU tests the forward pass of LeakyReLU for TrackedMatrix
func TestMatrixLeakyReLU(t *testing.T) {
	alpha := 0.01
	input := NewTrackedMatrixWithValues(2, 3, []float64{1.0, -2.0, 3.0, -0.5, 0.0, 2.5})

	output := input.LeakyReLU(alpha)

	// Expected: positive values unchanged, negative values multiplied by alpha
	expected := []float64{1.0, -0.02, 3.0, -0.005, 0.0, 2.5}

	for i := range expected {
		if !FloatEq(output.Values[i], expected[i]) {
			t.Errorf("Output[%d]: expected %.6f, got %.6f", i, expected[i], output.Values[i])
		}
	}

	t.Logf("✓ LeakyReLU forward pass correct")
}

// TestMatrixLeakyReLUBackward tests backprop through LeakyReLU for TrackedMatrix
func TestMatrixLeakyReLUBackward(t *testing.T) {
	alpha := 0.1
	input := NewTrackedMatrixWithValues(1, 4, []float64{2.0, -1.0, 0.5, -3.0})

	output := input.LeakyReLU(alpha)

	// Set upstream gradients
	output.Gradients.Values = []float64{1.0, 1.0, 1.0, 1.0}

	// Run backprop
	output.Backward()

	// Expected gradients:
	// input[0] = 2.0 > 0  => grad = 1.0
	// input[1] = -1.0 < 0 => grad = alpha = 0.1
	// input[2] = 0.5 > 0  => grad = 1.0
	// input[3] = -3.0 < 0 => grad = alpha = 0.1
	expectedGrads := []float64{1.0, 0.1, 1.0, 0.1}

	for i := range expectedGrads {
		if !FloatEq(input.Gradients.Values[i], expectedGrads[i]) {
			t.Errorf("Gradient[%d]: expected %.6f, got %.6f",
				i, expectedGrads[i], input.Gradients.Values[i])
		}
	}

	t.Logf("✓ LeakyReLU backward pass correct")
}

// TestMatrixLeakyReLUBackward_NumericalCheck verifies LeakyReLU gradients numerically
func TestMatrixLeakyReLUBackward_NumericalCheck(t *testing.T) {
	alpha := 0.01
	input := NewTrackedMatrixWithValues(1, 3, []float64{1.5, -0.5, 0.1})

	output := input.LeakyReLU(alpha)
	output.Gradients.Values = []float64{1.0, 2.0, 0.5}

	output.Backward()

	// Numerical gradient checking
	h := 1e-5
	tolerance := 1e-4

	for i := 0; i < 3; i++ {
		originalVal := input.Values[i]

		// f(x + h)
		input.Values[i] = originalVal + h
		outputPlus := input.LeakyReLU(alpha)
		lossPlus := outputPlus.Values[0]*1.0 + outputPlus.Values[1]*2.0 + outputPlus.Values[2]*0.5

		// f(x - h)
		input.Values[i] = originalVal - h
		outputMinus := input.LeakyReLU(alpha)
		lossMinus := outputMinus.Values[0]*1.0 + outputMinus.Values[1]*2.0 + outputMinus.Values[2]*0.5

		// Restore
		input.Values[i] = originalVal

		numericalGrad := (lossPlus - lossMinus) / (2 * h)
		analyticalGrad := input.Gradients.Values[i]

		diff := numericalGrad - analyticalGrad
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			t.Errorf("Gradient mismatch at index %d: numerical=%.6f, analytical=%.6f, diff=%.6f",
				i, numericalGrad, analyticalGrad, diff)
		} else {
			t.Logf("✓ Gradient at index %d: numerical=%.6f, analytical=%.6f (diff=%.8f)",
				i, numericalGrad, analyticalGrad, diff)
		}
	}
}

// TestClipGradients tests gradient clipping by value
func TestClipGradients(t *testing.T) {
	// Create a matrix with some gradients
	m := NewTrackedMatrixWithValues(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	// Set gradients with some large values
	m.Gradients.Values = []float64{5.0, -10.0, 2.0, -0.5, 8.0, -3.0}

	// Clip to [-3.0, 3.0]
	threshold := 3.0
	m.ClipGradients(threshold)

	// Expected: values clamped to [-3, 3]
	expected := []float64{3.0, -3.0, 2.0, -0.5, 3.0, -3.0}

	for i := range expected {
		if !FloatEq(m.Gradients.Values[i], expected[i]) {
			t.Errorf("Gradient[%d]: expected %.6f, got %.6f",
				i, expected[i], m.Gradients.Values[i])
		}
	}

	t.Logf("✓ Gradient clipping works correctly")
}

// TestClipGradients_AfterBackward tests clipping in realistic scenario
func TestClipGradients_AfterBackward(t *testing.T) {
	// Create a simple computation that might have large gradients
	a := NewTrackedMatrixWithValues(2, 2, []float64{10.0, 20.0, 30.0, 40.0})
	b := NewTrackedMatrixWithValues(2, 2, []float64{1.0, 2.0, 3.0, 4.0})

	// Multiply matrices
	c := a.MatMul(b)

	// Set large upstream gradients
	c.Gradients.Values = []float64{100.0, 100.0, 100.0, 100.0}

	// Run backprop
	c.Backward()

	// Check that gradients are large before clipping
	hasLargeGradient := false
	for _, g := range a.Gradients.Values {
		if g > 10.0 || g < -10.0 {
			hasLargeGradient = true
			break
		}
	}

	if !hasLargeGradient {
		t.Logf("Note: No large gradients generated, but test still valid")
	}

	// Clip gradients
	threshold := 10.0
	a.ClipGradients(threshold)
	b.ClipGradients(threshold)

	// Verify all gradients are within bounds
	for i, g := range a.Gradients.Values {
		if g > threshold || g < -threshold {
			t.Errorf("Gradient a[%d]=%.6f exceeds threshold %.6f", i, g, threshold)
		}
	}

	for i, g := range b.Gradients.Values {
		if g > threshold || g < -threshold {
			t.Errorf("Gradient b[%d]=%.6f exceeds threshold %.6f", i, g, threshold)
		}
	}

	t.Logf("✓ Gradient clipping after backward pass works correctly")
}

func TestSoftMaxCrossEntropy2(t *testing.T) {
	input := NewTrackedMatrixWithValues(
		2, 3,
		[]float64{
			0.25, 0.25, 0.5,
			0.40, 0.30, 0.30,
		})
	targets := NewTrackedMatrixWithValues(
		2, 1,
		[]float64{
			2.0, 0.0,
		},
	)

	output := input.SoftMaxCrossEntropy2(targets)

	fmt.Printf("Output: \n%s\n", output.Matrix.String())
}
