package main

import (
	"fmt"
	"math"
	"testing"
	"time"
)

func TestAddition(t *testing.T) {
	a := NewValue(2)
	b := NewValue(3)
	result := a.Add(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != 5 {
		t.Errorf("Expected result.Data = 5, got %.2f", result.Data)
	}
	if a.Gradient != 1 {
		t.Errorf("Expected a.Gradient = 1, got %.2f", a.Gradient)
	}
	if b.Gradient != 1 {
		t.Errorf("Expected b.Gradient = 1, got %.2f", b.Gradient)
	}
}

func TestSubtraction(t *testing.T) {
	a := NewValue(5)
	b := NewValue(2)
	result := a.Sub(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != 3 {
		t.Errorf("Expected result.Data = 3, got %.2f", result.Data)
	}
	if a.Gradient != 1 {
		t.Errorf("Expected a.Gradient = 1, got %.2f", a.Gradient)
	}
	if b.Gradient != -1 {
		t.Errorf("Expected b.Gradient = -1, got %.2f", b.Gradient)
	}
}

func TestMultiplication(t *testing.T) {
	a := NewValue(3)
	b := NewValue(4)
	result := a.Mul(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != 12 {
		t.Errorf("Expected result.Data = 12, got %.2f", result.Data)
	}
	if a.Gradient != 4 {
		t.Errorf("Expected a.Gradient = 4, got %.2f", a.Gradient)
	}
	if b.Gradient != 3 {
		t.Errorf("Expected b.Gradient = 3, got %.2f", b.Gradient)
	}
}

func TestValueReuseSameValue(t *testing.T) {
	// Test x + x where x = 5
	x := NewValue(5)
	result := x.Add(x)
	result.Gradient = 1
	result.Backward()

	if result.Data != 10 {
		t.Errorf("Expected result.Data = 10, got %.2f", result.Data)
	}
	if x.Gradient != 2 {
		t.Errorf("Expected x.Gradient = 2 (accumulated from both uses), got %.2f", x.Gradient)
	}
}

func TestValueReuseSquare(t *testing.T) {
	// Test x * x where x = 3
	// Derivative of x^2 is 2x = 6
	x := NewValue(3)
	result := x.Mul(x)
	result.Gradient = 1
	result.Backward()

	if result.Data != 9 {
		t.Errorf("Expected result.Data = 9, got %.2f", result.Data)
	}
	if x.Gradient != 6 {
		t.Errorf("Expected x.Gradient = 6 (derivative of x^2), got %.2f", x.Gradient)
	}
}

func TestComplexExpression(t *testing.T) {
	// Test 3 * (2 - 4) = -6
	a := NewValue(3)
	b := NewValue(2)
	c := NewValue(4)
	d := b.Sub(c)
	result := a.Mul(d)
	result.Gradient = 1
	result.Backward()

	if result.Data != -6 {
		t.Errorf("Expected result.Data = -6, got %.2f", result.Data)
	}
	if a.Gradient != -2 {
		t.Errorf("Expected a.Gradient = -2, got %.2f", a.Gradient)
	}
	if b.Gradient != 3 {
		t.Errorf("Expected b.Gradient = 3, got %.2f", b.Gradient)
	}
	if c.Gradient != -3 {
		t.Errorf("Expected c.Gradient = -3, got %.2f", c.Gradient)
	}
	if d.Gradient != 3 {
		t.Errorf("Expected d.Gradient = 3, got %.2f", d.Gradient)
	}
}

func TestChainedAdditions(t *testing.T) {
	// Test (a + b) + c
	a := NewValue(1)
	b := NewValue(2)
	c := NewValue(3)
	temp := a.Add(b)
	result := temp.Add(c)
	result.Gradient = 1
	result.Backward()

	if result.Data != 6 {
		t.Errorf("Expected result.Data = 6, got %.2f", result.Data)
	}
	if a.Gradient != 1 {
		t.Errorf("Expected a.Gradient = 1, got %.2f", a.Gradient)
	}
	if b.Gradient != 1 {
		t.Errorf("Expected b.Gradient = 1, got %.2f", b.Gradient)
	}
	if c.Gradient != 1 {
		t.Errorf("Expected c.Gradient = 1, got %.2f", c.Gradient)
	}
}

func TestChainedMultiplications(t *testing.T) {
	// Test (a * b) * c = (2 * 3) * 4 = 24
	a := NewValue(2)
	b := NewValue(3)
	c := NewValue(4)
	temp := a.Mul(b)
	result := temp.Mul(c)
	result.Gradient = 1
	result.Backward()

	if result.Data != 24 {
		t.Errorf("Expected result.Data = 24, got %.2f", result.Data)
	}
	if a.Gradient != 12 {
		t.Errorf("Expected a.Gradient = 12 (b*c), got %.2f", a.Gradient)
	}
	if b.Gradient != 8 {
		t.Errorf("Expected b.Gradient = 8 (a*c), got %.2f", b.Gradient)
	}
	if c.Gradient != 6 {
		t.Errorf("Expected c.Gradient = 6 (a*b), got %.2f", c.Gradient)
	}
}

func TestNegativeValues(t *testing.T) {
	// Test -2 * 3 = -6
	a := NewValue(-2)
	b := NewValue(3)
	result := a.Mul(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != -6 {
		t.Errorf("Expected result.Data = -6, got %.2f", result.Data)
	}
	if a.Gradient != 3 {
		t.Errorf("Expected a.Gradient = 3, got %.2f", a.Gradient)
	}
	if b.Gradient != -2 {
		t.Errorf("Expected b.Gradient = -2, got %.2f", b.Gradient)
	}
}

func TestTripleValueReuse(t *testing.T) {
	// Test x + x + x where x = 2
	x := NewValue(2)
	temp := x.Add(x)
	result := temp.Add(x)
	result.Gradient = 1
	result.Backward()

	if result.Data != 6 {
		t.Errorf("Expected result.Data = 6, got %.2f", result.Data)
	}
	if x.Gradient != 3 {
		t.Errorf("Expected x.Gradient = 3 (accumulated from 3 uses), got %.2f", x.Gradient)
	}
}

func TestFloatingPointGradients(t *testing.T) {
	// Test with non-integer values
	a := NewValue(2.5)
	b := NewValue(3.7)
	result := a.Mul(b)
	result.Gradient = 1
	result.Backward()

	expected := 2.5 * 3.7
	if math.Abs(result.Data-expected) > 0.0001 {
		t.Errorf("Expected result.Data = %.4f, got %.4f", expected, result.Data)
	}
	if math.Abs(a.Gradient-3.7) > 0.0001 {
		t.Errorf("Expected a.Gradient = 3.7, got %.4f", a.Gradient)
	}
	if math.Abs(b.Gradient-2.5) > 0.0001 {
		t.Errorf("Expected b.Gradient = 2.5, got %.4f", b.Gradient)
	}
}

// ReLU Tests

func TestReLUPositiveInput(t *testing.T) {
	// Test ReLU(5) = 5
	// Gradient should pass through (derivative = 1)
	x := NewValue(5)
	result := x.ReLU()
	result.Gradient = 1
	result.Backward()

	if result.Data != 5 {
		t.Errorf("Expected result.Data = 5, got %.2f", result.Data)
	}
	if x.Gradient != 1 {
		t.Errorf("Expected x.Gradient = 1 (gradient passes through), got %.2f", x.Gradient)
	}
}

func TestReLUNegativeInput(t *testing.T) {
	// Test ReLU(-3) = 0
	// Gradient should be killed (derivative = 0)
	x := NewValue(-3)
	result := x.ReLU()
	result.Gradient = 1
	result.Backward()

	if result.Data != 0 {
		t.Errorf("Expected result.Data = 0, got %.2f", result.Data)
	}
	if x.Gradient != 0 {
		t.Errorf("Expected x.Gradient = 0 (gradient killed), got %.2f", x.Gradient)
	}
}

func TestReLUZeroInput(t *testing.T) {
	// Test ReLU(0) = 0
	// Gradient typically treated as 0
	x := NewValue(0)
	result := x.ReLU()
	result.Gradient = 1
	result.Backward()

	if result.Data != 0 {
		t.Errorf("Expected result.Data = 0, got %.2f", result.Data)
	}
	if x.Gradient != 0 {
		t.Errorf("Expected x.Gradient = 0, got %.2f", x.Gradient)
	}
}

func TestReLUInChain(t *testing.T) {
	// Test 2 * ReLU(3) = 2 * 3 = 6
	// Chain rule: gradient flows back through ReLU, then to input
	x := NewValue(3)
	activated := x.ReLU()
	result := NewValue(2).Mul(activated)
	result.Gradient = 1
	result.Backward()

	if result.Data != 6 {
		t.Errorf("Expected result.Data = 6, got %.2f", result.Data)
	}
	if x.Gradient != 2 {
		t.Errorf("Expected x.Gradient = 2 (2 * 1 from ReLU), got %.2f", x.Gradient)
	}
}

func TestReLUBlocksNegativeChain(t *testing.T) {
	// Test 2 * ReLU(-3) = 2 * 0 = 0
	// Gradient should not flow back through ReLU
	x := NewValue(-3)
	activated := x.ReLU()
	result := NewValue(2).Mul(activated)
	result.Gradient = 1
	result.Backward()

	if result.Data != 0 {
		t.Errorf("Expected result.Data = 0, got %.2f", result.Data)
	}
	if x.Gradient != 0 {
		t.Errorf("Expected x.Gradient = 0 (blocked by ReLU), got %.2f", x.Gradient)
	}
}

func TestReLUWithAddition(t *testing.T) {
	// Test ReLU(2 + 3) = ReLU(5) = 5
	a := NewValue(2)
	b := NewValue(3)
	sum := a.Add(b)
	result := sum.ReLU()
	result.Gradient = 1
	result.Backward()

	if result.Data != 5 {
		t.Errorf("Expected result.Data = 5, got %.2f", result.Data)
	}
	if a.Gradient != 1 {
		t.Errorf("Expected a.Gradient = 1, got %.2f", a.Gradient)
	}
	if b.Gradient != 1 {
		t.Errorf("Expected b.Gradient = 1, got %.2f", b.Gradient)
	}
}

func TestReLUWithNegativeSum(t *testing.T) {
	// Test ReLU(2 - 5) = ReLU(-3) = 0
	// Gradients should be killed
	a := NewValue(2)
	b := NewValue(5)
	diff := a.Sub(b)
	result := diff.ReLU()
	result.Gradient = 1
	result.Backward()

	if result.Data != 0 {
		t.Errorf("Expected result.Data = 0, got %.2f", result.Data)
	}
	if a.Gradient != 0 {
		t.Errorf("Expected a.Gradient = 0 (killed by ReLU), got %.2f", a.Gradient)
	}
	if b.Gradient != 0 {
		t.Errorf("Expected b.Gradient = 0 (killed by ReLU), got %.2f", b.Gradient)
	}
}

func TestMultipleReLUs(t *testing.T) {
	// Test ReLU(3) + ReLU(-2) = 3 + 0 = 3
	a := NewValue(3)
	b := NewValue(-2)
	reluA := a.ReLU()
	reluB := b.ReLU()
	result := reluA.Add(reluB)
	result.Gradient = 1
	result.Backward()

	if result.Data != 3 {
		t.Errorf("Expected result.Data = 3, got %.2f", result.Data)
	}
	if a.Gradient != 1 {
		t.Errorf("Expected a.Gradient = 1 (positive, passes through), got %.2f", a.Gradient)
	}
	if b.Gradient != 0 {
		t.Errorf("Expected b.Gradient = 0 (negative, blocked), got %.2f", b.Gradient)
	}
}

func TestReLUValueReuse(t *testing.T) {
	// Test ReLU(x) + ReLU(x) where x = 2
	// Both ReLUs should contribute to gradient
	x := NewValue(2)
	relu1 := x.ReLU()
	relu2 := x.ReLU()
	result := relu1.Add(relu2)
	result.Gradient = 1
	result.Backward()

	if result.Data != 4 {
		t.Errorf("Expected result.Data = 4, got %.2f", result.Data)
	}
	if x.Gradient != 2 {
		t.Errorf("Expected x.Gradient = 2 (accumulated from both ReLUs), got %.2f", x.Gradient)
	}
}

// Power Tests

func TestPowSquare(t *testing.T) {
	// Test x^2 where x = 3
	// Forward: 3^2 = 9
	// Derivative: d/dx(x^2) = 2x = 2*3 = 6
	x := NewValue(3)
	result := x.Pow(NewValue(2))
	result.Gradient = 1
	result.Backward()

	if result.Data != 9 {
		t.Errorf("Expected result.Data = 9, got %.2f", result.Data)
	}
	if x.Gradient != 6 {
		t.Errorf("Expected x.Gradient = 6 (derivative 2*x), got %.2f", x.Gradient)
	}
}

func TestPowCube(t *testing.T) {
	// Test x^3 where x = 2
	// Forward: 2^3 = 8
	// Derivative: d/dx(x^3) = 3*x^2 = 3*4 = 12
	x := NewValue(2)
	result := x.Pow(NewValue(3))
	result.Gradient = 1
	result.Backward()

	if result.Data != 8 {
		t.Errorf("Expected result.Data = 8, got %.2f", result.Data)
	}
	if x.Gradient != 12 {
		t.Errorf("Expected x.Gradient = 12 (derivative 3*x^2), got %.2f", x.Gradient)
	}
}

func TestPowOne(t *testing.T) {
	// Test x^1 where x = 5
	// Forward: 5^1 = 5
	// Derivative: d/dx(x^1) = 1*x^0 = 1
	x := NewValue(5)
	result := x.Pow(NewValue(1))
	result.Gradient = 1
	result.Backward()

	if result.Data != 5 {
		t.Errorf("Expected result.Data = 5, got %.2f", result.Data)
	}
	if x.Gradient != 1 {
		t.Errorf("Expected x.Gradient = 1 (derivative of x^1), got %.2f", x.Gradient)
	}
}

func TestPowZero(t *testing.T) {
	// Test x^0 where x = 7
	// Forward: 7^0 = 1
	// Derivative: d/dx(x^0) = 0 (constant function)
	x := NewValue(7)
	result := x.Pow(NewValue(0))
	result.Gradient = 1
	result.Backward()

	if result.Data != 1 {
		t.Errorf("Expected result.Data = 1, got %.2f", result.Data)
	}
	if x.Gradient != 0 {
		t.Errorf("Expected x.Gradient = 0 (derivative of constant), got %.2f", x.Gradient)
	}
}

func TestPowNegativeExponent(t *testing.T) {
	// Test x^-1 where x = 2
	// Forward: 2^-1 = 0.5
	// Derivative: d/dx(x^-1) = -1*x^-2 = -1/4 = -0.25
	x := NewValue(2)
	result := x.Pow(NewValue(-1))
	result.Gradient = 1
	result.Backward()

	if math.Abs(result.Data-0.5) > 0.0001 {
		t.Errorf("Expected result.Data = 0.5, got %.4f", result.Data)
	}
	if math.Abs(x.Gradient-(-0.25)) > 0.0001 {
		t.Errorf("Expected x.Gradient = -0.25, got %.4f", x.Gradient)
	}
}

func TestPowFractional(t *testing.T) {
	// Test x^0.5 where x = 4 (square root)
	// Forward: 4^0.5 = 2
	// Derivative: d/dx(x^0.5) = 0.5*x^-0.5 = 0.5/2 = 0.25
	x := NewValue(4)
	result := x.Pow(NewValue(0.5))
	result.Gradient = 1
	result.Backward()

	if math.Abs(result.Data-2.0) > 0.0001 {
		t.Errorf("Expected result.Data = 2.0, got %.4f", result.Data)
	}
	if math.Abs(x.Gradient-0.25) > 0.0001 {
		t.Errorf("Expected x.Gradient = 0.25, got %.4f", x.Gradient)
	}
}

func TestPowInChain(t *testing.T) {
	// Test 2 * x^2 where x = 3
	// Forward: 2 * 9 = 18
	// Derivative: d/dx(2*x^2) = 2 * 2*x = 4*x = 12
	x := NewValue(3)
	squared := x.Pow(NewValue(2))
	result := NewValue(2).Mul(squared)
	result.Gradient = 1
	result.Backward()

	if result.Data != 18 {
		t.Errorf("Expected result.Data = 18, got %.2f", result.Data)
	}
	if x.Gradient != 12 {
		t.Errorf("Expected x.Gradient = 12 (chain rule: 2 * 2*x), got %.2f", x.Gradient)
	}
}

func TestPowWithAddition(t *testing.T) {
	// Test (x+1)^2 where x = 2
	// Forward: (2+1)^2 = 9
	// Derivative: d/dx((x+1)^2) = 2*(x+1) = 2*3 = 6
	x := NewValue(2)
	sum := x.Add(NewValue(1))
	result := sum.Pow(NewValue(2))
	result.Gradient = 1
	result.Backward()

	if result.Data != 9 {
		t.Errorf("Expected result.Data = 9, got %.2f", result.Data)
	}
	if x.Gradient != 6 {
		t.Errorf("Expected x.Gradient = 6 (chain rule), got %.2f", x.Gradient)
	}
}

func TestPowLargerExponent(t *testing.T) {
	// Test x^4 where x = 2
	// Forward: 2^4 = 16
	// Derivative: d/dx(x^4) = 4*x^3 = 4*8 = 32
	x := NewValue(2)
	result := x.Pow(NewValue(4))
	result.Gradient = 1
	result.Backward()

	if result.Data != 16 {
		t.Errorf("Expected result.Data = 16, got %.2f", result.Data)
	}
	if x.Gradient != 32 {
		t.Errorf("Expected x.Gradient = 32 (derivative 4*x^3), got %.2f", x.Gradient)
	}
}

func TestPowValueReuse(t *testing.T) {
	// Test x^2 + x^3 where x = 2
	// Forward: 4 + 8 = 12
	// Derivative: 2*x + 3*x^2 = 2*2 + 3*4 = 4 + 12 = 16
	x := NewValue(2)
	squared := x.Pow(NewValue(2))
	cubed := x.Pow(NewValue(3))
	result := squared.Add(cubed)
	result.Gradient = 1
	result.Backward()

	if result.Data != 12 {
		t.Errorf("Expected result.Data = 12, got %.2f", result.Data)
	}
	if x.Gradient != 16 {
		t.Errorf("Expected x.Gradient = 16 (accumulated: 2*x + 3*x^2), got %.2f", x.Gradient)
	}
}

func TestPowNegativeBase(t *testing.T) {
	// Test x^2 where x = -3
	// Forward: (-3)^2 = 9
	// Derivative: d/dx(x^2) = 2*x = 2*(-3) = -6
	x := NewValue(-3)
	result := x.Pow(NewValue(2))
	result.Gradient = 1
	result.Backward()

	if result.Data != 9 {
		t.Errorf("Expected result.Data = 9, got %.2f", result.Data)
	}
	if x.Gradient != -6 {
		t.Errorf("Expected x.Gradient = -6 (derivative 2*x with negative x), got %.2f", x.Gradient)
	}
}

func TestPowExponentGradient(t *testing.T) {
	// Test gradient with respect to the EXPONENT
	// For base^exp, d/d(exp) = base^exp * ln(base)
	// Example: 2^3 = 8
	// d/d(exp)(2^3) = 8 * ln(2) ≈ 8 * 0.693147 ≈ 5.545
	base := NewValue(2)
	exponent := NewValue(3)
	result := base.Pow(exponent)
	result.Gradient = 1
	result.Backward()

	if result.Data != 8 {
		t.Errorf("Expected result.Data = 8, got %.2f", result.Data)
	}

	// Gradient of base: exp * base^(exp-1) = 3 * 2^2 = 3 * 4 = 12
	if math.Abs(base.Gradient-12.0) > 0.0001 {
		t.Errorf("Expected base.Gradient = 12, got %.4f", base.Gradient)
	}

	// Gradient of exponent: base^exp * ln(base) = 8 * ln(2) ≈ 5.545177
	expectedExpGrad := 8 * math.Log(2)
	if math.Abs(exponent.Gradient-expectedExpGrad) > 0.0001 {
		t.Errorf("Expected exponent.Gradient = %.4f (result * ln(base)), got %.4f", expectedExpGrad, exponent.Gradient)
	}
}

func TestPowExponentGradientDifferentBase(t *testing.T) {
	// Test with different base: e^2 where e ≈ 2.71828
	// d/d(exp)(e^2) = e^2 * ln(e) = e^2 * 1 = e^2
	base := NewValue(math.E)
	exponent := NewValue(2)
	result := base.Pow(exponent)
	result.Gradient = 1
	result.Backward()

	expectedResult := math.E * math.E
	if math.Abs(result.Data-expectedResult) > 0.0001 {
		t.Errorf("Expected result.Data = %.4f, got %.4f", expectedResult, result.Data)
	}

	// Gradient of exponent: e^2 * ln(e) = e^2 * 1 = e^2
	if math.Abs(exponent.Gradient-expectedResult) > 0.0001 {
		t.Errorf("Expected exponent.Gradient = %.4f (e^2), got %.4f", expectedResult, exponent.Gradient)
	}
}

func TestPowBothGradientsWithReuse(t *testing.T) {
	// Test when both base and exponent are reused
	// x^x where x = 2 (result = 4)
	// This is tricky: gradient accumulates from both base and exponent roles
	// d/dx(x^x) = x^x * (ln(x) + 1)
	// For x=2: 2^2 * (ln(2) + 1) = 4 * (0.693 + 1) ≈ 4 * 1.693 ≈ 6.773
	x := NewValue(2)
	result := x.Pow(x)
	result.Gradient = 1
	result.Backward()

	if result.Data != 4 {
		t.Errorf("Expected result.Data = 4, got %.2f", result.Data)
	}

	// Total gradient = gradient as base + gradient as exponent
	// As base: x * x^(x-1) = 2 * 2^1 = 4
	// As exponent: x^x * ln(x) = 4 * ln(2) ≈ 2.773
	// Total: 4 + 2.773 ≈ 6.773
	expectedGrad := 4 * (math.Log(2) + 1)
	if math.Abs(x.Gradient-expectedGrad) > 0.0001 {
		t.Errorf("Expected x.Gradient = %.4f (accumulated from base and exponent), got %.4f", expectedGrad, x.Gradient)
	}
}

// Neuron Forward Tests
func TestNeuronForward(t *testing.T) {
	weights := []*Value{
		NewValue(1.0),
		NewValue(2.0),
	}
	bias := NewValue(5.0)
	neuron := NewNeuron(
		weights,
		bias,
	)
	inputs := []*Value{
		NewValue(3.0),
		NewValue(4.0),
	}
	result := neuron.Forward(inputs)
	if int(result.Data) != 16 {
		t.Errorf("Expected: 16, got: %.2f", result.Data)
	}
}

// Division Tests

func TestDivision(t *testing.T) {
	// Test 6 / 2 = 3
	// For c = a / b:
	// dc/da = 1/b = 1/2 = 0.5
	// dc/db = -a/b² = -6/4 = -1.5
	a := NewValue(6)
	b := NewValue(2)
	result := a.Div(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != 3 {
		t.Errorf("Expected result.Data = 3, got %.2f", result.Data)
	}
	if a.Gradient != 0.5 {
		t.Errorf("Expected a.Gradient = 0.5 (1/b), got %.2f", a.Gradient)
	}
	if b.Gradient != -1.5 {
		t.Errorf("Expected b.Gradient = -1.5 (-a/b²), got %.2f", b.Gradient)
	}
}

func TestDivisionFloatingPoint(t *testing.T) {
	// Test 10 / 4 = 2.5
	// For c = a / b:
	// dc/da = 1/b = 1/4 = 0.25
	// dc/db = -a/b² = -10/16 = -0.625
	a := NewValue(10)
	b := NewValue(4)
	result := a.Div(b)
	result.Gradient = 1
	result.Backward()

	if math.Abs(result.Data-2.5) > 0.0001 {
		t.Errorf("Expected result.Data = 2.5, got %.4f", result.Data)
	}
	if math.Abs(a.Gradient-0.25) > 0.0001 {
		t.Errorf("Expected a.Gradient = 0.25 (1/b), got %.4f", a.Gradient)
	}
	if math.Abs(b.Gradient-(-0.625)) > 0.0001 {
		t.Errorf("Expected b.Gradient = -0.625 (-a/b²), got %.4f", b.Gradient)
	}
}

func TestDivisionInChain(t *testing.T) {
	// Test 2 * (12 / 3) = 2 * 4 = 8
	// Let d = a / b, result = c * d
	// For division: dd/da = 1/b = 1/3 ≈ 0.333
	//               dd/db = -a/b² = -12/9 ≈ -1.333
	// For multiplication: dresult/dc = d = 4
	//                     dresult/dd = c = 2
	// Chain rule: dresult/da = dresult/dd * dd/da = 2 * 0.333 ≈ 0.667
	//             dresult/db = dresult/dd * dd/db = 2 * -1.333 ≈ -2.667
	a := NewValue(12)
	b := NewValue(3)
	c := NewValue(2)
	division := a.Div(b)
	result := c.Mul(division)
	result.Gradient = 1
	result.Backward()

	if result.Data != 8 {
		t.Errorf("Expected result.Data = 8, got %.2f", result.Data)
	}
	if math.Abs(a.Gradient-0.6667) > 0.001 {
		t.Errorf("Expected a.Gradient ≈ 0.667 (chain rule: 2 * 1/3), got %.4f", a.Gradient)
	}
	if math.Abs(b.Gradient-(-2.6667)) > 0.001 {
		t.Errorf("Expected b.Gradient ≈ -2.667 (chain rule: 2 * -12/9), got %.4f", b.Gradient)
	}
	if c.Gradient != 4 {
		t.Errorf("Expected c.Gradient = 4 (the division result), got %.2f", c.Gradient)
	}
}

func TestDivisionValueReuse(t *testing.T) {
	// Test x / x where x = 5
	// For c = a / b where a = b = x:
	// dc/da = 1/b = 1/5 = 0.2
	// dc/db = -a/b² = -5/25 = -0.2
	// Total gradient on x: 0.2 + (-0.2) = 0
	// This makes sense: d/dx(x/x) = d/dx(1) = 0
	x := NewValue(5)
	result := x.Div(x)
	result.Gradient = 1
	result.Backward()

	if result.Data != 1 {
		t.Errorf("Expected result.Data = 1, got %.2f", result.Data)
	}
	if math.Abs(x.Gradient-0) > 0.0001 {
		t.Errorf("Expected x.Gradient = 0 (derivative of constant 1), got %.4f", x.Gradient)
	}
}

func TestDivisionNegativeValues(t *testing.T) {
	// Test -8 / 2 = -4
	// For c = a / b:
	// dc/da = 1/b = 1/2 = 0.5
	// dc/db = -a/b² = -(-8)/4 = 8/4 = 2
	a := NewValue(-8)
	b := NewValue(2)
	result := a.Div(b)
	result.Gradient = 1
	result.Backward()

	if result.Data != -4 {
		t.Errorf("Expected result.Data = -4, got %.2f", result.Data)
	}
	if a.Gradient != 0.5 {
		t.Errorf("Expected a.Gradient = 0.5 (1/b), got %.2f", a.Gradient)
	}
	if b.Gradient != 2 {
		t.Errorf("Expected b.Gradient = 2 (-a/b² with negative a), got %.2f", b.Gradient)
	}
}

func TestDivisionWithSubtraction(t *testing.T) {
	// Test (10 - 2) / 4 = 8 / 4 = 2
	// Let s = a - b, result = s / c
	// For subtraction: ds/da = 1, ds/db = -1
	// For division: dresult/ds = 1/c = 1/4 = 0.25
	//               dresult/dc = -s/c² = -8/16 = -0.5
	// Chain rule: dresult/da = dresult/ds * ds/da = 0.25 * 1 = 0.25
	//             dresult/db = dresult/ds * ds/db = 0.25 * -1 = -0.25
	a := NewValue(10)
	b := NewValue(2)
	c := NewValue(4)
	subtraction := a.Sub(b)
	result := subtraction.Div(c)
	result.Gradient = 1
	result.Backward()

	if result.Data != 2 {
		t.Errorf("Expected result.Data = 2, got %.2f", result.Data)
	}
	if a.Gradient != 0.25 {
		t.Errorf("Expected a.Gradient = 0.25, got %.2f", a.Gradient)
	}
	if b.Gradient != -0.25 {
		t.Errorf("Expected b.Gradient = -0.25, got %.2f", b.Gradient)
	}
	if c.Gradient != -0.5 {
		t.Errorf("Expected c.Gradient = -0.5 (-s/c²), got %.2f", c.Gradient)
	}
}

func TestLnForwardPass(t *testing.T) {
	// Test ln(e) = 1
	x := NewValue(math.E)
	result := x.Ln()

	expected := 1.0
	if math.Abs(result.Data-expected) > 1e-10 {
		t.Errorf("Expected ln(e) = 1, got %.10f", result.Data)
	}

	// Test ln(1) = 0
	x2 := NewValue(1.0)
	result2 := x2.Ln()

	if math.Abs(result2.Data-0.0) > 1e-10 {
		t.Errorf("Expected ln(1) = 0, got %.10f", result2.Data)
	}

	// Test ln(2) ≈ 0.693147
	x3 := NewValue(2.0)
	result3 := x3.Ln()

	expected3 := math.Log(2.0)
	if math.Abs(result3.Data-expected3) > 1e-10 {
		t.Errorf("Expected ln(2) = %.10f, got %.10f", expected3, result3.Data)
	}
}

func TestLnBackwardPass(t *testing.T) {
	// Test gradient of ln(x)
	// Derivative: d/dx[ln(x)] = 1/x

	// Test with x = 2.0
	// d/dx[ln(2)] = 1/2 = 0.5
	x := NewValue(2.0)
	result := x.Ln()
	result.Gradient = 1.0
	result.Backward()

	expectedGradient := 1.0 / 2.0 // 1/x = 1/2 = 0.5
	if math.Abs(x.Gradient-expectedGradient) > 1e-10 {
		t.Errorf("Expected x.Gradient = %.10f (1/x), got %.10f", expectedGradient, x.Gradient)
	}

	// Test with x = e
	// d/dx[ln(e)] = 1/e ≈ 0.367879
	x2 := NewValue(math.E)
	result2 := x2.Ln()
	result2.Gradient = 1.0
	result2.Backward()

	expectedGradient2 := 1.0 / math.E
	if math.Abs(x2.Gradient-expectedGradient2) > 1e-10 {
		t.Errorf("Expected x2.Gradient = %.10f (1/e), got %.10f", expectedGradient2, x2.Gradient)
	}

	// Test with x = 10.0
	// d/dx[ln(10)] = 1/10 = 0.1
	x3 := NewValue(10.0)
	result3 := x3.Ln()
	result3.Gradient = 1.0
	result3.Backward()

	expectedGradient3 := 1.0 / 10.0
	if math.Abs(x3.Gradient-expectedGradient3) > 1e-10 {
		t.Errorf("Expected x3.Gradient = %.10f (1/10), got %.10f", expectedGradient3, x3.Gradient)
	}
}

func TestLnChainRule(t *testing.T) {
	// Test ln in a chain: ln(x * y)
	// d/dx[ln(x*y)] = d/d(xy)[ln(xy)] * d/dx[xy] = 1/(xy) * y = y/(xy) = 1/x
	// d/dy[ln(x*y)] = d/d(xy)[ln(xy)] * d/dy[xy] = 1/(xy) * x = x/(xy) = 1/y

	x := NewValue(3.0)
	y := NewValue(4.0)
	product := x.Mul(y)    // xy = 12
	result := product.Ln() // ln(12)

	// Check forward pass
	expected := math.Log(12.0)
	if math.Abs(result.Data-expected) > 1e-10 {
		t.Errorf("Expected ln(12) = %.10f, got %.10f", expected, result.Data)
	}

	// Check backward pass
	result.Gradient = 1.0
	result.Backward()

	expectedXGrad := 1.0 / 3.0 // 1/x
	expectedYGrad := 1.0 / 4.0 // 1/y

	if math.Abs(x.Gradient-expectedXGrad) > 1e-10 {
		t.Errorf("Expected x.Gradient = %.10f (1/x), got %.10f", expectedXGrad, x.Gradient)
	}
	if math.Abs(y.Gradient-expectedYGrad) > 1e-10 {
		t.Errorf("Expected y.Gradient = %.10f (1/y), got %.10f", expectedYGrad, y.Gradient)
	}
}

func TestLnNegativeInput(t *testing.T) {
	// ln(x) where x <= 0 should panic during backward pass
	x := NewValue(-2.0)

	// Forward pass will compute ln(-2) which Go's math.Log returns as NaN
	result := x.Ln()
	result.Gradient = 1.0

	// Backward pass should panic because derivative 1/x is undefined for x <= 0
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for ln derivative with negative input, but didn't panic")
		}
	}()

	result.Backward()
}

func TestLnZeroInput(t *testing.T) {
	// ln(0) should panic during backward pass
	x := NewValue(0.0)
	result := x.Ln()
	result.Gradient = 1.0

	// Backward pass should panic because derivative 1/0 is undefined
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for ln derivative with zero input, but didn't panic")
		}
	}()

	result.Backward()
}

func TestLnValidNegativeOutput(t *testing.T) {
	// ln(0.5) ≈ -0.693 is a valid negative OUTPUT
	// This should NOT panic - negative ln(x) is fine, as long as x > 0
	x := NewValue(0.5)
	result := x.Ln()

	// Check forward pass produces negative result
	if result.Data >= 0 {
		t.Errorf("Expected ln(0.5) < 0, got %.10f", result.Data)
	}

	// Backward pass should work fine
	result.Gradient = 1.0
	result.Backward()

	expectedGradient := 1.0 / 0.5 // 1/x = 2.0
	if math.Abs(x.Gradient-expectedGradient) > 1e-10 {
		t.Errorf("Expected x.Gradient = %.10f, got %.10f", expectedGradient, x.Gradient)
	}
}

func TestLeakyReLU(t *testing.T) {
	// Test forward pass
	alpha := 0.01

	// Test positive input
	x := NewValue(2.0)
	result := x.LeakyReLU(alpha)
	if result.Data != 2.0 {
		t.Errorf("LeakyReLU(2.0) expected 2.0, got %.4f", result.Data)
	}

	// Test negative input
	x2 := NewValue(-2.0)
	result2 := x2.LeakyReLU(alpha)
	expected := -2.0 * alpha
	if math.Abs(result2.Data-expected) > 1e-10 {
		t.Errorf("LeakyReLU(-2.0) expected %.4f, got %.4f", expected, result2.Data)
	}

	// Test backward pass - positive input
	x3 := NewValue(3.0)
	result3 := x3.LeakyReLU(alpha)
	result3.Gradient = 1.0
	result3.Backward()
	if math.Abs(x3.Gradient-1.0) > 1e-10 {
		t.Errorf("LeakyReLU backward (positive) expected gradient 1.0, got %.4f", x3.Gradient)
	}

	// Test backward pass - negative input
	x4 := NewValue(-3.0)
	result4 := x4.LeakyReLU(alpha)
	result4.Gradient = 1.0
	result4.Backward()
	if math.Abs(x4.Gradient-alpha) > 1e-10 {
		t.Errorf("LeakyReLU backward (negative) expected gradient %.4f, got %.4f", alpha, x4.Gradient)
	}
}

func TestGradientClipping(t *testing.T) {
	// Create a neuron with one weight
	weight := NewValue(0.5)
	bias := NewValue(0.1)
	neuron := NewNeuron([]*Value{weight}, bias)

	// Set a large gradient that should be clipped
	weight.Gradient = 5.0 // Should be clipped to 1.0
	bias.Gradient = -3.0  // Should be clipped to -1.0

	learningRate := 0.1
	originalWeight := weight.Data
	originalBias := bias.Data

	neuron.Tune(learningRate)

	// Weight update should use clipped gradient (1.0)
	expectedWeight := originalWeight - learningRate*1.0
	if math.Abs(weight.Data-expectedWeight) > 1e-10 {
		t.Errorf("Weight with clipped gradient: expected %.4f, got %.4f", expectedWeight, weight.Data)
	}

	// Bias update should use clipped gradient (-1.0)
	expectedBias := originalBias - learningRate*(-1.0)
	if math.Abs(bias.Data-expectedBias) > 1e-10 {
		t.Errorf("Bias with clipped gradient: expected %.4f, got %.4f", expectedBias, bias.Data)
	}

	// Gradients should be reset to 0
	if weight.Gradient != 0 {
		t.Errorf("Weight gradient should be reset to 0, got %.4f", weight.Gradient)
	}
	if bias.Gradient != 0 {
		t.Errorf("Bias gradient should be reset to 0, got %.4f", bias.Gradient)
	}
}

// Network Tests
func TestNetworkTraining(t *testing.T) {
	learningRate := 0.01

	network := NewRandomNetwork([]LayerDims{
		{numWeights: 2, numNeurons: 16},
		{numWeights: 16, numNeurons: 1}, // output layer
	})

	examples := []TrainingExample{
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(0.0),
			},
			output: NewValue(0.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(0.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(1.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(1.0),
			},
			output: NewValue(0.0),
		},
	}

	loss := NewValue(1.0)

	epochs := 1000
	for range epochs {
		for _, example := range examples {
			output := network.Forward(example.input)[0]
			expected := example.output

			loss = LossXOR(expected, output)

			fmt.Printf("loss: %.16f\n", loss.Data)

			loss.Gradient = 1
			loss.Backward()
			network.Tune(learningRate)
		}
	}
	network.Store(fmt.Sprintf("./network_run_%s.json", time.Now().Format(time.DateOnly)))
	fmt.Printf("\n\n\n final loss: %.16f\n", loss.Data)
}

func TestNetworkInference(t *testing.T) {
	network, err := LoadNetworkFromFile("./network_runs/network_run_1.json")
	if err != nil {
		t.Errorf("failed to load network from file: %s", err)
	}

	examples := []TrainingExample{
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(0.0),
			},
			output: NewValue(0.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(0.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(1.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(1.0),
			},
			output: NewValue(0.0),
		},
	}

	for _, example := range examples {
		output := network.Forward(example.input)[0]
		fmt.Printf("input: [%.2f, %.2f] -> output: %.2f\n", example.input[0].Data, example.input[1].Data, output.Data)
	}
}

func TestBiggerNetwork(t *testing.T) {
	learningRate := 0.01

	network := NewRandomNetwork([]LayerDims{
		{numWeights: 2, numNeurons: 784},
		{numWeights: 784, numNeurons: 64},
		{numWeights: 64, numNeurons: 10},
		{numWeights: 10, numNeurons: 1},
	})

	examples := []TrainingExample{
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(0.0),
			},
			output: NewValue(0.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(0.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(0.0),
				NewValue(1.0),
			},
			output: NewValue(1.0),
		},
		{
			input: []*Value{
				NewValue(1.0),
				NewValue(1.0),
			},
			output: NewValue(0.0),
		},
	}

	loss := NewValue(1.0)

	epochs := 1000
	for range epochs {
		for _, example := range examples {
			output := network.Forward(example.input)[0]
			expected := example.output

			loss = LossXOR(expected, output)

			fmt.Printf("loss: %.16f\n", loss.Data)

			loss.Gradient = 1
			loss.Backward()
			network.Tune(learningRate)
		}
	}
	network.Store(fmt.Sprintf("./network_run_%s.json", time.Now().Format(time.DateOnly)))
	fmt.Printf("\n\n\n final loss: %.16f\n", loss.Data)
}
