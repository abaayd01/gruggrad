package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
)

type Vector []float64

func NewVector(size int) *Vector {
	vec := Vector(make([]float64, size))
	return &vec
}

type Matrix struct {
	Values  []float64
	NumRows int
	NumCols int
}

func NewMatrix(numRows, numCols int) *Matrix {
	vals := make([]float64, numRows*numCols)
	return &Matrix{
		Values:  vals,
		NumRows: numRows,
		NumCols: numCols,
	}
}

func NewRandomMatrix(numRows, numCols int) *Matrix {
	cntElems := numRows * numCols
	m := Matrix{
		Values:  make([]float64, cntElems),
		NumRows: numRows,
		NumCols: numCols,
	}
	for i := range cntElems {
		m.Values[i] = rand.Float64() - 0.5
	}
	return &m
}

func (m *Matrix) String() string {
	var mStr strings.Builder
	for i := range len(m.Values) {
		remainder := i % m.NumCols
		if remainder == 0 && i > 0 {
			mStr.WriteString("\n")
		}
		mStr.WriteString(fmt.Sprintf("%.8f ", m.Values[i]))
	}
	return mStr.String()
}

func (m *Matrix) Row(i int) []float64 {
	if i >= m.NumRows || i < 0 {
		panic("error attempting to access a row index outside bounds of matrix")
	}
	startIdx := m.NumCols * i
	endIdx := startIdx + m.NumCols
	return m.Values[startIdx:endIdx]
}

func (m *Matrix) RowAsMatrix(i int) *Matrix {
	if i >= m.NumRows || i < 0 {
		panic("error attempting to access a row index outside bounds of matrix")
	}
	startIdx := m.NumCols * i
	endIdx := startIdx + m.NumCols
	values := m.Values[startIdx:endIdx]
	return &Matrix{
		Values:  values,
		NumRows: 1,
		NumCols: m.NumCols,
	}
}

func (m *Matrix) Col(j int) []float64 {
	if j >= m.NumCols || j < 0 {
		panic("error attempting to access a column index outside bounds of matrix")
	}
	result := make([]float64, 0, m.NumRows)
	for i := j; i < m.NumRows*m.NumCols; i = i + m.NumCols {
		result = append(result, m.Values[i])
	}
	return result
}

func (m *Matrix) Set(i, j int, val float64) {
	m.Values[i*m.NumCols+j] = val
}

const epsilon = 1e-9

func floatEq(a, b float64) bool {
	// Check if the absolute difference is less than or equal to the epsilon
	return math.Abs(a-b) <= epsilon
}

func (m *Matrix) Equal(n *Matrix) bool {
	if !EqualDims(m, n) {
		return false
	}
	for i := range m.NumRows * m.NumCols {
		if !floatEq(m.Values[i], n.Values[i]) {
			return false
		}
	}
	return true
}

func (m *Matrix) MultiplyScalar(scalar float64) *Matrix {
	return m.Map(func(val float64) float64 {
		return val * scalar
	})
}

func (m *Matrix) Apply(fn ApplyFn) {
	for i := range m.NumRows * m.NumCols {
		m.Values[i] = fn(m.Values[i])
	}
}

func (m *Matrix) Map(fn ApplyFn) *Matrix {
	resultMatrix := NewMatrix(m.NumRows, m.NumCols)
	for i := range m.NumRows * m.NumCols {
		resultMatrix.Values[i] = fn(m.Values[i])
	}
	return resultMatrix
}

type RowApplyFn func(vals []float64) []float64

func (m *Matrix) RowMap(fn RowApplyFn) *Matrix {
	resultMatrix := NewMatrix(m.NumRows, m.NumCols)
	for i := range m.NumRows {
		resultRow := fn(m.Values[i*m.NumCols : i*m.NumCols+m.NumCols])
		for j := range m.NumCols {
			resultMatrix.Values[i*m.NumCols+j] = resultRow[j]
		}
	}
	return resultMatrix
}

func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.NumCols, m.NumRows)
	for i := range m.NumRows {
		for j := range m.NumCols {
			result.Values[j*result.NumCols+i] = m.Values[i*m.NumCols+j]
		}
	}
	return result
}

type ApplyFn func(val float64) float64

func EqualDims(A *Matrix, B *Matrix) bool {
	if A.NumRows != B.NumRows || A.NumCols != B.NumCols {
		return false
	}
	return true
}

func (A *Matrix) Multiply(B *Matrix) *Matrix {
	if A.NumCols != B.NumRows {
		panic(fmt.Sprintf("mismatched inner dims of matmul, dimsA: [%d, %d], dimsB: [%d, %d]", A.NumRows, A.NumCols, B.NumRows, B.NumCols))
	}
	resultMatrix := NewMatrix(A.NumRows, B.NumCols)
	for i := range resultMatrix.NumRows {
		for j := range resultMatrix.NumCols {
			var sum float64
			for k := range A.NumCols {
				sum += A.Values[i*A.NumCols+k] * B.Values[k*B.NumCols+j]
			}
			resultMatrix.Values[i*resultMatrix.NumCols+j] = sum
		}
	}
	return resultMatrix
}

func (m *Matrix) Add(B *Matrix) *Matrix {
	if !EqualDims(m, B) {
		panic(fmt.Sprintf("matrices must be of same dimensions to perform Add, dimsA: [%d, %d], dimsB: [%d, %d]", m.NumRows, m.NumCols, B.NumRows, B.NumCols))
	}
	result := NewMatrix(m.NumRows, m.NumCols)
	for i := range m.NumRows {
		for j := range m.NumCols {
			k := i*m.NumCols + j
			result.Values[k] = m.Values[k] + B.Values[k]
		}
	}
	return result
}

// AddVector does broadcasting
func (m *Matrix) AddVector(vector *Matrix) *Matrix {
	if vector.NumRows > 1 {
		panic("AddVector must receive a matrix with only one row")
	}

	if m.NumCols != vector.NumCols {

		panic("AddVector m and vector cols must be matched")
	}

	result := NewMatrix(m.NumRows, m.NumCols)
	for i := range m.NumRows {
		for j := range m.NumCols {
			k := i*m.NumCols + j
			result.Values[k] = m.Values[k] + vector.Values[j]
		}
	}
	return result
}

func (A *Matrix) Subtract(B *Matrix) *Matrix {
	if !EqualDims(A, B) {
		panic(fmt.Sprintf("matrices must be of same dimensions to perform Subtract, dimsA: [%d, %d], dimsB: [%d, %d]", A.NumRows, A.NumCols, B.NumRows, B.NumCols))
	}
	result := NewMatrix(A.NumRows, A.NumCols)
	for i := range A.NumRows {
		for j := range A.NumCols {
			k := i*A.NumCols + j
			result.Values[k] = A.Values[k] - B.Values[k]
		}
	}
	return result

}

type TrackedMatrix struct {
	*Matrix
	LeftParent  *TrackedMatrix
	RightParent *TrackedMatrix
	Gradients   *Matrix
	Operation   Operation
	Targets     *TrackedMatrix
}

func NewTrackedMatrix(numRows, numCols int) *TrackedMatrix {
	m := NewMatrix(numRows, numCols)
	return &TrackedMatrix{
		Matrix:      m,
		LeftParent:  nil,
		RightParent: nil,
		Gradients:   NewMatrix(numRows, numCols),
		Operation:   OpNoop,
	}
}

func NewRandomTrackedMatrix(numRows, numCols int) *TrackedMatrix {
	m := NewRandomMatrix(numRows, numCols)
	return &TrackedMatrix{
		Matrix:      m,
		LeftParent:  nil,
		RightParent: nil,
		Gradients:   NewMatrix(numRows, numCols),
		Operation:   OpNoop,
	}
}

// NewRandomTrackedMatrixHeInit creates a matrix with He initialization
// Recommended for ReLU networks: weights ~ N(0, sqrt(2/fan_in))
// This prevents vanishing/exploding gradients in deep networks
func NewRandomTrackedMatrixHeInit(numRows, numCols int) *TrackedMatrix {
	m := NewMatrix(numRows, numCols)

	// He initialization: std = sqrt(2 / fan_in)
	// fan_in is the number of input units (numRows)
	stddev := math.Sqrt(2.0 / float64(numRows))

	for i := range m.Values {
		// Box-Muller transform to generate normal distribution
		u1 := rand.Float64()
		u2 := rand.Float64()
		z := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
		m.Values[i] = z * stddev
	}

	return &TrackedMatrix{
		Matrix:      m,
		LeftParent:  nil,
		RightParent: nil,
		Gradients:   NewMatrix(numRows, numCols),
		Operation:   OpNoop,
	}
}

func NewTrackedMatrixWithValues(numRows, numCols int, values []float64) *TrackedMatrix {
	m := NewTrackedMatrix(numRows, numCols)
	m.Values = values
	return m
}

func buildTopologicalMatrixOrder(v *TrackedMatrix) []*TrackedMatrix {
	visited := make(map[*TrackedMatrix]bool)
	var topo []*TrackedMatrix

	var build func(*TrackedMatrix)
	build = func(node *TrackedMatrix) {
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

func (t *TrackedMatrix) Backward() {
	// Build topological order
	topo := buildTopologicalMatrixOrder(t)

	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]

		switch node.Operation {
		case OpNoop:
			continue
		case OpAdd:
			// implies this is the bias node
			if node.LeftParent.Gradients.NumRows == 1 {
				for rowNum := range node.Gradients.NumRows {
					node.LeftParent.Gradients = node.LeftParent.Gradients.Add(node.Gradients.RowAsMatrix(rowNum))
				}
			} else {
				node.LeftParent.Gradients = node.LeftParent.Gradients.Add(node.Gradients)
			}

			// implies this is the bias node
			if node.RightParent.Gradients.NumRows == 1 {
				for rowNum := range node.Gradients.NumRows {
					node.RightParent.Gradients = node.RightParent.Gradients.Add(node.Gradients.RowAsMatrix(rowNum))
				}
			} else {
				node.RightParent.Gradients = node.RightParent.Gradients.Add(node.Gradients)
			}
		case OpSub:
			node.LeftParent.Gradients = node.LeftParent.Gradients.Add(node.Gradients)
			node.RightParent.Gradients = node.RightParent.Gradients.Add(node.Gradients.MultiplyScalar(-1))
		case OpMul:
			node.LeftParent.Gradients = node.LeftParent.Gradients.Add(node.Gradients.Multiply(node.RightParent.Matrix.Transpose()))
			node.RightParent.Gradients = node.RightParent.Gradients.Add(node.LeftParent.Matrix.Transpose().Multiply(node.Gradients))
		case OpRelu:
			for i := range node.Matrix.Values {
				if node.Matrix.Values[i] > 0 {
					node.LeftParent.Gradients.Values[i] += 1 * node.Gradients.Values[i]
				} else {
					node.LeftParent.Gradients.Values[i] += 0
				}
			}
		case OpLeakyRelu:
			// Gradient: 1 if x > 0, else alpha
			alpha := node.RightParent.Matrix.Values[0]
			for i := range node.LeftParent.Matrix.Values {
				if node.LeftParent.Matrix.Values[i] > 0 {
					node.LeftParent.Gradients.Values[i] += 1 * node.Gradients.Values[i]
				} else {
					node.LeftParent.Gradients.Values[i] += alpha * node.Gradients.Values[i]
				}
			}
		case OpSoftMaxCrossEntropy:
			// I don't exactly understand this derivative... it was AI generated
			// the other path is to express softmax cross-entropy as other more primitive
			// matrix operations, which support backprop as well.
			// This is what I did with the micrograd / Value approach.

			// Gradient: softmax(x) - one_hot(target), scaled by upstream gradient
			// Since we averaged the loss, upstream gradient is typically 1.0
			upstreamGrad := node.Gradients.Values[0]
			scaleFactor := upstreamGrad / float64(node.LeftParent.NumRows)

			// Compute softmax for each row and subtract one-hot
			for rowIdx := 0; rowIdx < node.LeftParent.NumRows; rowIdx++ {
				rowStart := rowIdx * node.LeftParent.NumCols
				rowEnd := rowStart + node.LeftParent.NumCols
				row := node.LeftParent.Values[rowStart:rowEnd]

				// Compute softmax for this row (same as forward pass)
				maxVal := row[0]
				for j := 1; j < node.LeftParent.NumCols; j++ {
					if row[j] > maxVal {
						maxVal = row[j]
					}
				}

				sumExp := 0.0
				for j := 0; j < node.LeftParent.NumCols; j++ {
					sumExp += math.Exp(row[j] - maxVal)
				}

				// Compute gradient for each element in this row
				target := int(node.Targets.Matrix.Values[rowIdx])
				for j := 0; j < node.LeftParent.NumCols; j++ {
					softmaxVal := math.Exp(row[j]-maxVal) / sumExp
					if j == target {
						// gradient = softmax - 1
						node.LeftParent.Gradients.Values[rowStart+j] += (softmaxVal - 1.0) * scaleFactor
					} else {
						// gradient = softmax
						node.LeftParent.Gradients.Values[rowStart+j] += softmaxVal * scaleFactor
					}
				}
			}
		case OpTranspose:
			node.LeftParent.Gradients = node.LeftParent.Gradients.Add(node.Gradients.Transpose())
		}
	}
}

func (t *TrackedMatrix) MatMul(b *TrackedMatrix) *TrackedMatrix {
	resultMatrix := t.Matrix.Multiply(b.Matrix)
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: b,
		Operation:   OpMul,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

func (t *TrackedMatrix) Add(b *TrackedMatrix) *TrackedMatrix {
	resultMatrix := t.Matrix.AddVector(b.Matrix)
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: b,
		Operation:   OpAdd,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

func (t *TrackedMatrix) Subtract(b *TrackedMatrix) *TrackedMatrix {
	resultMatrix := t.Matrix.Subtract(b.Matrix)
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: b,
		Operation:   OpSub,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

func ReluFn(in float64) float64 {
	if in <= 0 {
		return 0
	}
	return in
}

func (t *TrackedMatrix) ReLU() *TrackedMatrix {
	resultMatrix := t.Matrix.Map(ReluFn)
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: nil,
		Operation:   OpRelu,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

// LeakyReLU applies leaky ReLU activation with the given negative slope (alpha)
// f(x) = x if x > 0, else alpha * x
// Typical alpha is 0.01
func (t *TrackedMatrix) LeakyReLU(alpha float64) *TrackedMatrix {
	leakyReluFn := func(in float64) float64 {
		if in > 0 {
			return in
		}
		return alpha * in
	}

	resultMatrix := t.Matrix.Map(leakyReluFn)

	// Store alpha in the result matrix for use during backward pass
	// We'll store it as a 1x1 matrix in RightParent
	alphaMatrix := &Matrix{
		Values:  []float64{alpha},
		NumRows: 1,
		NumCols: 1,
	}
	alphaTracked := &TrackedMatrix{
		Matrix:      alphaMatrix,
		LeftParent:  nil,
		RightParent: nil,
		Gradients:   NewMatrix(1, 1),
		Operation:   OpNoop,
	}

	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: alphaTracked, // Store alpha here
		Operation:   OpLeakyRelu,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

func SoftMaxFn(vals []float64) []float64 {
	var exponentiatedInputs []float64
	var sumExponentiatedInputs float64
	for i := range vals {
		ei := math.Pow(math.E, vals[i])
		exponentiatedInputs = append(exponentiatedInputs, ei)
		sumExponentiatedInputs += ei
	}

	var output []float64
	for i := range exponentiatedInputs {
		output = append(output, exponentiatedInputs[i]/sumExponentiatedInputs)
	}
	return output
}

func (t *TrackedMatrix) SoftMax() *TrackedMatrix {
	resultMatrix := t.Matrix.RowMap(SoftMaxFn)
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: nil,
		Operation:   OpSoftMax,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

// SoftMaxCrossEntropy computes the combined softmax + cross-entropy loss
// targets is a slice of class indices, one per row in the input matrix
// Returns a scalar (1x1 matrix) containing the average loss across all rows
func (t *TrackedMatrix) SoftMaxCrossEntropy(targets []int) *TrackedMatrix {
	if len(targets) != t.NumRows {
		panic(fmt.Sprintf("targets length %d must match number of rows %d", len(targets), t.NumRows))
	}

	// Validate targets are in valid range
	for i, target := range targets {
		if target < 0 || target >= t.NumCols {
			panic(fmt.Sprintf("target[%d]=%d is out of range [0, %d)", i, target, t.NumCols))
		}
	}

	totalLoss := 0.0

	// Compute loss for each row
	for rowIdx := 0; rowIdx < t.NumRows; rowIdx++ {
		rowStart := rowIdx * t.NumCols
		rowEnd := rowStart + t.NumCols
		row := t.Values[rowStart:rowEnd]

		// Find max for numerical stability
		maxVal := row[0]
		for j := 1; j < t.NumCols; j++ {
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}

		// Compute log_sum_exp = log(sum(exp(x - max)))
		sumExp := 0.0
		for j := 0; j < t.NumCols; j++ {
			sumExp += math.Exp(row[j] - maxVal)
		}
		logSumExp := maxVal + math.Log(sumExp)

		// Loss = -x[target] + log_sum_exp
		target := targets[rowIdx]
		loss := -row[target] + logSumExp
		totalLoss += loss
	}

	// Average loss across batch
	avgLoss := totalLoss / float64(t.NumRows)

	// Return scalar result
	resultMatrix := &Matrix{
		Values:  []float64{avgLoss},
		NumRows: 1,
		NumCols: 1,
	}

	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: nil,
		Operation:   OpSoftMaxCrossEntropy,
		Gradients:   NewMatrix(1, 1),
		// Targets:     targets,
	}
}

func SoftMaxCrossEntropyVec(vec []float64, targetValIdx int) float64 {
	z_k := vec[targetValIdx]
	c := slices.Max(vec)
	var sum_exponents float64
	for _, val := range vec {
		sum_exponents += math.Pow(math.E, val-c)
	}
	return -z_k + c + math.Log(sum_exponents)
}

func Average(vals []float64) float64 {
	var sum float64
	for _, val := range vals {
		sum += val
	}
	return sum / float64(len(vals))
}

func (t *TrackedMatrix) SoftMaxCrossEntropy2(targets *TrackedMatrix) *TrackedMatrix {
	resultMatrix := NewMatrix(1, 1)

	var lossVals []float64
	for i := range t.NumRows {
		targetValIdx := int(targets.Matrix.Values[i])
		lossVal := SoftMaxCrossEntropyVec(t.Values[i*t.NumCols:i*t.NumCols+t.NumCols], targetValIdx)
		lossVals = append(lossVals, lossVal)
	}

	avgLoss := Average(lossVals)
	resultMatrix.Set(0, 0, avgLoss)

	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: nil,
		Operation:   OpSoftMaxCrossEntropy,
		Gradients:   NewMatrix(1, 1),
		Targets:     targets,
	}
}

func (t *TrackedMatrix) Transpose() *TrackedMatrix {
	resultMatrix := t.Matrix.Transpose()
	return &TrackedMatrix{
		Matrix:      resultMatrix,
		LeftParent:  t,
		RightParent: nil,
		Operation:   OpTranspose,
		Gradients:   NewMatrix(resultMatrix.NumRows, resultMatrix.NumCols),
	}
}

func (t *TrackedMatrix) ZeroGradients() {
	t.Gradients = NewMatrix(t.NumRows, t.NumCols)
}

// ClipGradients clips gradients by value to prevent exploding gradients
// Clamps each gradient element to the range [-threshold, threshold]
func (t *TrackedMatrix) ClipGradients(threshold float64) {
	if threshold <= 0 {
		panic("gradient clipping threshold must be positive")
	}

	for i := range t.Gradients.Values {
		if t.Gradients.Values[i] > threshold {
			t.Gradients.Values[i] = threshold
		} else if t.Gradients.Values[i] < -threshold {
			t.Gradients.Values[i] = -threshold
		}
	}
}
