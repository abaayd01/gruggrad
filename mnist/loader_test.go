package mnist

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestLoadMNISTData(t *testing.T) {
	// Arrange
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	// Act
	trainImg, trainLbl, testImg, testLbl, err := loader.LoadData()

	// Assert
	if err != nil {
		t.Fatalf("LoadData() failed: %v", err)
	}

	// Check training data dimensions
	if len(trainImg) != 60000 {
		t.Errorf("Expected 60000 training images, got %d", len(trainImg))
	}
	if len(trainLbl) != 60000 {
		t.Errorf("Expected 60000 training labels, got %d", len(trainLbl))
	}

	// Check test data dimensions
	if len(testImg) != 10000 {
		t.Errorf("Expected 10000 test images, got %d", len(testImg))
	}
	if len(testLbl) != 10000 {
		t.Errorf("Expected 10000 test labels, got %d", len(testLbl))
	}

	// Check image dimensions (28x28 = 784 pixels)
	if len(trainImg) > 0 && len(trainImg[0]) != 784 {
		t.Errorf("Expected 784 pixels per image, got %d", len(trainImg[0]))
	}

	// Check label is valid digit (0-9)
	if len(trainLbl) > 0 && (trainLbl[0] < 0 || trainLbl[0] > 9) {
		t.Errorf("Expected label in range 0-9, got %d", trainLbl[0])
	}
}

func TestReadImagesLabels(t *testing.T) {
	// Arrange
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	// Act
	images, labels, err := loader.ReadImagesLabels(trainingImages, trainingLabels)

	// Assert
	if err != nil {
		t.Fatalf("ReadImagesLabels() failed: %v", err)
	}

	if len(images) != len(labels) {
		t.Errorf("Number of images (%d) doesn't match number of labels (%d)", len(images), len(labels))
	}
}

func TestInvalidMagicNumber(t *testing.T) {
	// Arrange
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	// Act - try to read labels file as images (wrong magic number)
	_, _, err := loader.ReadImagesLabels(trainingLabels, trainingLabels)

	// Assert
	if err == nil {
		t.Error("Expected error for invalid magic number, got nil")
	}
}

func TestGenerateHTML(t *testing.T) {
	// Arrange
	testImages := [][]uint8{
		make([]uint8, 784), // First image (all zeros)
		make([]uint8, 784), // Second image (all zeros)
	}
	// Add some non-zero pixels to make it interesting
	testImages[0][100] = 255
	testImages[1][200] = 128

	testLabels := []uint8{5, 3}
	outputPath := "test_mnist_viewer.html"

	// Clean up any existing test file
	os.Remove(outputPath)
	defer os.Remove(outputPath)

	// Act
	err := GenerateHTML(testImages, testLabels, outputPath)

	// Assert
	if err != nil {
		t.Fatalf("GenerateHTML() failed: %v", err)
	}

	// Verify file was created
	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("Failed to read generated HTML file: %v", err)
	}

	htmlContent := string(content)

	// Verify HTML contains expected elements
	expectedElements := []string{
		"<html",
		"<canvas",
		"Label: 5",
		"Label: 3",
		"getContext",
		"createImageData",
		"putImageData",
	}

	for _, expected := range expectedElements {
		if !strings.Contains(htmlContent, expected) {
			t.Errorf("Generated HTML missing expected element: %s", expected)
		}
	}

	// Verify we have 2 canvases (one for each image)
	canvasCount := strings.Count(htmlContent, "<canvas")
	if canvasCount != 2 {
		t.Errorf("Expected 2 canvas elements, found %d", canvasCount)
	}
}

func TestReaderSandbox(t *testing.T) {
	trainingImages := "mnist/archive/train-images.idx3-ubyte"
	trainingLabels := "mnist/archive/train-labels.idx1-ubyte"
	testImages := "mnist/archive/t10k-images.idx3-ubyte"
	testLabels := "mnist/archive/t10k-labels.idx1-ubyte"

	loader := NewMnistDataloader(trainingImages, trainingLabels, testImages, testLabels)

	// Act
	trainImg, trainLbl, _, _, _ := loader.LoadData()

	fmt.Printf("label: %d\n", trainLbl[0])
	fmt.Printf("img:\n%x\n", trainImg[0])
}
