package main

import (
	"fmt"
	"os"

	"abaayd01/gruggrad/mnist"
)

func main() {
	fmt.Println("Loading MNIST dataset...")

	loader := mnist.NewMnistDataloader(
		"mnist/archive/train-images.idx3-ubyte",
		"mnist/archive/train-labels.idx1-ubyte",
		"mnist/archive/t10k-images.idx3-ubyte",
		"mnist/archive/t10k-labels.idx1-ubyte",
	)

	_, _, verificationImgs, verificationLabels, err := loader.LoadData()
	if err != nil {
		fmt.Printf("Error loading MNIST data: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d test images\n", len(verificationImgs))

	// Take first 20 images
	numImages := 20
	if len(verificationImgs) < numImages {
		numImages = len(verificationImgs)
	}

	images := verificationImgs[:numImages]
	labels := verificationLabels[:numImages]

	outputFile := "mnist_viewer.html"
	fmt.Printf("Generating HTML file with %d images...\n", numImages)

	err = mnist.GenerateHTML(images, labels, outputFile)
	if err != nil {
		fmt.Printf("Error generating HTML: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Success! Open '%s' in your browser to view the images.\n", outputFile)
}
