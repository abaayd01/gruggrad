package main

import (
	"flag"
	"fmt"

	"abaayd01/gruggrad/mnist"
)

func main() {
	mode := flag.String("mode", "milestone1", "Training mode: milestone1 or mnetwork")
	flag.Parse()

	switch *mode {
	case "milestone1":
		fmt.Println("Running Milestone 1 training...")
		mnist.Milestone1()
	case "mnetwork":
		fmt.Println("Running MNetwork training...")
		mnist.MnistMNetworkTraining()
	default:
		fmt.Printf("Unknown mode: %s\n", *mode)
		fmt.Println("Available modes: milestone1, mnetwork")
	}
}
