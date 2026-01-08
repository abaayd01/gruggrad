package main

import (
	"flag"
	"fmt"

	"abaayd01/gruggrad/mnist"
)

func main() {
	mode := flag.String("mode", "milestone1", "Verification mode: milestone1 or mnetwork")
	flag.Parse()

	switch *mode {
	case "milestone1":
		fmt.Println("Running Milestone 1 verification...")
		mnist.Milestone1Verify()
	case "mnetwork":
		fmt.Println("Running MNetwork verification...")
		mnist.MnistMNetworkVerify()
	default:
		fmt.Printf("Unknown mode: %s\n", *mode)
		fmt.Println("Available modes: milestone1, mnetwork")
	}
}
