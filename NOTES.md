 What You Need to Add

  1. Activation Functions

  Non-linear functions like:
  - ReLU: max(0, x) - outputs x if positive, 0 otherwise
  - Tanh: squashes values between -1 and 1
  - Sigmoid: squashes values between 0 and 1

  These need to be new operations on your Value type, with their derivatives for backprop.

  2. Parameter Management

  You need to track which Values are parameters (weights and biases that you'll update) vs which are data (inputs). Maybe a IsParameter flag or a separate list.

  3. The Training Loop Pattern

  1. Forward pass: Feed input through the network to get output
  2. Compute loss: How wrong is the output? (e.g., mean squared error)
  3. Zero gradients: Reset all parameter gradients to 0
  4. Backward pass: Compute gradients via backprop
  5. Update parameters: weights -= learning_rate * gradient
  6. Repeat

  Structuring It

  Neuron:
  - Has a list of weight Values (one per input)
  - Has one bias Value
  - Has a forward method that does the weighted sum + activation

  Layer:
  - A collection of neurons
  - Forward pass calls each neuron's forward

  Network:
  - A collection of layers
  - Forward pass chains layer outputs as inputs to next layer
  - Has a method to collect all parameters from all neurons

  The Beautiful Part

  Your Value implementation already handles the hard part (backprop through arbitrary computation graphs). Building a neural net is just:
  1. Adding a few more operations (activations)
  2. Creating convenient wrappers (Neuron, Layer, Network)
  3. Implementing the training loop pattern

  The gradient computation? Already solved. You just need to organize the Values into the neural network structure.

  Does this conceptual map make sense?

---

# Getting my simple NN to work for MNIST data set 

The Goal:
Use gruggrad to train on the MNIST dataset, and classify some numbers

How to get there?
- softmax + cross-entropy (output treatment)
- get the data in a shape I can feed into my NN (input treatment)
- design the 'parameters' of the NN
    - size, learning rate, activation functions
- training loop harness
- verification loop harness

Milestone 1: feed a single image into the NN
- [x] transform image pixels into a slice of Value ptrs
    - data should be normalised between 0 and 1
    - need to keep the label associated somehow for later?

Milestone 2: feed a single image into the NN, compute the loss using softmax + cross-entropy
- [x] implement softmax func
- [x] implement cross-entropy
- [x] compute loss -> backprop
    - cross-entropy loss function ideally requires data be processed by softmax func first
    - softmax is needed for interpretable outputs i.e. with what probability is this 'x' digit (instead of a vector of random floats)

- [x] set up training loop
- [x] set up verification harness to assess accuracy of trained model

# Learnings and next steps
- leaky relu and gradient clipping help with dead nuerons and the output 'jumping around the optimal loss' too aggresively
- also, when accumulating gradients within softmax, neded to be careful not to revisit nodes and explode out the gradient causing overflows / NaNs
- training is working, I can train models on about 2000 samples in roughly a minute
- moving to matrix based approach is needed next though, to go much faster
- I could also work on verifying against my own handwriting?

# 2025-12-24

Next Goal: implement NN with matrix based operations (still running on a single core on the CPU)

Steps:
- [x] implement forward pass
    - use the XOR trained network weights to verify it's working correctly
- matrix backprop
    - how to do a derivative of a matrix?
    - how to keep track of the operations over the matrix for chain rule?
        - is that even the right approach??
- set up a training loop

# 2025-12-25
Learn how to do backprop with matrices

What do I know from the backprop on scalars?
- i could keep track of the operations that produce a given matrix, and do local derivatives?
- accoring to Gemini, this is the right approach I keep a DAG of operations, then do local derivatives
    - figuring out local gradients for backprop is going to be confusing, but key is resulting grad matrix needs to be of same shape as layer

# 2025-12-27
- [x] convert matrix net to used tracked matrices for underlying structure instead of matrix
- [x] implement some unit tests for MatMul backprop
- [x] support Add operations in backprop
- [x] figure out how to support relu in backprop
- [x] train on the XOR problem

### Train + verify on the MNIST dataset
- get forward pass working on the MNIST dataset working using a pre-trained network
- cross-entropy loss + softmax func - matrix versions
