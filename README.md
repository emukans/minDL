# minDL
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Minimalistic set of tools for Deep Learning.

**!NB, this repository is for educational purposes and is not intended for production use.
For production use PyTorch, Tensorflow, etc.**

## Getting started
1. Create a new environment
    ```shell
    python -m venv venv
    ```
2. Activate the new environment (for Windows the activate script will have a different name)
    ```shell
    source venv/bin/activate
    ```
3. Install dependencies
    ```shell
    pip install -r requirements.txt
    ```

## Neural network
There are 2 implementations of a feedforward neural networks so far: sequential and vectorized.

### Sequential neural network
Written on pure python, without any dependencies.

#### An example of training a XOR function

```shell
from neuron import NeuralNetwork, Neuron, relu, relu_derivative


# Initialize neural network
nn = NeuralNetwork(
    activation_function=relu,
    activation_function_derivative=relu_derivative,
    learning_rate=0.01,
)

# Define an architecture. In this case it's 2-2-1 (input-hidden-output)
nn.stack([Neuron(0), Neuron(0)])
nn.stack([Neuron(), Neuron()])
nn.stack([Neuron()])

# Initialize XOR dataset
data = [((0, 0), 0), ((1, 0), 1), ((0, 1), 1), ((1, 1), 0)]

# Train the neural network
nn.fit(data, 10000)

# Predict the output
for X, y in data:
    print(f"Ground-truth: {y}, Predicted: {nn.forward(X)}")
```

### Vectorized neural network
This implementation is based on matrix computations and are more effective on large datasets and bigger networks.

#### An example of training a XOR function
```shell
import numpy as np
from neuron_vectorized import NeuronNetwork, relu, relu_derivative


nn = NeuronNetwork(
    [2, 2, 1],
    learning_rate=0.01,
    activation=relu,
    activation_derivative=relu_derivative,
)

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(
    f"Ground-truth: {y.squeeze(1)}, Predicted: {[round(a[0]) for a in nn.forward(X)]}"
)
nn.fit(X, y, 10000)

print(
    f"Ground-truth: {y.squeeze(1)}, Predicted: {[round(a[0]) for a in nn.forward(X)]}"
)
```