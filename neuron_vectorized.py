from typing import Iterable, Union, Sized, Callable

import numpy as np


np.random.seed(3)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    x = np.where(x > 0, 1, x)
    x = np.where(x < 0, 0, x)

    return x


def compute_loss(y: np.array, pred: np.array):
    return np.mean((y - pred) ** 2)


class NeuronNetwork:
    def __init__(
        self,
        shape: Union[Iterable[int], Sized],
        learning_rate: float,
        activation: Callable,
        activation_derivative: Callable,
    ):
        self.layer_length = len(shape)
        self.learning_rate = learning_rate
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.bias_list = [np.random.uniform(-1, 1, (1, layer)) for layer in shape[1:]]
        self.weight_list = [
            np.random.uniform(-1, 1, (current_layer, next_layer))
            for current_layer, next_layer in zip(shape[:-1], shape[1:])
        ]

        self.calculated_values = []

    def forward(self, X):
        self.calculated_values = [X]
        for weight, bias in zip(self.weight_list, self.bias_list):
            # !NB, using other activation functions could lead to corrupted input.
            # In this case, need to skip activating input neurons and pass the value as is.
            value = np.dot(self.activation(self.calculated_values[-1]), weight) + bias

            self.calculated_values.append(value)

        return self.activation(self.calculated_values[-1])

    def backprop(self, y):
        error_list = [(self.activation(self.calculated_values[-1]) - y) * 2]
        for i, value in enumerate(reversed(self.calculated_values[:-1]), 1):
            error = error_list[-1]

            # Calculate error for the next layer
            error_list.append(
                np.dot(error, self.weight_list[-i].T)
                * self.activation_derivative(value)
            )

            # Weight update
            self.weight_list[-i] -= (
                np.dot(self.activation(value.T), error) * self.learning_rate
            )
            # Bias update
            self.bias_list[-i] -= (
                np.sum(error, axis=0, keepdims=True) * self.learning_rate
            )

    def fit(self, X, y, iteration_count):
        for i in range(iteration_count):
            pred = self.forward(X)

            loss = compute_loss(y, pred)

            self.backprop(y)

            if i % 1000 == 0:
                print(f"Loss: {loss}")


if __name__ == "__main__":
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
