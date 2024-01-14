from typing import Union, Iterable, Sized

import numpy as np

from mindl.function import Function


class NeuralNetwork:
    def __init__(
        self,
        shape: Union[Iterable[int], Sized],
        learning_rate: float,
        activation: Function,
        loss: Function,
    ):
        self.layer_length = len(shape)
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss

        self.bias_list = [np.random.uniform(-1, 1, (1, layer)) for layer in shape[1:]]
        self.weight_list = [
            np.random.uniform(-1, 1, (current_layer, next_layer))
            for current_layer, next_layer in zip(shape[:-1], shape[1:])
        ]

        self.calculated_values = []

    def forward(self, X):
        self.calculated_values = [X]
        is_first = True
        for weight, bias in zip(self.weight_list, self.bias_list):
            if is_first:
                value = np.dot(self.calculated_values[-1], weight) + bias
                is_first = False
            else:
                value = np.dot(self.activation(self.calculated_values[-1]), weight) + bias

            self.calculated_values.append(value)

        return self.activation(self.calculated_values[-1])

    def backprop(self, y):
        error_list = [self.loss.derivative(y, self.activation(self.calculated_values[-1]))]
        for i, value in enumerate(reversed(self.calculated_values[:-1]), 1):
            error = error_list[-1]

            # Calculate error for the next layer
            error_list.append(
                np.dot(error, self.weight_list[-i].T)
                * self.activation.derivative(value)
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

            loss = np.mean(self.loss(y, pred))

            self.backprop(y)

            if i % 1000 == 0:
                print(f"Loss: {loss}")
