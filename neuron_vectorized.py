from typing import Iterable, Union, Sized

import numpy as np


np.random.seed(3)


def relu(x):
    return np.maximum(x, 0)
    # x = np.where(x >= 0, 1, x)
    # x = np.where(x < 0, 0, x)
    # return x


def relu_derivative(x):
    x = np.where(x > 0, 1, x)
    x = np.where(x < 0, 0, x)

    return x


def compute_loss(y: np.array, pred: np.array):
    return np.mean((y - pred) ** 2)
#    return ((pred - y)**2).sum() / (2*pred.size)


class NeuronNetwork:
    def __init__(self, shape: Union[Iterable[int], Sized]):
        self.layer_length = len(shape)

        self.bias_list = [np.random.uniform(-1, 1, (1, layer)) for layer in shape[1:]]
        # self.bias_list = [np.zeros((1, layer)) for layer in shape[1:]]
        self.weight_list = [np.random.uniform(-1, 1, (current_layer, next_layer)) for current_layer, next_layer in zip(shape[:-1], shape[1:])]

        self.calculated_values = []

    def forward(self, X):
        self.calculated_values = [X]
        for weight, bias in zip(self.weight_list, self.bias_list):
            value = np.dot(relu(self.calculated_values[-1]), weight) + bias  # if first, then not need relu

            self.calculated_values.append(value)

        return relu(value)

    def backprop(self, y):
        learning_rate = 0.01
        error = (relu(self.calculated_values[-1]) - y) * 2
        for i, value in enumerate(reversed(self.calculated_values[:-1]), 1):
            error_update = np.dot(error, self.weight_list[-i].T) * relu_derivative(value)
            self.weight_list[-i] -= np.dot(relu(value.T), error) * learning_rate
            self.bias_list[-i] -= np.sum(error, axis=0, keepdims=True) * learning_rate
            error = error_update

    def fit(self, X, y, iteration_count):
        for i in range(iteration_count):
            pred = self.forward(X)

            loss = compute_loss(y, pred)

            self.backprop(y)

            if i % 1000 == 0:
                print(f'Loss: {loss}')


nn = NeuronNetwork([2, 2, 1])

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(f'Ground-truth: {y.squeeze(1)}, Predicted: {[round(a[0]) for a in nn.forward(X)]}')
nn.fit(X, y, 10000)

print(f'Ground-truth: {y.squeeze(1)}, Predicted: {[round(a[0]) for a in nn.forward(X)]}')

