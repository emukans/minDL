from typing import Iterable, Union, Sized

import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    x = np.where(x > 0, 1, x)
    x = np.where(x < 0, 0, x)

    return x


class NeuronNetwork:
    def __init__(self, shape: Union[Iterable[int], Sized]):
        self.layer_length = len(shape)
        self.B = [np.random.randn(layer, 1) for layer in shape[1:]]
        self.W = [np.random.randn(current_layer, next_layer) for current_layer, next_layer in zip(shape[:-1], shape[1:])]

        self.z_list = []
        self.a_list = []

    def forward(self, X):
        self.z_list = []
        self.a_list = []
        for weight, bias in zip(self.W, self.B):
            if len(self.a_list):
                Z = np.dot(self.a_list[-1], weight) + bias
            else:
                Z = np.dot(X, weight) + bias

            self.z_list.append(Z)
            A = relu(Z)
            self.a_list.append(A)

        return self.a_list[-1]

    def backprop(self, y):
        e_list = []
        for A, Z in reversed(zip(self.a_list, self.z_list)):
            if len(e_list):
                E = A - y
            # else:
            #     np.dot(dW1, W2.T)
            # dW1 = E * A * (1 - A)
            #
            # E2 = np.dot(dW1, W2.T)
            # dW2 = E2 * A1 * (1 - A1)


nn = NeuronNetwork([2, 2, 1])
print(len(nn.B))
print(len(nn.W))
