from enum import StrEnum

import numpy as np

from mindl.function import Function


class ReLU(Function):
    def __call__(self, x: np.array):
        return np.maximum(x, 0)

    def derivative(self, x: np.array):
        x = np.where(x > 0, 1, x)
        x = np.where(x < 0, 0, x)
        return x


class Sigmoid(Function):
    def __call__(self, x: np.array):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array):
        return self(x) * (1 - self(x))


class Activation(StrEnum):
    ReLU = 'ReLU'
    Sigmoid = 'Sigmoid'

    def map_activation(self) -> Function:
        match self.value.lower():
            case 'relu':
                return ReLU()
            case 'sigmoid':
                return Sigmoid()

        raise ValueError('Unknown activation')
