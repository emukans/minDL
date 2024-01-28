from enum import StrEnum

import numpy as np

from mindl.function import Function


class MSE(Function):
    def __call__(self, y: np.array, pred: np.array):
        return (y - pred) ** 2

    def derivative(self, y: np.array, pred: np.array):
        return 2 * (pred - y)


class Loss(StrEnum):
    MSE = 'MSE'

    def map_loss(self):
        match self.value.lower():
            case 'mse':
                return MSE()

        raise ValueError('Unknown loss function')
