import numpy as np

from mindl.function import Function


class MSE(Function):
    def __call__(self, y: np.array, pred: np.array):
        return (y - pred) ** 2

    def derivative(self, y: np.array, pred: np.array):
        return 2 * (pred - y)

