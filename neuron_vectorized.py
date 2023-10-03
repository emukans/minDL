from typing import Iterable, Union, Sized

import numpy as np


class NeuronNetwork:
    def __init__(self, shape: Union[Iterable[int], Sized]):
        self.layer_length = len(shape)
        self.B = [np.random.randn(layer, 1) for layer in shape[1:]]
        self.W = [np.random.randn(current_layer, next_layer) for current_layer, next_layer in zip(shape[:-1], shape[1:])]

    # def feedforward(self, X, y):
    #     for weight, bias


nn = NeuronNetwork([2, 2, 1])
print(len(nn.B))
print(len(nn.W))
