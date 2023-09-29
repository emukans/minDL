from random import random
from typing import Callable, List


def step(x):
    return 1 if x > 0 else 0


class Neuron:
    def __init__(self, bias=None):
        self.bias = bias if bias is not None else random()
        self.connection_list = []

    def __call__(self, state_list: List[float]):
        result = 0
        for connection, state in zip(self.connection_list, state_list):  # type: (NeuronConnection, float)
            result += connection.weight * state

        return result + self.bias

    def connect(self, neuron, weight):
        self.connection_list.append(NeuronConnection(neuron, weight))


class NeuronConnection:
    def __init__(self, neuron: Neuron, weight=None):
        self.neuron = neuron
        self.weight = weight if weight else random()


class NeuralNetwork:
    def __init__(self):
        self.layer_list = []
        self.activation = None

    def set_activation(self, f: Callable):
        self.activation = f

    def stack(self, neuron_list: List[Neuron], weight_list: List[float] = None):
        self.layer_list.append(neuron_list)

        if len(self.layer_list) > 1:
            if not weight_list:
                weight_list = [None] * len(self.layer_list[-2]) * len(neuron_list)

            weight_list = iter(weight_list)
            for prev_neuron in self.layer_list[-2]:  # type: Neuron
                for curr_neuron in neuron_list:  # type: Neuron
                    curr_neuron.connect(prev_neuron, next(weight_list))

    def forward(self, inputs):
        if not len(self.layer_list):
            raise Exception('Network is empty')

        if len(inputs) != len(self.layer_list[0]):
            raise Exception('Input shape do not match')

        calculated_values = inputs
        for value, input_neuron in zip(calculated_values, self.layer_list[0]):
            input_neuron.bias += value

        for layer in self.layer_list:
            calculated_values = [self.activation(neuron(calculated_values)) for neuron in layer]

        return calculated_values


nn = NeuralNetwork()
nn.stack([Neuron(0), Neuron(0)])
nn.stack([Neuron(-1.6), Neuron(-0.3)], [1, 1, 1, 1])
nn.stack([Neuron(-0.4)], [-2, 1])

nn.set_activation(step)

print(nn.forward([0, 0]))
