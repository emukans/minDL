from random import random, uniform, seed
from typing import Callable, List, Optional


seed(5)


def step(x):
    # return 1 if x >= 0 else 0
    return max(x, 0)


def heviside_derivative(x):
    # return 1 if x == 0 else 0
    # return max(x, 0)
    return 1 if x > 0 else 0


class Neuron:
    def __init__(self, bias: Optional[float] = None):
        self.bias = bias if bias is not None else uniform(-1, 1)
        self.value = 0
        self.connection_list = []

    def __call__(self, state_list: List[float]):
        result = 0
        for connection, state in zip(self.connection_list, state_list):  # type: (NeuronConnection, float)
            result += connection.weight * state

        self.value = result + self.bias

        return self.value

    def connect(self, neuron, weight: Optional[float]):
        self.connection_list.append(NeuronConnection(neuron, weight))


class NeuronConnection:
    def __init__(self, neuron: Neuron, weight: Optional[float] = None):
        self.neuron = neuron
        self.weight = weight if weight else uniform(-1, 1)


class NeuralNetwork:
    def __init__(self, activation_function: Callable, learning_rate: float):
        self.layer_list = []
        self.activation = activation_function
        self.learning_rate = learning_rate

    def stack(self, neuron_list: List[Neuron], weight_list: List[float] = None):
        self.layer_list.append(neuron_list)

        if len(self.layer_list) > 1:
            if not weight_list:
                weight_list = [None] * len(self.layer_list[-2]) * len(neuron_list)

            weight_list = iter(weight_list)
            for prev_neuron in self.layer_list[-2]:  # type: Neuron
                for curr_neuron in neuron_list:  # type: Neuron
                    curr_neuron.connect(prev_neuron, next(weight_list))

    def fit(self, data, iteration_count):
        for i in range(iteration_count):
            total_loss = 0
            for X, y in data:
                pred = self.forward(X)

                loss = compute_loss(y, pred)
                total_loss += loss

                self.backprop(y)

            if i % 1000 == 0:
                print(f'Loss: {total_loss}')

    def forward(self, inputs):
        if not len(self.layer_list):
            raise Exception('Network is empty')

        if len(inputs) != len(self.layer_list[0]):
            raise Exception('Input shape do not match')

        calculated_values = inputs
        for layer in self.layer_list[1:]:
            calculated_values = [self.activation(neuron(calculated_values)) for neuron in layer]

        return calculated_values[0]

    def backprop(self, y):
        error_list = [(self.activation(self.layer_list[-1][0].value) - y)]
        for i, layer in reversed(list(enumerate(self.layer_list))):
            updated_error_list = []
            for j, neuron in enumerate(layer):
                if not len(neuron.connection_list):
                    continue

                for connection in neuron.connection_list:
                    k = list(self.layer_list[i - 1]).index(connection.neuron)
                    delta = error_list[j] * connection.weight * heviside_derivative(self.layer_list[i - 1][k].value)
                    updated_error_list.append(delta)
                    connection.weight -= self.learning_rate * error_list[j] * self.activation(self.layer_list[i - 1][k].value)

                neuron.bias -= self.learning_rate * error_list[j]
                error_list = updated_error_list


nn = NeuralNetwork(
    activation_function=step,
    learning_rate=0.01
)

# nn.stack([Neuron(0), Neuron(0)])
# nn.stack([Neuron(-1.6), Neuron(-0.3)], [1, 1, 1, 1])
# nn.stack([Neuron(-0.4)], [-2, 1])

nn.stack([Neuron(0), Neuron(0)])
nn.stack([Neuron(), Neuron()])
nn.stack([Neuron()])


def compute_loss(y: float, pred: float):
    return ((pred - y) ** 2) / 2


data = [((0, 0), 0),
     ((1, 0), 1),
     ((0, 1), 1),
     ((1, 1), 0)]

nn.fit(data, 10000)

for X, y in data:
    print(f'Ground-truth: {y}, Predicted: {nn.forward(X)}')
