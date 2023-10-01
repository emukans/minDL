from random import random, seed
from typing import Callable, List, Optional


seed(42)


def step(x):
    # return 1 if x >= 0 else 0
    return max(x, 0)


def heviside_derivative(x):
    # return 1 if x == 0 else 0
    # return max(x, 0)
    return 1 if x >= 0 else 0


class Neuron:
    def __init__(self, bias: Optional[float] = None):
        self.bias = bias if bias is not None else random()
        self.connection_list = []

    def __call__(self, state_list: List[float]):
        result = 0
        for connection, state in zip(self.connection_list, state_list):  # type: (NeuronConnection, float)
            result += connection.weight * state

        return result + self.bias

    def connect(self, neuron, weight: Optional[float]):
        self.connection_list.append(NeuronConnection(neuron, weight))


class NeuronConnection:
    def __init__(self, neuron: Neuron, weight: Optional[float] = None):
        self.neuron = neuron
        self.weight = weight if weight else random()


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

    def forward(self, inputs):
        if not len(self.layer_list):
            raise Exception('Network is empty')

        if len(inputs) != len(self.layer_list[0]):
            raise Exception('Input shape do not match')

        calculated_values = inputs
        result = []
        for layer in self.layer_list[1:]:
            calculated_values = [self.activation(neuron(calculated_values)) for neuron in layer]
            result.append(calculated_values)

        return result

    def backprop(self, y, y_prime):
        error_list = [(output[1][0] - expected) * heviside_derivative(output[1][0]) for expected, output in zip(y, y_prime)]

        for i, layer in reversed(list(enumerate(self.layer_list[1:]))):
            updated_error_list = []
            for j, neuron in enumerate(layer):
                neuron_error = 0
                for connection in neuron.connection_list:
                    delta = [connection.weight * error * heviside_derivative(output[j][i]) for error, output in zip(error_list, y_prime)]
                    neuron_error += sum(delta)

                    # print(self.learning_rate * sum(delta))
                    if len(updated_error_list):
                        updated_error_list = [e + d for e, d in zip(updated_error_list, delta)]
                    else:
                        updated_error_list = delta

                # neuron.bias -= self.learning_rate * neuron_error

                for connection in neuron.connection_list:
                    connection.weight -= self.learning_rate * sum([output[j][i] * delta for output, delta in zip(y_prime, updated_error_list)])

            # print('updated_error_list', updated_error_list)
            error_list = updated_error_list


        # [-0.02391072787787527, 0.0, 0.0, -0.02391072787787527]
        # [-0.31073327482590063, 0.0, 0.0, -0.31073327482590063]
        # [-0.019405681005008935, 0.0, 0.0, -0.019405681005008935]
        # [-0.28551427911356453, 0.0, 0.0, -0.28551427911356453]
        # error_list =
        # print(error_list)


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


def compute_loss(y: List[float], y_prime: List[List[float]]):
    diff = 0
    y_prime = [l[-1][0] for l in y_prime]
    for expected, result in zip(y, y_prime):
        diff += expected - result

    return diff / len(y)


X = [(0, 0),
     (1, 0),
     (0, 1),
     (1, 1)]

y = [0, 1, 1, 0]

for i in range(10000):
    y_prime = []
    for x in X:
        y_prime.append(nn.forward(x))

    print(compute_loss(y, y_prime))
    nn.backprop(y, y_prime)


# print([n.weight for n in nn.layer_list[-1][0].connection_list])
y_prime = []
for x in X:
    y_prime.append(nn.forward(x))

# nn.backprop(y, y_prime)
print([l[-1][0] for l in y_prime])