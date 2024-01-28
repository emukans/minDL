from random import uniform
from typing import Callable, List, Optional, Union, Iterable, Sized, Tuple


def relu(x):
    return max(x, 0)


def relu_derivative(x):
    return 1 if x > 0 else 0


def compute_loss(y: float, pred: float):
    return (pred - y) ** 2


def compute_loss_derivative(y: float, pred: float) -> float:
    return 2 * (pred - y)


class Neuron:
    def __init__(self, bias: Optional[float] = None):
        self.bias = bias if bias is not None else uniform(0, 1)
        self.value = 0
        self.connection_list = []

    def __call__(self, state_list: List[float]):
        result = 0
        for connection, state in zip(
            self.connection_list, state_list
        ):  # type: (NeuronConnection, float)
            result += connection.weight * state

        self.value = result + self.bias

        return self.value

    def connect(self, neuron, weight: Optional[float]):
        self.connection_list.append(NeuronConnection(neuron, weight))


class NeuronConnection:
    def __init__(self, neuron: Neuron, weight: Optional[float] = None):
        self.neuron = neuron
        self.weight = weight if weight else uniform(0, 1)


class NeuralNetwork:
    def __init__(
        self,
        shape: Union[Iterable[int], Sized],
        activation_function: Callable = relu,
        activation_function_derivative: Callable = relu_derivative,
        learning_rate: float = 0.01,
    ):
        self.layer_list = []
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.learning_rate = learning_rate

        is_first = True
        for neuron_count in shape:
            neuron_value = None
            if is_first:
                neuron_value = 0
                is_first = False

            self._stack([Neuron(neuron_value) for _ in range(neuron_count)])

    def _stack(self, neuron_list: List[Neuron], weight_list: List[float] = None):
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
            raise Exception("Network is empty")

        if len(inputs) != len(self.layer_list[0]):
            raise Exception("Input shape do not match")

        calculated_values = inputs
        for layer in self.layer_list[1:]:
            calculated_values = [
                self.activation(neuron(calculated_values)) for neuron in layer
            ]

        return calculated_values[0]

    def backprop(self, y):
        error_list = [compute_loss_derivative(y, self.activation(self.layer_list[-1][0].value))]
        for layer in reversed(self.layer_list[1:]):
            updated_error_list = []
            for neuron, prediction_error in zip(layer, error_list):
                for connection in neuron.connection_list:
                    # Calculate error for the next layer
                    delta = (
                        prediction_error
                        * connection.weight
                        * self.activation_derivative(connection.neuron.value)
                    )
                    updated_error_list.append(delta)

                    # Weight update
                    connection.weight -= (
                        self.learning_rate
                        * prediction_error
                        * self.activation(connection.neuron.value)
                    )

                # Bias update
                neuron.bias -= self.learning_rate * prediction_error
                error_list = updated_error_list

    def fit(self, X: List[Tuple], y: List[int], iteration_count: int):
        for i in range(iteration_count):
            total_loss = 0
            for x_, y_ in zip(X, y):
                pred = self.forward(x_)

                loss = compute_loss(y_, pred)
                total_loss += loss

                self.backprop(y_)

            if i % 1000 == 0:
                print(f"Loss: {total_loss}")
