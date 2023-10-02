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

    def forward(self, inputs):
        if not len(self.layer_list):
            raise Exception('Network is empty')

        if len(inputs) != len(self.layer_list[0]):
            raise Exception('Input shape do not match')

        calculated_values = inputs
        outputs = [calculated_values]
        inputs = [calculated_values]
        for layer in self.layer_list[1:]:
            calculated_values = [neuron(calculated_values) for neuron in layer]
            inputs.append(calculated_values)
            calculated_values = [self.activation(value) for value in calculated_values]
            outputs.append(calculated_values)

        return inputs, outputs

    def backprop(self, y_list, y_prime_list):
        for y, (y_inputs, y_outputs) in zip(y_list, y_prime_list):
            error_list = [(y_outputs[-1][0] - y)]
            for i, layer in reversed(list(enumerate(self.layer_list))):
                updated_error_list = []
                for j, neuron in enumerate(layer):
                    if not len(neuron.connection_list):
                        continue

                    for connection in neuron.connection_list:
                        k = list(self.layer_list[i - 1]).index(connection.neuron)
                        delta = error_list[j] * connection.weight * heviside_derivative(y_inputs[i - 1][k])
                        updated_error_list.append(delta)
                        connection.weight -= self.learning_rate * error_list[j] * y_outputs[i - 1][k]

                    neuron.bias -= self.learning_rate * error_list[j]
                    error_list = updated_error_list
#         d_output = 2 * (predicted_output - target)
#         d_output_bias = d_output
#
#         d_hidden = [0] * hidden_size
#         d_hidden_weights = [[0] * hidden_size for _ in range(input_size)]
#         d_hidden_biases = [0] * hidden_size
#
#         for i in range(hidden_size):
#             d_hidden[i] = d_output * output_weights[i] * relu_derivative(hidden_layer_input[i])
#             d_hidden_biases[i] = d_hidden[i]
#             for j in range(input_size):
#                 d_hidden_weights[j][i] = d_hidden[i] * input_data[j]
#
#         # Update weights and biases
#         for i in range(hidden_size):
#             output_weights[i] -= learning_rate * d_output * hidden_layer_output[i]
#             hidden_biases[i] -= learning_rate * d_hidden_biases[i]
#             for j in range(input_size):
#                 hidden_weights[j][i] -= learning_rate * d_hidden_weights[j][i]
#         output_bias -= learning_rate * d_output_bias



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
    y_prime = [l[1][-1][0] for l in y_prime]
    for expected, result in zip(y, y_prime):
        diff += (expected - result) ** 2

    return diff


X = [(0, 0),
     (1, 0),
     (0, 1),
     (1, 1)]

y = [0, 1, 1, 0]

for i in range(10000):
    y_prime = []
    for x in X:
        y_prime.append(nn.forward(x))

    if i % 1000 == 0:
        print(compute_loss(y, y_prime))
    nn.backprop(y, y_prime)
    # print(y_prime)


# print([n.weight for n in nn.layer_list[-1][0].connection_list])
# print([n.weight for n in nn.layer_list[-2][0].connection_list])
y_prime = []
for x in X:
    y_prime.append(nn.forward(x))

nn.backprop(y, y_prime)
# print(compute_loss(y, y_prime))
# print([l[-1][0] for l in y_prime])

print([round(l[1][-1][0], 2) for l in y_prime])

# import random
#
#
# random.seed(1)
#
#
# # Define the activation function (ReLU)
# def relu(x):
#     return max(0, x)
#     # return 1 if x >= 0 else 0
#
#
# # Define the derivative of the ReLU activation function
# def relu_derivative(x):
#     return 1 if x > 0 else 0
#     # return max(x, 0)
#
#
# # Initialize weights and biases
# input_size = 2
# hidden_size = 2
# output_size = 1
#
# # Initialize weights randomly
# hidden_weights = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
# output_weights = [random.uniform(-1, 1) for _ in range(hidden_size)]
# hidden_biases = [random.uniform(-1, 1) for _ in range(hidden_size)]
# output_bias = random.uniform(-1, 1)
#
#
# # Define the learning rate
# learning_rate = 0.01
#
# # Define the training data (input and target output)
# training_data = [
#     ([0, 0], 0),
#     ([0, 1], 1),
#     ([1, 0], 1),
#     ([1, 1], 0)
# ]
#
#
# # Training loop
# epochs = 10000
#
# for epoch in range(epochs):
#     total_loss = 0
#
#     for input_data, target in training_data:
#         # Forward pass
#         hidden_layer_input = [0] * hidden_size
#         hidden_layer_output = [0] * hidden_size
#
#         # forward pass
#         for i in range(hidden_size):
#             for j in range(input_size):
#                 hidden_layer_input[i] += input_data[j] * hidden_weights[j][i]
#             hidden_layer_input[i] += hidden_biases[i]
#             hidden_layer_output[i] = relu(hidden_layer_input[i])
#
#         output_layer_input = sum(hidden_layer_output[i] * output_weights[i] for i in range(hidden_size)) + output_bias
#         predicted_output = output_layer_input
#
#         # Calculate the loss (mean squared error)
#         loss = (predicted_output - target) ** 2
#         total_loss += loss
#
#         # Backpropagation
#         d_output = 2 * (predicted_output - target)
#         d_output_bias = d_output
#
#         d_hidden = [0] * hidden_size
#         d_hidden_weights = [[0] * hidden_size for _ in range(input_size)]
#         d_hidden_biases = [0] * hidden_size
#
#         for i in range(hidden_size):
#             d_hidden[i] = d_output * output_weights[i] * relu_derivative(hidden_layer_input[i])
#             d_hidden_biases[i] = d_hidden[i]
#             for j in range(input_size):
#                 d_hidden_weights[j][i] = d_hidden[i] * input_data[j]
#
#         # Update weights and biases
#         for i in range(hidden_size):
#             output_weights[i] -= learning_rate * d_output * hidden_layer_output[i]
#             hidden_biases[i] -= learning_rate * d_hidden_biases[i]
#             for j in range(input_size):
#                 hidden_weights[j][i] -= learning_rate * d_hidden_weights[j][i]
#         output_bias -= learning_rate * d_output_bias
#
#     # if epoch % 1000 == 0:
#     #     print(f"Epoch {epoch}: Loss = {total_loss}")
#
# # Test the trained network
# for input_data, target in training_data:
#     hidden_layer_input = [0] * hidden_size
#     hidden_layer_output = [0] * hidden_size
#
#     for i in range(hidden_size):
#         for j in range(input_size):
#             hidden_layer_input[i] += input_data[j] * hidden_weights[j][i]
#         hidden_layer_input[i] += hidden_biases[i]
#         hidden_layer_output[i] = relu(hidden_layer_input[i])
#
#     output_layer_input = sum(hidden_layer_output[i] * output_weights[i] for i in range(hidden_size)) + output_bias
#     predicted_output = output_layer_input
#
#     print(f"Input: {input_data}, Target: {target}, Predicted: {round(predicted_output, 2)}")
