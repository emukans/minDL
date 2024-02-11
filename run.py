from mindl.nn import NeuralNetwork
from plaindl.nn import NeuralNetwork as PlainNeuralNetwork
from mindl.function.activation import ReLU
from mindl.function.loss import MSE
from random import seed
import numpy as np


def set_seed(seed_value):
    seed(seed_value)
    np.random.seed(seed_value)


def print_output_and_target_comparison(nn, X, y):
    pred = [np.rint(p).astype('int') for p in nn.forward(X)]

    print()
    for y_, p in zip(y, pred):
        print(f"Ground-truth: {y_}, Predicted: {p}")


set_seed(3)

# Initialisation
nn = NeuralNetwork(
    [2, 2, 1],
    learning_rate=0.01,
    activation=ReLU(),
    loss=MSE(),
    log_frequency=50
)

# Define input-output structure
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print('Before:')
print_output_and_target_comparison(nn, X, y)

# Training the model
nn.fit(X, y, 800)

print('After:')
print_output_and_target_comparison(nn, X, y)

# Persist the model
nn.save('model/xor.json')
