from typing import Union, Iterable, Sized

import numpy as np

from mindl.function import Function


class NeuralNetwork:
    """
    The class represents the neural network for approximating functions.

    Usually the neural network implementation and training code is separated.
    In this realisation everything is merged into a single class for simplicity.

    An example of neural network (nn.Modules) representation in PyTorch (https://pytorch.org/docs/stable/notes/modules.html)
    Training algorithm realisation for machine learning (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit)
    and for deep learning an example could be Trainer from PyTorch Lightning (https://lightning.ai/docs/pytorch/stable/common/trainer.html#)
    """

    def __init__(
        self,
        shape: Union[Iterable[int], Sized],
        learning_rate: float,
        activation: Function,
        loss: Function,
        log_frequency: int = 1000
    ):
        self.layer_length = len(shape)
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss
        self.log_frequency = log_frequency

        self.bias_list = [np.random.uniform(-1, 1, (1, layer)) for layer in shape[1:]]
        self.weight_list = [
            np.random.uniform(-1, 1, (current_layer, next_layer))
            for current_layer, next_layer in zip(shape[:-1], shape[1:])
        ]

        self.calculated_values = []

    def forward(self, X: np.array):
        """
        Feedforward the input through the neural network.

        :param X:

        :return:
        """
        self.calculated_values = [X]
        is_first = True
        for weight, bias in zip(self.weight_list, self.bias_list):
            if is_first:
                value = np.dot(self.calculated_values[-1], weight) + bias
                is_first = False
            else:
                value = np.dot(self.activation(self.calculated_values[-1]), weight) + bias

            self.calculated_values.append(value)

        return self.activation(self.calculated_values[-1])

    def backprop(self, y: np.array):
        """
        Implementation of the backpropagation algorithm.

        :param np.array y: dataset output

        :return:
        """
        error_list = [self.loss.derivative(y, self.activation(self.calculated_values[-1]))]
        for i, value in enumerate(reversed(self.calculated_values[:-1]), 1):
            error = error_list[-1]

            # Calculate error for the next layer
            error_list.append(
                np.dot(error, self.weight_list[-i].T)
                * self.activation.derivative(value)
            )

            # Weight update
            self.weight_list[-i] -= (
                np.dot(self.activation(value.T), error) * self.learning_rate
            )
            # Bias update
            self.bias_list[-i] -= (
                np.sum(error, axis=0, keepdims=True) * self.learning_rate
            )

    def fit(self, X: np.array, y: np.array, iteration_count: int):
        """
        A method for training the neural network. Usually this method is implemented separately from the model.
        See examples in the class docstring above.

        :param np.array X: dataset input
        :param np.array y: dataset output
        :param int iteration_count: iteration count in the training loop

        :return:
        """
        for i in range(iteration_count):
            pred = self.forward(X)

            loss = np.mean(self.loss(y, pred))

            self.backprop(y)

            if i % self.log_frequency == 0:
                print(f"Loss: {loss}")
