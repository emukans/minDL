from dataclasses import dataclass
from typing import Callable, Union


def step(x):
    return 1 if x > 0 else 0


@dataclass
class Neuron:
    bias: float

    def __call__(self, f: Union[Callable, float]):
        if type(f) is float:
            return f + self.bias

        return f() + self.bias


x1 = Neuron(0)
x2 = Neuron(0)

h1 = Neuron(-1.6)
h2 = Neuron(-0.3)

y = Neuron(-0.4)

h1_v = h1(lambda: x1(1.) * 1 + x2(1.) * 1)
h2_v = h2(lambda: x1(1.) * 1 + x2(1.) * 1)

y_v = y(lambda: step(h1_v) * (-2) + step(h2_v) * 1)


print(y_v)
