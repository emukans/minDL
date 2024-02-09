import argparse
import json

from typing import Self

from manim import *


def calc_neuron(inputs, bias):
    return round(sum([w * max(i, 0) for w, i in inputs]) + bias, 1)


class Neuron(VMobject):
    def __init__(
        self,
        size=0.5,
        value=None,
    ):
        super().__init__()
        self.neuron = Circle(size, color=WHITE)
        self.add(self.neuron)
        self.size = size

        if value:
            self.initial_value = value
            self.value_unit = DecimalNumber(
                value.get_value(),
                font_size=DEFAULT_FONT_SIZE * size,
                num_decimal_places=1,
                color=WHITE,
            )
            self.value_unit.add_updater(lambda m: m.set_value(value.get_value()).move_to(self.neuron.get_center()))
            self.value_unit_updated = None
            self.value_unit.move_to(self.neuron.get_center())
            self.add(self.value_unit)

        self.connection_group = VGroup()
        self.add(self.connection_group)
        self.connection_line_group = VGroup()
        self.connection_label_group = VGroup()

    def draw_connection_to(
        self, object: Self, label=None, label_placement=RIGHT
    ):
        line = Line(self.neuron.get_right(), object.neuron.get_left(), z_index=-1)
        self.connection_line_group.add(line)
        group = VGroup(line)
        if label:
            label_unit = DecimalNumber(number=label.get_value(), font_size=DEFAULT_FONT_SIZE * self.size, color=YELLOW, z_index=100)
            label_unit.add_updater(lambda m: m.set_value(label.get_value()).next_to(line.get_center(), direction=label_placement, buff=0.5))
            label_unit.next_to(line.get_center(), direction=label_placement, buff=0.5)
            group.add(label_unit)
            self.connection_label_group.add(label_unit)

        self.connection_group.add(group)

    def set_value(self, value):
        self.value_unit_updated = self.value_unit.copy()
        self.value_unit_updated.set_value(value)
        self.value_unit_updated.move_to(self.neuron.get_center())

        return ReplacementTransform(self.value_unit, self.value_unit_updated)


class NeuralLayer(VGroup):
    def reset(self):
        return [neuron.reset() for neuron in self.submobjects]  # type: Neuron

    @property
    def neuron_list(self):
        return [
            neuron.neuron
            for neuron in self.submobjects
        ]  # type: Neuron

    @property
    def neuron_value_list(self):
        return [neuron.value_unit for neuron in self.submobjects]  # type: Neuron


class TrainingScene(Scene):
    def __init__(self, saved_model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(saved_model, 'r') as f:
            self.model_parameters = json.load(f)

    def construct(self):
        prev_layer = None
        all_layer_list = []
        layer_iterator = 0
        b_layer_list = []
        for neuron_count in self.model_parameters['shape']:
            neuron_layer = []
            for i in range(neuron_count):
                value = None
                if prev_layer is not None:
                    value = ValueTracker(self.model_parameters['bias_list'][layer_iterator][i])
                    b_layer_list.append(value)
                neuron_layer.append(Neuron(value=value))

            if prev_layer is not None:
                layer_iterator += 1
            current_layer = NeuralLayer(*neuron_layer)
            current_layer.arrange(DOWN * 15)
            all_layer_list.append(current_layer)

            prev_layer = current_layer

        nn = VGroup(*all_layer_list)
        nn.arrange(RIGHT * 15)
        nn.shift(LEFT * 1.5)

        w_layer_list = []

        for l, (prev_layer, current_layer) in enumerate(zip(nn[:-1], nn[1:])):
            for n, neuron_from in enumerate(prev_layer):
                for i, neuron_to in enumerate(current_layer):
                    value = ValueTracker(self.model_parameters['weight_list'][l][n][i])
                    w_layer_list.append(value)
                    neuron_from.draw_connection_to(
                        neuron_to,
                        label=value,
                        label_placement=UP * 0.8 if i % 2 else DOWN * 0.8,
                    )

        table_values_def = [
            ["1", "1", "0"],
            ["1", "0", "1"],
            ["0", "1", "1"],
            ["0", "0", "0"],
        ]
        table_def = Table(
            table_values_def, col_labels=[Tex("$X$"), Tex("$pred$"), Tex("$y$")]
        )
        table_def.scale(0.4)
        table_def.shift(RIGHT * 5.5 + DOWN * 2)

        self.add(nn, table_def)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render neural network animation')
    parser.add_argument('-m', '--saved-model')

    args = parser.parse_args()
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = TrainingScene(saved_model=args.saved_model)
        scene.render()
