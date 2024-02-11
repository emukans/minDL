import argparse
import json
import os

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
    def __init__(self, saved_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(saved_model, 'r') as f:
            self.model_parameters = json.load(f)

        self.model_name = os.path.splitext(os.path.basename(saved_model))[0].replace('_', ' ').upper()

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
            current_layer.arrange(DOWN * 15)  # todo: adjust me if you're changing the NN size
            all_layer_list.append(current_layer)

            prev_layer = current_layer

        nn = VGroup(*all_layer_list)
        nn.arrange(RIGHT * 13)  # todo: adjust me if you're changing the NN size
        nn.shift(LEFT * 2 + DOWN * 0.5)  # todo: adjust me if you're changing the NN size

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

        table_values_def = []
        input_len = len(self.model_parameters['snapshot_list'][0]['input'][0])
        pred_len = len(self.model_parameters['snapshot_list'][0]['prediction'][0])
        target_len = len(self.model_parameters['snapshot_list'][0]['target'][0])
        prediction_list = []
        for i in range(len(self.model_parameters['snapshot_list'][0]['input'])):
            snapshot = self.model_parameters['snapshot_list'][0]
            table_line = []
            for k in range(input_len):
                table_line.append(snapshot['input'][i][k])

            for k in range(pred_len):
                table_line.append(snapshot['prediction'][i][k])

            for k in range(target_len):
                table_line.append(snapshot['target'][i][k])
            table_values_def.append(table_line)

        column_labels = []
        for k in range(input_len):
            column_labels.append(Tex(f"$x_{k}$"))

        for k in range(pred_len):
            column_labels.append(Tex(f"$pred_{k}$"))

        for k in range(target_len):
            column_labels.append(Tex(f"$y_{k}$"))

        table_def = DecimalTable(
            table_values_def, col_labels=column_labels
        )
        table_def.scale(0.4)
        table_def.move_to(RIGHT * 5 + DOWN * 2.5)

        ax_contour = Axes(
            x_range=[0, len(self.model_parameters['snapshot_list'])],
            y_range=[0, 1.2],
            x_length=3,
            y_length=2,
            axis_config={"include_tip": False},
        ).move_to(RIGHT * 5 + UP * 0.5)
        labels = ax_contour.get_axis_labels(
            Tex("step").scale(0.5), Text("loss").scale(0.4)
        )

        title = Text(self.model_name, font_size=DEFAULT_FONT_SIZE)
        title.move_to(UP * 3.5)
        iteration_text = Text("Iteration: ", font_size=DEFAULT_FONT_SIZE / 2)
        iteration_text.move_to(UP * 2.5 + RIGHT * 5)
        iteration_variable = DecimalNumber(0, num_decimal_places=0, font_size=DEFAULT_FONT_SIZE * 0.75)
        iteration_variable.move_to(iteration_text.get_right() + RIGHT * 0.3)

        # todo: if you're adjusting the NN size, then uncomment self.add and comment everything below (this is to speedup the process)
        # self.add(title, nn, table_def, ax_contour, labels, iteration_variable, iteration_text)
        self.play(Write(title))
        self.play(FadeIn(nn))
        self.play(FadeIn(table_def, ax_contour), Write(labels), Write(iteration_text), Write(iteration_variable))
        prev_point = None

        for iteration, snapshot in enumerate(self.model_parameters['snapshot_list'][:2]):
            animation_list = []
            bias_iterator = 0
            for bias_list in snapshot['bias_list']:
                for bias_value in bias_list:
                    animation_list.append(b_layer_list[bias_iterator].animate.set_value(bias_value))
                    bias_iterator += 1

            weight_iterator = 0
            for weight_list in snapshot['weight_list']:
                for weight_layer in weight_list:
                    for weight_value in weight_layer:
                        animation_list.append(w_layer_list[weight_iterator].animate.set_value(weight_value))
                        weight_iterator += 1

            loss_point = Dot(ax_contour.c2p(iteration, snapshot['loss']), radius=DEFAULT_DOT_RADIUS / 2)
            fade_in_objects = [loss_point]
            if prev_point:
                fade_in_objects.append(Line(prev_point, loss_point, path_arc=0.5))

            prev_point = loss_point

            header_count = len(snapshot['input'][0]) + len(snapshot['target'][0]) + len(snapshot['prediction'][0])
            for i, prediction_list in enumerate(snapshot['prediction']):
                for k, prediction in enumerate(prediction_list):
                    animation_list.append(table_def[0][header_count * (i + 1) + len(snapshot['input'][0]) + k].animate.set_value(prediction))

            self.play(*animation_list, iteration_variable.animate.set_value(iteration + 1), FadeIn(*fade_in_objects))
            self.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render neural network animation')
    parser.add_argument('-m', '--saved-model')

    args = parser.parse_args()
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = TrainingScene(saved_model=args.saved_model)
        scene.render()
