from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import scipy.special


@dataclass
class MultiLayerPerceptronWeights:
    shape: list[int]
    weights: list[float]

    def num_layers(self):
        return len(self.shape)

    def num_inputs(self):
        return self.shape[0]

    def num_outputs(self):
        return self.shape[-1]

    @classmethod
    def create_random(
        cls, shape: list[int], boundaries=(-1, 1)
    ) -> MultiLayerPerceptronWeights:
        weights = []
        for i in range(len(shape) - 1):
            W = np.random.uniform(
                size=(shape[i + 1], shape[i] + 1), low=boundaries[0], high=boundaries[1]
            )
            weights.append(W)
        return cls(shape, weights)

    @classmethod
    def num_dimensions(cls, shape: list[int]) -> int:
        dim = 0
        for i in range(len(shape) - 1):
            dim = dim + (shape[i] + 1) * shape[i + 1]
        return dim

    @classmethod
    def from_particle_position(
        cls, particle_position: list[float], shape: list[int]
    ) -> MultiLayerPerceptronWeights:
        weights = []
        idx = 0
        for i in range(len(shape) - 1):
            r = shape[i + 1]
            c = shape[i] + 1
            idx_min = idx
            idx_max = idx + (r * c)
            W = particle_position[idx_min:idx_max].reshape(r, c)
            weights.append(W)
            idx = idx_max
        return cls(shape, weights)

    def to_particle_position(self) -> list[float]:
        w = np.asarray([])
        for i in range(len(self.weights)):
            v = self.weights[i].flatten()
            w = np.append(w, v)
        return w


class MultiLayerPerceptron:
    @staticmethod
    def run(weights: MultiLayerPerceptronWeights, inputs):
        layer = inputs
        for i in range(weights.num_layers() - 1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(weights.weights[i], prev_layer)
            # activation function: logistic sigmoid (output in ]0;1[)
            layer = scipy.special.expit(o)
        return layer

    @staticmethod
    def evaluate(
        weights: MultiLayerPerceptronWeights, inputs, expected_outputs
    ) -> float:
        y_pred = MultiLayerPerceptron.run(weights, inputs)
        print(
            "shape",
            np.sum((np.atleast_2d(expected_outputs) - y_pred) ** 2, axis=1).shape,
        )
        mse = (
            np.sum((np.atleast_2d(expected_outputs) - y_pred) ** 2, axis=1)
            / y_pred.shape[1]
        )
        print("mse mlp", mse, mse.shape)
        return mse

    # FOLLOWING FUNCTION HAS BEEN GENERATED BY CHATGPT
    @staticmethod
    def backpropagate(weights: MultiLayerPerceptronWeights, inputs, targets, alpha):
        # initialize array of errors
        errors = []

        # forward pass
        layer = inputs
        layers = [layer]
        for i in range(weights.num_layers() - 1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(weights.weights[i], prev_layer)
            # activation function: logistic sigmoid (output in ]0;1[)
            layer = scipy.special.expit(o)
            layers.append(layer)

        # output layer error
        layer_errors = targets - layers[-1]
        errors.append(layer_errors)

        # hidden layer errors
        for i in range(weights.num_layers() - 2, 0, -1):
            layer_errors = np.dot(weights.weights[i].T, layer_errors)
            layer_errors = layer_errors[1:]
            errors.append(layer_errors)

        # reverse errors array
        errors = errors[::-1]

        # update weights
        new_weights = []
        for i in range(weights.num_layers() - 1):
            layer = np.insert(layers[i], 0, 1, axis=0)
            delta_w = alpha * np.dot(errors[i], layer.T)
            new_weights.append(weights.weights[i] + delta_w)

        return replace(weights, weights=new_weights)