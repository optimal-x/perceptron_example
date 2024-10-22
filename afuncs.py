from collections.abc import Callable
import numpy as np
from abc import ABC


class Activation(ABC):
    def __init__(self, activation: Callable, derivative: Callable) -> None:
        self.__activation = activation
        self.__derivative = derivative

    def __call__(self, Z):
        return self.__activation(Z)

    def grad(self, Z):
        return self.__derivative(Z)


class ReLU(Activation):
    def __init__(self) -> None:
        def activation(Z: np.ndarray):
            return np.maximum(Z, 0)

        def derivative(Z: np.ndarray):
            return np.where(Z > 0, 1, 0)

        super().__init__(activation, derivative)


class Sigmoid(Activation):
    def __init__(self) -> None:
        def activation(Z: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-Z))

        def derivative(Z: np.ndarray) -> np.ndarray:
            sigmoid_Z = activation(Z)
            return sigmoid_Z * (1 - sigmoid_Z)

        super().__init__(activation, derivative)
