import numpy as np
from numpy import ndarray, floating


class MAE:
    def __call__(self):
        return 

    def grad(self, e):
        return 1 if e >= 0 else -1 


class Relu:
    def __call__(self, z, /):
        return z if z > 0 else 0

    def grad(self, z):
        return 1 if z > 0 else 0


class Forward:
    @staticmethod
    def z(X: ndarray, W: ndarray, B: ndarray, /) -> floating:
        return np.dot(X, W) + B

    @staticmethod
    def a(z: floating, fn=Relu(),/):
        return fn(z)

    @staticmethod
    def e(y: floating, yh, error=np.abs,/):
        return error(y - yh)


class Backward:
    @staticmethod
    def dZw(w: floating, /):
        return w

    @staticmethod
    def dZb():
        return 1

    @staticmethod
    def da(z: floating, dFn=Relu(), /):
        return dFn.grad(z)

    @staticmethod
    def dyha(yh: floating) -> ndarray:
        return -yh

    @staticmethod
    def de(e, dFn=MAE()):
        return dFn.grad(e)


def forward(X, W, B, /):
    """
    Returns the prediction given the inputs, weights, and biases
    """
    Z = Forward.z(X, W, B)
    return Forward.a(Z)


def grad(X, W, B, alpha=0.02, /):
    return  


def backwards(X, W, B, epoch=1000, alpha=0.02, /):
    """
    updates the weights and biases given some training data.
    """
    return




def main():
    MODEL_SHAPE = (2, 1)
    weights = np.random.uniform(
        low=0.0, high=1.0, size=(MODEL_SHAPE[0] * MODEL_SHAPE[1],)
    )
    bias = np.random.uniform(low=0.0, high=1.0, size=(MODEL_SHAPE[1],))

    print(weights)
    print(bias)


if __name__ == "__main__":
    main()
