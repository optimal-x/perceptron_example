import numpy as np
from numpy import ndarray, floating, array


class MAE:
    def __call__(self, e):
        return np.abs(e)

    def grad(self, e):
        return 1 if e >= 0 else -1


class Relu:
    def __call__(self, z, /) -> floating:
        return z if z > 0 else 0

    def grad(self, z):
        return 1 if z > 0 else 0


class Forward:
    @staticmethod
    def z(X: ndarray, W: ndarray, B: ndarray, /) -> floating:
        return np.dot(X, W) + B

    @staticmethod
    def a(z: floating, fn=Relu(), /) -> floating:
        return fn(z)

    @staticmethod
    def e(y: floating, yh, error=MAE(), /) -> floating:
        return error(y - yh)


class Backward:
    @staticmethod
    def dZw(w: ndarray, /) -> ndarray:
        return w

    @staticmethod
    def dZb() -> floating:
        return 1.0

    @staticmethod
    def __da(z: floating, dFn=Relu(), /) -> floating:
        return dFn.grad(z)

    @staticmethod
    def dyha(z: floating) -> floating:
        return -Backward.__da(z)

    @staticmethod
    def de(e, dFn=MAE()):
        return dFn.grad(e)


class Model:
    @staticmethod
    def forward(X: ndarray, W: ndarray, B: ndarray, /) -> floating:
        """
        Returns the prediction given the inputs, weights, and biases
        """
        Z = Forward.z(X, W, B)
        return Forward.a(Z)

    @staticmethod
    def backwards(X: ndarray, W: ndarray, B: ndarray, epoch=1000):
        """
        updates the weights and biases given some training data.
        """
        OUTPUT_INDEX = -1
        for i in range(epoch):
            # make prediction
            prediction: floating = Model.forward(X[:OUTPUT_INDEX], W, B)
            zh = Forward.z(X[:OUTPUT_INDEX], W, B)
            # check how wrong it is
            new_w = grad_w(W=W, y=X[OUTPUT_INDEX], yh=prediction, z=zh)
            new_b = grad_b(y=X[OUTPUT_INDEX], yh=prediction, z=zh)

            W -= new_w
            B -= new_b

        prediction: floating = Model.forward(X[:OUTPUT_INDEX], W, B)
        print(prediction)


def grad_w(
    W: ndarray,
    y: floating,
    yh: floating,
    z: floating,
    alpha=0.02,
) -> ndarray:
    return alpha * Backward.de(y - yh) * Backward.dyha(z) * Backward.dZw(W)


def grad_b(
    y: floating,
    yh: floating,
    z: floating,
    alpha=0.02,
) -> floating:
    return alpha * Backward.de(y - yh) * Backward.dyha(z) * Backward.dZb()


def main():
    MODEL_SHAPE = (2, 1)
    # matricies
    W = np.random.uniform(
        low=0.0, high=1.0, size=(MODEL_SHAPE[0] * MODEL_SHAPE[1],)
    )
    B = np.random.uniform(low=0.0, high=1.0, size=(MODEL_SHAPE[1],))
    X = array([1, 0])

    # values
    z = Forward.z(X, W, B)
    yh = Model.forward(X, W, B)

    X = array([1, 0, 1])
    Model.backwards(X, W, B, epoch=1000)


if __name__ == "__main__":
    main()
