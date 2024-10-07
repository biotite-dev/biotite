# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Implementation of the neural network layers used in ``foldseek``.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["Layer", "DenseLayer", "CentroidLayer", "Model"]

import abc
import functools
import numpy


class Layer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, weights, biases=None, activation: bool = True):
        self.activation = activation
        self.weights = numpy.asarray(weights)
        if biases is None:
            self.biases = numpy.zeros(self.weights.shape[1])
        else:
            self.biases = numpy.asarray(biases)

    def __call__(self, x):
        x = numpy.asarray(x)
        out = x @ self.weights
        out += self.biases

        if self.activation:
            return _relu(out, out=out)
        else:
            return out


class CentroidLayer(Layer):
    def __init__(self, centroids) -> None:
        self.centroids = numpy.asarray(centroids)
        self.r2 = numpy.sum(self.centroids**2, axis=1).reshape(-1, 1).T

    def __call__(self, x):
        # compute pairwise squared distance matrix
        r1 = numpy.sum(x**2, axis=1).reshape(-1, 1)
        D = r1 - 2 * x @ self.centroids.T + self.r2
        # find closest centroid
        states = numpy.empty(D.shape[0], dtype=numpy.uint8)
        D.argmin(axis=1, out=states)
        return states


class Model:
    def __init__(self, layers=()):
        self.layers = list(layers)

    def __call__(self, x):
        return functools.reduce(lambda x, f: f(x), self.layers, x)


def _relu(
    x,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return numpy.maximum(
        0.0,
        x,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )
