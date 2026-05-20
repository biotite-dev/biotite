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
from collections.abc import Iterable
from typing import Literal
import numpy
import numpy.typing as npt


class Layer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: npt.ArrayLike) -> numpy.ndarray:
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(
        self,
        weights: npt.ArrayLike,
        biases: npt.ArrayLike | None = None,
        activation: bool = True,
    ) -> None:
        self.activation = activation
        self.weights = numpy.asarray(weights)
        if biases is None:
            self.biases = numpy.zeros(self.weights.shape[1])
        else:
            self.biases = numpy.asarray(biases)

    def __call__(self, x: npt.ArrayLike) -> numpy.ndarray:
        x = numpy.asarray(x)
        out = x @ self.weights
        out += self.biases

        if self.activation:
            return _relu(out, out=out)
        else:
            return out


class CentroidLayer(Layer):
    def __init__(self, centroids: npt.ArrayLike) -> None:
        self.centroids = numpy.asarray(centroids)
        self.r2 = numpy.sum(self.centroids**2, axis=1).reshape(-1, 1).T

    def __call__(self, x: npt.ArrayLike) -> numpy.ndarray:
        x = numpy.asarray(x)
        # compute pairwise squared distance matrix
        r1 = numpy.sum(x**2, axis=1).reshape(-1, 1)
        D = r1 - 2 * x @ self.centroids.T + self.r2
        # find closest centroid
        states = numpy.empty(D.shape[0], dtype=numpy.uint8)
        D.argmin(axis=1, out=states)
        return states


class Model:
    def __init__(self, layers: Iterable[Layer] = ()) -> None:
        self.layers = list(layers)

    def __call__(self, x: npt.ArrayLike) -> numpy.ndarray:
        return functools.reduce(lambda x, f: f(x), self.layers, numpy.asarray(x))


def _relu(
    x: npt.ArrayLike,
    out: numpy.ndarray | None = None,
    *,
    where: bool | npt.NDArray[numpy.bool_] = True,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    order: Literal["K", "A", "C", "F"] = "K",
    dtype: npt.DTypeLike | None = None,
    subok: bool = True,
) -> numpy.ndarray:
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
