from numpy import (
    float64,
    ndarray,
)
from typing import Union


def distance(v1: ndarray, v2: ndarray) -> Union[float64, ndarray]: ...


def norm_vector(v: ndarray) -> None: ...


def vector_dot(v1: ndarray, v2: ndarray) -> Union[float64, ndarray]: ...
