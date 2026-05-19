# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Type annotation helpers for *Biotite*.

Exposes the :class:`TypeVar`\\ s used as shape parameters of the
generic atom containers and a family of dimension-specific NumPy array
aliases.
"""

__name__ = "biotite.typing"
__author__ = "Patrick Kunzmann"
__all__ = [
    "M",
    "N",
    "XYZ",
    "C1",
    "C2",
    "C3",
    "C4",
    "NDArray1",
    "NDArray2",
    "NDArray3",
]

from typing import TypeAlias, TypeVar
import numpy as np

# Depth and length of atom containers.
M = TypeVar("M")
N = TypeVar("N")
# Channel axes for use as the "minor" dimension of index ndarrays and
# similar tabular data.
# The numeric suffix should be equal to the actual number of channels.
C1 = TypeVar("C1")
C2 = TypeVar("C2")
C3 = TypeVar("C3")
C4 = TypeVar("C4")
# Coordinate axis (typically of size 3).
# Using a TypeVar :data:`XYZ` instead of ``Literal[3]`` for the
# coordinate axis is deliberate: ``Literal[3]`` interacts badly with
# the invariance of :class:`numpy.ndarray`'s shape parameter in mypy.
XYZ = TypeVar("XYZ")

# Private TypeVars used to parametrize the rank-specific aliases.
_A = TypeVar("_A")
_B = TypeVar("_B")
_C = TypeVar("_C")
_ScalarT = TypeVar("_ScalarT", bound=np.generic)


#: NumPy array type annotations for different number of dimensions
NDArray1: TypeAlias = np.ndarray[tuple[_A], np.dtype[_ScalarT]]
NDArray2: TypeAlias = np.ndarray[tuple[_A, _B], np.dtype[_ScalarT]]
NDArray3: TypeAlias = np.ndarray[tuple[_A, _B, _C], np.dtype[_ScalarT]]
