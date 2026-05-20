# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Type annotation helpers for *Biotite*.

Exposes the :class:`TypeVar`\\ s used as shape parameters of the
generic atom containers and a family of dimension-specific NumPy array
aliases.
"""

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = [
    "M",
    "N",
    "K",
    "XYZ",
    "C1",
    "C2",
    "C3",
    "C4",
    "S",
    "S1",
    "S2",
    "NDArray1",
    "NDArray2",
    "NDArray3",
    "NDArray4",
    "MplColor",
]

from typing import TypeAlias, TypeVar
import numpy as np

# Matplotlib color spec: name string ("red"), hex string ("#RRGGBB"),
# RGB / RGBA tuple of floats, or array form.
MplColor: TypeAlias = str | tuple[float, ...] | np.ndarray

# Depth and length of atom containers. All shape TypeVars are bounded
# by `int` so they satisfy `numpy.ndarray`'s shape constraint
# (`_ShapeType_co: tuple[int, ...]`).
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
# Generic dynamic count for axes whose length is determined by the
# operation (e.g. residue / chain / base-pair / hbond / contact count)
K = TypeVar("K", bound=int)
# Channel axes for use as the "minor" dimension of index ndarrays and
# similar tabular data.
# The numeric suffix should be equal to the actual number of channels.
C1 = TypeVar("C1", bound=int)
C2 = TypeVar("C2", bound=int)
C3 = TypeVar("C3", bound=int)
C4 = TypeVar("C4", bound=int)
# Coordinate axis (typically of size 3).
# Using a TypeVar :data:`XYZ` instead of ``Literal[3]`` for the
# coordinate axis is deliberate: ``Literal[3]`` interacts badly with
# the invariance of :class:`numpy.ndarray`'s shape parameter in mypy.
XYZ = TypeVar("XYZ", bound=int)

# Symbol type for sequence alphabets (e.g. `str` for nucleotide/protein
# alphabets, `tuple[str, ...]` for k-mer alphabets). `S1`/`S2` are used
# when two independent alphabets meet, e.g. in
# :class:`SubstitutionMatrix`.
S = TypeVar("S")
S1 = TypeVar("S1")
S2 = TypeVar("S2")

# Private TypeVars used to parametrize the rank-specific aliases.
# Each axis is bounded by `int` so the NDArray aliases satisfy
# `numpy.ndarray`'s shape constraint (`_ShapeType_co: tuple[int, ...]`)
# under pyright/mypy.
_A = TypeVar("_A", bound=int)
_B = TypeVar("_B", bound=int)
_C = TypeVar("_C", bound=int)
_D = TypeVar("_D", bound=int)
_ScalarT = TypeVar("_ScalarT", bound=np.generic)

#: NumPy array type annotations for different number of dimensions
NDArray1: TypeAlias = np.ndarray[tuple[_A], np.dtype[_ScalarT]]
NDArray2: TypeAlias = np.ndarray[tuple[_A, _B], np.dtype[_ScalarT]]
NDArray3: TypeAlias = np.ndarray[tuple[_A, _B, _C], np.dtype[_ScalarT]]
NDArray4: TypeAlias = np.ndarray[tuple[_A, _B, _C, _D], np.dtype[_ScalarT]]
