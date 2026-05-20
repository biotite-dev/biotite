# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Type stubs for :mod:`biotite.structure.celllist`.

`CellList` and `CellListResult` come from the Rust extension
`biotite.rust.structure`. The runtime `.py` performs a couple of
dynamic attribute rewrites (``CellListResult.__name__ = "Result"``,
``CellList.Result = CellListResult``) so that the enum appears as the
more ergonomic ``CellList.Result``. This stub declares that surface
directly.

`CellList` is generic over the number of atoms ``N`` it was
constructed with. That parameter feeds into the shapes of
:meth:`create_adjacency_matrix` and the ``MASK`` form of
:meth:`get_atoms` / :meth:`get_atoms_in_cells`.
"""

from enum import IntEnum
from typing import Any, Generic, Literal, TypeVar, overload
import numpy as np
from biotite.structure.atoms import AtomArray
from biotite.typing import C2, XYZ, K, N, NDArray1, NDArray2

__all__ = ["CellList"]

# Number of query coordinates passed to `get_atoms` / `get_atoms_in_cells`
# when the user provides a 2-D `(m, 3)` array. Local because it has no
# fixed channel semantics.
_M = TypeVar("_M", bound=int)

class CellList(Generic[N]):
    class Result(IntEnum):
        """
        The format of the result returned by :meth:`CellList.get_atoms`
        and :meth:`CellList.get_atoms_in_cells`.

        - ``MAPPING`` -- atom indices per query, padded with ``-1``.
          Shape ``(k,)`` for a single query, ``(m, k)`` for ``m`` queries.
        - ``MASK`` -- boolean mask over the atoms.
          Shape ``(n,)`` for a single query, ``(m, n)`` for ``m`` queries.
        - ``PAIRS`` -- ``(k, 2)`` ``(query_index, atom_index)`` pairs for
          the multi-query case; for a single query the shape collapses to
          ``(k,)`` (just the atom indices, equivalent to ``MAPPING``).
        """

        MAPPING = 0
        MASK = 1
        PAIRS = 2

    def __init__(
        self,
        atom_array: AtomArray[N] | NDArray2[N, XYZ, np.floating],
        cell_size: float,
        periodic: bool = ...,
        box: NDArray2[XYZ, XYZ, np.floating] | None = ...,
        selection: NDArray1[N, np.bool_] | None = ...,
    ) -> None: ...
    def create_adjacency_matrix(
        self, threshold_distance: float
    ) -> NDArray2[N, N, np.bool_]: ...
    # Single coord + PAIRS -> 1-D atom indices (equivalent to MAPPING)
    @overload
    def get_atoms(
        self,
        coord: NDArray1[XYZ, np.floating],
        radius: float,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.PAIRS],
    ) -> NDArray1[K, np.integer]: ...
    # Multi coord + PAIRS -> (k, 2) integer pairs
    @overload
    def get_atoms(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        radius: float | NDArray1[_M, np.floating],
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.PAIRS],
    ) -> NDArray2[K, C2, np.integer]: ...
    # Single coord + MASK -> (n,) bool
    @overload
    def get_atoms(
        self,
        coord: NDArray1[XYZ, np.floating],
        radius: float,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.MASK],
    ) -> NDArray1[N, np.bool_]: ...
    # Multi coord + MASK -> (m, n) bool
    @overload
    def get_atoms(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        radius: float | NDArray1[_M, np.floating],
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.MASK],
    ) -> NDArray2[_M, N, np.bool_]: ...
    # Multi coord + MAPPING -> (m, k) integer (rectangular, -1 padded)
    @overload
    def get_atoms(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        radius: float | NDArray1[_M, np.floating],
        as_mask: Literal[False] = ...,
        result_format: Literal[Result.MAPPING] = ...,
    ) -> NDArray2[_M, K, np.integer]: ...
    # Single coord + MAPPING (default) -> (k,) integer indices
    @overload
    def get_atoms(
        self,
        coord: NDArray1[XYZ, np.floating],
        radius: float,
        as_mask: Literal[False] = ...,
        result_format: Literal[Result.MAPPING] = ...,
    ) -> NDArray1[K, np.integer]: ...
    # Fallback when `result_format` is a non-literal `Result` value.
    @overload
    def get_atoms(
        self,
        coord: NDArray1[XYZ, np.floating] | NDArray2[Any, XYZ, np.floating],
        radius: float | NDArray1[Any, np.floating],
        as_mask: bool = ...,
        result_format: Result = ...,
    ) -> np.ndarray: ...
    # Same overload structure as `get_atoms`; only the second positional
    # parameter changes (`cell_radius` instead of Euclidean `radius`).
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray1[XYZ, np.floating],
        cell_radius: int | None = ...,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.PAIRS],
    ) -> NDArray1[K, np.integer]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        cell_radius: int | NDArray1[_M, np.integer] | None = ...,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.PAIRS],
    ) -> NDArray2[K, C2, np.integer]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray1[XYZ, np.floating],
        cell_radius: int | None = ...,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.MASK],
    ) -> NDArray1[N, np.bool_]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        cell_radius: int | NDArray1[_M, np.integer] | None = ...,
        as_mask: bool = ...,
        *,
        result_format: Literal[Result.MASK],
    ) -> NDArray2[_M, N, np.bool_]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray2[_M, XYZ, np.floating],
        cell_radius: int | NDArray1[_M, np.integer] | None = ...,
        as_mask: Literal[False] = ...,
        result_format: Literal[Result.MAPPING] = ...,
    ) -> NDArray2[_M, K, np.integer]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray1[XYZ, np.floating],
        cell_radius: int | None = ...,
        as_mask: Literal[False] = ...,
        result_format: Literal[Result.MAPPING] = ...,
    ) -> NDArray1[K, np.integer]: ...
    @overload
    def get_atoms_in_cells(
        self,
        coord: NDArray1[XYZ, np.floating] | NDArray2[Any, XYZ, np.floating],
        cell_radius: int | NDArray1[Any, np.integer] | None = ...,
        as_mask: bool = ...,
        result_format: Result = ...,
    ) -> np.ndarray: ...
