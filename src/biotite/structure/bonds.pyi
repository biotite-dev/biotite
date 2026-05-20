# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from collections.abc import Iterable, Iterator
from enum import IntEnum
from typing import Any, Generic, TypeVar, overload
import numpy as np
from biotite.typing import C1, C2, N, NDArray1, NDArray2

__all__ = ["BondList", "BondType"]

# A second atom-count TypeVar used in methods whose result has a
# different atom count than the receiver (e.g. ``concatenate`` /
# ``__getitem__``).
_N2 = TypeVar("_N2", bound=int)

class BondType(IntEnum):
    ANY = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4
    AROMATIC_SINGLE = 5
    AROMATIC_DOUBLE = 6
    AROMATIC_TRIPLE = 7
    COORDINATION = 8
    AROMATIC = 9

    def without_aromaticity(self) -> BondType: ...

class BondList(Generic[N]):
    """
    Generic over ``N``, the number of atoms the bond list refers to.
    An :class:`AtomArray[N]` is expected to carry a :class:`BondList[N]`.
    """

    def __init__(
        self,
        atom_count: int,
        bonds: NDArray2[C1, C2, np.integer] | None = ...,
    ) -> None: ...

    # Concatenation produces a `BondList` over the *sum* of the input
    # atom counts; the resulting `N` is unrelated to any individual
    # input.
    @staticmethod
    def concatenate(
        bond_lists: Iterable[BondList[Any]],
    ) -> BondList[Any]: ...
    def offset_indices(self, offset: int) -> None: ...
    def as_array(self) -> NDArray2[C1, C2, np.uint32]: ...
    def as_set(self) -> set[tuple[int, int, int]]: ...

    # Returns a :class:`networkx.Graph`; left as ``Any`` to avoid a
    # hard dependency on a `networkx` type stub.
    def as_graph(self) -> Any: ...
    def remove_aromaticity(self) -> None: ...
    def remove_kekulization(self) -> None: ...
    def remove_bond_order(self) -> None: ...
    def convert_bond_type(
        self,
        original_bond_type: BondType,
        new_bond_type: BondType,
    ) -> None: ...
    def get_atom_count(self) -> N: ...
    def get_bond_count(self) -> int: ...
    def get_bonds(
        self, atom_index: int
    ) -> tuple[
        NDArray1[C1, np.uint32],
        NDArray1[C1, np.uint8],
    ]: ...
    def get_all_bonds(
        self,
    ) -> tuple[
        NDArray2[N, C1, np.int32],
        NDArray2[N, C1, np.int8],
    ]: ...
    def adjacency_matrix(self) -> NDArray2[N, N, np.bool_]: ...
    def bond_type_matrix(self) -> NDArray2[N, N, np.int8]: ...
    def add_bond(
        self,
        atom_index1: int,
        atom_index2: int,
        bond_type: BondType = ...,
    ) -> None: ...
    def remove_bond(self, atom_index1: int, atom_index2: int) -> None: ...
    def remove_bonds_to(self, atom_index: int) -> None: ...

    # Only meaningful if the other bond list refers to the same atom
    # count.
    def remove_bonds(self, bond_list: BondList[N]) -> None: ...
    def merge(self, bond_list: BondList[Any]) -> BondList[Any]: ...
    def copy(self) -> BondList[N]: ...

    # Concatenation along the atom axis; the resulting bond list spans
    # a different atom count than either operand individually.
    def __add__(self, other: BondList[Any]) -> BondList[Any]: ...
    def __eq__(self, other: object) -> bool: ...
    def __contains__(self, item: tuple[int, int]) -> bool: ...

    # Indexing yields a sub-bond-list over a potentially different
    # atom count, so the return is `BondList[_N2]` where `_N2` is the
    # new (unrelated) atom count.
    @overload
    def __getitem__(self, index: int) -> BondList[Any]: ...
    @overload
    def __getitem__(self, index: slice) -> BondList[Any]: ...
    @overload
    def __getitem__(self, index: NDArray1[_N2, np.integer]) -> BondList[_N2]: ...
    @overload
    def __getitem__(self, index: NDArray1[N, np.bool_]) -> BondList[Any]: ...
    def __iter__(self) -> Iterator[tuple[int, int, BondType]]: ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> BondList[N]: ...
    def __deepcopy__(self, memo: object) -> BondList[N]: ...

def bond_type_members() -> list[tuple[str, int]]: ...
