# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Type stubs for :mod:`biotite.structure.atoms`.

The runtime module stores annotation columns (``chain_id``, ``res_id``,
``coord``, ``bonds``, etc.) in a hidden ``_annot`` dict and dispatches
attribute access through ``__getattr__``. Static type checkers cannot
see those dynamic attributes; this stub declares them explicitly so
that consumers of the public API (filters, geometry, IO, ...) can be
narrowed properly.

Note that ``.pyi`` files completely shadow the ``.py`` for static
analysis, so every public name listed in ``atoms.py.__all__`` must
also be declared here.
"""

from collections.abc import Iterable, Iterator
from types import EllipsisType
from typing import Any, Generic, TypeAliasType, overload
import numpy as np
from biotite.copyable import Copyable
from biotite.structure.bonds import BondList
from biotite.typing import XYZ, M, N, NDArray1, NDArray2, NDArray3, NDArray4

__all__ = [
    "Atom",
    "AtomArray",
    "AtomArrayStack",
    "SingleCoord",
    "Coord",
    "MultiCoord",
    "concatenate",
    "array",
    "stack",
    "repeat",
    "from_template",
    "coord",
    "set_print_limits",
]

class _AtomArrayBase(Copyable):
    def __init__(self, length: int) -> None: ...
    def array_length(self) -> int: ...
    def add_annotation(self, category: str, dtype: Any) -> None: ...
    def del_annotation(self, category: str) -> None: ...
    def get_annotation(self, category: str) -> np.ndarray: ...
    def set_annotation(self, category: str, array: np.ndarray) -> None: ...
    def get_annotation_categories(self) -> list[str]: ...


class AtomArrayStack(_AtomArrayBase, Generic[M, N]):
    # --- Mandatory annotation categories ---
    chain_id: NDArray1[N, np.str_]
    res_id: NDArray1[N, np.integer]
    ins_code: NDArray1[N, np.str_]
    res_name: NDArray1[N, np.str_]
    hetero: NDArray1[N, np.bool_]
    atom_name: NDArray1[N, np.str_]
    element: NDArray1[N, np.str_]
    # --- Optional annotation categories ---
    atom_id: NDArray1[N, np.integer]
    b_factor: NDArray1[N, np.floating]
    occupancy: NDArray1[N, np.floating]
    charge: NDArray1[N, np.integer]
    sym_id: NDArray1[N, np.str_]
    entity_id: NDArray1[N, np.str_]
    # --- Misc ---
    coord: NDArray3[M, N, int, np.floating]
    bonds: BondList[N] | None
    box: NDArray2[int, int, np.floating] | NDArray3[M, int, int, np.floating] | None

    def __init__(self, depth: int, length: int) -> None: ...
    def array_length(self) -> N: ...
    def stack_depth(self) -> M: ...
    @property
    def shape(self) -> tuple[M, N]: ...
    @overload
    def __getitem__(self, index: int) -> AtomArray[N]: ...
    @overload
    def __getitem__(
        self,
        index: (slice | NDArray1[Any, np.integer] | NDArray1[M, np.bool_]),
    ) -> AtomArrayStack[int, N]: ...
    @overload
    def __getitem__(
        self,
        index: tuple[
            slice | NDArray1[Any, np.integer] | NDArray1[M, np.bool_],
            slice | NDArray1[Any, np.integer] | NDArray1[N, np.bool_],
        ],
    ) -> AtomArrayStack[int, int]: ...
    @overload
    def __getitem__(
        self,
        index: tuple[
            EllipsisType,
            slice | NDArray1[Any, np.integer] | NDArray1[N, np.bool_],
        ],
    ) -> AtomArrayStack[M, int]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[AtomArray[N]]: ...
    def __add__(self, other: AtomArrayStack[M, int]) -> AtomArrayStack[M, int]: ...
    def __copy_create__(self) -> AtomArrayStack[M, N]: ...


class AtomArray(_AtomArrayBase, Generic[N]):
    # --- Mandatory annotation categories ---
    chain_id: NDArray1[N, np.str_]
    res_id: NDArray1[N, np.integer]
    ins_code: NDArray1[N, np.str_]
    res_name: NDArray1[N, np.str_]
    hetero: NDArray1[N, np.bool_]
    atom_name: NDArray1[N, np.str_]
    element: NDArray1[N, np.str_]
    # --- Optional annotation categories ---
    atom_id: NDArray1[N, np.integer]
    b_factor: NDArray1[N, np.floating]
    occupancy: NDArray1[N, np.floating]
    charge: NDArray1[N, np.integer]
    sym_id: NDArray1[N, np.str_]
    entity_id: NDArray1[N, np.str_]
    # --- Misc ---
    coord: NDArray2[N, int, np.floating]
    bonds: BondList[N] | None
    box: NDArray2[int, int, np.floating] | None

    def __init__(self, length: int) -> None: ...
    def array_length(self) -> N: ...
    @property
    def shape(self) -> tuple[N]: ...
    @overload
    def __getitem__(self, index: int) -> Atom: ...
    @overload
    def __getitem__(
        self,
        index: (slice | NDArray1[Any, np.integer] | NDArray1[N, np.bool_]),
    ) -> AtomArray[int]: ...
    @overload
    def __getitem__(
        self,
        index: tuple[
            EllipsisType,
            slice | NDArray1[Any, np.integer] | NDArray1[N, np.bool_],
        ],
    ) -> AtomArray[int]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Atom]: ...
    def __add__(self, other: AtomArray[int]) -> AtomArray[int]: ...
    def __copy_create__(self) -> AtomArray[N]: ...


class Atom(Copyable):
    # --- Mandatory annotation categories ---
    chain_id: str
    res_id: int
    ins_code: str
    res_name: str
    hetero: bool
    atom_name: str
    element: str
    # --- Optional annotation categories ---
    # See the table in `biotite.structure.__init__.py` for semantics.
    # Declared as non-`Optional` for static-checker convenience; at
    # runtime the attribute may be absent if it was never added.
    atom_id: int
    b_factor: float
    occupancy: float
    charge: int
    sym_id: str
    entity_id: str
    # --- Misc ---
    coord: NDArray1[int, np.floating]

    def __init__(self, coord: Any, **kwargs: Any) -> None: ...
    @property
    def shape(self) -> tuple[()]: ...
    def __eq__(self, item: object) -> bool: ...
    def __ne__(self, item: object) -> bool: ...
    def __copy_create__(self) -> Atom: ...


# Re-declared because the `.pyi` shadows the `.py`; the runtime values
# still live in `atoms.py` and are what `beartype` resolves at call
# time.
SingleCoord = TypeAliasType(
    "SingleCoord",
    Atom | NDArray1[int, np.floating],
)
Coord = TypeAliasType(
    "Coord",
    AtomArray[N] | NDArray2[N, int, np.floating],
    type_params=(N,),
)
MultiCoord = TypeAliasType(
    "MultiCoord",
    AtomArrayStack[M, N] | NDArray3[M, N, int, np.floating],
    type_params=(M, N),
)

def array(atoms: Iterable[Atom]) -> AtomArray[int]: ...

def stack(arrays: Iterable[AtomArray[N]]) -> AtomArrayStack[int, N]: ...

@overload
def concatenate(atoms: Iterable[AtomArray[Any]]) -> AtomArray[int]: ...
@overload
def concatenate(
    atoms: Iterable[AtomArrayStack[M, Any]],
) -> AtomArrayStack[M, int]: ...

@overload
def repeat(
    atoms: AtomArray[N],
    coord: NDArray3[Any, N, XYZ, np.floating],
) -> AtomArray[Any]: ...
@overload
def repeat(
    atoms: AtomArrayStack[M, N],
    coord: NDArray4[Any, M, N, XYZ, np.floating],
) -> AtomArrayStack[M, Any]: ...
@overload
def from_template(
    template: AtomArray[N] | AtomArrayStack[Any, N],
    coord: NDArray2[N, XYZ, np.floating],
    box: NDArray2[XYZ, XYZ, np.floating] | None = ...,
) -> AtomArray[N]: ...
@overload
def from_template(
    template: AtomArray[N] | AtomArrayStack[Any, N],
    coord: NDArray3[M, N, XYZ, np.floating],
    box: (
        NDArray2[XYZ, XYZ, np.floating] | NDArray3[M, XYZ, XYZ, np.floating] | None
    ) = ...,
) -> AtomArrayStack[M, N]: ...

@overload
def coord(item: SingleCoord) -> NDArray1[XYZ, np.floating]: ...
@overload
def coord(item: Coord[N]) -> NDArray2[N, XYZ, np.floating]: ...
@overload
def coord(item: MultiCoord[M, N]) -> NDArray3[M, N, XYZ, np.floating]: ...

def set_print_limits(max_models: int = ..., max_atoms: int = ...) -> None: ...
