from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from numpy import ndarray
from typing import (
    Callable,
    Tuple,
    Union,
)


def apply_residue_wise(
    array: AtomArray,
    data: ndarray,
    function: Callable,
    axis: None = None
) -> ndarray: ...


def get_residue_count(array: AtomArray) -> int: ...


def get_residue_starts(
    array: Union[AtomArrayStack, AtomArray]
) -> ndarray: ...


def get_residues(array: AtomArray) -> Tuple[ndarray, ndarray]: ...


def spread_residue_wise(array: AtomArray, input_data: ndarray) -> ndarray: ...
