from biotite.structure.atoms import AtomArray
from typing import (
    List,
    Tuple,
    Union,
)


def rotate(
    atoms: AtomArray,
    angles: Union[List[float], Tuple[int, int, int]]
) -> AtomArray: ...


def rotate_centered(atoms: AtomArray, angles: List[float]) -> AtomArray: ...


def translate(
    atoms: AtomArray,
    vector: Tuple[int, int, int]
) -> AtomArray: ...
