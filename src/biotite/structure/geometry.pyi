from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from numpy import (
    float64,
    ndarray,
)
from typing import (
    Tuple,
    Union,
)


def angle(atom1: ndarray, atom2: ndarray, atom3: ndarray) -> Union[float64, ndarray]: ...


def centroid(
    atoms: Union[AtomArrayStack, ndarray, AtomArray]
) -> ndarray: ...


def dihedral(
    atom1: ndarray,
    atom2: ndarray,
    atom3: ndarray,
    atom4: ndarray
) -> Union[float64, ndarray]: ...


def dihedral_backbone(
    atom_array: Union[AtomArrayStack, AtomArray],
    chain_id: str
) -> Tuple[ndarray, ndarray, ndarray]: ...


def distance(atoms1: ndarray, atoms2: ndarray) -> float64: ...
