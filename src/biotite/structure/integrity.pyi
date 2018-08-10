from biotite.structure.atoms import AtomArray
from numpy import ndarray


def check_bond_continuity(
    array: AtomArray,
    min_len: float = 1.2,
    max_len: float = 1.8
) -> ndarray: ...


def check_duplicate_atoms(array: AtomArray) -> ndarray: ...


def check_id_continuity(array: AtomArray) -> ndarray: ...
