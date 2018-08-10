from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from numpy import ndarray
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)


def _superimpose(fix_centered: ndarray, mob_centered: ndarray) -> ndarray: ...


def superimpose(
    fixed: AtomArray,
    mobile: Union[AtomArray, AtomArrayStack],
    atom_mask: Optional[ndarray] = None
) -> Union[Tuple[AtomArrayStack, List[Tuple[ndarray, ndarray, ndarray]]], Tuple[AtomArray, Tuple[ndarray, ndarray, ndarray]]]: ...


def superimpose_apply(
    atoms: AtomArray,
    transformation: Tuple[ndarray, ndarray, ndarray]
) -> AtomArray: ...
