from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from numpy import (
    float64,
    ndarray,
)
from typing import Union


def _sq_euclidian(
    reference: AtomArray,
    subject: Union[AtomArrayStack, AtomArray]
) -> ndarray: ...


def average(atom_arrays: AtomArrayStack) -> AtomArray: ...


def rmsd(
    reference: AtomArray,
    subject: Union[AtomArrayStack, AtomArray]
) -> Union[float64, ndarray]: ...


def rmsf(reference: AtomArray, subject: AtomArrayStack) -> ndarray: ...
