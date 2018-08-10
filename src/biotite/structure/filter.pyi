from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
)
from numpy import ndarray
from typing import (
    Any,
    List,
    Optional,
    Union,
)


def filter_amino_acids(
    array: Union[AtomArrayStack, AtomArray]
) -> ndarray: ...


def filter_backbone(
    array: Union[AtomArrayStack, AtomArray]
) -> ndarray: ...


def filter_inscode_and_altloc(
    array: Union[AtomArrayStack, AtomArray],
    inscode: List[Any] = [],
    altloc: List[Any] = [],
    inscode_array: Optional[ndarray] = None,
    altloc_array: Optional[ndarray] = None
) -> ndarray: ...


def filter_intersection(
    array: AtomArray,
    intersect: AtomArray
) -> ndarray: ...


def filter_monoatomic_ions(array: AtomArray) -> ndarray: ...


def filter_solvent(array: AtomArray) -> ndarray: ...
