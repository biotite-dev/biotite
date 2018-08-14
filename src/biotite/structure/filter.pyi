# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Optional, Tuple, Iterable, Sequence
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def filter_monoatomic_ions(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def filter_solvent(array: Union[AtomArrayStack, AtomArray]) -> np.ndarray: ...

def filter_amino_acids(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def filter_backbone(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def filter_intersection(
    array: Union[AtomArrayStack, AtomArray],
    intersect: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def filter_inscode_and_altloc(
    array: Union[AtomArrayStack, AtomArray],
    inscode: Iterable[Tuple[int, str]] = [],
    altloc:  Iterable[Tuple[int, str]] = [],
    inscode_array: Optional[Sequence[str]] = None,
    altloc_array:  Optional[Sequence[str]] = None
) -> np.ndarray: ...
