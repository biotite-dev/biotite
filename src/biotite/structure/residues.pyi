# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, Callable, Any, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def get_residue_starts(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

@overload
def apply_residue_wise(
    array: Union[AtomArrayStack, AtomArray],
    data: np.ndarray,
    function: Callable[[np.ndarray], Any],
    axis: None = ...
) -> np.ndarray: ...
@overload
def apply_residue_wise(
    array: Union[AtomArrayStack, AtomArray],
    data: np.ndarray,
    function: Callable[[np.ndarray, int], Any],
    axis: int = ...
) -> np.ndarray: ...

def spread_residue_wise(
    array: Union[AtomArrayStack, AtomArray], input_data: np.ndarray
) -> np.ndarray: ...

def get_residues(
    array: Union[AtomArrayStack, AtomArray]
) -> Tuple[np.ndarray, np.ndarray]: ...

def get_residue_count(array: Union[AtomArrayStack, AtomArray]) -> int: ...

