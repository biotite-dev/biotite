# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def check_id_continuity(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def check_bond_continuity(
    array: AtomArray, min_len: float = 1.2, max_len: float = 1.8
) -> np.ndarray: ...

def check_duplicate_atoms(
    array: Union[AtomArrayStack, AtomArray]
) -> np.ndarray: ...

def check_in_box(array: AtomArray) -> np.ndarray: ...
