# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Optional, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack


@overload
def gyration_radius(
    array: AtomArray, masses: Optional[np.ndarray] = None
) -> float: ...
@overload
def gyration_radius(
    array: AtomArrayStack, masses: Optional[np.ndarray] = None
) -> np.ndarray: ...

@overload
def mass_center(
    array: AtomArrayStack, masses:  Optional[np.ndarray] = None
) -> float: ...
@overload
def mass_center(
    array: AtomArray, masses:  Optional[np.ndarray] = None
) -> np.ndarray: ...

def atom_masses(array: Union[AtomArrayStack, AtomArray]) -> np.ndarray: ...

def mass_of_element(element: str) -> float: ...

