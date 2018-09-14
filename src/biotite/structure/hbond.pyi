# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, Optional, Sequence, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack

@overload
def hbond(
    atoms: AtomArray,
    selection1: Optional[np.dnarray] = None,
    selection2: Optional[np.dnarray] = None,
    selection1_type: str = "both",
    cutoff_dist: float = 2.5,
    cutoff_angle: float = 120,
    donor_elements: Sequence = ('O', 'N', 'S'),
    acceptor_elements: Sequence = ('O', 'N', 'S'),
) -> np.ndarray: ...
@overload
def hbond(
    atoms: AtomArrayStack,
    donor_selection: Optional[np.dnarray] = None,
    acceptor_selection: Optional[np.dnarray] = None,
    cutoff_dist: float = 2.5,
    cutoff_angle: float = 120,
    donor_elements: Sequence = ('O', 'N', 'S'),
    acceptor_elements: Sequence = ('O', 'N', 'S')
) -> Tuple[np.ndarray, np.ndarray]: ...

def hbond_frequency(mask: np.ndarray) -> np.ndarray: ...