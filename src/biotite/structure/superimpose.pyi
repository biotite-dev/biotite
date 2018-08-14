# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Tuple, List, Optional, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack


@overload
def superimpose(
    fixed: AtomArray,
    mobile: AtomArray,
    atom_mask: Optional[np.ndarray] = None
) -> Tuple[AtomArray,
           Tuple[np.ndarray, np.ndarray, np.ndarray]]: ...
@overload
def superimpose(
    fixed: AtomArray,
    mobile: AtomArrayStack,
    atom_mask: Optional[np.ndarray] = None
) -> Tuple[AtomArrayStack,
           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]: ...

def superimpose_apply(
    atoms: AtomArray,
    transformation: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> AtomArray: ...
