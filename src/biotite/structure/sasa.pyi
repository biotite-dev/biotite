# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Optional, Callable
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def sasa(
    array: AtomArray,
    probe_radius: float = 1.4,
    atom_filter: Optional[np.ndarray] = None,
    gnore_ions: bool = True,
    point_number: int = 1000,
    point_distr: Union[str, Callable[[int], np.ndarray]] = "Fibonacci",
    vdw_radii: Union[str, np.ndarray] = "ProtOr"
) -> np.ndarray: ...