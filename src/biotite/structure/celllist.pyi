# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Optional
import numpy as np
from .atoms import AtomArray


class CellList:
    def __init__(
        self,
        atom_array: Union[AtomArray, np.ndarray],
        cell_size: float,
        periodic: bool = False
        box: Optional[np.ndarray] = None
    ) -> None: ...
    def create_adjacency_matrix(
        self, threshold_distance: float
    ) -> np.ndarray: ...
    def get_atoms(
        self, coord: np.ndarray, radius: Union[float, np.ndarray]
    ) -> np.ndarray: ...
    def get_atoms_in_cells(
        self,coord: np.ndarray, cell_radius: Union[float, np.ndarray] = 1
    ) -> np.ndarray: ...