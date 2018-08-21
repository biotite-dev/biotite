# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def distance(
    atoms1: Union[AtomArrayStack, AtomArray, np.ndarray],
    atoms2: Union[AtomArrayStack, AtomArray, np.ndarray]
) -> Union[np.ndarray, float]: ...

def centroid(
    atoms: Union[AtomArrayStack, AtomArray, np.ndarray]
) -> Union[np.ndarray, float]: ...

def angle(
    atom1: Union[AtomArrayStack, AtomArray, np.ndarray],
    atom2: Union[AtomArrayStack, AtomArray, np.ndarray],
    atom3: Union[AtomArrayStack, AtomArray, np.ndarray]
) -> Union[np.ndarray, float]: ...

def dihedral(
    atom1: Union[AtomArrayStack, AtomArray, np.ndarray],
    atom2: Union[AtomArrayStack, AtomArray, np.ndarray],
    atom3: Union[AtomArrayStack, AtomArray, np.ndarray],
    atom4: Union[AtomArrayStack, AtomArray, np.ndarray]
) -> Union[np.ndarray, float]: ...

def dihedral_backbone(
    atom_array: Union[AtomArrayStack, AtomArray],
    chain_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
