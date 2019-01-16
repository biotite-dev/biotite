# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def vectors_from_unitcell(
    len_a: float,
    len_b: float,
    len_c: float,
    alpha: float,
    beta:  float,
    gamma: float
) -> np.ndarray: ...

def unitcell_from_vectors(
    box: np.ndarray
) -> Tuple[float, float, float, float, float, float]: ...

def box_volume(box: np.ndarray) -> Union[float, np.ndarray]: ...

@overload
def repeat_box(
    atoms: AtomArray,
    amount: int = 1
) -> Tuple(AtomArray, np.ndarray): ...
@overload
def repeat_box(
    atoms: AtomArrayStack,
    amount: int = 1
) -> Tuple(AtomArrayStack, np.ndarray): ...

def repeat_box(
    coord: np.ndarray,
    box: np.ndarray,
    amount: int = 1
) -> Tuple(np.ndarray, np.ndarray): ...

def move_inside_box(coord: np.ndarray, box: np.ndarray) -> np.ndarray: ...

def coord_to_fraction(coord: np.ndarray, box: np.ndarray) -> np.ndarray: ...

def fraction_to_coord(coord: np.ndarray, box: np.ndarray) -> np.ndarray: ...

def is_orthogonal(box: np.ndarray) -> Union[float, np.ndarray]: ...
