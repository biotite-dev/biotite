# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple, Sequence
import numpy as np
from .atoms import AtomArray, AtomArrayStack, Atom


def translate(
    atoms: Union[AtomArrayStack, AtomArray, Atom], vector: Sequence[float]
) -> AtomArray: ...

def rotate(
    atoms: Union[AtomArrayStack, AtomArray, Atom], angles: Sequence[float]
) -> AtomArray: ...

def rotate_centered(
    atoms: Union[AtomArrayStack, AtomArray], angles: Sequence[float]
) -> AtomArray: ...
