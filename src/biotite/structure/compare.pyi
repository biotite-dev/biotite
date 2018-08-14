# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, overload
import numpy as np
from .atoms import AtomArray, AtomArrayStack


@overload
def rmsd(reference: AtomArray, subject: AtomArray) -> float: ...
@overload
def rmsd(reference: AtomArray, subject: AtomArrayStack) -> np.ndarray: ...

def rmsf(reference: AtomArray, subject: AtomArrayStack) -> np.ndarray: ...

def average(atom_arrays: AtomArrayStack) -> AtomArray: ...
