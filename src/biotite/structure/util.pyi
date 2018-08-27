# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Tuple
import numpy as np
from .atoms import AtomArray, AtomArrayStack


def vector_dot(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]: ...

def norm_vector(v: np.ndarray) -> None: ...

def distance(v1: np.ndarray, v2: np.ndarray) -> Union[float, np.ndarray]: ...