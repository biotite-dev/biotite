# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Iterable, TypeVar
import numpy as np


_T = TypeVar("_T")

# Generic placeholder for numpy arrays
class Array(np.ndarray, Iterable[_T]):
    ...