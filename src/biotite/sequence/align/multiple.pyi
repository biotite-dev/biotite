# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Union, Optional, Tuple, List
import numpy as np
from .alignment import Alignment
from .matrix import SubstitutionMatrix
from ..pyhlo.tree import Tree
from ..sequence import Sequence


def align_multiple(
    sequences: List[Sequence]:
    matrix: SubstitutionMatrix,
    gap_penalty: Union[Tuple[int, int], int] = -10,
    terminal_penalty: bool = True,
    distances: Optional[np.ndarray]
    guide_tree: Optional[Tree]
) -> Tuple[Alignment, np.ndarray, Tree, np.ndarray]: ...