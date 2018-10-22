# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Tuple, Union, List, overload
import numpy as np
from ..sequence import Sequence
from .matrix import SubstitutionMatrix
from .alignment import Alignment


@overload
def align_ungapped(
    seq1: Sequence, seq2: Sequence, matrix: SubstitutionMatrix
) -> Alignment: ...
@overload
def align_ungapped(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix,
    score_only = False
) -> Union[Alignment, int]: ...

def align_optimal(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix,
    gap_penalty: Union[Tuple[int, int], int] = -10,
    terminal_penalty: bool = True,
    local: bool = False
) -> List[Alignment]: ...