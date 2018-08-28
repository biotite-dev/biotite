# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import List, Tuple, Union
import numpy as np
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.align.Alignment import Alignment


class BlastAlignment(Alignment):
    def __init__(
        self,
        sequences: Union[List[ProteinSequence], List[NucleotideSequence]],
        trace: np.ndarray,
        score: int,
        e_value: float,
        query_interval: Tuple[int, int],
        hit_interval: Tuple[int, int],
        hit_id: str,
        hit_definition: str
    ) -> None: ...
