from biotite.sequence.seqtypes import (
    NucleotideSequence,
    ProteinSequence,
)
from numpy import ndarray
from typing import (
    List,
    Tuple,
    Union,
)


class BlastAlignment:
    def __init__(
        self,
        sequences: Union[List[ProteinSequence], List[NucleotideSequence]],
        trace: ndarray,
        score: int,
        e_value: float,
        query_interval: Tuple[int, int],
        hit_interval: Tuple[int, int],
        hit_id: str,
        hit_definition: str
    ) -> None: ...
