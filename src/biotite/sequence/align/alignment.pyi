from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.seqtypes import (
    NucleotideSequence,
    ProteinSequence,
)
from numpy import (
    int32,
    int64,
    ndarray,
    str_,
)
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)


def _identify_terminal_gaps(seq_code: ndarray) -> Tuple[int, int]: ...


def get_codes(alignment: Alignment) -> ndarray: ...


def get_sequence_identity(alignment: Alignment, mode: str = 'not_terminal') -> float: ...


def get_symbols(alignment: Alignment) -> List[List[Union[str_, None]]]: ...


def score(
    alignment: Alignment,
    matrix: SubstitutionMatrix,
    gap_penalty: Union[Tuple[int, int], int] = -10,
    terminal_penalty: bool = True
) -> int64: ...


class Alignment:
    def __init__(
        self,
        sequences: Union[List[NucleotideSequence], List[ProteinSequence]],
        trace: ndarray,
        score: Optional[Union[int32, int]]
    ) -> None: ...
    def __str__(self) -> str: ...
    def _gapped_str(self, seq_index: int) -> str: ...
    def get_gapped_sequences(self) -> List[str]: ...
    @staticmethod
    def trace_from_strings(seq_str_list: List[str]) -> ndarray: ...
