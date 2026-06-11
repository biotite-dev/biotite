__all__ = ["align_ungapped", "align_optimal"]

from typing import Literal, overload
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.sequence import Sequence
from biotite.typing import S1, S2

@overload
def align_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    score_only: Literal[False] = False,
) -> Alignment: ...
@overload
def align_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    score_only: Literal[True],
) -> int: ...
def align_optimal(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
    local: bool = False,
    max_number: int = 1000,
) -> list[Alignment]: ...
