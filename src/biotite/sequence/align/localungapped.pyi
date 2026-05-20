__all__ = ["align_local_ungapped"]

from typing import Literal, overload
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.sequence import Sequence
from biotite.typing import S1, S2

@overload
def align_local_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    seed: tuple[int, int],
    threshold: int,
    direction: Literal["both", "upstream", "downstream"] = "both",
    score_only: Literal[False] = False,
    check_matrix: bool = True,
) -> Alignment: ...
@overload
def align_local_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    seed: tuple[int, int],
    threshold: int,
    direction: Literal["both", "upstream", "downstream"] = "both",
    score_only: Literal[True] = ...,
    check_matrix: bool = True,
) -> int: ...
