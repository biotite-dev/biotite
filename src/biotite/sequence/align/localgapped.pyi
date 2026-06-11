__all__ = ["align_local_gapped"]

from typing import Literal, overload
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.sequence import Sequence
from biotite.typing import S1, S2

@overload
def align_local_gapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    seed: tuple[int, int],
    threshold: int,
    gap_penalty: int | tuple[int, int] = -10,
    max_number: int = 1,
    direction: Literal["both", "upstream", "downstream"] = "both",
    score_only: Literal[False] = False,
    max_table_size: int | None = None,
) -> list[Alignment]: ...
@overload
def align_local_gapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    seed: tuple[int, int],
    threshold: int,
    gap_penalty: int | tuple[int, int] = -10,
    max_number: int = 1,
    direction: Literal["both", "upstream", "downstream"] = "both",
    score_only: Literal[True] = ...,
    max_table_size: int | None = None,
) -> int: ...
