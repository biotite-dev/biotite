__all__ = ["align_banded"]

from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.sequence import Sequence
from biotite.typing import S1, S2

def align_banded(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2],
    band: tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    local: bool = False,
    max_number: int = 1000,
) -> list[Alignment]: ...
