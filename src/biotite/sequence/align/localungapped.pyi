# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Generic, Literal, overload
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.sequence import Sequence
from biotite.typing import S1, S2

__all__ = ["SeedExtension", "align_local_ungapped"]

class SeedExtension(Generic[S1, S2]):
    def __init__(
        self,
        matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
        threshold: int,
        direction: Literal["both", "upstream", "downstream"] = "both",
    ) -> None: ...
    @overload
    def align(
        self,
        seq1: Sequence[S1],
        seq2: Sequence[S2],
        seed: tuple[int, int],
        score_only: Literal[False] = False,
    ) -> Alignment: ...
    @overload
    def align(
        self,
        seq1: Sequence[S1],
        seq2: Sequence[S2],
        seed: tuple[int, int],
        score_only: Literal[True],
    ) -> int: ...

@overload
def align_local_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
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
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    seed: tuple[int, int],
    threshold: int,
    direction: Literal["both", "upstream", "downstream"] = "both",
    *,
    score_only: Literal[True],
    check_matrix: bool = True,
) -> int: ...
