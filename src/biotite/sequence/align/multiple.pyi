__all__ = ["align_multiple"]

import numpy as np
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.phylo.tree import Tree
from biotite.sequence.sequence import Sequence
from biotite.typing import K, N, NDArray1, NDArray2, S

def align_multiple(
    sequences: list[Sequence[S]],
    matrix: SubstitutionMatrix[S, S],
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
    distances: NDArray2[N, N, np.floating] | None = None,
    guide_tree: Tree | None = None,
) -> tuple[
    Alignment,
    NDArray1[K, np.integer],
    Tree,
    NDArray2[N, N, np.floating],
]: ...
