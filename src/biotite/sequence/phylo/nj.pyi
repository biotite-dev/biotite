__all__ = ["neighbor_joining"]

import numpy as np
from biotite.sequence.phylo.tree import Tree
from biotite.typing import N, NDArray2

def neighbor_joining(distances: NDArray2[N, N, np.floating]) -> Tree: ...
