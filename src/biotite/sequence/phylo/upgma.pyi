__all__ = ["upgma"]

import numpy as np
from biotite.sequence.phylo.tree import Tree
from biotite.typing import N, NDArray2

def upgma(distances: NDArray2[N, N, np.floating]) -> Tree: ...
