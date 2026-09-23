# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann"
__all__ = ["upgma"]

import networkx as nx
import numpy as np
from biotite.rust.sequence.phylo import upgma as rust_upgma
from biotite.sequence.phylo.util import graph_from_edges
from biotite.typing import N, NDArray2


def upgma(distances: NDArray2[N, N, np.floating]) -> nx.DiGraph:
    """
    Perform hierarchical clustering using the
    *unweighted pair group method with arithmetic mean* (UPGMA).

    This algorithm produces leaf nodes with the same distance to the
    root node.
    In the context of evolution this means a constant evolution rate
    (molecular clock).

    Parameters
    ----------
    distances : ndarray, shape=(n,n)
        Pairwise distance matrix.

    Returns
    -------
    tree : DiGraph
        A rooted binary tree.
        The leaf nodes ``0`` to ``n-1`` refer to the indices of
        `distances`, the intermediate nodes continue the numbering in
        the order of their creation.
        Hence, the root is the node with the highest number.
        Each edge points from the parent to the child node and has a
        ``"distance"`` attribute.

    Raises
    ------
    ValueError
        If the distance matrix is not symmetric
        or if any matrix entry is below 0.

    Examples
    --------

    >>> distances = np.array([
    ...     [0, 1, 7, 7, 9],
    ...     [1, 0, 7, 6, 8],
    ...     [7, 7, 0, 2, 4],
    ...     [7, 6, 2, 0, 3],
    ...     [9, 8, 4, 3, 0],
    ... ])
    >>> tree = upgma(distances)
    >>> print(to_newick(tree, include_distance=False))
    ((4,(3,2)),(1,0));
    """
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    parents, children, edge_distances = rust_upgma(distances)
    return graph_from_edges(len(distances), parents, children, edge_distances)
