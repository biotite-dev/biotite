# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann"
__all__ = ["neighbor_joining"]

import networkx as nx
import numpy as np
from biotite.rust.sequence.phylo import neighbor_joining as _rust_neighbor_joining
from biotite.sequence.phylo.tree import graph_from_edges
from biotite.typing import N, NDArray2


def neighbor_joining(distances: NDArray2[N, N, np.floating]) -> nx.DiGraph:
    """
    Perform hierarchical clustering using the
    *neighbor joining* algorithm. :footcite:`Saitou1987, Studier1988`

    In contrast to UPGMA this algorithm does not assume a constant
    evolution rate. The resulting tree is considered to be unrooted.

    Parameters
    ----------
    distances : ndarray, shape=(n,n)
        Pairwise distance matrix.

    Returns
    -------
    tree : DiGraph
        A rooted tree.
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

    Notes
    -----
    The created tree is binary except for the root node, that has three
    child nodes.
    As the tree is actually unrooted, :meth:`networkx.DiGraph.to_undirected()`
    gives a more faithful representation.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> distances = np.array([
    ...     [0, 1, 7, 7, 9],
    ...     [1, 0, 7, 6, 8],
    ...     [7, 7, 0, 2, 4],
    ...     [7, 6, 2, 0, 3],
    ...     [9, 8, 4, 3, 0],
    ... ])
    >>> tree = neighbor_joining(distances)
    >>> print(to_newick(tree, include_distance=False))
    (3,(2,(1,0)),4);
    """
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    parents, children, edge_distances = _rust_neighbor_joining(distances)
    return graph_from_edges(len(distances), parents, children, edge_distances)
