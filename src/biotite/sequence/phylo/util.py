# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Utility functions for internal use in the `biotite.sequence.phylo` package.
"""

from __future__ import annotations

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann"
__all__ = ["graph_from_edges"]

import networkx as nx
import numpy as np
from biotite.typing import N, NDArray1


def graph_from_edges(
    n_leaves: int,
    parents: NDArray1[N, np.integer],
    children: NDArray1[N, np.integer],
    distances: NDArray1[N, np.floating],
) -> nx.DiGraph:
    """
    Create a tree from edges given as arrays.

    Parameters
    ----------
    n_leaves : int
        The number of leaf nodes.
        The leaf nodes ``0`` to ``n_leaves-1`` are added to the graph
        before the edges to ensure that they exist even in the absence
        of edges.
    parents, children : ndarray, shape=(m,), dtype=int
        The parent and child node of each edge.
    distances : ndarray, shape=(m,), dtype=float
        The distance of each edge.

    Returns
    -------
    tree : DiGraph
        The tree.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(range(n_leaves))
    tree.add_weighted_edges_from(
        zip(parents.tolist(), children.tolist(), distances.tolist()),
        weight="distance",
    )
    return tree
