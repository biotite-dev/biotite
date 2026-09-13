# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides functions for creating and handling
(phylogenetic) trees.

A tree is represented as a rooted :class:`networkx.DiGraph`, whose
edges point from a parent node to its child node.
Each edge has a ``"distance"`` attribute, that gives the distance
between the two nodes.
The leaf nodes are the integers ``0`` to ``n-1``.
They are reference indices into a separate list or array, containing
the actual objects the tree represents, e.g. species names or
sequences.
The intermediate nodes may be any other hashable objects.
By convention, the trees created by this subpackage use the integers
``n`` upwards for the intermediate nodes, so that the root is the node
with the highest number.
This subpackage adds a few convenience functions on top of *NetworkX*
for common tree queries, namely :func:`get_root()`,
:func:`get_leaves()`, :func:`get_distance()` and
:func:`get_leaf_distances()`.
Other queries are directly available from *NetworkX*, for example
:func:`networkx.lowest_common_ancestor()` or
:meth:`networkx.DiGraph.successors()` and
:meth:`networkx.DiGraph.predecessors()` for the children and the parent
of a node, respectively.

A tree can be created from or exported to a *Newick* notation,
using the :func:`from_newick()` or :func:`to_newick()` function,
respectively.

A tree can be built from a pairwise distance matrix using the
popular *UPGMA* (:func:`upgma()`) and *Neighbor-Joining*
(:func:`neighbor_joining()`) algorithms.
"""

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann"

from .nj import *
from .tree import *
from .upgma import *
