# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides functions and data structures for creating
(phylogenetic) trees.

The :class:`Tree` is the central class in this subpackage.
It wraps a *root* :class:`TreeNode` object.
A :class:`TreeNode` is either an intermediate node, if it has child
:class:`TreeNode` objects, or otherwise a leaf node.

A :class:`Tree` is not a container itself:
Objects, e.g species names or sequences, that are represented by the
nodes, cannot be stored directly in a :class:`Tree` or
:class:`TreeNode`.
Instead, each leaf node has a reference index:
These indices refer to a separate list or array, containing the actual
reference objects.

A :class:`Tree` can be created from or exported to a *Newick* notation,
usingthe :func:`Tree.from_newick()` or :func:`Tree.to_newick()` method,
respectively.

A :class:`Tree` can be build from a pairwise distance matrix using the
popular *UPGMA* (:func:`upgma()`) and *Neighbor-Joining*
(:func:`neighbor_joining()`) algorithms.
"""

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann"

from .nj import *
from .tree import *
from .upgma import *
