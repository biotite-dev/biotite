# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Tree", "TreeNode"]

cimport cython
cimport numpy as np

import copy
import numpy as np


class Tree:
    
    def __init__(self, TreeNode root not None):
        root.as_root()
        self._root = root
        
        cdef list leaves_unsorted = self._root.get_leaves()
        cdef np.ndarray indices = np.array(
            [leaf.index for leaf in leaves_unsorted]
        )
        self._leaves = [None] * len(leaves_unsorted)
        cdef int i
        for i in range(len(indices)):
            self._leaves[indices[i]] = leaves_unsorted[i]
    
    @property
    def root(self):
        return self._root
    
    @property
    def leaves(self):
        return copy.copy(self._leaves)

    def get_distance(self, index1, index2):
        return self._leaves[index1].distance_to(self._leaves[index2])
    
    def to_newick(self, labels=None, bint include_distance=True):
        return self._root.to_newick(labels, include_distance) + ";"

    def __str__(self):
        return self.to_newick()


cdef class TreeNode:
    """
    __init__(child1=None, child2=None,
             child1_distance=None, child2_distance=None,
             index=None)
    
    `TreeNode` objects are part of a rooted binary tree
    (e.g. phylogentic tree).
    There are two `TreeNode` subtypes.:
        
        - Leaf node - Cannot have child nodes but has an index referring
          to an array-like reference object.
        - Intermediate node - Has two child nodes but no reference index
    
    This type is determined by the time of the object's creation.
    
    Every `TreeNode` has a reference to its parent node.
    A root node is node without a parent node.
    In order to prevent that a root node cannot be used as child,
    the `as_root()` method can be called.

    `TreeNode` objects are semi-immutable:
    The child nodes or the reference index are fixed at the time of
    creation.
    Only the parent can be set once, when the parent node is created. 
    `TreeNode` objects that are finalized using `as_root()` are
    completely immutable.

    All object attributes are read-only.

    Parameters
    ----------
    child1, child2: TreeNode, optional
        The childs of this node.
        As this causes the creation of an intermediate node,
        this parameter cannot be used in combination with `index`.
    child1_distance, child2_distance: float, optional
        The distances of the child nodes to this node.
        Must be set if `child1` and `child2` are set.
    index: int, optional
        Index to a reference array-like object
        (e.g. list of sequences or labels).
        Must be a positive integer.
        As this causes the creation of a leaf node, this parameter
        cannot be used in combination with the other parameters.
    
    Attributes
    ----------
    parent : TreeNode
        The parent node.
        `None` if node has no parent.
    childs : tuple(TreeNode, TreeNode)
        The child nodes.
        `None` if node has no child nodes.
    index : int
        The index to a reference array-like object.
        `None` if node is not a leaf node.
    distance : float
        Distance to parent node.
        `None` if `parent` is `Ǹone`.

    Examples
    --------
    
    >>> leaf1 = TreeNode(index=0)
    >>> leaf2 = TreeNode(index=1)
    >>> leaf3 = TreeNode(index=2)
    >>> inter = TreeNode(leaf1, leaf2, 5.0, 7.0)
    >>> root  = TreeNode(inter, leaf3, 3.0, 10.0)
    >>> print(root)
    ((0:5.0,1:7.0):3.0,2:10.0):0.0
    """

    cdef int _index
    cdef float _distance
    cdef bint _is_root
    cdef TreeNode _parent
    cdef TreeNode _child1
    cdef TreeNode _child2

    def __cinit__(self, TreeNode child1=None, TreeNode child2=None,
                  child1_distance=None, child2_distance=None, index=None):
        self._is_root = False
        self._distance = 0
        self._parent = None
        if index is None:
            # Node is intermediate -> has childs
            if child1 is None or child2 is None or \
                child1_distance is None or child2_distance is None:
                    raise TypeError(
                        "Either reference index (for terminal node) or "
                        "child nodes including the distance "
                        "(for intermediate node) must be set"
                    )
            if child1 is child2:
                raise ValueError("The child nodes cannot be the same object")
            self._index = -1
            self._child1 = child1
            self._child2 = child2
            self._child1._set_parent(self, child1_distance)
            self._child2._set_parent(self, child2_distance)
        elif index < 0:
            raise ValueError("Index cannot be negative")
        else:
            # Node is terminal -> has no childs
            if child1 is not None or child2 is not None:
                raise TypeError(
                    "Reference index and child nodes are mutually exclusive"
                )
            self._index = index
            self._child1 = None
            self._child2 = None
    
    def _set_parent(self, TreeNode parent not None, float distance):
        if self._parent is not None:
            raise TreeError("Node already has a parent")
        self._parent = parent
        self._distance = distance

    @property
    def index(self):
        return None if self._index == -1 else self._index
    
    @property
    def childs(self):
        # If child1 is None child2 is also None
        if self._child1 is not None:
            return self._child1, self._child2
        else:
            return None
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def distance(self):
        return None if self._parent is None else self._distance

    def is_leaf(self):
        return False if self._index == -1 else True
    
    def is_root(self):
        return bool(self._is_root)
    
    def as_root(self):
        if self._parent is not None:
            raise TreeError("Node has parent, cannot be a root node")
        self._is_root = True
    
    def distance_to(self, TreeNode node):
        # Sum distances until LCA has been reached
        cdef float distance = 0
        cdef TreeNode current_node = None
        cdef TreeNode lca = self.lowest_common_ancestor(node)
        if lca is None:
            raise TreeError("The nodes do not have a common ancestor")
        current_node = self
        while current_node is not lca:
            distance += current_node._distance
            current_node = current_node._parent
        current_node = node
        while current_node is not lca:
            distance += current_node._distance
            current_node = current_node._parent
        return distance
    
    def lowest_common_ancestor(self, TreeNode node):
        cdef int i
        cdef TreeNode lca = None
        # Create two paths from the leaves to root
        cdef list self_path = _create_path_to_root(self)
        cdef list other_path = _create_path_to_root(node)
        # Reverse Iteration through path (beginning from root)
        # until the paths diverge
        for i in range(-1, -min(len(self_path), len(other_path))-1, -1):
            if self_path[i] is other_path[i]:
                # Same node -> common ancestor
                lca = self_path[i]
            else:
                # Different node -> Not common ancestor
                # -> return last common ancewstor found
                break
        return lca
    
    def get_indices(self):
        return np.array(
            [leaf._index for leaf in self.get_leaves()], dtype=np.int32
        )

    def get_leaves(self):
        cdef list leaf_list = []
        _get_leaves(self, leaf_list)
        return leaf_list
    
    def to_newick(self, labels=None, bint include_distance=True):
        if self.is_leaf():
            if labels is not None:
                label = labels[self._index]
            else:
                label = str(self._index)
            if include_distance:
                return f"{label}:{self._distance}"
            else:
                return f"{label}"
        else:
            child1_str = self._child1.to_newick(labels, include_distance)
            child2_str = self._child2.to_newick(labels, include_distance)
            if include_distance:
                return f"({child1_str},{child2_str}):{self._distance}"
            else:
                return f"({child1_str},{child2_str})"

    def __str__(self):
        return self.to_newick()
    

cdef _get_leaves(TreeNode node, list leaf_list):
    if node._index == -1:
        # Intermediatte node -> Recursive calls
        _get_leaves(node._child1, leaf_list)
        _get_leaves(node._child2, leaf_list)
    else:
        # Terminal node -> add node -> terminate
        leaf_list.append(node)


cdef list _create_path_to_root(TreeNode node):
    cdef list path = []
    cdef TreeNode current_node = node
    while current_node is not None:
        path.append(current_node)
        current_node = current_node._parent
    return path


class TreeError(Exception):
    """
    An exception that occurs in context of tree topology.
    """
    pass