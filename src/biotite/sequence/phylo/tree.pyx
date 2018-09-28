# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Tree", "TreeNode"]

cimport cython


class Tree:
    
    def __init__(self, TreeNode root not None):
        root.as_root()
        self._root = root
    
    @property
    def root(self):
        return self._root

    def get_distance(index1, index2):
        raise NotImplementedError()
    
    def get_tree_depth():
        raise NotImplementedError()
    
    def __str__(self):
        return str(self._root)


cdef class TreeNode:

    cdef int _index
    cdef float _distance
    cdef bint _is_root
    cdef TreeNode _parent
    cdef TreeNode _child1
    cdef TreeNode _child2

    def __cinit__(self, index=None,
                  TreeNode child1=None, TreeNode child2=None,
                  child1_distance=None, child2_distance=None):
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
    
    def _set_parent(self, TreeNode parent not None, float distance):
        self._parent = parent
        self._distance = distance

    @property
    def index(self):
        return None if self._index == -1 else self._index
    
    @property
    def childs(self):
        return self._child1, self._child2
    
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
            raise ValueError("Node has parent, cannot be a root node")
        self._is_root = True
    
    def distance_to(self, TreeNode node):
        raise NotImplementedError()
    
    def get_indices(self):
        index_list = []
        _get_indices(self, index_list)
        return index_list
    
    def __str__(self):
        if self.is_leaf():
            return f"{self._index}:{self._distance}"
        else:
            return f"({str(self._child1)},{str(self._child2)})" \
                   f":{self._distance}"
    

cdef _get_indices(TreeNode node, list index_list):
    if node._index == -1:
        # Intermediatte node -> Recursive calls
        _get_indices(node._child1, index_list)
        _get_indices(node._child2, index_list)
    else:
        # Terminal node -> add index -> terminate
        index_list.append(node._index)