# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Tree", "TreeNode", "TreeError"]

cimport cython
cimport numpy as np

import copy
import numpy as np
from ...copyable import Copyable


class Tree(Copyable):
    """
    __init__(root)
    
    A `Tree` represents a rooted tree
    (e.g. alignment guide tree).
    The tree itself is represented by `TreeNode` objects.
    The root node is accessible via the `root` property.
    The property `leaves` contains a list of the leaf nodes, where the
    index of the leaf node in this list is equal to the reference index
    of the leaf node (``leaf.index``).

    Two `Tree` objects are equal if they are the same object,
    so the ``==`` operator is equal to the ``is`` operator.

    The amount of leaves in a tree can be determined via the `len()`
    function.

    Objects of this class are immutable.

    Parameters
    ----------
    root: TreeNode
        The root of the tree.
        The constructor calls the node's `as_root()` method,
        in order to make it immutable.
    
    Attributes
    ----------
    root : TreeNode
        The root node of the tree.
    leaves : list of TreeNode
        The leaf nodes of the tree.
        The index of the leaf node in this list is equal to the
        reference index of the leaf node.
        This attribute is a shallow copy of the repsective internal
        object.

    Examples
    --------
    
    >>> leaf1 = TreeNode(index=0)
    >>> leaf2 = TreeNode(index=1)
    >>> leaf3 = TreeNode(index=2)
    >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
    >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
    >>> tree  = Tree(root)
    >>> print(tree)
    ((0:5.0,1:7.0):3.0,2:10.0):0.0;
    """
    
    def __init__(self, TreeNode root not None):
        root.as_root()
        self._root = root
        
        cdef list leaves_unsorted = self._root.get_leaves()
        cdef int leaf_count = len(leaves_unsorted)
        cdef np.ndarray indices = np.array(
            [leaf.index for leaf in leaves_unsorted]
        )
        self._leaves = [None] * leaf_count
        cdef int i
        cdef int index
        for i in range(len(indices)):
            index = indices[i]
            if index >= leaf_count or index < 0:
                raise TreeError("The tree's indices are out of range")
            self._leaves[index] = leaves_unsorted[i]
    
    def __copy_create__(self):
        return Tree(self._root.copy())
    
    @property
    def root(self):
        return self._root
    
    @property
    def leaves(self):
        return copy.copy(self._leaves)

    def get_distance(self, index1, index2, bint topological=False):
        """
        get_distance(index1, index2, topological=False)
        
        Get the distance between two leaf nodes.

        The distance is the sum of all distances from the each of the
        two nodes to their lowest common ancestor.

        Parameters
        ----------
        index1, index2 : int
            The reference indices of the two leaf nodes, to calculate
            the distance for.
        topological : bool, optional
            If True the topological distance is measured, i.e. all
            child-parent distance is 1.
            Otherwise, the distances from the `distance` attribute are
            used.

        Returns
        -------
        distance : float
            The distance between the nodes.
        
        Examples
        --------

        >>> leaf1 = TreeNode(index=0)
        >>> leaf2 = TreeNode(index=1)
        >>> leaf3 = TreeNode(index=2)
        >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
        >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
        >>> tree = Tree(root)
        >>> print(tree.get_distance(0,1))
        12.0
        >>> print(tree.get_distance(0,2))
        18.0
        >>> print(tree.get_distance(1,2))
        20.0
        """
        return self._leaves[index1].distance_to(
            self._leaves[index2], topological
        )
    
    def to_newick(self, labels=None, bint include_distance=True):
        """
        to_newick(labels=None, include_distance=True)

        Obtain the Newick notation of the tree.

        Parameters
        ----------
        labels : iterable object of str
            The labels the indices in the leaf nodes srefer to
        include_distance : bool
            If true, the distances are displayed in the newick notation,
            otherwise they are omitted.
        
        Returns
        -------
        newick : str
            The Newick notation of the tree.

        Examples
        --------
        
        >>> leaf1 = TreeNode(index=0)
        >>> leaf2 = TreeNode(index=1)
        >>> leaf3 = TreeNode(index=2)
        >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
        >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
        >>> tree = Tree(root)
        >>> print(tree.to_newick())
        ((0:5.0,1:7.0):3.0,2:10.0):0.0;
        >>> print(tree.to_newick(include_distance=False))
        ((0,1),2);
        >>> labels = ["foo", "bar", "foobar"]
        >>> print(tree.to_newick(labels=labels, include_distance=False))
        ((foo,bar),foobar);
        """
        return self._root.to_newick(labels, include_distance) + ";"
    
    @staticmethod
    def from_newick(str newick, list labels=None):
        """
        from_newick(newick, labels=None)
        
        Create a tree from a Newick notation.

        Parameters
        ----------
        newick : str
            The Newick notation to create the tree from.
        labels : list of str, optional
            If the Newick notation contains labels, that are not
            parseable into reference indices,
            i.e. they are not integers, this parameter can be provided
            to convert these labels into reference indices.
            The corresponding index is the position of the label in the
            provided list.

        Returns
        -------
        tree : Tree
            A tree created from the Newick notation
        
        Notes
        -----
        This function does accept but does not require the Newick string
        to have the terminal semicolon.
        Keep in mind that the `Tree` class does support any labels on
        intermediate nodes.
        If the string contains such labels, they are discarded.
        """
        newick = newick.strip()
        # Remove terminal colon as required by 'TreeNode.from_newick()'
        if newick[-1] == ";":
            newick = newick[:-1]
        root, distance = TreeNode.from_newick(newick, labels)
        return Tree(root)

    def __str__(self):
        return self.to_newick()
    
    def __len__(self):
        return len(self._leaves)


cdef class TreeNode:
    """
    __init__(child1=None, child2=None,
             child1_distance=None, child2_distance=None,
             index=None)
    
    `TreeNode` objects are part of a rooted tree
    (e.g. alignment guide tree).
    There are two `TreeNode` subtypes:
        
        - Leaf node - Cannot have child nodes but has an index referring
          to an array-like reference object.
        - Intermediate node - Has child nodes but no reference index
    
    This type is determined by the time of the object's creation.
    
    Every `TreeNode` has a reference to its parent node.
    A root node is node without a parent node, that is finalized
    using `as_root()`.
    The call of this function prevents that a the node can be used as
    child.

    `TreeNode` objects are semi-immutable:
    The child nodes or the reference index are fixed at the time of
    creation.
    Only the parent can be set once, when the parent node is created. 
    `TreeNode` objects that are finalized using `as_root()` are
    completely immutable.

    All object properties are read-only.

    Two `TreeNode` objects are equal if they are the same object,
    so the ``==`` operator is equal to the ``is`` operator.

    Parameters
    ----------
    children: array-like object of TreeNode, length=n, optional
        The children of this node.
        As this causes the creation of an intermediate node,
        this parameter cannot be used in combination with `index`.
    distances: array-like object of float, length=n, optional
        The distances of the child nodes to this node.
        Must be set if `children` is set.
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
    children : tuple of TreeNode
        The child nodes.
        `None` if node is a leaf node.
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
    >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
    >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
    >>> print(root)
    ((0:5.0,1:7.0):3.0,2:10.0):0.0
    """

    cdef int _index
    cdef float _distance
    cdef bint _is_root
    cdef TreeNode _parent
    cdef tuple _children

    def __cinit__(self, children=None, distances=None, index=None):
        self._is_root = False
        self._distance = 0
        self._parent = None
        cdef TreeNode child
        cdef float distance
        if index is None:
            # Node is intermediate -> has children
            if children is None or distances is None:
                raise TypeError(
                    "Either reference index (for terminal node) or "
                    "child nodes including the distance "
                    "(for intermediate node) must be set"
                )
            for item in children:
                if not isinstance(item, TreeNode):
                    raise TypeError(
                        f"Expected 'TreeNode', but got '{type(item).__name__}'"
                    )
            for item in distances:
                if not isinstance(item, float) and not isinstance(item, int):
                    raise TypeError(
                        f"Expected 'float' or 'int', "
                        f"but got '{type(item).__name__}'"
                    )
            if len(children) == 0:
                raise TreeError(
                    "Intermediate nodes must at least contain one child node"
                )
            if len(children) != len(distances):
                raise ValueError(
                    "The number of children must equal the number of distances"
                )
            for i in range(len(children)):
                for j in range(len(children)):
                    if i != j and children[i] is children[j]:
                        raise TreeError(
                            "Two child nodes cannot be the same object"
                        )
            self._index = -1
            self._children = tuple(children)
            for child, distance in zip(children, distances):
                child._set_parent(self, distance)
        elif index < 0:
            raise ValueError("Index cannot be negative")
        else:
            # Node is terminal -> has no children
            if children is not None or distances is not None:
                raise TypeError(
                    "Reference index and child nodes are mutually exclusive"
                )
            self._index = index
            self._children = None
    
    def _set_parent(self, TreeNode parent not None, float distance):
        if self._parent is not None or self._is_root:
            raise TreeError("Node already has a parent")
        self._parent = parent
        self._distance = distance
    
    def copy(self):
        """
        copy()

        Create a deep copy of this `TreeNode`.

        The copy includes this node, its reference index and deep copies
        of its child nodes.
        The parent node and the distance to it is not included.
        """
        if self.is_leaf():
            return TreeNode(index=self._index)
        else:
            distances = [child.distance for child in self._children]
            children_clones = [child.copy() for child in self._children]
            return TreeNode(children_clones, distances)

    @property
    def index(self):
        return None if self._index == -1 else self._index
    
    @property
    def children(self):
        return self._children
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def distance(self):
        return None if self._parent is None else self._distance

    def is_leaf(self):
        """
        is_leaf()
        
        Check if the node is a leaf node.

        Returns
        -------
        is_leaf : bool
            True if the node is a leaf node, false otherwise.
        """
        return False if self._index == -1 else True
    
    def is_root(self):
        """
        is_root()
        
        Check if the node is a root node.

        Returns
        -------
        is_root : bool
            True if the node is a root node, false otherwise.
        """
        return bool(self._is_root)
    
    def as_root(self):
        """
        as_root()
        
        Convert the node into a root node.

        When a root node is used as `child` parameter in the
        construction of a potential parent node, a `TreeError` is
        raised.
        """
        if self._parent is not None:
            raise TreeError("Node has parent, cannot be a root node")
        self._is_root = True
    
    def distance_to(self, TreeNode node, bint topological=False):
        """
        distance_to(node, topological=False)
        
        Get the distance of this node to another node.

        The distance is the sum of all distances from this and the other
        node to the lowest common ancestor.

        Parameters
        ----------
        node : TreeNode
            The second node for distance calculation.
        

        Returns
        -------
        distance : float
            The distance of this node to `node`.
        
        Raises
        ------
        TreeError
            If the nodes have no common ancestor.
        
        Examples
        --------

        >>> leaf1 = TreeNode(index=0)
        >>> leaf2 = TreeNode(index=1)
        >>> leaf3 = TreeNode(index=2)
        >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
        >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
        >>> print(leaf1.distance_to(leaf2))
        12.0
        >>> print(leaf1.distance_to(leaf3))
        18.0
        """
        # Sum distances until LCA has been reached
        cdef float distance = 0
        cdef TreeNode current_node = None
        cdef TreeNode lca = self.lowest_common_ancestor(node)
        if lca is None:
            raise TreeError("The nodes do not have a common ancestor")
        current_node = self
        while current_node is not lca:
            if topological:
                distance += 1
            else:
                distance += current_node._distance
            current_node = current_node._parent
        current_node = node
        while current_node is not lca:
            if topological:
                distance += 1
            else:
                distance += current_node._distance
            current_node = current_node._parent
        return distance
    
    def lowest_common_ancestor(self, TreeNode node):
        """
        lowest_common_ancestor(node)
        
        Get the lowest common ancestor of this node and another node.

        Parameters
        ----------
        node : TreeNode
            The node to get the lowest common ancestor with.

        Returns
        -------
        ancestor : TreeNode or None
            The lowest common ancestor. `None` if the nodes have no
            common ancestor, i.e. they are not in the same tree
        """
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
        """
        get_indices()
        
        Get an array of reference indices that leaf nodes of this node
        contain.

        This method identifies all leaf nodes, which have this node as
        ancestor and puts the contained indices into an array.
        If this node is a leaf node itself, the array contains the
        reference index of this node as single element.

        Returns
        -------
        indices : ndarray, dtype=int32
            The reference indices of direct and indirect child leaf
            nodes of this node.

        Examples
        --------

        >>> leaf0 = TreeNode(index=0)
        >>> leaf1 = TreeNode(index=1)
        >>> leaf2 = TreeNode(index=2)
        >>> leaf3 = TreeNode(index=3)
        >>> intr0 = TreeNode([leaf0, leaf2], [0, 0])
        >>> intr1 = TreeNode([leaf1, leaf3], [0, 0])
        >>> root  = TreeNode([intr0, intr1], [0, 0])
        >>> print(leaf0.get_indices())
        [0]
        >>> print(intr0.get_indices())
        [0 2]
        >>> print(intr1.get_indices())
        [1 3]
        >>> print(root.get_indices())
        [0 2 1 3]
        """
        cdef TreeNode leaf
        return np.array(
            [leaf._index for leaf in self.get_leaves()], dtype=np.int32
        )

    def get_leaves(self):
        """
        get_leaves()
        Get a list of leaf nodes that are direct or indirect child nodes
        of this node.

        This method identifies all leaf nodes, which have this node as
        ancestor.
        If this node is a leaf node itself, the list contains this node
        as single element.

        Returns
        -------
        leaf_nodes : list
            The leaf nodes, that are direct or indirect child nodes
            of this node.
        """
        cdef list leaf_list = []
        # delegate to 'cdef' method
        # to reduce overhead of recursive function calling
        _get_leaves(self, leaf_list)
        return leaf_list
    
    def get_leaf_count(self):
        """"
        get_leaf_count()

        Get the number of direct or indirect leaves of this ńode.

        This method identifies all leaf nodes, which have this node as
        ancestor.
        If this node is a leaf node itself, 1 is returned.
        """
        return _get_leaf_count(self)
    
    def to_newick(self, labels=None, bint include_distance=True):
        """
        to_newick(labels=None, include_distance=True)
        
        Obtain the node represented in Newick notation.

        The terminal semicolon is not included.

        Parameters
        ----------
        labels : iterable object of str
            The labels the indices in the leaf nodes refer to
        include_distance : bool
            If true, the distances are displayed in the newick notation,
            otherwise they are omitted.
        
        Returns
        -------
        newick : str
            The Newick notation of the node.

        Examples
        --------
        
        >>> leaf1 = TreeNode(index=0)
        >>> leaf2 = TreeNode(index=1)
        >>> leaf3 = TreeNode(index=2)
        >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
        >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
        >>> print(root.to_newick())
        ((0:5.0,1:7.0):3.0,2:10.0):0.0
        >>> print(root.to_newick(include_distance=False))
        ((0,1),2)
        >>> labels = ["foo", "bar", "foobar"]
        >>> print(root.to_newick(labels=labels, include_distance=False))
        ((foo,bar),foobar)
        """
        if self.is_leaf():
            if labels is not None:
                for label in labels:
                    label = labels[self._index]
                    # Characters that are part of the Newick syntax
                    # are illegal
                    illegal_chars = [",",":",";","(",")"]
                    for char in illegal_chars:
                        if char in label:
                            raise ValueError(
                                f"Label '{label}' contains "
                                f"illegal character '{char}'"
                            )
            else:
                label = str(self._index)
            if include_distance:
                return f"{label}:{self._distance}"
            else:
                return f"{label}"
        else:
            # Build string in a recursive way
            child_strings = [child.to_newick(labels, include_distance)
                             for child in self._children]
            if include_distance:
                return f"({','.join(child_strings)}):{self._distance}"
            else:
                return f"({','.join(child_strings)})"
    
    @staticmethod
    def from_newick(str newick, list labels=None):
        """
        from_newick(newick, labels=None)

        Create a node and all its child nodes from a Newick notation.

        Parameters
        ----------
        newick : str
            The Newick notation to create the node from.
        labels : list of str, optional
            If the Newick notation contains labels, that are not
            parseable into reference indices,
            i.e. they are not integers, this parameter can be provided
            to convert these labels into reference indices.
            The corresponding index is the position of the label in the
            provided list.

        Returns
        -------
        node : TreeNode
            The tree node parsed from the Newick notation.
        distance : float
            Distance of the node to its parent. If the newick notation
            does not provide a distance, it is set to 0 by default.
        
        Notes
        -----
        The provided Newick notation must not have a terminal semicolon.
        If you have a Newick notation that covers an entire tree, you
        may use the same method in the `Tree` class instead.
        Keep in mind that the `TreeNode` class does support any labels
        on intermediate nodes.
        If the string contains such labels, they are discarded.
        """
        cdef int i
        cdef int subnewick_start_i = -1
        cdef int subnewick_stop_i  = -1
        cdef int level = 0
        cdef list comma_pos
        cdef list children
        cdef list distances
        cdef int pos
        cdef int next_pos
        
        # Ignore any whitespace
        newick = "".join(newick.split())

        # Find brackets belonging to sub-newick
        # e.g. (A:0.1,B:0.2):0.5
        #      ^           ^
        for i in range(len(newick)):
            char = newick[i]
            if char == "(":
                subnewick_start_i = i
                break
            if char == ")":
                raise ValueError("Bracket closed before it was opened")
        for i in reversed(range(len(newick))):
            char = newick[i]
            if char == ")":
                subnewick_stop_i = i+1
                break
            if char == "(":
                raise ValueError("Bracket was opened but not closed")
        
        if subnewick_start_i == -1 and subnewick_stop_i == -1:
            # No brackets -> no sub-newwick -> Leaf node
            label_and_distance = newick
            try:
                label, distance = label_and_distance.split(":")
                distance = float(distance)
            except ValueError:
                # No colon -> No distance is provided
                distance = 0
                label = label_and_distance
            index = int(label) if labels is None else labels.index(label)
            return TreeNode(index=index), distance
        
        else:
            # Intermediate node
            if subnewick_stop_i == len(newick):
                # Node with neither distance nor label
                label = None
                distance = 0
            else:
                label_and_distance = newick[subnewick_stop_i:]
                try:
                    label, distance = label_and_distance.split(":")
                    distance = float(distance)
                except ValueError:
                    # No colon -> No distance is provided
                    distance = 0
                    label = label_and_distance
                # Label of intermediate nodes is discarded 
                distance = float(distance)
            
            subnewick = newick[subnewick_start_i+1 : subnewick_stop_i-1]
            # Parse childs
            # Split subnewick at ',' if ',' is at current level
            # (not in a subsubnewick)
            comma_pos = []
            for i, char in enumerate(subnewick):
                if char == "(":
                    level += 1
                elif char == ")":
                    level -= 1
                elif char == ",":
                    if level == 0:
                        comma_pos.append(i)
                if level < 0:
                    raise ValueError("Bracket closed before it was opened")
        
            children = []
            distances = []
            # Recursive tree construction
            for i, pos in enumerate(comma_pos):
                if i == 0:
                    # (A,B),(C,D),(E,F)
                    # -----
                    child, dist = TreeNode.from_newick(
                        subnewick[:pos], labels=labels
                    )
                else:
                    # (A,B),(C,D),(E,F)
                    #       -----
                    prev_pos = comma_pos[i-1]
                    child, dist = TreeNode.from_newick(
                        subnewick[prev_pos+1 : pos], labels=labels
                    )
                children.append(child)
                distances.append(dist)
            # Node after last comma
            # (A,B),(C,D),(E,F)
            #             -----
            child, dist = TreeNode.from_newick(
                subnewick[comma_pos[-1]+1:], labels=labels
            )
            children.append(child)
            distances.append(dist)
            return TreeNode(children, distances), distance

    def __str__(self):
        return self.to_newick()


cdef _get_leaves(TreeNode node, list leaf_list):
    cdef TreeNode child
    if node._index == -1:
        # Intermediate node -> Recursive calls
        for child in node._children:
            _get_leaves(child, leaf_list)
    else:
        # Node itself is leaf node -> add node -> terminate
        leaf_list.append(node)


cdef int _get_leaf_count(TreeNode node):
    cdef TreeNode child
    cdef int count = 0
    if node._index == -1:
        # Intermediate node -> Recursive calls
        for child in node._children:
            count += _get_leaf_count(child)
        return count
    else:
        # Leaf node -> return count of itself = 1
        return 1


cdef list _create_path_to_root(TreeNode node):
    """
    Create a list of nodes representing the path from this node to the
    specified node
    """
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