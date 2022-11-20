# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann, Tom David Müller"
__all__ = ["Tree", "TreeNode", "as_binary", "TreeError"]

cimport cython
cimport numpy as np

import copy
import numpy as np
import networkx as nx
from ...file import InvalidFileError
from ...copyable import Copyable


class Tree(Copyable):
    """
    __init__(root)
    
    A :class:`Tree` represents a rooted tree
    (e.g. alignment guide tree or phylogenetic tree).

    The tree itself wraps a *root* :class:`TreeNode` object,
    accessible via the :attr:`root` property.

    A :class:`Tree` is not a container itself:
    Objects, e.g species names or sequences, that are represented by the
    nodes, cannot be stored directly in a :class:`Tree` or its nodes.
    Instead, each leaf :class:`TreeNode` has a reference index:
    These indices refer to a separate list or array, containing the
    actual reference objects.

    The property :attr:`leaves` contains a list of the leaf nodes,
    where the index of the leaf node in this list is equal to the
    reference index of the leaf node (``leaf.index``).

    The amount of leaves in a tree can be determined via the
    :func:`len()` function.

    Objects of this class are immutable.

    Parameters
    ----------
    root: TreeNode
        The root of the tree.
        The constructor calls the node's :func:`as_root()` method,
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

    >>> objects = ["An object", "Another object", "Yet another one"]
    >>> leaf1 = TreeNode(index=0)
    >>> leaf2 = TreeNode(index=1)
    >>> leaf3 = TreeNode(index=2)
    >>> inter = TreeNode([leaf1, leaf2], [5.0, 7.0])
    >>> root  = TreeNode([inter, leaf3], [3.0, 10.0])
    >>> tree  = Tree(root)
    >>> print(tree)
    ((0:5.0,1:7.0):3.0,2:10.0):0.0;
    >>> print([objects[node.index] for node in tree.leaves])
    ['An object', 'Another object', 'Yet another one']
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
    
    def as_graph(self):
        """
        as_graph()
        
        Obtain a graph representation of the :class:`Tree`.

        Returns
        -------
        bond_set : DiGraph
            A *NetworkX* directed graph.
            For a leaf node the graph node is its reference index.
            For an intermediate and root node the graph node is a tuple
            containing it children nodes.
            Each edge has a ``"distance"`` attribute depicting the
            distance between the nodes.
            Each edge starts from the parent ends at its child.
        
        Examples
        --------

        >>> leaves = [TreeNode(index=i) for i in range(3)]
        >>> intermediate = TreeNode([leaves[0], leaves[1]], [2.0, 3.0])
        >>> root = TreeNode([intermediate, leaves[2]], [1.0, 5.0])
        >>> tree = Tree(root)
        >>> graph = tree.as_graph()
        >>> for node_i, node_j in graph.edges:
        ...     print(f"{str(node_i):12}  ->  {str(node_j):12}")
        (0, 1)        ->  0
        (0, 1)        ->  1
        ((0, 1), 2)   ->  (0, 1)
        ((0, 1), 2)   ->  2
        """
        cdef tuple children
        cdef bint children_already_handled
        cdef TreeNode node, child, parent

        graph = nx.DiGraph()
        
        # This dict maps a TreeNode to its corresponding int or tuple
        cdef dict node_repr = {}

        # A First-In-First-Out queue for iterative handling of each node
        # Starting with all leaf nodes 
        cdef list queue = copy.copy(self._leaves)
        # A set representation of the same queue for efficient
        # '__contains__()' operation
        cdef set queue_set = set(self._leaves)
        while len(queue) > 0:
            node = queue.pop(0)
            
            if node.is_leaf():
                node_repr[node] = node.index
            else:
                children = node.children
                children_handled = True
                for child in children:
                    if child not in node_repr:
                        children_handled = False
                # If the node representation of any child of this node
                # is not calculated yet, put this node to the end of the
                # queue and handle it later
                if not children_handled:
                    queue.append(node)
                    continue
                else:
                    repr = tuple(node_repr[child] for child in children)
                    node_repr[node] = repr
                    # Add adges to children in graph
                    for child in children:
                        graph.add_edge(
                            repr, node_repr[child], distance=child.distance
                        )
            
            # This leads finally to termination of the loop:
            # When the root node is handled the last element in the
            # queue is handled and no new node is added to the queue
            if not node.is_root():
                parent = node.parent
                # The parent node might be already in the queue from
                # handling another child node
                if parent not in queue_set:
                    queue.append(parent)
                    queue_set.add(parent)
            
            # Node is handled
            # -> not in 'queue' anymore
            # -> remove also from 'queue_set'
            queue_set.remove(node)
        
        return graph

        

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
    
    def to_newick(self, labels=None, bint include_distance=True, 
                  round_distance=None):
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
        round_distance : int, optional
            If set, the distances are rounded to the given number of
            digits.
        
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
        return self._root.to_newick(
            labels, include_distance, round_distance
        ) + ";"
    
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
        
        Keep in mind that the :class:`Tree` class does not support any
        labels on intermediate nodes.
        If the string contains such labels, they are discarded.
        """
        newick = newick.strip()
        if len(newick) == 0:
            raise InvalidFileError("Newick string is empty")
        # Remove terminal colon as required by 'TreeNode.from_newick()'
        if newick[-1] == ";":
            newick = newick[:-1]
        root, distance = TreeNode.from_newick(newick, labels)
        return Tree(root)

    def __str__(self):
        return self.to_newick()
    
    def __len__(self):
        return len(self._leaves)
    
    def __eq__(self, item):
        if not isinstance(item, Tree):
            return False
        return self._root == item._root
    
    def __hash__(self):
        return hash(self._root)


cdef class TreeNode:
    """
    __init__(children=None, distances=None, index=None)
    
    :class:`TreeNode` objects are part of a rooted tree
    (e.g. alignment guide tree).
    There are two :class:`TreeNode` subtypes:
        
        - Leaf node - Cannot have child nodes but has an index referring
          to an array-like reference object.
        - Intermediate node - Has child nodes but no reference index
    
    This subtype is determined based on whether child nodes were given
    to the constructor.
    
    Every :class:`TreeNode` has a reference to its parent node.
    A root node is node without a parent node, that is finalized
    using `as_root()`.
    The call of this function prevents that a the node can be used as
    child.

    :class:`TreeNode` objects are semi-immutable:
    The child nodes or the reference index are fixed at the time of
    creation.
    Only the parent can be set once, when the parent node is created. 
    :class:`TreeNode` objects that are finalized using `as_root()` are
    completely immutable.

    All object properties are read-only.

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
    Creating leaf nodes:
    
    >>> leaf1 = TreeNode(index=0)
    >>> leaf2 = TreeNode(index=1)
    >>> leaf3 = TreeNode(index=2)

    Creating intermediate nodes as parent of those leaf nodes:

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

        Create a deep copy of this :class:`TreeNode`.

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
        construction of a potential parent node, a :class:`TreeError` is
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
    
    def to_newick(self, labels=None, bint include_distance=True, 
                  round_distance=None):
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
        round_distance : int, optional
            If set, the distances are rounded to the given number of
            digits.
        
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
                if round_distance is None:
                    return f"{label}:{self._distance}"
                else:
                    return f"{label}:{self._distance:.{round_distance}f}"
            else:
                return f"{label}"
        else:
            # Build string in a recursive way
            child_strings = [child.to_newick(
                labels, include_distance, round_distance
            ) for child in self._children]
            if include_distance:
                if round_distance is None:
                    return f"({','.join(child_strings)}):{self._distance}"
                else:
                    return (
                        f"({','.join(child_strings)}):"
                        f"{self._distance:.{round_distance}f}"
                    )
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
        may use the same method in the :class:`Tree` class instead.
        Keep in mind that the :class:`TreeNode` class does support any
        labels on intermediate nodes.
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
                raise InvalidFileError("Bracket closed before it was opened")
        for i in reversed(range(len(newick))):
            char = newick[i]
            if char == ")":
                subnewick_stop_i = i+1
                break
            if char == "(":
                raise InvalidFileError("Bracket was opened but not closed")
        
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
            if len(subnewick) == 0:
                raise InvalidFileError(
                    "Intermediate node must at least have one child"
                )
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
                    raise InvalidFileError(
                        "Bracket closed before it was opened"
                    )
        
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
            if len(comma_pos) != 0:
                child, dist = TreeNode.from_newick(
                    subnewick[comma_pos[-1]+1:], labels=labels
                )
            else:
                # Single child node:
                child, dist = TreeNode.from_newick(
                    subnewick, labels=labels
                )
            children.append(child)
            distances.append(dist)
            return TreeNode(children, distances), distance

    def __str__(self):
        return self.to_newick()
    
    def __eq__(self, item):
        if not isinstance(item, TreeNode):
            return False
        cdef TreeNode node = item
        if self._distance != node._distance:
            return False
        if self._index !=-1:
            if self._index != node._index:
                return False
        else:
            if frozenset(self._children) != frozenset(node._children):
                return False
        return True
    
    def __hash__(self):
        # Order of children is not important -> set
        children_set = frozenset(self._children) \
                       if self._children is not None else None
        return hash((self._index, children_set, self._distance))


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



def as_binary(tree_or_node):
    """
    as_binary(tree_or_node)

    Convert a tree into a binary tree.

    In general a :class:`TreeNode` can have more or less than two
    children.
    However guide trees usually expect each intermediate node to have
    exactly two child nodes.
    This function creates a binary :class:`Tree` (or :class:`TreeNode`)
    for the given :class:`Tree` (or :class:`TreeNode`):
    Intermediate nodes that have only a single child are deleted and its
    parent node is directly connected to its child node.
    Intermediate nodes that have more than two childs are divided into
    multiple nodes (distances are preserved).
    
    Parameters
    ----------
    tree_or_node : Tree or TreeNode
        The tree or node to be converted into a binary tree or node.
    
    Returns
    -------
    binary_tree_or_node : Tree or TreeNode
        The converted tree or node.
    """
    if isinstance(tree_or_node, Tree):
        node, _ = _as_binary(tree_or_node.root)
        return Tree(node)
    elif isinstance(tree_or_node, TreeNode):
        node, _ = _as_binary(tree_or_node)
        return _as_binary(node)
    else:
        raise TypeError(
            f"Expected 'Tree' or 'TreeNode', not {type(tree_or_node).__name__}"
        )

cdef _as_binary(TreeNode node):
    """
    The actual logic wrapped by :func:`as_binary()`.
    
    Parameters
    ----------
    node : TreeNode
        The node to be converted.
    
    Returns
    -------
    binary_node: TreeNode
        The converted node.
    distance : float
        The distance of the converted node to its parent
    """
    cdef TreeNode child
    cdef TreeNode current_div_node
    cdef tuple children
    cdef list rem_children
    cdef list distances
    cdef float distance

    children = node.children
    if children is None:
        # Leaf node
        return TreeNode(index=node.index), node.distance
    elif len(children) == 1:
        # Intermediate node with one child
        # -> Omit node and directly connect its child to its parent
        # The distances are added
        #
        #      |--            |--   
        #      |              |   
        # --|--|--   ->   ----|--  
        #      |              |   
        #      |--            |-- 
        #
        child, distance = _as_binary(node.children[0])
        if node.is_root():
            # Child is new root -> No distance to parent
            return child, None
        else:
            return child, node.distance + distance
    elif len(children) > 2:
        # Intermediate node with more than two childs
        # -> Create a new node having two childs:
        #    - One of the childs of the original node
        #    - The original node with one child less (distance = 0)
        # Repeat until all children are put into binary nodes
        #
        #   |--          |--
        #   |          --|  |--
        # --|--   ->     |--|
        #   |               |--
        #   |--
        #
        # The remaining children
        rem_children, distances = [list(tup) for tup in zip(
            *[_as_binary(child) for child in children]
        )]
        current_div_node = None
        while len(rem_children) > 0:
            if current_div_node is None:
                # The bottom-most node is created
                #-> Gets two of the remaining childs
                current_div_node = TreeNode(
                    rem_children[:2],
                    distances[:2]
                )
                # Pop the two utilized remaining childs from the list
                rem_children.pop(0)
                rem_children.pop(0)
                distances.pop(0)
                distances.pop(0)
            else:
                # A node is created that gets one remaining child
                # and the intermediate node from the last step
                current_div_node = TreeNode(
                    (current_div_node, rem_children[0]),
                    (0, distances[0]) 
                )
                # Pop the utilized remaining child from the list
                rem_children.pop(0)
                distances.pop(0)
        return current_div_node, node.distance
    else:
        # Intermediate node with exactly two childs
        # -> Keep node unchanged
        binary_children, distances = [list(tup) for tup in zip(
            *[_as_binary(child) for child in children]
        )]
        return TreeNode(binary_children, distances), node.distance



class TreeError(Exception):
    """
    An exception that occurs in context of tree topology.
    """
    pass