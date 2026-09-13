# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.phylo"
__author__ = "Patrick Kunzmann, Tom David Müller"
__all__ = [
    "get_root",
    "get_leaves",
    "get_distance",
    "get_leaf_distances",
    "to_newick",
    "from_newick",
    "as_binary",
]

from collections.abc import Sequence
from typing import Any
import networkx as nx
import numpy as np
from biotite.file import InvalidFileError
from biotite.typing import NDArray1, NDArray2

# Characters that are part of the Newick syntax and hence must not appear in labels
_NEWICK_SYNTAX_CHARS = "(),:;"


def get_root(tree: nx.DiGraph) -> Any:
    """
    Get the root node of a tree.

    Parameters
    ----------
    tree : DiGraph
        The tree.

    Returns
    -------
    root : hashable
        The root node, i.e. the only node without a parent.

    Raises
    ------
    ValueError
        If the tree has not exactly one node without a parent.

    Examples
    --------

    >>> tree = from_newick("((0,1),2);")
    >>> print(get_root(tree))
    4
    """
    roots = [node for node, degree in tree.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root node, but found {len(roots)}")
    return roots[0]


def get_leaves(tree: nx.DiGraph, node: Any | None = None) -> list[Any]:
    """
    Get the leaf nodes that are direct or indirect children of the
    given node.

    Parameters
    ----------
    tree : DiGraph
        The tree.
    node : hashable, optional
        The node whose descendant leaves are returned.
        If the node is a leaf itself, the list contains only this node.
        By default, the root is used, i.e. all leaves are returned.

    Returns
    -------
    leaves : list of hashable
        The leaf nodes in depth-first order, i.e. leaves from the first
        child are listed before leaves from the second child etc.

    Examples
    --------

    >>> tree = from_newick("((0,1),(2,3));")
    >>> print(get_leaves(tree))
    [0, 1, 2, 3]
    >>> intermediate = next(tree.predecessors(2))
    >>> print(get_leaves(tree, intermediate))
    [2, 3]
    """
    if node is None:
        node = get_root(tree)
    # A depth-first traversal visits the children in insertion order,
    # so the leaves are returned in the left-to-right order of the tree
    # (e.g. as they appear in the Newick notation),
    # in contrast to the unordered set from `nx.descendants()`
    return [n for n in nx.dfs_preorder_nodes(tree, node) if tree.out_degree(n) == 0]


def get_distance(
    tree: nx.DiGraph, node1: Any, node2: Any, topological: bool = False
) -> float:
    """
    Get the distance between two nodes in a tree.

    The distance is the sum of all edge distances from each of the two
    nodes to their lowest common ancestor.

    Parameters
    ----------
    tree : DiGraph
        The tree.
    node1, node2 : hashable
        The two nodes to calculate the distance for.
    topological : bool, optional
        If true, the topological distance is measured, i.e. each edge
        has a distance of 1.
        Otherwise, the ``"distance"`` attribute of the edges is used.

    Returns
    -------
    distance : float
        The distance between the nodes.

    Raises
    ------
    ValueError
        If the nodes have no common ancestor,
        i.e. they are not part of the same tree.

    Examples
    --------

    >>> tree = from_newick("((0:5.0,1:7.0):3.0,2:10.0);")
    >>> print(get_distance(tree, 0, 1))
    12.0
    >>> print(get_distance(tree, 0, 2))
    18.0
    >>> print(get_distance(tree, 0, 2, topological=True))
    3.0
    """
    # Cumulative distance from `node1` to each of its ancestors
    ancestor_distances = {node1: 0.0}
    node = node1
    distance = 0.0
    while True:
        parent, edge_distance = _parent(tree, node, topological)
        if parent is None:
            break
        distance += edge_distance
        ancestor_distances[parent] = distance
        node = parent
    # Walk up from `node2` until the path of `node1` is hit
    node = node2
    distance = 0.0
    while node not in ancestor_distances:
        parent, edge_distance = _parent(tree, node, topological)
        if parent is None:
            raise ValueError("The nodes do not have a common ancestor")
        distance += edge_distance
        node = parent
    return distance + ancestor_distances[node]


def get_leaf_distances(
    tree: nx.DiGraph, topological: bool = False
) -> NDArray2[int, int, np.float64]:
    """
    Get the pairwise distances between all leaf nodes of a tree.

    Parameters
    ----------
    tree : DiGraph
        The tree.
        The leaf nodes must be the integers ``0`` to ``n-1``.
    topological : bool, optional
        If true, the topological distance is measured, i.e. each edge
        has a distance of 1.
        Otherwise, the ``"distance"`` attribute of the edges is used.

    Returns
    -------
    distances : ndarray, shape=(n,n), dtype=float
        The distance between each pair of leaf nodes, indexed by the
        leaf node.

    See Also
    --------
    get_distance : Distance between two arbitrary nodes.

    Examples
    --------

    >>> tree = from_newick("((0:5.0,1:7.0):3.0,2:10.0);")
    >>> print(get_leaf_distances(tree))
    [[ 0. 12. 18.]
     [12.  0. 20.]
     [18. 20.  0.]]
    """
    root = get_root(tree)
    n_leaves = _get_leaf_count(tree, root)
    distances = np.zeros((n_leaves, n_leaves), dtype=np.float64)
    # For each processed node the leaves in its subtree
    # and their distances to this node
    subtrees: dict[Any, tuple[NDArray1, NDArray1]] = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        if len(children) == 0:
            subtrees[node] = (np.array([node]), np.zeros(1))
            continue
        leaves_per_child = []
        distances_per_child = []
        for child in children:
            leaves, leaf_distances = subtrees.pop(child)
            edge_distance = _edge_distance(tree, node, child, topological)
            leaves_per_child.append(leaves)
            distances_per_child.append(leaf_distances + edge_distance)
        # The distance between leaves in different subtrees is the sum of
        # their distances to this node, which is their lowest common ancestor
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pair_distances = (
                    distances_per_child[i][:, np.newaxis]
                    + distances_per_child[j][np.newaxis, :]
                )
                distances[np.ix_(leaves_per_child[i], leaves_per_child[j])] = (
                    pair_distances
                )
                distances[np.ix_(leaves_per_child[j], leaves_per_child[i])] = (
                    pair_distances.T
                )
        subtrees[node] = (
            np.concatenate(leaves_per_child),
            np.concatenate(distances_per_child),
        )
    return distances


def to_newick(
    tree: nx.DiGraph,
    labels: Sequence[str] | None = None,
    include_distance: bool = True,
    round_distance: int | None = None,
) -> str:
    """
    Obtain the Newick notation of a tree.

    Parameters
    ----------
    tree : DiGraph
        The tree.
    labels : sequence of str, optional
        The labels the leaf nodes refer to.
        By default, the leaf nodes themselves are used as labels.
    include_distance : bool, optional
        If true, the ``"distance"`` attribute of the edges is written
        into the Newick notation, otherwise it is omitted.
    round_distance : int, optional
        If set, the distances are rounded to the given number of
        digits.

    Returns
    -------
    newick : str
        The Newick notation of the tree.

    Examples
    --------

    >>> tree = from_newick("((0:5.0,1:7.0):3.0,2:10.0);")
    >>> print(to_newick(tree))
    ((0:5.0,1:7.0):3.0,2:10.0):0.0;
    >>> print(to_newick(tree, include_distance=False))
    ((0,1),2);
    >>> labels = ["foo", "bar", "foobar"]
    >>> print(to_newick(tree, labels=labels, include_distance=False))
    ((foo,bar),foobar);
    """
    root = get_root(tree)
    # Build the strings of the subtrees bottom-up
    subtree_strings: dict[Any, str] = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        if len(children) == 0:
            if labels is None:
                string = str(node)
            else:
                string = str(labels[node])
                for char in _NEWICK_SYNTAX_CHARS:
                    if char in string:
                        raise ValueError(
                            f"Label '{string}' contains illegal character '{char}'"
                        )
        else:
            string = "(" + ",".join(subtree_strings.pop(child) for child in children)
            string += ")"
        if include_distance:
            parent, distance = _parent(tree, node, topological=False)
            if parent is None:
                distance = 0.0
            if round_distance is None:
                string += f":{distance}"
            else:
                string += f":{distance:.{round_distance}f}"
        subtree_strings[node] = string
    return subtree_strings[root] + ";"


def from_newick(newick: str, labels: Sequence[str] | None = None) -> nx.DiGraph:
    """
    Create a tree from a Newick notation.

    Parameters
    ----------
    newick : str
        The Newick notation to create the tree from.
    labels : sequence of str, optional
        If the Newick notation contains leaf labels that are not
        parseable into integers, this parameter can be provided to
        convert these labels into leaf nodes.
        The corresponding leaf node is the position of the label in the
        given sequence.

    Returns
    -------
    tree : DiGraph
        The tree created from the Newick notation.
        The leaf nodes are the integers ``0`` to ``n-1``, the
        intermediate nodes continue the numbering in the order they
        are closed in the Newick notation.
        Hence, the root is the node with the highest number.
        Each edge points from the parent to the child node and has a
        ``"distance"`` attribute, that is 0 if the Newick notation
        does not provide a distance.

    Raises
    ------
    InvalidFileError
        If the Newick notation is malformed or the leaf nodes are not
        the integers ``0`` to ``n-1``.

    Notes
    -----
    This function does accept but does not require the Newick string
    to have the terminal semicolon.

    Labels of intermediate nodes are discarded.

    Examples
    --------

    >>> tree = from_newick("((0:5.0,1:7.0):3.0,2:10.0);")
    >>> print(tree.nodes)
    [0, 1, 2, 3, 4]
    >>> for parent, child, distance in tree.edges(data="distance"):
    ...     print(parent, child, distance)
    3 0 5.0
    3 1 7.0
    4 3 3.0
    4 2 10.0
    >>> tree = from_newick("((A,B),C);", labels=["A", "B", "C"])
    >>> print(tree.nodes)
    [0, 1, 2, 3, 4]
    """
    # Ignore any whitespace
    newick = "".join(newick.split())
    if newick.endswith(";"):
        newick = newick[:-1]
    if len(newick) == 0:
        raise InvalidFileError("Newick string is empty")

    # The children of the intermediate nodes that are not closed yet,
    # as (node, distance) tuples
    open_nodes: list[list[tuple[int, float]]] = []
    # Edges as (parent, child, distance) tuples,
    # where intermediate nodes have temporary negative numbers
    edges: list[tuple[int, int, float]] = []
    leaves: list[int] = []
    n_intermediate = 0
    root: int | None = None
    # Whether the next token must be a node, i.e. a leaf or an opening
    # bracket, as it is the case at the start and after '(' and ','
    node_expected = True
    i = 0
    while i < len(newick):
        match newick[i]:
            case "(":
                if not node_expected:
                    raise InvalidFileError(f"Missing ',' before '(' at position {i}")
                open_nodes.append([])
                i += 1
            case ")":
                if node_expected:
                    raise InvalidFileError(f"Empty node before ')' at position {i}")
                if len(open_nodes) == 0:
                    raise InvalidFileError("Bracket closed before it was opened")
                children = open_nodes.pop()
                # Labels of intermediate nodes are discarded
                _, distance, i = _parse_label_and_distance(newick, i + 1)
                node = -1 - n_intermediate
                n_intermediate += 1
                for child, child_distance in children:
                    edges.append((node, child, child_distance))
                root = _attach_node(open_nodes, root, node, distance)
                node_expected = False
            case ",":
                if node_expected:
                    raise InvalidFileError(f"Empty node before ',' at position {i}")
                node_expected = True
                i += 1
            case _:
                # Any other character starts a leaf label
                # A label directly after ')' was already consumed there,
                # so this token is a node if and only if one is expected
                if not node_expected:
                    raise InvalidFileError(f"Unexpected label at position {i}")
                label, distance, i = _parse_label_and_distance(newick, i)
                node = _leaf_from_label(label, labels)
                leaves.append(node)
                root = _attach_node(open_nodes, root, node, distance)
                node_expected = False
    if len(open_nodes) != 0:
        raise InvalidFileError("Bracket was opened but not closed")
    if root is None:
        raise InvalidFileError("Newick notation contains no node")

    n_leaves = len(leaves)
    if sorted(leaves) != list(range(n_leaves)):
        raise InvalidFileError(
            f"The leaf nodes must be the integers 0 to {n_leaves - 1} "
            "without duplicates"
        )
    # Replace the temporary numbers of the intermediate nodes with
    # numbers that continue the leaf node numbering
    tree = nx.DiGraph()
    tree.add_nodes_from(range(n_leaves))
    tree.add_weighted_edges_from(
        (
            (_final_node(parent, n_leaves), _final_node(child, n_leaves), distance)
            for parent, child, distance in edges
        ),
        weight="distance",
    )
    return tree


def as_binary(tree: nx.DiGraph) -> nx.DiGraph:
    """
    Convert a tree into a binary tree.

    In general a node in a tree can have more or less than two children.
    However, guide trees usually expect each intermediate node to have
    exactly two children.
    This function creates a binary tree for the given tree:
    Intermediate nodes that have only a single child are deleted and its
    parent node is directly connected to its child node.
    Intermediate nodes that have more than two children are divided into
    multiple nodes (distances are preserved).

    Parameters
    ----------
    tree : DiGraph
        The tree to be converted into a binary tree.
        The leaf nodes must be the integers ``0`` to ``n-1``.

    Returns
    -------
    binary_tree : DiGraph
        The converted tree.
        The leaf nodes are the same as in the input tree.
        The intermediate nodes are renumbered from ``n`` (the number of
        leaves) upwards in the order they are visited in post-order
        traversal, i.e. the root is the node with the highest number.

    Examples
    --------

    >>> tree = from_newick("(0:1.0,1:2.0,2:3.0);")
    >>> print(to_newick(as_binary(tree)))
    ((0:1.0,1:2.0):0.0,2:3.0):0.0;
    >>> tree = from_newick("((0:1.0,1:2.0):5.0):3.0;")
    >>> print(to_newick(as_binary(tree)))
    (0:1.0,1:2.0):0.0;
    """
    root = get_root(tree)
    next_node = _get_leaf_count(tree, root)
    edges: list[tuple[int, Any, float]] = []
    # For each processed node the node that replaces it in the binary tree
    # and the distance of omitted single-child nodes to be added to the
    # distance to its parent
    converted: dict[Any, tuple[Any, float]] = {}
    for node in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(node))
        if len(children) == 0:
            # Leaf node
            converted[node] = (node, 0.0)
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
            child, extra_distance = converted.pop(children[0])
            edge_distance = _edge_distance(tree, node, children[0], False)
            converted[node] = (child, edge_distance + extra_distance)
        else:
            # Intermediate node with more than two children
            # -> Create a new node having two children:
            #    - One of the children of the original node
            #    - The original node with one child less (distance = 0)
            # Repeat until all children are put into binary nodes
            #
            #   |--          |--
            #   |          --|  |--
            # --|--   ->     |--|
            #   |               |--
            #   |--
            #
            # Intermediate nodes with exactly two children are also
            # handled by this branch, the loop below is not executed then
            child_nodes = []
            child_distances = []
            for child in children:
                child_node, extra_distance = converted.pop(child)
                child_nodes.append(child_node)
                child_distances.append(
                    _edge_distance(tree, node, child, False) + extra_distance
                )
            current_node = next_node
            next_node += 1
            edges.append((current_node, child_nodes[0], child_distances[0]))
            edges.append((current_node, child_nodes[1], child_distances[1]))
            for child_node, child_distance in zip(child_nodes[2:], child_distances[2:]):
                parent_node = next_node
                next_node += 1
                edges.append((parent_node, current_node, 0.0))
                edges.append((parent_node, child_node, child_distance))
                current_node = parent_node
            converted[node] = (current_node, 0.0)

    binary_tree = nx.DiGraph()
    binary_root, _ = converted[root]
    # In the case of a single-node tree, no edge adds the root
    binary_tree.add_node(binary_root)
    binary_tree.add_weighted_edges_from(edges, weight="distance")
    return binary_tree


def _parent(tree: nx.DiGraph, node: Any, topological: bool) -> tuple[Any | None, float]:
    """
    Get the parent of a node and the distance to it.

    The parent is ``None`` for the root node.
    """
    predecessors = tree.pred[node]
    if len(predecessors) == 0:
        return None, 0.0
    parent = next(iter(predecessors))
    return parent, 1.0 if topological else predecessors[parent]["distance"]


def _edge_distance(
    tree: nx.DiGraph, parent: Any, child: Any, topological: bool
) -> float:
    """
    Get the distance of the edge from `parent` to `child`.
    """
    return 1.0 if topological else tree.edges[parent, child]["distance"]


def _get_leaf_count(tree: nx.DiGraph, root: Any) -> int:
    """
    Get the number of leaves below `root` and check that these are the
    integers ``0`` to ``n-1``.
    """
    leaves = get_leaves(tree, root)
    if sorted(leaves) != list(range(len(leaves))):
        raise ValueError(f"The leaf nodes must be the integers 0 to {len(leaves) - 1}")
    return len(leaves)


def _parse_label_and_distance(newick: str, start: int) -> tuple[str, float, int]:
    """
    Parse the label and the optional distance (separated by ``:``)
    beginning at `start`, up to the next syntax character.

    Returns the label, the distance (0 if absent) and the position after
    the parsed part.
    """
    stop = start
    while stop < len(newick) and newick[stop] not in _NEWICK_SYNTAX_CHARS:
        stop += 1
    label = newick[start:stop]
    distance = 0.0
    if stop < len(newick) and newick[stop] == ":":
        start = stop + 1
        stop = start
        while stop < len(newick) and newick[stop] not in _NEWICK_SYNTAX_CHARS:
            stop += 1
        try:
            distance = float(newick[start:stop])
        except ValueError:
            raise InvalidFileError(f"Invalid distance '{newick[start:stop]}'")
    return label, distance, stop


def _leaf_from_label(label: str, labels: Sequence[str] | None) -> int:
    """
    Convert a leaf label into the leaf node number.
    """
    if labels is None:
        try:
            return int(label)
        except ValueError:
            raise InvalidFileError(
                f"Leaf label '{label}' is not an integer, "
                "provide 'labels' to convert it"
            )
    else:
        try:
            return labels.index(label)
        except ValueError:
            raise ValueError(f"Leaf label '{label}' is not in 'labels'")


def _attach_node(
    open_nodes: list[list[tuple[int, float]]],
    root: int | None,
    node: int,
    distance: float,
) -> int | None:
    """
    Add a parsed node as child to the innermost open intermediate node
    or make it the root, if no intermediate node is open.

    Returns the (potentially updated) root.
    """
    if len(open_nodes) != 0:
        open_nodes[-1].append((node, distance))
        return root
    if root is not None:
        raise InvalidFileError("Newick notation contains more than one root node")
    return node


def _final_node(node: int, n_leaves: int) -> int:
    """
    Convert the temporary negative number of an intermediate node into
    its final number, that continues the leaf numbering.
    """
    return node if node >= 0 else n_leaves - 1 - node
