# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import networkx as nx
import numpy as np
import pytest
import biotite
import biotite.sequence.phylo as phylo
from tests.util import data_dir


@pytest.fixture
def distances():
    # Distances are based on the example
    # "Dendrogram of the BLOSUM62 matrix"
    # with the small modification M[i,j] += i+j
    # to reduce ambiguity in the tree construction.
    return np.loadtxt(data_dir("sequence") / "distances.txt", dtype=int)


@pytest.fixture
def upgma_newick():
    # Newick notation of the tree created from 'distances.txt',
    # created via DendroUPGMA
    with open(data_dir("sequence") / "newick_upgma.txt", "r") as file:
        newick = file.read().strip()
    return newick


@pytest.fixture
def tree(distances):
    return phylo.upgma(distances)


def _random_tree(n_leaves, rng, ultrametric):
    """
    Create a random binary tree with random edge distances.

    If `ultrametric` is true, all leaves have the same distance to the
    root, as required for exact reconstruction via UPGMA.
    Otherwise the tree is merely additive, as required for exact
    reconstruction via neighbor joining.
    """
    tree = nx.DiGraph()
    tree.add_nodes_from(range(n_leaves))
    # Distance of each cluster from its leaves
    heights = {i: 0.0 for i in range(n_leaves)}
    clusters = list(range(n_leaves))
    next_node = n_leaves
    while len(clusters) > 1:
        i, j = rng.choice(len(clusters), size=2, replace=False)
        node_i, node_j = clusters[i], clusters[j]
        if ultrametric:
            height = max(heights[node_i], heights[node_j]) + rng.uniform(0.1, 1.0)
            dist_i = height - heights[node_i]
            dist_j = height - heights[node_j]
        else:
            height = 0.0
            dist_i, dist_j = rng.uniform(0.1, 1.0, size=2)
        tree.add_edge(next_node, node_i, distance=dist_i)
        tree.add_edge(next_node, node_j, distance=dist_j)
        heights[next_node] = height
        clusters = [c for c in clusters if c not in (node_i, node_j)] + [next_node]
        next_node += 1
    return tree


def test_upgma(tree, upgma_newick):
    """
    Compare the results of `upgma()` with DendroUPGMA.
    """
    ref_tree = phylo.from_newick(upgma_newick)
    # The topology and distances must be equal,
    # but distances might slightly differ due to floating point rounding
    assert phylo.get_leaf_distances(tree) == pytest.approx(
        phylo.get_leaf_distances(ref_tree), abs=1e-3
    )
    assert np.array_equal(
        phylo.get_leaf_distances(tree, topological=True),
        phylo.get_leaf_distances(ref_tree, topological=True),
    )


def test_neighbor_joining():
    """
    Compare the results of `neighbor_joining()` with a known tree.
    """
    dist = np.array([
        [ 0,  5,  4,  7,  6,  8],
        [ 5,  0,  7, 10,  9, 11],
        [ 4,  7,  0,  7,  6,  8],
        [ 7, 10,  7,  0,  5,  9],
        [ 6,  9,  6,  5,  0,  8],
        [ 8, 11,  8,  9,  8,  0],
    ])  # fmt: skip
    ref_tree = phylo.from_newick("(((0:1,1:4):1,2:2):1,(3:3,4:2):1,5:5);")

    test_tree = phylo.neighbor_joining(dist)

    # The leaf distances determine the unrooted tree,
    # the leaves below each child of the root determine its placement
    assert phylo.get_leaf_distances(test_tree) == pytest.approx(
        phylo.get_leaf_distances(ref_tree)
    )
    assert np.array_equal(
        phylo.get_leaf_distances(test_tree, topological=True),
        phylo.get_leaf_distances(ref_tree, topological=True),
    )
    root = phylo.get_root(test_tree)
    assert sorted(
        sorted(phylo.get_leaves(test_tree, child))
        for child in test_tree.successors(root)
    ) == [[0, 1, 2], [3, 4], [5]]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_leaves", [3, 4, 10, 50])
@pytest.mark.parametrize(
    "method", [phylo.upgma, phylo.neighbor_joining], ids=lambda x: x.__name__
)
def test_clustering_reconstruction(method, n_leaves, seed):
    """
    Check that clustering a distance matrix derived from a random tree
    reconstructs the leaf distances of that tree.
    This works exactly for UPGMA on ultrametric trees and for neighbor
    joining on additive trees.
    Furthermore, check the structural properties of the created tree.
    """
    rng = np.random.default_rng(seed)
    ref_tree = _random_tree(n_leaves, rng, ultrametric=method is phylo.upgma)
    ref_distances = phylo.get_leaf_distances(ref_tree)

    test_tree = method(ref_distances)

    # Check if the method actually creates a tree
    assert nx.is_arborescence(test_tree)
    assert sorted(phylo.get_leaves(test_tree)) == list(range(n_leaves))
    assert phylo.get_root(test_tree) == max(test_tree.nodes)
    child_counts = [test_tree.out_degree(node) for node in test_tree.nodes]
    if method is phylo.upgma:
        assert set(child_counts) == {0, 2}
    else:
        assert test_tree.out_degree(phylo.get_root(test_tree)) == 3
        assert set(child_counts) <= {0, 2, 3}
    assert phylo.get_leaf_distances(test_tree) == pytest.approx(ref_distances, rel=1e-4)


@pytest.mark.parametrize(
    "method", [phylo.upgma, phylo.neighbor_joining], ids=lambda x: x.__name__
)
@pytest.mark.parametrize(
    "distances",
    [
        np.zeros((5, 4)),
        np.arange(25).reshape(5, 5),
        np.full((5, 5), np.nan),
        np.full((5, 5), np.inf),
        -np.ones((5, 5)),
    ],
    ids=["not_square", "asymmetric", "nan", "inf", "negative"],
)
def test_invalid_distances(method, distances):
    """
    Check that invalid distance matrices raise an exception.
    """
    with pytest.raises(ValueError):
        method(distances)


def test_distances(tree):
    """
    Check that `get_distance()` is consistent with `get_leaf_distances()`,
    and that the distances of the UPGMA tree have the expected properties.
    """
    root = phylo.get_root(tree)
    leaf_distances = phylo.get_leaf_distances(tree)
    topological_distances = phylo.get_leaf_distances(tree, topological=True)
    for i in range(len(leaf_distances)):
        for j in range(len(leaf_distances)):
            assert leaf_distances[i, j] == pytest.approx(phylo.get_distance(tree, i, j))
            assert topological_distances[i, j] == phylo.get_distance(
                tree, i, j, topological=True
            )
    # Tree is created via UPGMA
    # -> The distances to root should be equal for all leaf nodes
    root_distances = [
        phylo.get_distance(tree, leaf, root) for leaf in phylo.get_leaves(tree)
    ]
    assert root_distances == pytest.approx([root_distances[0]] * len(root_distances))
    # Example topological distances
    assert phylo.get_distance(tree, 0, 19, topological=True) == 9
    assert phylo.get_distance(tree, 4, 2, topological=True) == 10

    # All pairwise leaf node distances should be sufficient
    # to reconstruct the same tree via UPGMA
    new_tree = phylo.upgma(leaf_distances)
    assert phylo.get_leaf_distances(new_tree) == pytest.approx(leaf_distances)


def test_get_leaves(tree):
    """
    Check `get_leaves()` on manual example cases.
    """
    assert sorted(phylo.get_leaves(tree)) == list(range(20))
    parent_of_6 = next(tree.predecessors(6))
    assert set(phylo.get_leaves(tree, parent_of_6)) == set(
        [6, 11, 2, 3, 13, 8, 14, 5, 0, 15, 16]
    )
    assert phylo.get_leaves(tree, 10) == [10]


@pytest.mark.parametrize(
    ["newick", "labels", "error"],
    [
        # Reference index out of range
        ("((1,0),4),2);", None, biotite.InvalidFileError),
        # Duplicate reference index
        ("((1,0),1);", None, biotite.InvalidFileError),
        # Empty string
        ("", None, biotite.InvalidFileError),
        # Empty node
        ("();", None, biotite.InvalidFileError),
        # Missing brackets
        ("((0,1,(2,3));", None, biotite.InvalidFileError),
        # Non-numeric distance
        ("(0:x,1);", None, biotite.InvalidFileError),
        # Non-integer label without labels given
        ("(A,B);", None, biotite.InvalidFileError),
        # Empty nodes
        ("(0,,1);", None, biotite.InvalidFileError),
        ("(0,1,);", None, biotite.InvalidFileError),
        # Missing separator
        ("(0(1,2));", None, biotite.InvalidFileError),
        ("(0,1)(2,3);", None, biotite.InvalidFileError),
        # A node with three leaves
        ("((0,1),(2,3),(4,5));", None, None),
        # A node with one leaf
        ("((0,1),(2,3),(4));", None, None),
        # Named intermediate nodes
        ("((0,1,3)A,2)B;", None, None),
        # Named intermediate nodes and distances
        ("((0:1.0,1:3.0,3:5.0)A:2.0,2:5.0)B;", None, None),
        # Nodes with labels
        ("((((A,B),(C,D)),E),F);", ["A", "B", "C", "D", "E", "F"], None),
        # Nodes with labels and distances
        ("((((A:1,B:2),(C:3,D:4)),E:5),F:6);", ["A", "B", "C", "D", "E", "F"], None),
        # Newick with spaces
        (" ( 0 : 1.0 , 1 : 3.0 ) A ; ", None, None),
        # Single leaf
        ("0;", None, None),
    ],
)
def test_newick_simple(newick, labels, error):
    """
    Read, write and read again a Newick notation and expect the same
    result from both reads.
    """
    if error is None:
        tree1 = phylo.from_newick(newick, labels)
        newick = phylo.to_newick(tree1, labels, include_distance=True)
        tree2 = phylo.from_newick(newick, labels)
        assert nx.utils.graphs_equal(tree1, tree2)
    else:
        with pytest.raises(error):
            phylo.from_newick(newick, labels)


@pytest.mark.parametrize("use_labels", [False, True])
def test_newick_complex(upgma_newick, use_labels):
    """
    Same as above with more complex string.
    """
    if use_labels:
        labels = [str(i) for i in range(20)]
    else:
        labels = None
    tree1 = phylo.from_newick(upgma_newick, labels)
    newick = phylo.to_newick(tree1, labels, include_distance=True)
    tree2 = phylo.from_newick(newick, labels)
    assert nx.utils.graphs_equal(tree1, tree2)


def test_newick_deep():
    """
    Check that a deeply nested Newick notation can be parsed and written
    without hitting the recursion limit.
    """
    n_leaves = 5000
    newick = "0"
    for i in range(1, n_leaves):
        newick = f"({newick},{i})"
    newick += ";"

    tree = phylo.from_newick(newick)

    assert phylo.get_leaves(tree) == list(range(n_leaves))
    assert phylo.to_newick(tree, include_distance=False) == newick
    assert phylo.get_distance(tree, 0, n_leaves - 1, topological=True) == n_leaves


def test_newick_rounding():
    """
    Check that distances are correctly rounded in the Newick notation.
    """
    distances = np.array(
        [
            [0.0, 0.53, 0.93, 0.78, 0.38, 0.99, 1.02, 0.76],
            [0.53, 0.0, 0.59, 0.41, 0.35, 0.87, 1.03, 0.83],
            [0.93, 0.59, 0.0, 0.16, 0.58, 0.55, 1.59, 1.19],
            [0.78, 0.41, 0.16, 0.0, 0.42, 0.69, 1.4, 1.18],
            [0.38, 0.35, 0.58, 0.42, 0.0, 1.02, 1.11, 0.89],
            [0.99, 0.87, 0.55, 0.69, 1.02, 0.0, 1.47, 1.26],
            [1.02, 1.03, 1.59, 1.4, 1.11, 1.47, 0.0, 1.39],
            [0.76, 0.83, 1.19, 1.18, 0.89, 1.26, 1.39, 0.0],
        ]
    )
    tree = phylo.neighbor_joining(distances)

    assert (
        phylo.to_newick(tree, include_distance=True, round_distance=2)
        == "((((5:0.42,(3:0.03,2:0.13):0.12):0.24,1:0.14):0.04,4:0.17):0.09,"
        "6:0.82,(7:0.57,0:0.19):0.01):0.00;"
    )
    assert (
        phylo.to_newick(tree, include_distance=True) == "((((5:0.4175000786781311,(3"
        ":0.0341666080057621,2:0.1258333921432495):0.12249992787837982):0.2384374"
        "588727951,1:0.1365625113248825):0.03979162126779556,4:0.1727083921432495"
        "):0.08937495946884155,6:0.8162499666213989,(7:0.5737498998641968,0:0.186"
        "25009059906006):0.008749991655349731):0.0;"
    )


def test_newick_illegal_label():
    """
    Check that labels containing Newick syntax characters are rejected.
    """
    tree = phylo.from_newick("(0,1);")
    with pytest.raises(ValueError):
        phylo.to_newick(tree, labels=["A", "B:C"])


@pytest.mark.parametrize(
    ["newick_in", "exp_newick_out"],
    [
        ("(0:1.0, 1:2.0);", "(0:1.0,1:2.0):0.0;"),
        ("(0:1.0, 1:2.0, 2:3.0);", "((0:1.0,1:2.0):0.0,2:3.0):0.0;"),
        ("(((0:1.0, 1:2.0):10.0):5.0, 2:8.0);", "((0:1.0,1:2.0):15.0,2:8.0):0.0;"),
        ("((0:1.0, 1:2.0):10.0):5.0;", "(0:1.0,1:2.0):0.0;"),
        ("0;", "0:0.0;"),
    ],
)
def test_as_binary_cases(newick_in, exp_newick_out):
    """
    Test the `as_binary()` function based on known cases.
    """
    tree = phylo.from_newick(newick_in)
    bin_tree = phylo.as_binary(tree)
    assert phylo.to_newick(bin_tree) == exp_newick_out


@pytest.mark.parametrize("seed", range(5))
def test_as_binary_distances(seed):
    """
    Test the preservation of all pairwise leaf distances after calling
    `as_binary()` on a random tree with arbitrary numbers of children.
    """
    rng = np.random.default_rng(seed)
    n_leaves = 30
    tree = nx.DiGraph()
    # Repeatedly merge a random number of clusters into a new node
    clusters = list(range(n_leaves))
    next_node = n_leaves
    while len(clusters) > 1:
        n_children = min(len(clusters), rng.integers(1, 5))
        children = rng.choice(clusters, size=n_children, replace=False)
        for child in children:
            tree.add_edge(next_node, int(child), distance=rng.uniform(0, 10))
        clusters = [c for c in clusters if c not in children] + [next_node]
        next_node += 1
    ref_distances = phylo.get_leaf_distances(tree)

    bin_tree = phylo.as_binary(tree)

    assert nx.is_arborescence(bin_tree)
    assert all(bin_tree.out_degree(node) in (0, 2) for node in bin_tree.nodes)
    assert sorted(phylo.get_leaves(bin_tree)) == list(range(n_leaves))
    assert phylo.get_leaf_distances(bin_tree) == pytest.approx(ref_distances)
