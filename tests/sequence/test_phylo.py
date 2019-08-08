# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite
import biotite.sequence.phylo as phylo
from .util import data_dir


@pytest.fixture
def distances():
    # Distances are based on the example
    # "Dendrogram of the BLOSUM62 matrix"
    # with the small modification M[i,j] += i+j
    # to reduce ambiguity in the tree construction.
    return np.loadtxt(join(data_dir, "distances.txt"), dtype=int)


@pytest.fixture
def upgma_newick():
    # Newick notation of the tree created from 'distances.txt',
    # created via DendroUPGMA
    with open(join(data_dir, "newick_upgma.txt"), "r") as file:
        newick = file.read().strip()
    return newick


@pytest.fixture
def nj_newick():
    # Newick notation of the tree created from 'distances.txt',
    # created via Trex-online (http://trex.uqam.ca/)
    with open(join(data_dir, "newick.txt"), "r") as file:
        newick = file.read().strip()
    return newick


@pytest.fixture
def tree(distances):
    return phylo.upgma(distances)


@pytest.fixture
def nj_tree(distances):
    return phylo.neighbour_join(distances)


def test_upgma(tree, upgma_newick):
    """
    Compare the results of `upgma()` with DendroUPGMA.
    """
    ref_tree = phylo.Tree.from_newick(upgma_newick)
    assert _tree_equal(tree, ref_tree)


def test_neighbor_join(nj_tree, nj_newick):
    """
    Compare the results of `neighbor_join()` with Trex-online.
    """
    ref_tree = phylo.Tree.from_newick(nj_newick)
    assert _tree_equal(nj_tree, ref_tree)


def test_node_distance(tree):
    """
    Test whether the `distance_to()` and `lowest_common_ancestor()` work
    correctly.
    """
    # Tree is created via UPGMA
    # -> The distances to root should be equal for all leaf nodes
    dist = tree.root.distance_to(tree.leaves[0])
    for leaf in tree.leaves:
        assert leaf.distance_to(tree.root) == dist
    # Example topological distances
    assert tree.get_distance(0, 19, True) == 9
    assert tree.get_distance(4,  2, True) == 10
    
    # All pairwise leaf node distances should be sufficient
    # to reconstruct the same tree via UPGMA
    ref_dist_mat = np.zeros((len(tree), len(tree)))
    for i in range(len(tree)):
        for j in range(len(tree)):
            ref_dist_mat[i,j] = tree.get_distance(i,j)
    assert np.allclose(ref_dist_mat, ref_dist_mat.T)
    new_tree = phylo.upgma(ref_dist_mat)
    test_dist_mat = np.zeros((len(tree), len(tree)))
    for i in range(len(tree)):
        for j in range(len(tree)):
            test_dist_mat[i,j] = new_tree.get_distance(i,j)
    assert np.allclose(test_dist_mat, ref_dist_mat)


def test_leaf_list(tree):
    for i, leaf in enumerate(tree.leaves):
        assert i == leaf.index


def test_distances(tree):
    # Tree is created via UPGMA
    # -> The distances to root should be equal for all leaf nodes
    dist = tree.root.distance_to(tree.leaves[0])
    for leaf in tree.leaves:
        assert leaf.distance_to(tree.root) == dist
    # Example topological distances
    assert tree.get_distance(0, 19, True) == 9
    assert tree.get_distance(4,  2, True) == 10


def test_get_leaves(tree):
    # Manual example cases
    node = tree.leaves[6]
    assert set(tree.leaves[6].parent.get_indices()) == set(
        [6,11,2,3,13,8,14,5,0,15,16]
    )
    assert set(tree.leaves[10].get_indices()) == set([10])
    assert tree.root.get_leaf_count() == 20

 
def test_copy(tree):
    assert _tree_equal(tree, tree.copy())


def test_immutability():
    node = phylo.TreeNode(index=0)
    # Attributes are not writable
    with pytest.raises(AttributeError):
        node.children = None
    with pytest.raises(AttributeError):
        node.parent = None
    with pytest.raises(AttributeError):
        node.index = None
    # A root node cannot be child
    node1 = phylo.TreeNode(index=0)
    node2 = phylo.TreeNode(index=1)
    node1.as_root()
    with pytest.raises(phylo.TreeError):
        phylo.TreeNode([node1, node2], [0, 0])
    # A child node cannot be root
    node1 = phylo.TreeNode(index=0)
    node2 = phylo.TreeNode(index=1)
    phylo.TreeNode([node1, node2], [0, 0])
    with pytest.raises(phylo.TreeError):
        node1.as_root()
    # A node cannot be child of a two nodes
    node1 = phylo.TreeNode(index=0)
    node2 = phylo.TreeNode(index=1)
    phylo.TreeNode([node1, node2], [0, 0])
    with pytest.raises(phylo.TreeError):
        phylo.TreeNode([node1, node2], [0, 0])
    # Tree cannot be constructed from child nodes
    node1 = phylo.TreeNode(index=0)
    node2 = phylo.TreeNode(index=0)
    # node1 and node2 have now a parent
    phylo.TreeNode([node1, node2], [0, 0])
    with pytest.raises(phylo.TreeError):
        phylo.Tree(node1)


@pytest.mark.parametrize("newick, labels, error", [
    # Reference index out of range
    ("((1,0),4),2);", None, biotite.InvalidFileError),
    # Empty string
    ("", None, biotite.InvalidFileError),
    # Empty node
    ("();", None, biotite.InvalidFileError),
    # Missing brackets
    ("((0,1,(2,3));", None, biotite.InvalidFileError),
    # A node with three leaves
    ("((0,1),(2,3),(4,5));", None, None),
    # A node with one leaf
    ("((0,1),(2,3),(4));", None, None),
    # Named intermediate nodes
    ("((0,1,3)A,2)B;", None, None),
    # Named intermediate nodes and distances
    ("((0:1.0,1:3.0,3:5.0)A:2.0,2:5.0)B;", None, None),
    # Nodes with labels
    ("((((A,B),(C,D)),E),F);", ["A","B","C","D","E","F"], None),
    # Nodes with labels and distances
    ("((((A:1,B:2),(C:3,D:4)),E:5),F:6);", ["A","B","C","D","E","F"], None),
    # Newick with spaces
    (" ( 0 : 1.0 , 1 : 3.0 ) A ; ", None, None),
])
def test_newick_simple(newick, labels, error):
    # Read, write and read again a Newick notation and expect
    # the same reult from both reads
    if error is None:
        tree1 = phylo.Tree.from_newick(newick, labels)
        newick = tree1.to_newick(labels, include_distance=True)
        tree2 = phylo.Tree.from_newick(newick, labels)
        assert _tree_equal(tree1, tree2)
    else:
         with pytest.raises(error):
             tree1 = phylo.Tree.from_newick(newick, labels)


@pytest.mark.parametrize("use_labels", [False, True])
def test_newick_complex(upgma_newick, use_labels):
    # Same as above with more complex string
    if use_labels:
        labels = [str(i) for i in range(20)]
    else:
        labels = None
    tree1 = phylo.Tree.from_newick(upgma_newick, labels)
    newick = tree1.to_newick(labels, include_distance=True)
    tree2 = phylo.Tree.from_newick(newick, labels)
    assert _tree_equal(tree1, tree2)


@pytest.mark.parametrize("newick_in, exp_newick_out", [
    ("(0:1.0, 1:2.0);",                     "(0:1.0,1:2.0):0.0;"             ),
    ("(0:1.0, 1:2.0, 2:3.0);",              "((0:1.0,1:2.0):0.0,2:3.0):0.0;" ),
    ("(((0:1.0, 1:2.0):10.0):5.0, 2:8.0);", "((0:1.0,1:2.0):15.0,2:8.0):0.0;"),
    ("((0:1.0, 1:2.0):10.0):5.0;",          "(0:1.0,1:2.0):0.0;"             ),
])
def test_as_binary_cases(newick_in, exp_newick_out):
    """
    Test the `as_binary()` function based on known cases.
    """
    tree = phylo.Tree.from_newick(newick_in)
    bin_tree = phylo.as_binary(tree)
    assert bin_tree.to_newick() == exp_newick_out


def test_as_binary_distances():
    """
    Test the preservation of all pairwise leaf distances after calling
    `as_binary()`.
    """
    # Some random newick
    newick = "((((0:5, 1:1, 2:13, 5:9):4, (4:2, 6:9):7):18), 3:12);"
    tree = phylo.Tree.from_newick(newick)
    ref_dist_mat = np.zeros((len(tree), len(tree)))
    for i in range(len(tree)):
        for j in range(len(tree)):
            ref_dist_mat[i,j] = tree.get_distance(i,j)
    
    bin_tree = phylo.as_binary(tree)
    test_dist_mat = np.zeros((len(tree), len(tree)))
    for i in range(len(tree)):
        for j in range(len(tree)):
            test_dist_mat[i,j] = bin_tree.get_distance(i,j)
    assert np.allclose(test_dist_mat, ref_dist_mat)


def _tree_equal(t1, t2):
    # The topological and actual distances between all nodes should
    # be identical for both trees
    shape = (len(t1.leaves),) * 2
    dist = np.zeros(shape)
    dist_top = np.zeros(shape)
    ref_dist = np.zeros(shape)
    ref_dist_top = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[0]):
            dist[i,j]         = t1.get_distance(i, j)
            dist_top[i,j]     = t1.get_distance(i, j, True)
            ref_dist[i,j]     = t2.get_distance(i, j)
            ref_dist_top[i,j] = t2.get_distance(i, j, True)
    return np.allclose(dist, ref_dist, rtol=1e-3) & \
           np.array_equal(dist_top, ref_dist_top)


def _show_tree(tree):
    import biotite.sequence.graphics as graphics
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    graphics.plot_dendrogram(ax, tree)
    plt.show()