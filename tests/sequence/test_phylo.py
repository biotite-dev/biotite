# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite
import biotite.sequence.phylo as phylo
from ..util import data_dir


@pytest.fixture
def distances():
    # Distances are based on the example
    # "Dendrogram of the BLOSUM62 matrix"
    # with the small modification M[i,j] += i+j
    # to reduce ambiguity in the tree construction.
    return np.loadtxt(join(data_dir("sequence"), "distances.txt"), dtype=int)


@pytest.fixture
def upgma_newick():
    # Newick notation of the tree created from 'distances.txt',
    # created via DendroUPGMA
    with open(join(data_dir("sequence"), "newick_upgma.txt"), "r") as file:
        newick = file.read().strip()
    return newick


@pytest.fixture
def tree(distances):
    return phylo.upgma(distances)


def test_upgma(tree, upgma_newick):
    """
    Compare the results of `upgma()` with DendroUPGMA.
    """
    ref_tree = phylo.Tree.from_newick(upgma_newick)
    # Cannot apply direct tree equality assertion because the distance
    # might not be exactly equal due to floating point rounding errors
    for i in range(len(tree)):
        for j in range(len(tree)):
            # Check for equal distances and equal topologies
            assert                   tree.get_distance(i,j) \
                == pytest.approx(ref_tree.get_distance(i,j), abs=1e-3)
            assert     tree.get_distance(i,j, topological=True) \
                == ref_tree.get_distance(i,j, topological=True)


def test_neighbor_joining():
    """
    Compare the results of `neighbor_join()` with a known tree.
    """
    dist = np.array([
        [ 0,  5,  4,  7,  6,  8],
        [ 5,  0,  7, 10,  9, 11],
        [ 4,  7,  0,  7,  6,  8],
        [ 7, 10,  7,  0,  5,  9],
        [ 6,  9,  6,  5,  0,  8],
        [ 8, 11,  8,  9,  8,  0],
    ])

    ref_tree = phylo.Tree(phylo.TreeNode(
        [
            phylo.TreeNode(
                [
                    phylo.TreeNode(
                        [
                            phylo.TreeNode(index=0),
                            phylo.TreeNode(index=1),
                        ],
                        [1,4]
                    ),
                    phylo.TreeNode(index=2),
                ],
                [1, 2]
            ),
            phylo.TreeNode(
                [
                    phylo.TreeNode(index=3),
                    phylo.TreeNode(index=4),
                ],
                [3,2]
            ),
            phylo.TreeNode(index=5),
        ],
        [1,1,5]
    ))

    test_tree = phylo.neighbor_joining(dist)

    assert test_tree == ref_tree


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
    assert tree is not tree.copy()
    assert tree == tree.copy()


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
        assert tree1 == tree2
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
    assert tree1 == tree2


def test_newick_rounding():
    # Create the distance matrix
    distances = np.array(
        [[0.  , 0.53, 0.93, 0.78, 0.38, 0.99, 1.02, 0.76],
         [0.53, 0.  , 0.59, 0.41, 0.35, 0.87, 1.03, 0.83],
         [0.93, 0.59, 0.  , 0.16, 0.58, 0.55, 1.59, 1.19],
         [0.78, 0.41, 0.16, 0.  , 0.42, 0.69, 1.4 , 1.18],
         [0.38, 0.35, 0.58, 0.42, 0.  , 1.02, 1.11, 0.89],
         [0.99, 0.87, 0.55, 0.69, 1.02, 0.  , 1.47, 1.26],
         [1.02, 1.03, 1.59, 1.4 , 1.11, 1.47, 0.  , 1.39],
         [0.76, 0.83, 1.19, 1.18, 0.89, 1.26, 1.39, 0.  ]]
    )
    # Create the tree
    tree = phylo.neighbor_joining(distances)

    # Check if rounding omission of rounding works
    assert (
        tree.to_newick(include_distance=True, round_distance=2)
        == "(6:0.82,(((5:0.42,(3:0.03,2:0.13):0.12):0.24,1:0.14):0.04,4:0.17):"
        "0.09,(7:0.57,0:0.19):0.01):0.00;"
    )
    assert (
        tree.to_newick(include_distance=True) == "(6:0.8162499666213989,(((5:0"
        ".4175001084804535,(3:0.0341666080057621,2:0.1258333921432495):0.12249"
        "992787837982):0.23843751847743988,1:0.13656245172023773):0.0397916249"
        "9308586,4:0.1727083921432495):0.08937492966651917,(7:0.57375001907348"
        "63,0:0.1862499713897705):0.008749991655349731):0.0;"
    )


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


def test_equality(tree):
    """
    Assert that equal trees equal each other, and non-equal trees do not
    equal each other.
    """
    assert tree == tree.copy()
    # Order of children is not important
    assert tree == phylo.Tree(phylo.TreeNode(
        [tree.root.children[1].copy(), tree.root.children[0].copy()],
        [tree.root.children[1].distance, tree.root.children[0].distance]
    ))
    # Different distance -> Unequal tree
    assert tree != phylo.Tree(phylo.TreeNode(
        [tree.root.children[0].copy(), tree.root.children[1].copy()],
        [tree.root.children[0].distance, 42]
    ))
    # Additional node -> Unequal tree
    assert tree != phylo.Tree(phylo.TreeNode(
        [
            tree.root.children[0].copy(),
            tree.root.children[1].copy(),
            phylo.TreeNode(index=len(tree))
        ],
        [
            tree.root.children[0].distance,
            tree.root.children[1].distance,
            42
        ]
    ))

