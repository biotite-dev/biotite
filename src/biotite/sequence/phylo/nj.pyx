# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["neighbor_joining"]

cimport cython
cimport numpy as np

from .tree import Tree, TreeNode
import numpy as np

ctypedef np.float32_t float32
ctypedef np.uint8_t uint8
ctypedef np.uint32_t uint32


cdef float32 MAX_FLOAT = np.finfo(np.float32).max


def neighbor_joining(np.ndarray distances):
    """
    neighbor_join(distances)
    
    Perform hierarchical clustering using the
    *neighbor joining* algorithm [1,2]_.

    In contrast to UPGMA this algorithm does not assume a constant
    evolution rate. The resulting tree is considered to be unrooted.

    Parameters
    ----------
    distances : ndarray, shape=(n,n)
        Pairwise distance matrix.

    Returns
    -------
    tree : Tree
        A rooted tree. The `index` attribute in the leaf
        `TreeNode` objects refer to the indices of `distances`.

    Raises
    ------
    ValueError
        If the distance matrix is not symmetric
        or if any matrix entry is below 0.
    
    Notes
    -----
    The created tree is binary except for the root node, that has three
    child notes
    
    References
    ----------
    
    .. [1] N Saitou, M Nei,
       "The neighbor-joining method: a new method for reconstructing
       phylogenetic trees."
       Mol Biol Evol, 4, 406-425 (1987).
    .. [2] JA Studier, KJ Keppler,
       "A note on the neighbor-joining algorithm of Saitou and Neil."
       Mol Biol Evol, 5, 729-731 (1988).

    Examples
    --------
    
    >>> distances = np.array([
    ...     [0, 1, 7, 7, 9],
    ...     [1, 0, 7, 6, 8],
    ...     [7, 7, 0, 2, 4],
    ...     [7, 6, 2, 0, 3],
    ...     [9, 8, 4, 3, 0],
    ... ])
    >>> tree = neighbor_join(distances)
    >>> print(tree.to_newick(include_distance=False))
    """
    cdef int i=0, j=0, k=0, u=0
    cdef int i_min=0, j_min=0
    cdef float32 dist, dist_min, dist_sum
    cdef float32 node_dist_i, node_dist_j
    

    if distances.shape[0] != distances.shape[1] \
        or not np.allclose(distances.T, distances):
            raise ValueError("Distance matrix must be symmetric")
    if distances.shape[0] < 4:
        raise ValueError("At least 4 nodes are required")
    if (distances < 0).any():
        raise ValueError("Distances must be positive")


    # Keep track on clustered indices
    cdef np.ndarray nodes = np.array(
        [TreeNode(index=i) for i in range(distances.shape[0])]
    )
    # Indicates whether an index in the distance matrix has already been
    # clustered and the repsective rows and columns can be ignored
    cdef uint8[:] is_clustered_v = np.full(
        distances.shape[0], False, dtype=np.uint8
    )
    cdef int n_rem_nodes = \
        len(distances) - np.count_nonzero(np.asarray(is_clustered_v))
    # The divergence of of a 'taxum'
    # describes the relative evolution rate
    cdef float32[:] divergence_v = np.zeros(
        distances.shape[0], dtype=np.float32
    )
    # Triangular matrix for storing the divergence corrected distances
    cdef float32[:,:] corr_distances_v = np.zeros(
        (distances.shape[0],) * 2, dtype=np.float32
    )
    cdef float32[:,:] distances_v = distances.astype(np.float32, copy=True)

    # Cluster indices

    # Exit loop via 'return'
    while True:

        # Calculate divergence
        for i in range(distances_v.shape[0]):
            if is_clustered_v[i]:
                continue
            dist_sum = 0
            for k in range(distances_v.shape[0]):
                if is_clustered_v[k]:
                    continue
                dist_sum += distances_v[i,k]
            divergence_v[i] = dist_sum
        
        # Calculate corrected distance matrix
        for i in range(distances_v.shape[0]):
            for j in range(i):
                corr_distances_v[i,j] = \
                    (n_rem_nodes - 2) * distances_v[i,j] \
                    - divergence_v[i] - divergence_v[j]

        # Find minimum corrected distance
        dist_min = MAX_FLOAT
        i_min = -1
        j_min = -1
        for i in range(corr_distances_v.shape[0]):
            if is_clustered_v[i]:
                    continue
            for j in range(i):
                if is_clustered_v[j]:
                    continue
                dist = corr_distances_v[i,j]
                if dist < dist_min:
                    dist_min = dist
                    i_min = i
                    j_min = j
        
        # Check if all nodes have been clustered
        if i_min == -1 or j_min == -1:
            # No distance found -> all leaf nodes are clustered
            # -> exit loop
            break
        
        # Cluster the nodes with minimum distance
        # replacing the node at position i_min
        # leaving the node at position j_min empty
        # (is_clustered_v -> True)
        node_dist_i = 0.5 * (
            distances_v[i_min,j_min]
            + 1/(n_rem_nodes-2) * (divergence_v[i_min] - divergence_v[j_min])
        )
        node_dist_j = 0.5 * (
            distances_v[i_min,j_min]
            + 1/(n_rem_nodes-2) * (divergence_v[j_min] - divergence_v[i_min])
        )
        nodes[i_min] = TreeNode(
            (nodes[i_min], nodes[j_min]),
            (node_dist_i, node_dist_j)
        )
        # Mark position j_min as clustered
        nodes[j_min] = None
        is_clustered_v[j_min] = True
        
        # Calculate distances of new node to all other nodes
        for k in range(distances_v.shape[0]):
            if not is_clustered_v[k] and k != i_min:
                dist = 0.5 * (
                    distances_v[i_min,k] + distances_v[j_min,k]
                    - distances_v[i_min,j_min]
                )
                distances_v[i_min,k] = dist
                distances_v[k,i_min] = dist

        # Update the amount of remaining nodes
        n_rem_nodes = \
            len(distances) - np.count_nonzero(np.asarray(is_clustered_v))
        
        # Check if all nodes have been clustered
        if n_rem_nodes <= 3:
            # The 'root' node has the 3 remianing nodes as child nodes
            rem_nodes = nodes[
                np.where(~np.asarray(is_clustered_v, dtype=bool))
            ]
            return Tree(TreeNode(rem_nodes, [0,0,0]))