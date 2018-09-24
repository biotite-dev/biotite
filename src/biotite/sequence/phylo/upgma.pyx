# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["upgma"]

cimport cython
cimport numpy as np

from .tree import Tree, TreeNode
import numpy as np

ctypedef np.float32_t float32
ctypedef np.uint8_t uint8


cdef float32 MAX_FLOAT = np.finfo(np.float32).max

def upgma(np.ndarray distances):
    cdef int i=0, j=0, k=0
    cdef int i_min=0, j_min=0
    cdef float32 dist, dist_min
    cdef float mean
    

    if distances.shape[0] != distances.shape[1] \
        or not np.array_equal(distances.T, distances):
            raise ValueError("Distance matrix must be symmetric")
    if (distances < 0).any():
        raise ValueError("Distances must be positive")


    # Keep track on clustered indices
    cdef np.ndarray nodes = np.array(
        [TreeNode(index=i) for i in range(distances.shape[0])]
    )
    cdef uint8[:] is_clustered_v = np.full(
        distances.shape[0], False, dtype=np.uint8
    )


    # Cluster indices
    cdef float32[:,:] distances_v = distances.astype(np.float32, copy=False)
    
    # Exit loop via 'break'
    while True:
        ###
        print(distances)
        print(", ".join([str(e) for e in nodes]))
        print(np.asarray(is_clustered_v))
        print("\n")
        ###

        # Find minimum distance
        dist_min = MAX_FLOAT
        i_min = -1
        j_min = -1
        for i in range(distances_v.shape[0]):
            if is_clustered_v[i]:
                    continue
            for j in range(i):
                if is_clustered_v[j]:
                    continue
                dist = distances_v[i,j]
                if dist < dist_min:
                    dist_min = dist
                    i_min = i
                    j_min = j
        
        if i_min == -1 or j_min == -1:
            # No distance found -> all terminal nodes are clustered
            # -> exit loop
            break
        
        # Cluster the nodes with minimum distance
        # replacing the node at position i_min
        # leaving the node at position j_min empty
        # (is_clustered_v -> True)
        nodes[i_min] = TreeNode(
            child1=nodes[i_min], child2=nodes[j_min],
            child1_distance=0, child2_distance=0
        )
        nodes[j_min] = None
        is_clustered_v[j_min] = True
        # Calculate arithmetic mean distances of child nodes
        # as distances for new node and update matrix
        for k in range(distances_v.shape[0]):
            mean = (distances_v[i_min,k] + distances_v[j_min,k]) / 2
            distances_v[i_min,k] = mean
            distances_v[k,i_min] = mean
    

    # As each higher level node is always created on position i_min
    # and i is always higher than j in minimum distance calculation,
    # the root node must be at the last index
    return Tree(nodes[-1])