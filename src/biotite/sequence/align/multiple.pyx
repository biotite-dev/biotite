# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["align_multiple"]

cimport cython
cimport numpy as np
from libc.math cimport log

from .matrix import SubstitutionMatrix
from .alignment import Alignment
from .pairwise import align_optimal
from ..sequence import Sequence
from ..phylo.upgma import upgma
from ..phylo.tree import Tree, TreeNode
import numpy as np


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32

ctypedef fused CodeType:
    uint8
    uint16
    uint32
    uint64


def align_multiple(sequences, matrix, gap_penalty=-10, terminal_penalty=True):
    if not matrix.is_symmetric():
        raise ValueError("A symmetric substitution matrix is required")
    if not matrix.get_alphabet1().extends(sequences[0].get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    for seq in sequences:
        if seq.code is None:
            raise ValueError("sequence code must not be None")
        dtype = sequences[0].code.dtype
        if seq.code.dtype != dtype:
            raise ValueError(
                "The sequence codes must have the same data type "
                "for each sequence"
            )
    
    distances = _get_distance_matrix(
        sequences[0].code, sequences, matrix, gap_penalty, terminal_penalty
    )
    tree = upgma(distances)
    print(tree)


def _get_distance_matrix(CodeType[:] _T, sequences, matrix,
                         gap_penalty, terminal_penalty):
    cdef int i, j

    cdef np.ndarray scores = np.zeros(
        (len(sequences), len(sequences)), dtype=np.int32
    )
    cdef np.ndarray alignments = np.full(
        (len(sequences), len(sequences)), None, dtype=object
    )
    for i in range(len(sequences)):
        # Inclusive range
        for j in range(i+1):
            # For this method we only consider one alignment:
            # Score is equal for all alignments
            # Alignment length is equal for most alignments
            alignment = align_optimal(
                sequences[i], sequences[j], matrix,
                gap_penalty, terminal_penalty, max_number=1
            )[0]
            scores[i,j] = alignment.score
            alignments[i,j] = alignment
    print(scores)
    
    ### Distance calculation from similarity scores ###
    # Calculate the occurences of each symbol code in each sequence
    # This is used later for the random score
    # Both alphabets are the same
    cdef CodeType alphabet_size = len(matrix.get_alphabet1())
    cdef np.ndarray code_count = np.zeros(
        (len(sequences), alphabet_size), dtype=np.int32
    )
    cdef int32[:,:] code_count_v = code_count
    for i in range(len(sequences)):
        code_count[i] = np.bincount(sequences[i].code, minlength=alphabet_size)

    cdef int gap_open=0, gap_ext=0
    if type(gap_penalty) == int:
        gap_open = gap_penalty
        gap_ext = gap_penalty
    elif type(gap_penalty) == tuple:
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
    else:
        raise TypeError("Gap penalty must be either integer or tuple")

    cdef int32[:,:] score_matrix = matrix.score_matrix()
    cdef int32[:,:] scores_v = scores
    cdef np.ndarray distances = np.zeros(
        (scores.shape[0], scores.shape[1]), dtype=np.float32
    )
    cdef float32[:,:] distances_v = distances
    cdef CodeType[:] seq_code1, seq_code2
    cdef CodeType code1, code2
    cdef float32 score_rand, score_max
    
    # Calculate distance
    # i and j are indicating the alignment between the sequences i and j
    for i in range(scores_v.shape[0]):
        #for j in range(i):
        for j in range(i+1):
            score_max =  (scores_v[i,i] + scores_v[j,j]) / 2.0
            score_rand = 0
            for code1 in range(alphabet_size):
                for code2 in range(alphabet_size):
                    score_rand += score_matrix[code1,code2] \
                                  * code_count[i,code1] \
                                  * code_count[j,code2]
            score_rand /= alignments[i,j].trace.shape[0]
            print("Index", i,j)
            print("")
            print(alignments[i,j])
            print("")
            gap_open_count, gap_ext_count = _count_gaps(
                alignments[i,j].trace.astype(np.int64, copy=False),
                terminal_penalty
            )
            score_rand += gap_open_count * gap_open
            score_rand += gap_ext_count * gap_ext
            distances[i,j] = -log(
                (scores_v[i,j] - score_rand) / (score_max - score_rand)
            )
            # Pairwise distance matrix is symmetric
            distances[i,j] = distances[j,i]
    print(distances)
    return distances


def _count_gaps(int64[:,:] trace_v, bint terminal_penalty):
    cdef int i, j
    cdef int gap_open_count=0, gap_ext_count=0
    cdef int start_index=-1, stop_index=-1

    if not terminal_penalty:
        # Ignore terminal gaps
        # -> get start and exclusive stop column of the trace
        # excluding terminal gaps
        for i in range(trace_v.shape[0]):
            # Check if all sequences have no gap at the given position
            if trace_v[i,0] != -1 and trace_v[i,1] != -1:
                start_index = i
                break
        # Reverse iteration
        for i in range(trace_v.shape[0]-1, -1, -1):
            # Check if all sequences have no gap at the given position
             if trace_v[i,0] != -1 and trace_v[i,1] != -1:
                stop_index = i+1
                break
        if start_index == -1 or stop_index == -1:
            return 0, 0
        trace_v = trace_v[start_index : stop_index]
    
    if trace_v[0,0] == -1:
        gap_open_count += 1
    if trace_v[0,1] == -1:
        gap_open_count += 1
    for i in range(1, trace_v.shape[0]):
        # trace_v.shape[1] = 2 due to pairwise alignemt
        for j in range(trace_v.shape[1]):
            if trace_v[i,j] == -1:
                if trace_v[i-1,j] == -1:
                    gap_ext_count += 1
                else:
                    gap_open_count += 1
    print(gap_open_count, gap_ext_count)
    return gap_open_count, gap_ext_count

