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
from ..alphabet import Alphabet
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


cdef float32 MAX_FLOAT = np.finfo(np.float32).max


class GapSymbol:
    
    _instance = None
    
    def __init__(self):
        if GapSymbol._instance is not None:
            raise ValueError(
                "Cannot instantiate this singleton more than one time"
            )
        else:
            GapSymbol._instance = self
    
    @staticmethod
    def instance():
        if GapSymbol._instance is None:
            GapSymbol._instance = GapSymbol()
        return GapSymbol._instance
    
    def __str__(self):
        return "-"
    
    def __hash__(self):
        return 0


def align_multiple(sequences, matrix, gap_penalty=-10, terminal_penalty=True,
                   distances=None, guide_tree=None):
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
    
    # Create guide tree
    # Template parameter workaround
    _T = sequences[0].code
    if distances is None:
        distances = _get_distance_matrix(
            _T, sequences, matrix, gap_penalty, terminal_penalty
        )
    else:
        distances = distances.astype(np.float32, copy=True)
    if guide_tree is None:
        guide_tree = upgma(distances)
    
    # Create new matrix with neutral gap symbol
    gap_symbol = GapSymbol.instance()
    new_alphabet = Alphabet(
        matrix.get_alphabet1().get_symbols() + [gap_symbol]
    )
    new_score_matrix = np.zeros(
        (len(new_alphabet), len(new_alphabet)), dtype=np.int32
    )
    # New substitution matrix is the same as the old one,
    # except the neutral ghap symbol,
    # that scores 0 with all other symbols
    new_score_matrix[:-1,:-1] = matrix.score_matrix()
    new_matrix = SubstitutionMatrix(
        new_alphabet, new_alphabet, new_score_matrix
    )

    # Progressive alignment
    gap_symbol_code = new_alphabet.encode(gap_symbol)
    order, aligned_seqs = _progressive_align(
        _T, sequences, guide_tree.root, distances, new_matrix,
        gap_symbol_code, gap_penalty, terminal_penalty
    )
    aligned_seq_codes = [seq.code for seq in aligned_seqs]

    # Remove neutral gap symbols and create actual trace
    seq_i = np.zeros(len(aligned_seqs))
    trace = np.full(
        (len(aligned_seqs[0]), len(aligned_seqs)), -1, dtype=np.int64)
    for j in range(trace.shape[1]):
        seq_code = aligned_seq_codes[j]
        seq_i = 0
        for i in range(trace.shape[0]):
            if seq_code[i] == gap_symbol_code:
                trace[i,j] = -1
            else:
                trace[i,j] = seq_i
                seq_i += 1
    aligned_seq_codes = [
        code[code != gap_symbol_code] for code in aligned_seq_codes
    ]
    for i in range(len(aligned_seqs)):
        aligned_seqs[i].code = aligned_seq_codes[i]
    
    # Reorder alignmets into original alignemnt
    new_order = np.argsort(order)
    aligned_seqs = [aligned_seqs[pos] for pos in new_order]
    trace = trace[:, new_order]

    return Alignment(aligned_seqs, trace), order, guide_tree, distances


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
            gap_open_count, gap_ext_count = _count_gaps(
                alignments[i,j].trace.astype(np.int64, copy=False),
                terminal_penalty
            )
            score_rand += gap_open_count * gap_open
            score_rand += gap_ext_count * gap_ext
            distances_v[i,j] = -log(
                (scores_v[i,j] - score_rand) / (score_max - score_rand)
            )
            # Pairwise distance matrix is symmetric
            distances_v[j,i] = distances_v[i,j]
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
    return gap_open_count, gap_ext_count


def _progressive_align(CodeType[:] _T, sequences, tree_node,
                       float32[:,:]distances_v, matrix,
                       int gap_symbol_code, gap_penalty, terminal_penalty):
    cdef int i=0, j=0
    cdef int i_min=0, j_min=0
    cdef float32 dist_min, dist
    cdef int32[:] indices1_v, indices2_v
    cdef np.ndarray incides1, incides2
    cdef list aligned_seqs1, aligned_seqs2
    
    if tree_node.is_terminal():
        # Copy sequences to avoid modification of input sequences
        # when neutral gap character is inserted
        return np.array([tree_node.index], dtype=np.int32), \
               [sequences[tree_node.index].copy()]
    
    else:
        child1, child2 = tree_node.childs
        incides1, aligned_seqs1 = _progressive_align(
            _T, sequences, child1, distances_v, matrix,
            gap_symbol_code, gap_penalty, terminal_penalty
        )
        indices1_v = incides1
        incides2, aligned_seqs2 = _progressive_align(
            _T, sequences, child2, distances_v, matrix,
            gap_symbol_code, gap_penalty, terminal_penalty
        )
        indices2_v = incides2
        # Find sequence pair with lowest distance
        dist_min = MAX_FLOAT
        for i in range(indices1_v.shape[0]):
            for j in range(indices2_v.shape[0]):
                dist = distances_v[indices1_v[i], indices2_v[j]]
                if dist < dist_min:
                    dist_min = dist
                    i_min = i
                    j_min = j
        # Alignment of sequence pair with lowest distance
        # For this method we only consider one alignment:
        alignment = align_optimal(
            aligned_seqs1[i_min], aligned_seqs2[j_min], matrix,
            gap_penalty, terminal_penalty, max_number=1
        )[0]
        # Place neutral gap symbol for position of new gaps
        # in both sequence groups 
        for i in range(len(aligned_seqs1)):
            seq = aligned_seqs1[i]
            seq.code = _replace_gaps(
                _T, alignment.trace[:,0], seq.code, gap_symbol_code
            )
        for i in range(len(aligned_seqs2)):
            seq = aligned_seqs2[i]
            seq.code = _replace_gaps(
                _T, alignment.trace[:,1], seq.code, gap_symbol_code
            )
        return np.append(incides1, incides2), \
               aligned_seqs1 + aligned_seqs2



def _replace_gaps(CodeType[:] _T,
                  int64[:] partial_trace_v,
                  np.ndarray seq_code,
                  int gap_symbol_code):
    cdef int i
    cdef int64 index
    cdef CodeType code

    cdef CodeType[:] seq_code_v = seq_code
    cdef np.ndarray new_seq_code = np.zeros(
        partial_trace_v.shape[0], dtype=seq_code.dtype
    )
    cdef CodeType[:] new_seq_code_v = new_seq_code
    
    for i in range(partial_trace_v.shape[0]):
        index = partial_trace_v[i]
        if index == -1:
            new_seq_code_v[i] = gap_symbol_code
        else:
            new_seq_code_v[i] = seq_code[index]
    
    return new_seq_code
